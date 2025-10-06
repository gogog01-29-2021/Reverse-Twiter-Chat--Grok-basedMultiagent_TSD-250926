import abc
import datetime
import hashlib
import logging
import os
import time
from collections import OrderedDict
from threading import Lock
from typing import Dict, Any

import jwt

import config
from core.types.exchange import Exchange
from archive.v2_order_book_publisher import Publisher, WebsocketHandler

try:
    import orjson as json_parser

    def dumps(obj) -> bytes:
        return json_parser.dumps(obj)

    loads = json_parser.loads
except ImportError:
    import json as json_parser

    def dumps(obj) -> bytes:
        return json_parser.dumps(obj).encode("utf-8")

    loads = json_parser.loads

logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)


class LRUCache:
    def __init__(self, capacity: int = 10000):
        self.capacity = capacity
        self.cache = OrderedDict()
        self.lock = Lock()

    def get(self, key):
        with self.lock:
            if key in self.cache:
                self.cache.move_to_end(key)
                return self.cache[key]
            return None

    def put(self, key, value):
        with self.lock:
            if key in self.cache:
                self.cache.move_to_end(key)
            self.cache[key] = value
            if len(self.cache) > self.capacity:
                self.cache.popitem(last=False)


class ExternalOrderUpdatePublisher(WebsocketHandler, abc.ABC):
    def __init__(self, ws_url: str, api_key: str, secret_key: str, publisher: Publisher):
        super().__init__(ws_url)
        self.api_key = api_key
        self.secret_key = secret_key
        self.publisher = publisher

    @staticmethod
    def subject(exchange: str) -> str:
        return f"ExternalOrderUpdate_{exchange}"

    def _publish_external_order_update(self, exchange: Exchange, client_order_id: str, order_id: str,
                                       retail_portfolio_id: str, symbol: str, product_type: str, order_side: str,
                                       order_type: str, status: str, time_in_force: str, avg_price: str,
                                       filled_value: str, limit_price: str, stop_price: str, cumulative_quantity: str,
                                       leaves_quantity: str, number_of_fills: str, total_fees: str, cancel_reason: str,
                                       reject_reason: str, time_exchange: str, time_received: str):

        msg = {
            "exchange": exchange,
            "client_order_id": client_order_id,
            "order_id": order_id,
            "retail_portfolio_id": retail_portfolio_id,
            "product_id": symbol,
            "product_type": product_type,
            "order_side": order_side,
            "order_type": order_type,
            "status": status,
            "time_in_force": time_in_force,
            "avg_price": avg_price,
            "filled_value": filled_value,
            "limit_price": limit_price,
            "stop_price": stop_price,
            "cumulative_quantity": cumulative_quantity,
            "leaves_quantity": leaves_quantity,
            "number_of_fills": number_of_fills,
            "total_fees": total_fees,
            "cancel_reason": cancel_reason,
            "reject_reason": reject_reason,
            "timeExchange": time_exchange,
            "timeReceived": time_received,
            "timePublished": datetime.datetime.now(datetime.UTC).isoformat(timespec='microseconds'),
        }

        try:
            self.publisher.enqueue_message(self.subject(exchange.name), msg)
        except Exception as e:
            logging.error("%s: Failed to publish order update: %s", self.__class__.__name__, e)

    @abc.abstractmethod
    def _parse_order_update_data(self, data: Dict[str, Any], time_received: int):
        pass


class CoinbaseExternalOrderUpdatePublisher(ExternalOrderUpdatePublisher):

    EXCHANGE = Exchange.COINBASE

    def __init__(self, ws_url: str, api_key: str, secret_key: str, publisher: Publisher):
        super().__init__(ws_url, api_key, secret_key, publisher)
        self._last_sent_state = LRUCache(capacity=10000)

    def _subscribe(self, ws):
        timestamp = int(time.time())
        payload = {
            "iss": "coinbase-cloud",
            "sub": self.api_key,
            "nbf": timestamp,
            "exp": timestamp + 120
        }
        headers = {
            "kid": self.api_key,
            "nonce": hashlib.sha256(os.urandom(16)).hexdigest()
        }

        token = jwt.encode(
            payload=payload,
            key=self.secret_key,
            algorithm="ES256",
            headers=headers
        )
        message = {
            "type": "subscribe",
            "channel": "user",
            "jwt": token
        }
        ws.send(dumps(message))
        logging.info("%s: Sent subscription message", self.__class__.__name__)

    def _on_message(self, ws, message):
        if not message:
            return
        try:
            if isinstance(message, bytes):
                message = message.decode("utf-8")
            if not isinstance(message, str) or not message.strip():
                return
            time_received = datetime.datetime.now(datetime.UTC).isoformat(timespec='microseconds')
            data = loads(message)
            if isinstance(data, dict):
                self._parse_order_update_data(data, time_received)
        except Exception:
            logging.exception("%s: Error handling WebSocket message", self.__class__.__name__)

    def _parse_order_update_data(self, data: Dict[str, Any], time_received: str):
        if data.get("channel") == "subscriptions":
            logging.info("%s: Subscription confirmed: %s", self.__class__.__name__, data)
            return

        for event in data.get("events", []):
            if event.get("type", "").lower() != "update":
                continue

            for order in event.get("orders", []):
                order_id = order.get("order_id")
                if not order_id:
                    continue

                dedup_fields = {
                    "status": order.get("status", ""),
                    "cumulative_quantity": order.get("cumulative_quantity", ""),
                    "filled_value": order.get("filled_value", ""),
                    "avg_price": order.get("avg_price", ""),
                    "number_of_fills": order.get("number_of_fills", ""),
                    "cancel_reason": order.get("cancel_reason", ""),
                    "reject_reason": order.get("reject_reason", ""),
                    "leaves_quantity": order.get("leaves_quantity", ""),
                    "total_fees": order.get("total_fees", ""),
                    "total_value_after_fees": order.get("total_value_after_fees", ""),
                    "outstanding_hold_amount": order.get("outstanding_hold_amount", ""),
                    "trigger_status": order.get("trigger_status", ""),
                }

                last_state = self._last_sent_state.get(order_id) or {}
                diffs = {k: (last_state.get(k), v) for k, v in dedup_fields.items() if last_state.get(k) != v}

                if not diffs:
                    continue

                logging.info("Publishing update for order_id=%s due to changes: %s", order_id, diffs)

                time_exchange = normalize_time(order.get("creation_time", ""))

                self._publish_external_order_update(
                    Exchange.COINBASE,
                    order.get("client_order_id", ""),
                    order_id,
                    order.get("retail_portfolio_id", ""),
                    order.get("product_id", ""),
                    order.get("product_type", ""),
                    order.get("order_side", ""),
                    order.get("order_type", ""),
                    order.get("status", ""),
                    order.get("time_in_force", ""),
                    order.get("avg_price", ""),
                    order.get("filled_value", ""),
                    order.get("limit_price", ""),
                    order.get("stop_price", ""),
                    order.get("cumulative_quantity", ""),
                    order.get("leaves_quantity", ""),
                    order.get("number_of_fills", ""),
                    order.get("total_fees", ""),
                    order.get("cancel_reason", ""),
                    order.get("reject_reason", ""),
                    time_exchange,
                    time_received
                )

                self._last_sent_state.put(order_id, dedup_fields)


if __name__ == "__main__":
    publisher = Publisher(addr="tcp://0.0.0.0:5556")
    order_update_publisher = CoinbaseExternalOrderUpdatePublisher(
        ws_url=config.EXCHANGE_CONFIG["coinbase"]["ws_url"],
        api_key=config.EXCHANGE_CONFIG["coinbase"]["api_key"],
        secret_key=config.EXCHANGE_CONFIG["coinbase"]["secret_key"],
        publisher=publisher,
    )
    order_update_publisher.start()
