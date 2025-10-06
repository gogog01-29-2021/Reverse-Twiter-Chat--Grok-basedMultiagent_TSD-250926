import abc
import hashlib
import logging
import os
import threading
import time
from typing import Dict, Any

import jwt
from cachetools import LRUCache
from orjson import dumps, loads

import config
from core.types.exchange import Exchange
from dsm.utils.conversion_utils import iso_to_epoch_ms, now_epoch_ms
from archive.v3_order_book_publisher import Publisher

logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)


class ExternalOrderUpdatePublisher(Publisher, abc.ABC):
    """
    Base class for publishing external order updates over ZMQ.
    """

    def __init__(self, ws_url: str, zmq_addr: str, api_key: str, secret_key: str, verbose: bool, debug: bool = False,
                 log_on_change: bool = False):
        super().__init__(ws_url, zmq_addr, verbose, debug, log_on_change)
        self.api_key = api_key
        self.secret_key = secret_key

    @staticmethod
    def subject(exchange: Exchange) -> str:
        return f"ExternalOrderUpdate_{exchange.value}"

    def publish_order_update(self, exchange: Exchange, msg: dict):
        msg["time_published"] = now_epoch_ms()
        try:
            self.publish(self.subject(exchange), msg)
        except Exception as e:
            logging.error("%s: Failed to publish order update: %s", self.__class__.__name__, e)

    def _on_message(self, ws, message):
        if isinstance(message, bytes):
            message = message.decode()

        if not message or not isinstance(message, str):
            return

        try:
            data = loads(message)
            time_received = now_epoch_ms()
            if isinstance(data, dict):
                self._handle_order_update(data, time_received)
        except Exception:
            logging.exception("%s: Failed to process WebSocket message", self.__class__.__name__)

    @abc.abstractmethod
    def _handle_order_update(self, data: Dict[str, Any], time_received: int):
        pass


class CoinbaseExternalOrderUpdatePublisher(ExternalOrderUpdatePublisher):
    """
    Publisher for Coinbase external order updates.
    """

    EXCHANGE = Exchange.COINBASE

    def __init__(self, ws_url: str, zmq_addr: str, api_key: str, secret_key: str, verbose: bool = False,
                 debug: bool = False, log_on_change: bool = False):
        super().__init__(ws_url, zmq_addr, api_key, secret_key, verbose, debug, log_on_change)
        self._last_sent_state = LRUCache(maxsize=10000)

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

        token = jwt.encode(payload, self.secret_key, algorithm="ES256", headers=headers)
        ws.send(dumps({
            "type": "subscribe",
            "channel": "user",
            "jwt": token
        }))
        logging.info("%s: Sent subscription message", self.__class__.__name__)

    def _handle_order_update(self, data: Dict[str, Any], time_received: int):
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
                    "status": order.get("status"),
                    "cumulative_quantity": order.get("cumulative_quantity"),
                    "filled_value": order.get("filled_value"),
                    "avg_price": order.get("avg_price"),
                    "number_of_fills": order.get("number_of_fills"),
                    "cancel_reason": order.get("cancel_reason"),
                    "reject_reason": order.get("reject_reason"),
                    "leaves_quantity": order.get("leaves_quantity"),
                    "total_fees": order.get("total_fees"),
                }

                last_state = self._last_sent_state.get(order_id, {})
                diffs = {k: (last_state.get(k), v) for k, v in dedup_fields.items() if last_state.get(k) != v}
                if not diffs:
                    continue

                logging.info("%s: Publishing update for order_id=%s with changes: %s",
                             self.__class__.__name__, order_id, diffs)

                msg = {
                    "exchange": self.EXCHANGE.value,
                    "client_order_id": order.get("client_order_id", ""),
                    "order_id": order_id,
                    "retail_portfolio_id": order.get("retail_portfolio_id", ""),
                    "product_id": order.get("product_id", ""),
                    "product_type": order.get("product_type", ""),
                    "order_side": order.get("order_side", ""),
                    "order_type": order.get("order_type", ""),
                    "status": order.get("status", ""),
                    "time_in_force": order.get("time_in_force", ""),
                    "avg_price": order.get("avg_price", ""),
                    "filled_value": order.get("filled_value", ""),
                    "limit_price": order.get("limit_price", ""),
                    "stop_price": order.get("stop_price", ""),
                    "cumulative_quantity": order.get("cumulative_quantity", ""),
                    "leaves_quantity": order.get("leaves_quantity", ""),
                    "number_of_fills": order.get("number_of_fills", ""),
                    "total_fees": order.get("total_fees", ""),
                    "cancel_reason": order.get("cancel_reason", ""),
                    "reject_reason": order.get("reject_reason", ""),
                    "time_exchange": iso_to_epoch_ms(order.get("creation_time", "")),
                    "time_received": time_received,
                }

                self.publish_order_update(self.EXCHANGE, msg)
                self._last_sent_state[order_id] = dedup_fields


if __name__ == "__main__":
    coinbase = CoinbaseExternalOrderUpdatePublisher(
        ws_url=config.EXCHANGE_CONFIG["coinbase"]["ws_url"],
        zmq_addr="tcp://0.0.0.0:5556",
        api_key=config.EXCHANGE_CONFIG["coinbase"]["api_key"],
        secret_key=config.EXCHANGE_CONFIG["coinbase"]["secret_key"],
        verbose=True,
        debug=True,
        log_on_change=True
    )

    thread = threading.Thread(target=coinbase.start, kwargs={"block": False})
    thread.start()

    try:
        time.sleep(600)
    finally:
        coinbase.end()
        thread.join()
