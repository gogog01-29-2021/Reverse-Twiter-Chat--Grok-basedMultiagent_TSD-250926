import hashlib
import logging
import os
import threading
import time
from abc import ABC, abstractmethod
from typing import List, Tuple

import jwt
from cachetools import LRUCache
from orjson import dumps, loads

from config import get_exchange_credentials, TransportAddressBuilder
from config.routing_config import AddressBuilder
from core.types.exchange import Exchange
from dsm.core.pubsub_base import Publisher, PubTransport, MessageParser, ZmqPubTransport
from dsm.utils.conversion_utils import iso_to_epoch_ms, now_epoch_ms

logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)


class OrderUpdateParser(MessageParser, ABC):
    pass


class CoinbaseOrderUpdateParser(OrderUpdateParser):
    def __init__(self, exchange: Exchange):
        self.exchange = exchange
        self._last_sent_state = LRUCache(maxsize=10000)

    def parse(self, raw_message: str, time_received: int) -> List[Tuple[str, dict]]:
        updates = []
        data = loads(raw_message)
        if data.get("channel") == "subscriptions":
            logging.info("CoinbaseOrderUpdateParser: Subscription confirmed: %s", data)
            return []

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

                logging.info("CoinbaseOrderUpdateParser: Publishing update for order_id=%s with changes: %s",
                             order_id, diffs)

                msg = {
                    "exchange": self.exchange.value,
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
                    "time_published": now_epoch_ms(),
                }

                topic = f"ExternalOrderUpdate_{self.exchange.value}"
                updates.append((topic, msg))
                self._last_sent_state[order_id] = dedup_fields

        return updates


class ExecutionReportPublisher(Publisher, ABC):
    def __init__(
            self,
            ws_url: str,
            exchange: Exchange,
            transport: PubTransport,
            parser: MessageParser,
            verbose: bool = False,
            debug: bool = False,
            log_on_change: bool = False
    ):
        self._exchange = exchange
        super().__init__(ws_url, transport, parser, verbose, debug, log_on_change)

    @staticmethod
    def topic(exchange: Exchange) -> str:
        return f"ExecutionReport_{exchange.value}"

    @abstractmethod
    def on_open(self, ws):
        pass


class CoinbaseExecutionReportPublisher(ExecutionReportPublisher):
    def __init__(self, transport: PubTransport, api_key: str, secret_key: str,
                 verbose: bool = False, debug: bool = False, log_on_change: bool = False):
        self._api_key = api_key
        self._secret_key = secret_key
        parser = CoinbaseOrderUpdateParser(Exchange.COINBASE)
        super().__init__(
            ws_url=AddressBuilder.websocket(Exchange.COINBASE),
            exchange=Exchange.COINBASE,
            transport=transport,
            parser=parser,
            verbose=verbose,
            debug=debug,
            log_on_change=log_on_change
        )

    def on_open(self, ws):
        timestamp = int(time.time())
        payload = {
            "iss": "coinbase-cloud",
            "sub": self._api_key,
            "nbf": timestamp,
            "exp": timestamp + 120
        }
        headers = {
            "kid": self._api_key,
            "nonce": hashlib.sha256(os.urandom(16)).hexdigest()
        }
        token = jwt.encode(payload, self._secret_key, algorithm="ES256", headers=headers)
        ws.send(dumps({
            "type": "subscribe",
            "channel": "user",
            "jwt": token
        }))
        logging.info("%s: Sent subscription message", self.__class__.__name__)


if __name__ == "__main__":
    coinbase = CoinbaseExecutionReportPublisher(
        transport=ZmqPubTransport(
            zmq_bind_addr=TransportAddressBuilder.external_order_update(Exchange.COINBASE, "tcp://0.0.0.0"),
            debug=True),
        api_key=get_exchange_credentials(Exchange.COINBASE)["api_key"],
        secret_key=get_exchange_credentials(Exchange.COINBASE)["secret_key"],
        verbose=True,
        debug=False,
        log_on_change=True
    )

    thread = threading.Thread(target=coinbase.start)
    thread.start()

    try:
        time.sleep(600)
    finally:
        coinbase.stop()
        thread.join()
