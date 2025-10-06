import abc
from abc import abstractmethod
import hashlib
import logging
import os
import threading
import time

import jwt

import config
from core.types.exchange import Exchange
from core.utils.timeutils import datetime_str_to_nanoseconds

MAX_LEVELS = 10

logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)

from archive.v1_nng_market_data_publisher import Publisher

try:
    import orjson as json_parser


    def dumps(obj):
        return json_parser.dumps(obj).decode("utf-8")


    loads = json_parser.loads
except ImportError:
    import json as json_parser

    dumps = json_parser.dumps
    loads = json_parser.loads


class OrderUpdatePublisher(Publisher, abc.ABC):
    def __init__(self, ws_url, api_key, secret_key):
        super().__init__(ws_url)
        self.api_key = api_key
        self.secret_key = secret_key

    def _on_message(self, ws, message):
        """Receive and parse incoming WebSocket message as market data."""
        logging.debug("%s: WebSocket message received: %s", self.__class__.__name__, message)
        if not isinstance(message, str):
            return
        try:
            timeReceived = time.time_ns()
            data = loads(message)
            self._parse_order_update_data(data, timeReceived, max_levels=MAX_LEVELS)
        except Exception as e:
            logging.error("%s: Error processing WebSocket message: %s", self.__class__.__name__, e)

    @abstractmethod
    def _parse_order_update_data(self, data, timeReceived, max_levels):
        """Parse and extract structured data from raw market feed."""
        pass

    @staticmethod
    def subject(exchange, symbol):
        """Generate NNG topic string."""
        return f"OrderUpdate_{exchange}_{symbol}"

    def publish_order_update_data(self,
                                  exchange,
                                  client_order_id,
                                  order_id,
                                  retail_portfolio_id,
                                  symbol,
                                  product_type,
                                  order_side,
                                  order_type,
                                  status,
                                  time_in_force,
                                  avg_price,
                                  filled_value,
                                  limit_price,
                                  stop_price,
                                  cumulative_quantity,
                                  leaves_quantity,
                                  number_of_fills,
                                  total_fees,
                                  cancel_reason,
                                  reject_Reason,
                                  time_exchange,
                                  time_received):
        """Publish a parsed order update message."""
        subject = self.subject(exchange, symbol)

        msg = {
            "exchange": exchange,
            "client_order_id": client_order_id,
            "order_id": order_id,
            "retail_portfolio_id": retail_portfolio_id,
            "product_id": symbol, # "BTC-USD"
            "product_type": product_type, # "SPOT" TODO: need to support future
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
            "reject_Reason": reject_Reason,
            "timeExchange": time_exchange, # "creation_time"
            "timeReceived": time_received,
            "timePublished": time.time_ns(),
        }

        self.publisher_thread.publish(subject, msg)


class CoinbaseOrderUpdatePublisher(OrderUpdatePublisher):

    def _parse_order_update_data(self, data, time_received, max_levels):
        time_exchange = datetime_str_to_nanoseconds(data.get("creation_time"), format="%Y-%m-%dT%H:%M:%S.%fZ")

        if "events" in data:
            for event in data["events"]:
                if event.get("type", "").lower() == "update":
                    for order in event.get("orders", []):

                        self.publish_order_update_data(
                            Exchange.COINBASE,
                            order.get("client_order_id", ""),
                            order.get("order_id", ""),
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
                            order.get("reject_Reason", ""),
                            time_exchange,
                            time_received)

    def _subscribe(self, ws):
        """
        Authenticate and subscribe to user channels for given symbols.
        """
        timestamp = int(time.time())
        payload = {
            "iss": "coinbase-cloud",
            "nbf": timestamp,
            "exp": timestamp + 120,
            "sub": self.api_key,
        }
        headers = {
            "kid": self.api_key,
            "nonce": hashlib.sha256(os.urandom(16)).hexdigest()
        }
        token = jwt.encode(payload, self.secret_key, algorithm="ES256", headers=headers)
        message = {
            "type": "subscribe",
            "channel": "user",
            "jwt": token
        }
        ws.send(dumps(message))
        logging.info("%s: Sent subscription message on channel %s", self.__class__.__name__, "user")


if __name__ == "__main__":
    coinbase_market_data_publisher = CoinbaseOrderUpdatePublisher(
        ws_url="wss://advanced-trade-ws.coinbase.com",
        api_key=config.EXCHANGE_CONFIG["coinbase"]["api_key"],
        secret_key=config.EXCHANGE_CONFIG["coinbase"]["secret_key"]
    )

    coinbase_market_data_publisher.wait_for_subscriber_handshake(control_url="tcp://127.0.0.1:6666")

    coinbase_thread = threading.Thread(target=coinbase_market_data_publisher.start, kwargs={'block': False})
    coinbase_thread.start()

    time.sleep(60)
    coinbase_market_data_publisher.end()
    coinbase_thread.join()