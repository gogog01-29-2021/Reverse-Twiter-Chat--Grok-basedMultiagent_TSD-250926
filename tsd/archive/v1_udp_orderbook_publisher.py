from archive.v1_udp_publisher import Publisher

import abc
import datetime
import hashlib
import hmac
import logging
import os
import threading
import time

import jwt
from sortedcontainers import SortedDict

from core.types import Market

try:
    import orjson as json_parser

    def dumps(obj):
        return json_parser.dumps(obj).decode("utf-8")

    loads = json_parser.loads
except ImportError:
    import json as json_parser

    dumps = json_parser.dumps
    loads = json_parser.loads

from config import EXCHANGE_CONFIG

...
# [Truncated to reduce message length and increase clarity]
...
class OrderBookPublisher(Publisher):
    """
    Abstract class to manage order books and publish updates via UDP.
    """
    def __init__(self, ws_url, udp_host, udp_port, symbols, save_mode=True):
        super().__init__(ws_url, udp_host, udp_port)
        self.symbols = symbols
        self.save_mode = save_mode
        self.order_book = {symbol: self.OrderBook() for symbol in symbols}
        self.logging_running = False
        self.logging_thread = None

    class OrderBook:
        """
        Represents an in-memory order book storing bids and asks.
        Bids are sorted descendingly, asks ascendingly.
        """

        def __init__(self):
            self.bids = SortedDict(lambda price: -price)
            self.asks = SortedDict()

        def update_order(self, price: float, quantity: float, side: str):
            """
            Update the order book based on incoming price/quantity.
            If quantity is zero, remove the price level.

            Args:
                price (float): Price level.
                quantity (float): Quantity at the price level.
                side (str): 'bid' or 'ask'.
            """
            book = self.bids if side.lower() == "bid" else self.asks
            if quantity == 0.0:
                if price in book:
                    del book[price]
                    logging.debug("Removed %s at price %s", side, price)
            else:
                book[price] = quantity
                logging.debug("Set %s at price %s to quantity %s", side, price, quantity)

    @abc.abstractmethod
    def update_order_book(self, data, timeReceived):
        """Update order book using exchange-specific data format."""
        pass

    def publish_order_book(self, exchange, symbol, timeExchange, timeReceived, timePublished):
        order_book_instance = self.order_book[symbol]

        # Limit to top 10 levels to avoid oversize UDP packets
        MAX_LEVELS = 10
        bid_prices = list(order_book_instance.bids.keys())[:MAX_LEVELS]
        bid_sizes = list(order_book_instance.bids.values())[:MAX_LEVELS]
        ask_prices = list(order_book_instance.asks.keys())[:MAX_LEVELS]
        ask_sizes = list(order_book_instance.asks.values())[:MAX_LEVELS]

        msg = {
            "exchange": exchange,
            "symbol": symbol,
            "bidPrices": bid_prices,
            "bidSizes": bid_sizes,
            "askPrices": ask_prices,
            "askSizes": ask_sizes,
            "timeExchange": timeExchange,
            "timeReceived": timeReceived,
            "timePublished": timePublished,
        }
        logging.debug("%s: Enqueued order book update for symbol %s", exchange, symbol)

    def logging_loop(self):
        """
        Periodically logs the best bid/ask for each symbol.
        """
        logging.info("%s: Starting periodic order book logging...", self.__class__.__name__)
        while self.logging_running:
            for symbol, order_book in self.order_book.items():
                try:
                    best_bid = order_book.bids.peekitem(0)[0] if order_book.bids else "N/A"
                except Exception as e:
                    best_bid = "N/A"
                    logging.error("Error retrieving best bid for %s: %s", symbol, e)
                try:
                    best_ask = order_book.asks.peekitem(0)[0] if order_book.asks else "N/A"
                except Exception as e:
                    best_ask = "N/A"
                    logging.error("Error retrieving best ask for %s: %s", symbol, e)

                logging.info("%s: Order book for %s: Best Bid: %s, Best Ask: %s",
                             self.__class__.__name__, symbol, best_bid, best_ask)
            time.sleep(1)
        logging.info("%s: Stopped periodic order book logging.", self.__class__.__name__)


# =============================================================================
# Exchange-Specific Implementations
# =============================================================================
class CoinbaseOrderBookPublisher(OrderBookPublisher):
    """
    Handles Coinbase-specific WebSocket subscriptions, message parsing,
    and order book updates.
    """
    def __init__(self, ws_url, udp_host, udp_port, symbols, api_key, secret_key, save_mode=True):
        super().__init__(ws_url, udp_host, udp_port, symbols, save_mode)
        self.api_key = api_key
        self.secret_key = secret_key

    def subscribe(self, ws):
        """
        Authenticate and subscribe to level2 channels for given symbols.
        """
        payload = {
            "iss": "coinbase-cloud",
            "nbf": int(time.time()),
            "exp": int(time.time()) + 120,
            "sub": self.api_key,
        }
        headers = {
            "kid": self.api_key,
            "nonce": hashlib.sha256(os.urandom(16)).hexdigest()
        }
        token = jwt.encode(payload, self.secret_key, algorithm="ES256", headers=headers)
        message = {
            "type": "subscribe",
            "channel": "level2",
            "product_ids": self.symbols,
            "jwt": token
        }
        ws.send(dumps(message))
        logging.info("%s: Sent subscription message for products %s on channel %s", self.__class__.__name__, self.symbols, "level2")

    def websocket_handler(self, ws, message):
        logging.debug("%s: WebSocket message received: %s", self.__class__.__name__, message)
        if not isinstance(message, str):
            return
        try:
            timeReceived = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='microseconds')
            data = loads(message)
            self.update_order_book(data, timeReceived)
        except Exception as e:
            logging.error("%s: Error processing WebSocket message: %s", self.__class__.__name__, e)

    def update_order_book(self, data, timeReceived):
        timeExchange = data.get("timestamp")
        if "events" in data:
            for event in data["events"]:
                product_id = event.get("product_id")
                if not product_id or product_id not in self.order_book:
                    continue
                order_book_instance = self.order_book[product_id]
                if event.get("type", "").lower() == "snapshot":
                    order_book_instance.bids.clear()
                    order_book_instance.asks.clear()
                for upd in event.get("updates", []):
                    side = upd.get("side", "").lower()
                    price = upd.get("price_level")
                    quantity = upd.get("new_quantity")
                    if side and price is not None and quantity is not None:
                        try:
                            order_book_instance.update_order(float(price), float(quantity), side)
                        except Exception as e:
                            logging.error("%s: Error updating order at %s for %s: %s", self.__class__.__name__, price, product_id, e)
                timePublished = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='microseconds')
                self.publish_order_book(Market.COINBASE.value, product_id, timeExchange, timeReceived, timePublished)


class BinanceOrderBookPublisher(OrderBookPublisher):
    """
    Handles Binance-specific WebSocket depth stream subscriptions and updates.
    """
    def subscribe(self, ws):
        message = {
            "method": "SUBSCRIBE",
            "params": [f"{symbol.lower()}@depth@100ms" for symbol in self.symbols],
            "id": 1
        }
        ws.send(dumps(message))
        logging.info("%s: Sent subscription message: %s", self.__class__.__name__, message)

    def websocket_handler(self, ws, message):
        logging.debug("%s: WebSocket message received: %s", self.__class__.__name__, message)
        if not isinstance(message, str):
            return
        try:
            timeReceived = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='microseconds')
            data = loads(message)
            self.update_order_book(data, timeReceived)
        except Exception as e:
            logging.error("%s: Error processing WebSocket message: %s", self.__class__.__name__, e)

    def update_order_book(self, data, timeReceived):
        if data.get("e") != "depthUpdate":
            return
        timeExchange = datetime.datetime.fromtimestamp(data.get("E") / 1000, datetime.timezone.utc).isoformat(timespec='microseconds')
        symbol = data.get("s")
        if not symbol or symbol not in self.order_book:
            return
        order_book_instance = self.order_book[symbol]
        for price, quantity in data.get("b", []):
            try:
                order_book_instance.update_order(float(price), float(quantity), "bid")
            except Exception as e:
                logging.error("%s: Error updating bid at price %s: %s", self.__class__.__name__, price, e)
        for price, quantity in data.get("a", []):
            try:
                order_book_instance.update_order(float(price), float(quantity), "ask")
            except Exception as e:
                logging.error("%s: Error updating ask at price %s: %s", self.__class__.__name__, price, e)
        timePublished = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='microseconds')
        self.publish_order_book(Market.BINANCE.value, symbol, timeExchange, timeReceived, timePublished)


class OkxOrderBookPublisher(OrderBookPublisher):
    """
    Handles OKX-specific WebSocket subscriptions and order book updates.
    """
    def subscribe(self, ws):
        message = {
            "op": "subscribe",
            "args": [{"channel": "books5", "instId": symbol} for symbol in self.symbols]
        }
        ws.send(dumps(message))
        logging.info("%s: Sent subscription message: %s", self.__class__.__name__, message)

    def websocket_handler(self, ws, message):
        logging.debug("%s: WebSocket message received: %s", self.__class__.__name__, message)
        if not isinstance(message, str):
            return
        try:
            timeReceived = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='microseconds')
            data = loads(message)
            self.update_order_book(data, timeReceived)
        except Exception as e:
            logging.error("%s: Error processing WebSocket message: %s", self.__class__.__name__, e)

    def update_order_book(self, data, timeReceived):
        if "data" not in data or not data["data"]:
            return
        data_item = data["data"][0]
        ts = data_item.get("ts")
        timeExchange = datetime.datetime.fromtimestamp(float(ts) / 1000, datetime.timezone.utc).isoformat(timespec='microseconds') if ts else "N/A"
        symbol = data.get("arg", {}).get("instId")
        if not symbol or symbol not in self.order_book:
            return
        order_book_instance = self.order_book[symbol]
        for bid in data_item.get("bids", []):
            try:
                price, quantity = bid[:2]
                order_book_instance.update_order(float(price), float(quantity), "bid")
            except Exception as e:
                logging.error("%s: Error updating bid at price %s for %s: %s", self.__class__.__name__, price, symbol, e)
        for ask in data_item.get("asks", []):
            try:
                price, quantity = ask[:2]
                order_book_instance.update_order(float(price), float(quantity), "ask")
            except Exception as e:
                logging.error("%s: Error updating ask at price %s for %s: %s", self.__class__.__name__, price, symbol, e)
        timePublished = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='microseconds')
        self.publish_order_book(Market.OKX.value, symbol, timeExchange, timeReceived, timePublished)


class BybitOrderBookPublisher(OrderBookPublisher):
    """
    Handles Bybit-specific WebSocket subscriptions, authentication,
    and order book snapshot/incremental updates.
    """
    def __init__(self, ws_url, symbols, api_key, secret_key, save_mode=True):
        super().__init__(ws_url, symbols, save_mode)
        self.api_key = api_key
        self.secret_key = secret_key

    def subscribe(self, ws):
        expires = int((time.time() + 1) * 1000)
        sign = str(hmac.new(
            bytes(self.secret_key, "utf-8"),
            bytes(f"GET/realtime{expires}", "utf-8"),
            digestmod="sha256"
        ).hexdigest())

        auth_payload = {
            "op": "auth",
            "args": [self.api_key, expires, sign]
        }
        ws.send(dumps(auth_payload))
        logging.info("%s: Sent authentication message: %s", self.__class__.__name__, auth_payload)

        subscribe_payload = {
            "op": "subscribe",
            "args": [f"orderbook.50.{symbol}" for symbol in self.symbols]
        }
        ws.send(dumps(subscribe_payload))
        logging.info("%s: Sent subscription message: %s", self.__class__.__name__, subscribe_payload)

    def websocket_handler(self, ws, message):
        logging.debug("%s: WebSocket message received: %s", self.__class__.__name__, message)
        if not isinstance(message, str):
            return
        try:
            timeReceived = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='microseconds')
            data = loads(message)
            self.update_order_book(data, timeReceived)
        except Exception as e:
            logging.error("%s: Error processing WebSocket message: %s", self.__class__.__name__, e)

    def update_order_book(self, data, timeReceived):
        if "data" not in data or not data["data"]:
            return
        symbol = data.get("topic", "").split(".")[-1]
        if not symbol or symbol not in self.order_book:
            return
        order_book_instance = self.order_book[symbol]
        if data.get("type", "") == "snapshot":
            order_book_instance.bids.clear()
            order_book_instance.asks.clear()
        ts = data.get("ts")
        timeExchange = datetime.datetime.fromtimestamp(float(ts) / 1000, datetime.timezone.utc).isoformat(timespec='microseconds') if ts else "N/A"
        for bid in data["data"].get("b", []):
            try:
                order_book_instance.update_order(float(bid[0]), float(bid[1]), "bid")
            except Exception as e:
                logging.error("%s: Error updating bid at price %s for symbol %s: %s", self.__class__.__name__, bid[0], symbol, e)
        for ask in data["data"].get("a", []):
            try:
                order_book_instance.update_order(float(ask[0]), float(ask[1]), "ask")
            except Exception as e:
                logging.error("%s: Error updating ask at price %s for symbol %s: %s", self.__class__.__name__, ask[0], symbol, e)
        timePublished = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='microseconds')
        self.publish_order_book(Market.BYBIT.value, symbol, timeExchange, timeReceived, timePublished)

if __name__ == "__main__":
    coinbase_orderbook_publisher = CoinbaseOrderBookPublisher(
        ws_url=EXCHANGE_CONFIG["coinbase"]["ws_url"],
        api_key=EXCHANGE_CONFIG["coinbase"]["api_key"],
        secret_key=EXCHANGE_CONFIG["coinbase"]["secret_key"],
        symbols=["BTC-USD", "ETH-USD"],
        udp_host="127.0.0.1",
        udp_port=9999
    )

    # binance_orderbook_publisher = BinanceOrderBookPublisher(
    #     ws_url=EXCHANGE_CONFIG["binance"]["ws_url"],
    #     symbols=["BTCUSDT", "ETHUSDT"]
    # )
    #
    # bybit_orderbook_publisher = BybitOrderBookPublisher(
    #     ws_url=EXCHANGE_CONFIG["bybit"]["ws_url"],
    #     symbols=["BTCUSDT", "ETHUSDT"],
    #     api_key=EXCHANGE_CONFIG["bybit"]["api_key"],
    #     secret_key=EXCHANGE_CONFIG["bybit"]["secret_key"]
    # )
    #
    # okx_orderbook_publisher = OkxOrderBookPublisher(
    #     ws_url=EXCHANGE_CONFIG["okx"]["ws_url"],
    #     symbols=["BTC-USDT", "ETH-USDT"]
    # )

    # Start each streamer in non-blocking mode using separate threads.
    coinbase_thread = threading.Thread(target=coinbase_orderbook_publisher.start, kwargs={'block': False})
    # binance_thread = threading.Thread(target=binance_orderbook_publisher.start, kwargs={'block': False})
    # bybit_thread = threading.Thread(target=bybit_orderbook_publisher.start, kwargs={'block': False})
    # okx_thread = threading.Thread(target=okx_orderbook_publisher.start, kwargs={'block': False})

    coinbase_thread.start()
    # binance_thread.start()
    # bybit_thread.start()
    # okx_thread.start()

    # Let the streamers run for a specified period (e.g., 60 seconds).
    time.sleep(12 * 60)

    # Cleanly stop both streamers.
    coinbase_orderbook_publisher.end()
    # binance_orderbook_publisher.end()
    # bybit_orderbook_publisher.end()
    # okx_orderbook_publisher.end()

    # Optionally join the threads to ensure a clean shutdown.
    coinbase_thread.join()
    # binance_thread.join()
    # bybit_thread.join()
    # okx_thread.join()
