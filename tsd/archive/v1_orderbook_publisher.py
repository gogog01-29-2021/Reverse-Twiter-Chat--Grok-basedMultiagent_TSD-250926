import abc
import datetime
import hashlib
import hmac
import logging
import os
import threading
import time
import pandas as pd
import queue
from collections import defaultdict

import jwt
from sortedcontainers import SortedDict

from config import EXCHANGE_CONFIG
from archive.v1_messaging import Publisher, loads, dumps
from core.types.exchange import Exchange
from core.types.instrument import Spot, Currency
from core.types.exchange.utils import instrument_to_symbol

MAX_LEVELS = 10


class OrderBookPublisher(Publisher):
    """
    Abstract base class for publishing real-time order book data from various exchanges.

    Responsibilities:
    - Maintains in-memory order book state (bids and asks) for each subscribed symbol.
    - Interfaces with WebSocket data feed.
    - Publishes top-of-book snapshots to NATS (or another messaging backend).
    - Optionally persists snapshots to disk at periodic intervals for historical analysis.

    Args:
        ws_url (str): WebSocket URL for the exchange feed.
        symbols (List[str]): List of trading symbols to track.
        save_mode (bool): If True, enables disk-based snapshot storage.
    """

    def __init__(self, ws_url, symbols, save_mode=True):
        super().__init__(ws_url)
        self.symbols = symbols
        self.save_mode = save_mode
        self.order_book = {symbol: self.OrderBook() for symbol in symbols}

        if self.save_mode:
            self._save_queue = queue.Queue()
            self.save_buffers = defaultdict(list)  # Buffer to group snapshots by 10-minute intervals
            self.saving_shutdown_event = threading.Event()
            self._saving_thread = threading.Thread(target=self.saving_loop, daemon=True)
            self._saving_thread.start()

    class OrderBook:
        """
        Represents an in-memory order book for a single trading symbol.

        Attributes:
            bids (SortedDict): Bid side of the book, sorted in descending order.
            asks (SortedDict): Ask side of the book, sorted in ascending order.
        """

        def __init__(self):
            self.bids = SortedDict(lambda price: -price)
            self.asks = SortedDict()

        def update_order(self, price: float, quantity: float, side: str):
            """
            Insert or update a price level in the order book, or remove it if quantity is 0.

            Args:
                price (float): Price level.
                quantity (float): Quantity at the given price level.
                side (str): 'bid' or 'ask'.
            """
            book = self.bids if side.lower() == "bid" else self.asks
            if quantity == 0.0:
                book.pop(price, None)  # Remove price level if it exists
                logging.debug("Removed %s at price %s", side, price)
            else:
                book[price] = quantity
                logging.debug("Set %s at price %s to quantity %s", side, price, quantity)

    @abc.abstractmethod
    def update_order_book(self, data, timeReceived):
        """
        Abstract method for handling incoming WebSocket data and updating the order book.

        This must be implemented by each exchange-specific subclass.

        Args:
            data (dict): Parsed message from WebSocket.
            timeReceived (int): Timestamp (ns) when the message was received.
        """
        pass

    @staticmethod
    def subject(exchange, symbol):
        """
        Construct a subject name for publishing order book updates.

        Args:
            exchange (str): Exchange name.
            symbol (str): Trading symbol.

        Returns:
            str: Formatted subject string.
        """
        return f"ORDERBOOK_{exchange}_{symbol}"

    def on_message(self, ws, message):
        """
        Handler for incoming WebSocket messages.

        Parses the message and passes it to the update_order_book method.

        Args:
            ws: WebSocket connection.
            message (str): Raw message string from WebSocket.
        """
        logging.debug("%s: WebSocket message received: %s", self.__class__.__name__, message)
        if not isinstance(message, str):
            return
        try:
            timeReceived = time.time_ns()
            data = loads(message)
            self.update_order_book(data, timeReceived)
        except Exception as e:
            logging.error("%s: Error processing WebSocket message: %s", self.__class__.__name__, e)

    def publish_order_book(self, exchange, symbol, timeExchange, timeReceived, timePublished, max_levels=MAX_LEVELS):
        """
        Publish a top-of-book snapshot to the messaging system and optionally to disk.

        Args:
            exchange (str): Exchange name.
            symbol (str): Trading symbol.
            timeExchange (int): Exchange-provided timestamp (ns).
            timeReceived (int): Local receipt timestamp (ns).
            timePublished (int): Time of publication (ns).
            max_levels (int): Maximum depth of book to publish.
        """
        ob = self.order_book[symbol]
        subject = self.subject(exchange, symbol)

        msg = {
            "exchange": exchange,
            "symbol": symbol,
            "bidPrices": list(ob.bids.keys())[:max_levels],
            "bidSizes": list(ob.bids.values())[:max_levels],
            "askPrices": list(ob.asks.keys())[:max_levels],
            "askSizes": list(ob.asks.values())[:max_levels],
            "timeExchange": timeExchange,
            "timeReceived": timeReceived,
            "timePublished": timePublished,
        }

        self.publisher_thread.publish(subject, msg)
        self.save_order_book(exchange, symbol, timeExchange, timeReceived, timePublished)
        logging.debug("%s: Enqueued order book update for symbol %s", exchange, symbol)

    def logging_loop(self):
        """
        Periodically logs best bid/ask for each tracked symbol. Useful for live diagnostics.
        """
        logging.info("%s: Starting periodic order book logging...", self.__class__.__name__)
        while self.logging_running:
            for symbol, ob in self.order_book.items():
                try:
                    best_bid = ob.bids.peekitem(0)[0] if ob.bids else "N/A"
                except Exception as e:
                    best_bid = "N/A"
                    logging.error("Error retrieving best bid for %s: %s", symbol, e)
                try:
                    best_ask = ob.asks.peekitem(0)[0] if ob.asks else "N/A"
                except Exception as e:
                    best_ask = "N/A"
                    logging.error("Error retrieving best ask for %s: %s", symbol, e)

                logging.info("%s: Order book for %s: Best Bid: %s, Best Ask: %s",
                             self.__class__.__name__, symbol, best_bid, best_ask)
            time.sleep(1)
        logging.info("%s: Stopped periodic order book logging.", self.__class__.__name__)

    def save_order_book(self, exchange, symbol, timeExchange, timeReceived, timePublished, max_levels=MAX_LEVELS):
        """
        Adds a snapshot to the internal buffer queue for periodic saving.

        Snapshots are grouped in 10-minute intervals for efficient disk writes.
        """
        if not self.save_mode:
            return

        ob = self.order_book[symbol]

        # Bucket snapshot into 10-minute aligned intervals
        ts = datetime.datetime.utcfromtimestamp(timeExchange / 1e9).replace(second=0, microsecond=0)
        bucket_start = ts.replace(minute=(ts.minute // 10) * 10)
        bucket_end = bucket_start + datetime.timedelta(minutes=10)

        snapshot = {
            "timeExchange": timeExchange,
            "timeReceived": timeReceived,
            "timePublished": timePublished,
            "bidPrices": list(ob.bids.keys())[:max_levels],
            "bidSizes": list(ob.bids.values())[:max_levels],
            "askPrices": list(ob.asks.keys())[:max_levels],
            "askSizes": list(ob.asks.values())[:max_levels],
        }

        self._save_queue.put((exchange, symbol, bucket_start, bucket_end, snapshot))

    def saving_loop(self):
        """
        Background thread that monitors the save queue and periodically flushes
        data to disk in 10-minute buckets.
        """
        logging.info("Saving thread started...")
        while not self.saving_shutdown_event.is_set():
            try:
                exchange, symbol, bucket_start, bucket_end, snapshot = self._save_queue.get(timeout=1)
                key = (exchange, symbol, bucket_start, bucket_end)
                self.save_buffers[key].append(snapshot)
            except queue.Empty:
                pass

            now = datetime.datetime.utcnow()
            flushable_keys = [key for key in self.save_buffers if now >= key[3]]

            for key in flushable_keys:
                self.flush_buffer(key)

        logging.info("Saving thread shutting down. Final flush...")
        self.flush_all_buffers()

    def flush_buffer(self, key):
        """
        Write buffered snapshots for a given time bucket to disk as a Parquet file.

        Args:
            key (tuple): (exchange, symbol, bucket_start, bucket_end)
        """
        exchange, symbol, bucket_start, bucket_end = key
        records = self.save_buffers.get(key, [])
        if not records:
            return

        df = pd.DataFrame(records)
        data_dir = os.path.join(os.getcwd(), "data")
        os.makedirs(data_dir, exist_ok=True)

        filename = (
            f"{exchange}_{symbol}_"
            f"{bucket_start.strftime('%Y-%m-%d_%H-%M-%S')}_"
            f"{bucket_end.strftime('%Y-%m-%d_%H-%M-%S')}.parquet"
        )
        filepath = os.path.join(data_dir, filename)
        df.to_parquet(filepath, index=False)

        logging.info(f"Flushed {len(records)} records to {filepath}")
        del self.save_buffers[key]

    def flush_all_buffers(self):
        """
        Flush all remaining data in memory to disk, regardless of time bucket.
        Useful during shutdown to ensure no data loss.
        """
        for key in list(self.save_buffers.keys()):
            self.flush_buffer(key)

    def shutdown_saving(self):
        """
        Gracefully shut down the saving thread and flush remaining buffers.
        """
        if self.save_mode:
            self.saving_shutdown_event.set()
            self._saving_thread.join()
            logging.info("%s: Saving thread shut down gracefully.", self.__class__.__name__)

    def end(self):
        """
        Clean shutdown procedure for the order book publisher:
        - Stops WebSocket client
        - Terminates background threads
        - Flushes any unsaved data
        """
        logging.info("%s: Stopping publisher", self.__class__.__name__)
        self.stop_logging()

        if self.ws_app:
            self.ws_app.keep_running = False
            self.ws_app.close()

        self.publisher_thread.stop()

        if self.ws_thread:
            self.ws_thread.join(timeout=2)

        self.loop.call_soon_threadsafe(self.loop.stop)
        self.loop_thread.join(timeout=2)

        self.shutdown_saving()

        logging.info("%s: Publisher stopped", self.__class__.__name__)

# =============================================================================
# Exchange-Specific Implementations
# =============================================================================
class CoinbaseOrderBookPublisher(OrderBookPublisher):
    """
    Order book publisher for Coinbase exchange.

    Handles JWT-based authentication, subscription to the level2 feed,
    and parsing of snapshot/update messages.
    """

    def __init__(self, ws_url, symbols, api_key, secret_key, save_mode=True):
        super().__init__(ws_url, symbols, save_mode)
        self.api_key = api_key
        self.secret_key = secret_key

    def subscribe(self, ws):
        """
        Authenticate and subscribe to level2 channels for specified symbols.
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
            "channel": "level2",
            "product_ids": self.symbols,
            #"jwt": token
        }
        ws.send(dumps(message))
        logging.info("%s: Subscribed to Coinbase level2 for symbols: %s", self.__class__.__name__, self.symbols)

    def update_order_book(self, data, timeReceived):
        timestamp_str = data.get("timestamp")
        try:
            timestamp_str = timestamp_str.rstrip("Z")
            if "." in timestamp_str:
                dt_str, frac = timestamp_str.split(".")
                frac = frac[:6].ljust(6, "0")
                timestamp_str = f"{dt_str}.{frac}"
                dt = datetime.datetime.strptime(timestamp_str, "%Y-%m-%dT%H:%M:%S.%f")
            else:
                dt = datetime.datetime.strptime(timestamp_str, "%Y-%m-%dT%H:%M:%S")
            dt = dt.replace(tzinfo=datetime.timezone.utc)
            timeExchange = int(dt.timestamp() * 1e9)
        except Exception as e:
            logging.error("%s: Failed to parse timestamp '%s': %s", self.__class__.__name__, timestamp_str, e)
            timeExchange = time.time_ns()

        for event in data.get("events", []):
            product_id = event.get("product_id")
            if product_id not in self.order_book:
                continue
            ob = self.order_book[product_id]
            if event.get("type") == "snapshot":
                ob.bids.clear()
                ob.asks.clear()
            for upd in event.get("updates", []):
                side = upd.get("side")
                price = upd.get("price_level")
                quantity = upd.get("new_quantity")
                if side and price and quantity:
                    try:
                        ob.update_order(float(price), float(quantity), side)
                    except Exception as e:
                        logging.error("%s: Failed to update %s %s: %s", self.__class__.__name__, side, price, e)
            self.publish_order_book(Exchange.COINBASE.value, product_id, timeExchange, timeReceived, time.time_ns())


class BinanceOrderBookPublisher(OrderBookPublisher):
    """
    Order book publisher for Binance exchange.

    Subscribes to depth@100ms stream and handles snapshot updates.
    """

    def subscribe(self, ws):
        message = {
            "method": "SUBSCRIBE",
            "params": [f"{symbol.lower()}@depth@100ms" for symbol in self.symbols],
            "id": 1
        }
        ws.send(dumps(message))
        logging.info("%s: Subscribed to Binance depth streams: %s", self.__class__.__name__, self.symbols)

    def update_order_book(self, data, timeReceived):
        if data.get("e") != "depthUpdate":
            return

        symbol = data.get("s")
        if not symbol or symbol not in self.order_book:
            return

        ob = self.order_book[symbol]
        timeExchange = int(data.get("E", 0) * 1e6)

        for price, quantity in data.get("b", []):
            ob.update_order(float(price), float(quantity), "bid")
        for price, quantity in data.get("a", []):
            ob.update_order(float(price), float(quantity), "ask")

        self.publish_order_book(Exchange.BINANCE.value, symbol, timeExchange, timeReceived, time.time_ns())


class OkxOrderBookPublisher(OrderBookPublisher):
    """
    Order book publisher for OKX exchange.

    Subscribes to books5 channel and updates the local book on new events.
    """

    def subscribe(self, ws):
        message = {
            "op": "subscribe",
            "args": [{"channel": "books5", "instId": symbol} for symbol in self.symbols]
        }
        ws.send(dumps(message))
        logging.info("%s: Subscribed to OKX books5: %s", self.__class__.__name__, self.symbols)

    def update_order_book(self, data, timeReceived):
        if "data" not in data or not data["data"]:
            return

        data_item = data["data"][0]
        symbol = data.get("arg", {}).get("instId")
        if not symbol or symbol not in self.order_book:
            return

        ob = self.order_book[symbol]
        ts = data_item.get("ts")
        timeExchange = int(ts) * 1_000_000 if ts else time.time_ns()

        for bid in data_item.get("bids", []):
            price, quantity = bid[:2]
            ob.update_order(float(price), float(quantity), "bid")
        for ask in data_item.get("asks", []):
            price, quantity = ask[:2]
            ob.update_order(float(price), float(quantity), "ask")

        self.publish_order_book(Exchange.OKX.value, symbol, timeExchange, timeReceived, time.time_ns())


class BybitOrderBookPublisher(OrderBookPublisher):
    """
    Order book publisher for Bybit exchange.

    Uses HMAC authentication and orderbook.50 topic.
    """

    def __init__(self, ws_url, symbols, api_key, secret_key, save_mode=True):
        super().__init__(ws_url, symbols, save_mode)
        self.api_key = api_key
        self.secret_key = secret_key

    def subscribe(self, ws):
        expires = int((time.time() + 1) * 1000)
        sign = hmac.new(
            bytes(self.secret_key, "utf-8"),
            bytes(f"GET/realtime{expires}", "utf-8"),
            digestmod="sha256"
        ).hexdigest()

        auth_payload = {
            "op": "auth",
            "args": [self.api_key, expires, sign]
        }
        ws.send(dumps(auth_payload))
        logging.info("%s: Sent Bybit auth payload.", self.__class__.__name__)

        subscribe_payload = {
            "op": "subscribe",
            "args": [f"orderbook.50.{symbol}" for symbol in self.symbols]
        }
        ws.send(dumps(subscribe_payload))
        logging.info("%s: Subscribed to Bybit orderbook.50: %s", self.__class__.__name__, self.symbols)

    def update_order_book(self, data, timeReceived):
        if "data" not in data or not data["data"]:
            return

        symbol = data.get("topic", "").split(".")[-1]
        if not symbol or symbol not in self.order_book:
            return

        ob = self.order_book[symbol]
        if data.get("type") == "snapshot":
            ob.bids.clear()
            ob.asks.clear()

        ts = data.get("ts")
        timeExchange = int(float(ts)) * 1_000_000 if ts else time.time_ns()

        for bid in data["data"].get("b", []):
            ob.update_order(float(bid[0]), float(bid[1]), "bid")
        for ask in data["data"].get("a", []):
            ob.update_order(float(ask[0]), float(ask[1]), "ask")

        self.publish_order_book(Exchange.BYBIT.value, symbol, timeExchange, timeReceived, time.time_ns())


if __name__ == "__main__":
    coinbase_orderbook_publisher = CoinbaseOrderBookPublisher(
        ws_url=EXCHANGE_CONFIG["coinbase"]["ws_url"],
        api_key=EXCHANGE_CONFIG["coinbase"]["api_key"],
        secret_key=EXCHANGE_CONFIG["coinbase"]["secret_key"],
        symbols=[
            instrument_to_symbol(Exchange.COINBASE, Spot(base=Currency.BTC, term=Currency.USD)),
            instrument_to_symbol(Exchange.COINBASE, Spot(base=Currency.ETH, term=Currency.USD))
        ]
    )

    binance_orderbook_publisher = BinanceOrderBookPublisher(
        ws_url=EXCHANGE_CONFIG["binance"]["ws_url"],
        symbols=["BTCUSDT", "ETHUSDT"] # TODO: input = Instrument -> map to symbol inside the class
    )

    bybit_orderbook_publisher = BybitOrderBookPublisher(
        ws_url=EXCHANGE_CONFIG["bybit"]["ws_url"],
        symbols=["BTCUSDT", "ETHUSDT"], # TODO: input = Instrument -> map to symbol inside the class
        api_key=EXCHANGE_CONFIG["bybit"]["api_key"],
        secret_key=EXCHANGE_CONFIG["bybit"]["secret_key"]
    )

    okx_orderbook_publisher = OkxOrderBookPublisher(
        ws_url=EXCHANGE_CONFIG["okx"]["ws_url"],
        symbols=["BTC-USDT", "ETH-USDT"] # TODO: input = Instrument -> map to symbol inside the class
    )

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
    time.sleep(20 * 60)

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
