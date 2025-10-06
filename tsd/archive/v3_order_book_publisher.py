import abc
import hashlib
import json
import logging
import threading
import time
from abc import ABC, abstractmethod
from collections import deque
from datetime import datetime, timezone
from typing import Dict, List, Tuple

import orjson as json_parser
import websocket
import zmq
from sortedcontainers import SortedDict

from core.types.exchange import Exchange
from core.types.instrument import Spot, Currency, Instrument
from core.types.marketdata import Side
from dsm.utils.conversion_utils import symbol_to_instrument, instrument_to_exchange_symbol, now_epoch_ms, \
    iso_to_epoch_ms

logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)


class ZmqSocketHandler(ABC):
    def __init__(self):
        self._context = zmq.Context()
        self._socket = None

    def stop(self):
        if self._socket:
            self._socket.close()
        self._context.term()


class ZmqPublisherSocketHandler(ZmqSocketHandler):
    """
    Transport layer using ZeroMQ to publish messages to subscribers.

    Attributes:
        zmq_bind_addr (str): ZMQ bind address for PUB socket.

    This class runs a background thread that dequeues messages and sends them
    via ZMQ PUB socket. It's designed for low-latency communication.

    Assumptions:
        - Consumers are already subscribed using ZMQ SUB sockets.
        - Messages are serialized before sending.

    Warning:
        - High Water Mark is set to zero for minimal delay, but may drop messages if overwhelmed.
    """

    def __init__(self, zmq_bind_addr: str, debug: bool = False):
        super().__init__()
        self._socket = self._context.socket(zmq.PUB)
        self._socket.setsockopt(zmq.SNDHWM, 0)
        self._socket.bind(zmq_bind_addr)

        self.queue = deque()
        self.condition = threading.Condition()
        self.running = threading.Event()
        self.running.set()

        self.debug = debug
        if self.debug:
            self._last_hashes = deque(maxlen=50)

        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def publish(self, subject: str, msg: dict):
        """
        Add a message to the queue to be sent asynchronously.

        Args:
            subject (str): Message topic/subject (used by subscribers).
            msg (dict): JSON-serializable message payload.
        """
        with self.condition:
            self.queue.append((subject, msg))
            self.condition.notify()

    def _run(self):
        """
        Internal loop to send messages from the queue over ZMQ.

        This function runs in a dedicated thread.
        """
        while self.running.is_set():
            with self.condition:
                while not self.queue and self.running.is_set():
                    self.condition.wait()
                if not self.running.is_set():
                    break
                subject, msg = self.queue.popleft()
            try:
                if self.debug:
                    msg_str = json.dumps(msg, sort_keys=True)
                    hash_val = hashlib.md5((subject + msg_str).encode()).hexdigest()
                    if hash_val in self._last_hashes:
                        logging.warning("ZmqPublisherBase duplicate detected for subject %s", subject)
                    self._last_hashes.append(hash_val)

                self._socket.send_multipart([subject.encode(), json_parser.dumps(msg)], copy=False)
            except Exception as e:
                logging.error("ZmqPublisherBase error: %s", e)

    def stop(self):
        """
        Gracefully shutdown the ZMQ transport.

        Ensures the background thread exits cleanly and sockets are closed.
        """
        with self.condition:
            self.running.clear()
            self.condition.notify()
        self.thread.join(timeout=2)
        super().stop()


class WebsocketHandler(ABC):
    """
    Abstract base class to manage WebSocket lifecycle and message handling.

    Responsibilities:
        - Establish connection to a WebSocket server.
        - Register hooks for message handling and subscriptions.

    Attributes:
        ws_url (str): WebSocket endpoint URL.

    Subclasses must implement:
        - _subscribe(ws): Define what to send after opening the connection.
        - _on_message(ws, message): Define how to process incoming messages.
    """

    def __init__(self, ws_url: str):
        """
        Initialize WebSocket URL.

        Args:
            ws_url (str): The WebSocket URL to connect to.
        """
        self.ws_url = ws_url
        self.ws_app = None
        self.ws_thread = None

    def start(self, block: bool = True):
        """
        Start the WebSocket streaming connection.

        Args:
            block (bool): If True, blocks main thread until interrupted.
        """
        self._start_streaming()
        logging.info("%s: WebSocket streaming started", self.__class__.__name__)
        if block:
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                self.end()

    def _start_streaming(self):
        """
        Create and run WebSocketApp in a separate background thread.

        Hooks into user-defined _subscribe and _on_message methods.
        """
        self.ws_app = websocket.WebSocketApp(
            self.ws_url,
            on_open=self._subscribe,
            on_message=self._on_message
        )
        self.ws_thread = threading.Thread(target=self.ws_app.run_forever, daemon=True)
        self.ws_thread.start()

    def end(self):
        """
        Close the WebSocket connection and thread gracefully.
        """
        logging.info("%s: Closing WebSocket connection", self.__class__.__name__)
        if self.ws_app:
            self.ws_app.keep_running = False
            self.ws_app.close()
        if self.ws_thread:
            self.ws_thread.join(timeout=2)

    @abc.abstractmethod
    def _subscribe(self, ws):
        """
        Define subscription message(s) to be sent once the socket is open.

        Args:
            ws: WebSocket connection instance.
        """
        pass

    @abc.abstractmethod
    def _on_message(self, ws, message):
        """
        Handle incoming message from the WebSocket.

        Args:
            ws: WebSocket connection instance.
            message (str | bytes): Raw message from the WebSocket.
        """
        pass


class Publisher(WebsocketHandler, abc.ABC):
    """
    WebSocket publisher base that emits messages to ZMQ and logs them optionally.

    Attributes:
        ws_url (str): WebSocket endpoint URL.
        zeromq_addr (str): ZMQ address to bind PUB socket.
        verbose (bool): If True, logs messages per second.

    Thread-safe logging and message counting for performance monitoring.
    """

    def __init__(self, ws_url: str, zeromq_addr: str, verbose: bool, debug: bool = False, log_on_change: bool = False):
        """
        Initialize WebSocket publisher.

        Args:
            ws_url (str): WebSocket endpoint URL.
            zeromq_addr (str): Address to bind the ZMQ PUB socket.
            verbose (bool): Enable verbose logging.
            debug (bool): Enable duplicate detection.
            log_on_change (bool): If True, log on message change; otherwise, log every second.
        """
        super().__init__(ws_url)
        self.debug = debug
        self.transport = ZmqPublisherSocketHandler(zeromq_addr, debug=debug)

        self._verbose = verbose
        self._log_on_change = log_on_change
        self._msg_count = 0
        self._last_subject = None
        self._last_msg = None
        self._lock = threading.Lock()

        if self._verbose and not self._log_on_change:
            self._start_logger_thread()

        if self.debug:
            self._published_hashes = deque(maxlen=10)

    def publish(self, subject: str, msg: dict):
        """
        Push message to ZMQ transport and optionally track for logging.

        Args:
            subject (str): ZMQ topic header.
            msg (dict): JSON-serializable message to publish.
        """

        if self.debug:
            msg_str = json.dumps(msg, sort_keys=True)
            hash_val = hashlib.md5((subject + msg_str).encode()).hexdigest()
            if hash_val in self._published_hashes:
                logging.warning("Publisher duplicate detected for subject %s", subject)
            self._published_hashes.append(hash_val)

        with self._lock:
            changed = (self._last_subject != subject) or (self._last_msg != msg)
            self._msg_count += 1

            if self._verbose and self._log_on_change and changed:
                timestamp = datetime.now(timezone.utc).isoformat()
                logging.info(
                    "%s: [%s] Subject: %s | Msg: %s",
                    self.__class__.__name__, timestamp, subject, str(msg)
                )

            self._last_subject = subject
            self._last_msg = msg

        self.transport.publish(subject, msg)

    def _start_logger_thread(self):
        """
        Background thread to log publishing rate and sample messages.
        Useful for debugging and monitoring message throughput.
        """

        def logger_loop():
            has_started = False
            while True:
                time.sleep(1)
                with self._lock:
                    count = self._msg_count
                    subject = self._last_subject
                    msg = self._last_msg
                    self._msg_count = 0

                if count > 0:
                    has_started = True
                if not has_started:
                    continue

                timestamp = datetime.now(timezone.utc).isoformat()
                logging.info(
                    "%s: [%s] %d msg/sec | Subject: %s | Msg: %s",
                    self.__class__.__name__, timestamp, count, subject or "N/A", str(msg) if msg else "N/A"
                )

        threading.Thread(target=logger_loop, daemon=True, name="PublisherLogger").start()

    def end(self):
        """
        Shut down WebSocket and transport layer.
        Ensures clean termination of background threads.
        """
        super().end()
        self.transport.stop()


class OrderBookData:
    """
    Maintains a local view of an order book with capped depth for bids and asks.

    This class is used to manage and apply updates from market data feeds,
    ensuring the book always maintains up to `max_levels` entries per side.

    Attributes:
        max_levels (int): Maximum number of levels to retain for bids and asks.
        bids (SortedDict): Descending price-sorted map of bid prices to quantities.
        asks (SortedDict): Ascending price-sorted map of ask prices to quantities.

    Notes:
        - Assumes updates arrive in sorted order or are quickly trimmed.
        - This structure is not thread-safe.
    """

    def __init__(self, max_levels: int):
        self.max_levels = max_levels
        self.bids = SortedDict(lambda x: -x)
        self.asks = SortedDict()

    def apply_snapshot(self, updates: List[Tuple[float, float, Side]]):
        """
        Apply a full snapshot to reset the order book.

        Args:
            updates (List[Tuple[float, float, Side]]):
                A list of (price, quantity, side) tuples representing the book.

        Warning:
            Entries with qty=0 are ignored. Bids/asks capped to `max_levels`.
        """
        self.bids.clear()
        self.asks.clear()
        for price, qty, side in updates:
            if qty == 0:
                continue
            book = self.bids if side == Side.BID else self.asks
            if len(book) < self.max_levels:
                book[price] = qty
            if len(self.bids) >= self.max_levels and len(self.asks) >= self.max_levels:
                break

    def apply_update_one(self, price: float, qty: float, side: Side) -> bool:
        """
        Apply a single level update to the order book.

        Args:
            price (float): The price level being updated.
            qty (float): The new quantity at that level. If 0, the level is removed.
            side (Side): BID or ASK side.

        Returns:
            bool: False if the update violates the level limit, else True.
        """
        book = self.bids if side == Side.BID else self.asks
        if qty == 0:
            book.pop(price, None)
        else:
            book[price] = qty

        if len(book) > self.max_levels:
            worst_price = next(reversed(book))
            if (side == Side.BID and price < worst_price) or (side == Side.ASK and price > worst_price):
                return False
        return True

    def trim(self):
        """
        Ensure bids and asks contain no more than max_levels entries each.
        """
        while len(self.bids) > self.max_levels:
            self.bids.popitem()
        while len(self.asks) > self.max_levels:
            self.asks.popitem()

    def get_book(self) -> Dict[str, List[Dict[str, float]]]:
        """
        Export the order book as a dictionary with sorted bid/ask levels.

        Returns:
            dict: {"bids": [...], "asks": [...]} sorted by price.
        """
        return {
            "bids": [{"price": p, "qty": self.bids[p]} for p in self.bids],
            "asks": [{"price": p, "qty": self.asks[p]} for p in self.asks],
        }


class OrderBookPublisher(Publisher, ABC):
    """
    Base class for publishers that stream and broadcast order book data.

    Provides the standard message schema for broadcasting order books to ZMQ.

    Methods:
        publish_order_book(...): Publishes a single snapshot to subscribers.
    """

    @staticmethod
    def subject(exchange: Exchange, instrument: Instrument) -> str:
        """
        Compose a ZMQ subject string for the given exchange and instrument.

        Args:
            exchange (Exchange): Exchange identifier.
            instrument (Instrument): Market instrument (e.g., spot pair).

        Returns:
            str: ZMQ topic string.
        """
        return f"OrderBook_{exchange.value}_{instrument.symbol}"

    def publish_order_book(
            self,
            exchange: Exchange,
            instrument: Instrument,
            bids: List[List[float]],
            asks: List[List[float]],
            time_exchange: int,
            time_received: int,
            time_published: int
    ):
        """
        Format and publish an order book snapshot to ZMQ.

        Args:
            exchange (Exchange): Source exchange.
            instrument (Instrument): The instrument being quoted.
            bids (List[List[float]]): Top bid levels [[price, qty], ...].
            asks (List[List[float]]): Top ask levels [[price, qty], ...].
            time_exchange (int): Exchange-reported event time.
            time_received (int): Local epoch milliseconds when data was received.
            time_published (int):
        """
        msg = {
            "exchange": exchange.value,
            "symbol": instrument.symbol,
            "bids": bids,
            "asks": asks,
            "time_exchange": time_exchange,
            "time_received": time_received,
            "time_published": time_published,
        }
        try:
            self.publish(self.subject(exchange, instrument), msg)
        except Exception as e:
            logging.error("%s: Failed to publish: %s", self.__class__.__name__, e)

    def _on_message(self, ws, message):
        """
        Handle incoming WebSocket message.

        Args:
            ws: Active WebSocketApp connection.
            message (str): Raw JSON-encoded message.
        """
        if not isinstance(message, str):
            return
        try:
            time_received = now_epoch_ms()
            data = json_parser.loads(message)
            self._handle_market_data(data, time_received)
        except Exception as e:
            logging.error("%s: Error processing message: %s", self.__class__.__name__, e)

    @abstractmethod
    def _handle_market_data(self, market_data: Dict, time_received: int):
        pass


class CoinbaseOrderBookPublisher(OrderBookPublisher):
    """
    Concrete implementation of OrderBookPublisher for Coinbase exchange.

    Subscribes to Coinbase level2 WebSocket, processes L2 updates,
    and broadcasts snapshots via ZMQ.

    Attributes:
        max_levels (int): Max order book depth per side.
        _order_books (dict): symbol → OrderBookData mapping.
    """

    EXCHANGE = Exchange.COINBASE

    def __init__(self, ws_url: str, zeromq_addr: str, instruments: List[Instrument], max_levels: int, verbose: bool):
        """
        Initialize Coinbase order book publisher.

        Args:
            ws_url (str): WebSocket endpoint URL for Coinbase.
            zeromq_addr (str): ZMQ bind address.
            instruments (List[Instrument]): List of instruments to subscribe.
            max_levels (int): Maximum number of levels to store.
            verbose (bool): Enable verbose logging.
        """
        super().__init__(ws_url, zeromq_addr, verbose)
        self._symbols = [instrument_to_exchange_symbol(self.EXCHANGE, instrument) for instrument in instruments]
        self.max_levels = max_levels
        self._order_books = {symbol: OrderBookData(self.max_levels) for symbol in self._symbols}

    def _subscribe(self, ws):
        """
        Send subscription message to Coinbase WebSocket.

        Args:
            ws: Active WebSocketApp connection.
        """
        ws.send(json_parser.dumps({
            "type": "subscribe",
            "channel": "level2",
            "product_ids": self._symbols,
        }))
        logging.info("%s: Subscribed to Coinbase level2 for %s", self.__class__.__name__, self._symbols)

    def _handle_market_data(self, market_data: Dict, time_received: int):
        """
        Parse and apply order book snapshot or updates.

        Args:
            market_data (dict): Parsed WebSocket message.
            time_received (int): Local epoch milliseconds of receipt.
        """
        if market_data.get("channel") != "l2_data":
            return

        time_exchange = iso_to_epoch_ms(market_data.get("timestamp", ""))

        for event in market_data.get("events", []):
            symbol = event.get("product_id")
            book = self._order_books.get(symbol)
            if not book:
                continue

            updates = event.get("updates", [])
            if event["type"] == "snapshot":
                parsed = [
                    (float(u["price_level"]), float(u["new_quantity"]),
                     Side.BID if u["side"] == "bid" else Side.ASK)
                    for u in self._filter_snapshot(updates)
                    if "price_level" in u and "new_quantity" in u
                ]
                book.apply_snapshot(parsed)
            elif event["type"] == "update":
                for u in updates:
                    try:
                        price = float(u["price_level"])
                        qty = float(u["new_quantity"])
                        side = Side.BID if u["side"] == "bid" else Side.ASK
                        keep_going = book.apply_update_one(price, qty, side)
                        if not keep_going:
                            break
                    except Exception:
                        continue
                book.trim()

            book_data = book.get_book()
            self.publish_order_book(
                self.EXCHANGE,
                symbol_to_instrument(symbol),
                [[p["price"], p["qty"]] for p in book_data["bids"]],
                [[p["price"], p["qty"]] for p in book_data["asks"]],
                time_exchange,
                time_received,
                now_epoch_ms()
            )

    def _filter_snapshot(self, updates: List[Dict]) -> List[Dict]:
        """
        Trim the snapshot to the top N levels per side.

        Args:
            updates (List[Dict]): Raw snapshot update messages.

        Returns:
            List[Dict]: Filtered updates preserving top bid/ask levels.
        """
        filtered, bids, asks = [], 0, 0
        for update in updates:
            side = update.get("side")
            if side == "bid" and bids < self.max_levels:
                bids += 1
                filtered.append(update)
            elif side == "offer" and asks < self.max_levels:
                asks += 1
                filtered.append(update)
            if bids >= self.max_levels and asks >= self.max_levels:
                break
        return filtered


if __name__ == "__main__":

    coinbase = CoinbaseOrderBookPublisher(
        ws_url="wss://advanced-trade-ws.coinbase.com",
        zeromq_addr="tcp://0.0.0.0:5556",
        instruments=[
            Spot(base=Currency.BTC, term=Currency.USD),
            Spot(base=Currency.ETH, term=Currency.USD)],
        max_levels=10,
        verbose=True
    )

    publishers = [
        coinbase,
        # binance,
        # BybitMarketDataPublisher("wss://stream.bybit.com/realtime", ["BTCUSD"], shared_publisher),
        # UpbitMarketDataPublisher("wss://api.upbit.com/websocket/v1", ["KRW-BTC"], shared_publisher),
        # OKXMarketDataPublisher("wss://ws.okx.com:8443/ws/v5/public", ["BTC-USDT"], shared_publisher),
        # BitgetMarketDataPublisher("wss://ws.bitget.com/spot/v1/stream", ["btcusdt"], shared_publisher),
        # MEXCMarketDataPublisher("wss://wbs.mexc.com/ws", ["btcusdt"], shared_publisher),
        # CryptoComMarketDataPublisher("wss://stream.crypto.com/v2/market", ["BTC_USDT"], shared_publisher),
        # HTXMarketDataPublisher("wss://api.huobi.pro/ws", ["btcusdt"], shared_publisher),
        # GateIOMarketDataPublisher("wss://api.gate.io/ws/v4/", ["BTC_USDT"], shared_publisher)
    ]

    threads = [
        threading.Thread(target=p.start, kwargs={"block": False})
        for p in publishers
    ]

    for t in threads:
        t.start()

    try:
        time.sleep(600)
    finally:
        for p in publishers:
            p.end()
        for t in threads:
            t.join()
