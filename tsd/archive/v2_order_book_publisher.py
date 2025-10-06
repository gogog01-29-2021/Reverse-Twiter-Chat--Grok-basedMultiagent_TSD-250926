import abc
import logging
import threading
import time
from abc import ABC
from collections import deque
from datetime import datetime, timezone
from typing import Dict, List

import websocket
import zmq
from sortedcontainers import SortedDict

from core.types.instrument import Instrument, Currency, Spot
from core.types.exchange.utils import instrument_to_symbol
from core.types.exchange import Exchange
from core.utils.timeutils import SECOND_IN_NANOS, datetime_to_string

# Try to use high-performance JSON parser
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


class WebsocketHandler(abc.ABC):
    """
    Abstract base class for managing WebSocket connections.
    """

    def __init__(self, ws_url: str):
        self.ws_url = ws_url
        self.ws_app = None
        self.ws_thread = None

    def start(self, block: bool = True):
        """
        Start the WebSocket client and optionally block the main thread.
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
        Initialize and run the WebSocket client in a separate thread.
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
        Gracefully terminate the WebSocket connection and thread.
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
        Abstract method for subscription logic when WebSocket opens.
        """
        pass

    @abc.abstractmethod
    def _on_message(self, ws, message):
        """
        Abstract method for handling incoming WebSocket messages.
        """
        pass


class Publisher:
    """
    Threaded ZeroMQ publisher for asynchronous message dispatch.

    e.g. addr = "tcp://0.0.0.0:5555"
    """

    def __init__(self, addr: str):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PUB)
        self.socket.setsockopt(zmq.SNDHWM, 0)
        self.socket.bind(addr)
        self.publisher_thread = self.PublisherThread(self.socket)
        self.publisher_thread.start()

    def enqueue_message(self, subject: str, msg: dict):
        """
        Enqueue message to be published on a given subject.
        """
        self.publisher_thread.enqueue(subject, msg)

    def stop(self):
        """
        Stop publisher thread and clean up resources.
        """
        self.publisher_thread.stop()
        self.socket.close()
        self.context.term()

    class PublisherThread(threading.Thread):
        """
        Internal thread class responsible for sending messages via ZMQ.
        """

        def __init__(self, socket):
            super().__init__(daemon=True)
            self.socket = socket
            self.queue = deque()
            self.running = threading.Event()
            self.running.set()
            self.condition = threading.Condition()
            self._last_log_time_ns = 0

        def run(self):
            while self.running.is_set():
                with self.condition:
                    while not self.queue and self.running.is_set():
                        self.condition.wait()
                    if not self.running.is_set():
                        break
                    item = self.queue.popleft()

                self.send_message(item)

        def send_message(self, data: Dict):
            """
            Send a message through the ZeroMQ socket.
            """
            try:
                self.socket.send_multipart(
                    [data["subject"].encode(), dumps(data["msg"])],
                    copy=False
                )
                if self._should_log():
                    logging.info("Publishing [%s]: %s", data["subject"], data["msg"])
            except Exception as e:
                logging.error("Failed to send message: %s", e)

        def enqueue(self, subject: str, msg: dict):
            """
            Add message to publishing queue and notify thread.
            """
            with self.condition:
                self.queue.append({"subject": subject, "msg": msg})
                self.condition.notify()

        def _should_log(self) -> bool:
            now_ns = time.time_ns()
            if now_ns - self._last_log_time_ns >= SECOND_IN_NANOS:
                self._last_log_time_ns = now_ns
                return True
            return False

        def stop(self):
            """
            Stop the publishing thread gracefully.
            """
            with self.condition:
                self.running.clear()
                self.condition.notify()


class OrderBookPublisher(WebsocketHandler, ABC):
    """
    Abstract WebSocket handler for publishing market data to ZMQ.
    """

    def __init__(self, ws_url: str, addr: str):
        super().__init__(ws_url)
        self.publisher = Publisher(addr=addr)

    @staticmethod
    def subject(exchange: Exchange, instrument: Instrument) -> str:
        return f"OrderBook_{exchange.value}_{instrument.symbol}"

    def _publish_order_book(
            self,
            exchange: Exchange,
            instrument: Instrument,
            bids: List[List[float]],
            asks: List[List[float]],
            time_exchange: str,
            time_received: str
    ):
        msg = {
            "exchange": exchange.value,
            "symbol": instrument.symbol,
            "bids": bids,
            "asks": asks,
            "time_exchange": time_exchange,
            "time_received": time_received,
            "time_published": datetime_to_string(datetime.now(timezone.utc)),
        }
        try:
            self.publisher.enqueue_message(self.subject(exchange, instrument), msg)
        except Exception as e:
            logging.error("%s: Failed to publish: %s", self.__class__.__name__, e)

    class OrderBook:
        def __init__(self, max_levels):
            self.max_levels = max_levels
            self.bids = SortedDict(lambda x: -x)
            self.asks = SortedDict()

        def apply_snapshot(self, updates: List[Dict]):
            self.bids.clear()
            self.asks.clear()
            b_count = a_count = 0

            for u in updates:
                try:
                    price, qty, side = float(u["price_level"]), float(u["new_quantity"]), u["side"]
                    if qty == 0:
                        continue
                    if side == "bid" and b_count < self.max_levels:
                        self.bids[price], b_count = qty, b_count + 1
                    elif side == "offer" and a_count < self.max_levels:
                        self.asks[price], a_count = qty, a_count + 1
                    if b_count >= self.max_levels and a_count >= self.max_levels:
                        break
                except (KeyError, ValueError, TypeError):
                    continue

        def apply_update(self, updates: List[Dict]):
            for u in updates:
                try:
                    price, qty, side = float(u["price_level"]), float(u["new_quantity"]), u["side"]
                    book = self.bids if side == "bid" else self.asks
                    if qty == 0:
                        book.pop(price, None)
                    else:
                        book[price] = qty
                except (KeyError, ValueError, TypeError):
                    continue
            self._trim()

        def _trim(self):
            while len(self.bids) > self.max_levels:
                self.bids.popitem()
            while len(self.asks) > self.max_levels:
                self.asks.popitem()

        def get_top_n(self, n: int = None) -> Dict[str, List[Dict[str, float]]]:
            n = n or self.max_levels
            return {
                "bids": [{"price": p, "qty": self.bids[p]} for i, p in enumerate(self.bids) if i < n],
                "asks": [{"price": p, "qty": self.asks[p]} for i, p in enumerate(self.asks) if i < n],
            }

        def get_top_of_book(self) -> Dict[str, float]:
            return {
                "bid": next(iter(self.bids), None),
                "ask": next(iter(self.asks), None)
            }

        def get_book(self) -> Dict[str, List[Dict[str, float]]]:
            return {
                "bids": [{"price": p, "qty": self.bids[p]} for p in self.bids],
                "asks": [{"price": p, "qty": self.asks[p]} for p in self.asks]
            }


class CoinbaseOrderBookPublisher(OrderBookPublisher):
    """
    Coinbase WebSocket market data handler that only tracks top N order book levels.
    """

    EXCHANGE = Exchange.COINBASE

    def __init__(self, ws_url: str, addr: str, spots: List[Spot], max_levels):
        super().__init__(ws_url, addr)
        self._symbols = [instrument_to_symbol(self.EXCHANGE, spot) for spot in spots]
        self.max_levels = max_levels

        self._order_books = {
            symbol: self.OrderBook(self.max_levels) for symbol in self._symbols
        }

    def _subscribe(self, ws):
        ws.send(dumps({
            "type": "subscribe",
            "channel": "level2",
            "product_ids": self._symbols,
        }))
        logging.info("%s: Subscribed to Coinbase level2 for %s", self.__class__.__name__, self._symbols)

    def _on_message(self, ws, message):
        if not isinstance(message, str):
            return
        try:
            time_received = datetime_to_string(datetime.now(timezone.utc))
            data = loads(message)
            self._handle_market_data(data, time_received)
        except Exception as e:
            logging.error("%s: Error processing message: %s", self.__class__.__name__, e)

    def _handle_market_data(self, data: Dict, time_received: str):
        if data.get("channel") != "l2_data":
            return

        time_exchange = data.get("timestamp", "")

        for event in data.get("events", []):
            symbol = event.get("product_id")
            book = self._order_books.get(symbol)
            if not book:
                continue

            updates = event.get("updates", [])
            if event["type"] == "snapshot":
                book.apply_snapshot(self._filter_snapshot(updates))
            elif event["type"] == "update":
                book.apply_update(updates)

            book_data = book.get_book()
            self._publish_order_book(
                self.EXCHANGE,
                symbol_to_spot(symbol, self.EXCHANGE),
                [[p["price"], p["qty"]] for p in book_data["bids"]],
                [[p["price"], p["qty"]] for p in book_data["asks"]],
                time_exchange,
                time_received
            )

    def _filter_snapshot(self, updates: List[Dict]) -> List[Dict]:
        """
        Limit the snapshot to MAX_LEVELS per side.
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
        addr="tcp://0.0.0.0:5555",
        spots=[
            Spot(base=Currency.BTC, term=Currency.USD),
            Spot(base=Currency.BTC, term=Currency.USD)],
        max_levels=10
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
