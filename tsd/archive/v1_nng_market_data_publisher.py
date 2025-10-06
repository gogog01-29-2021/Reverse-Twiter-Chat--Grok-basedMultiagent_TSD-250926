import abc
import logging
import queue
import threading
import time
from abc import abstractmethod

import pynng
import websocket

from core.types.exchange import Exchange
from core.utils.timeutils import datetime_str_to_nanoseconds, SECOND_IN_NANOS

MAX_LEVELS = 10

logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)

try:
    import orjson as json_parser


    def dumps(obj):
        return json_parser.dumps(obj).decode("utf-8")


    loads = json_parser.loads
except ImportError:
    import json as json_parser

    dumps = json_parser.dumps
    loads = json_parser.loads


class WebsocketHandler(abc.ABC):
    """
    Abstract base class to manage WebSocket connection lifecycle and dispatch incoming messages.
    """

    def __init__(self, ws_url: str):
        self.ws_url = ws_url
        self.ws_app = None
        self.ws_thread = None

    def start(self, block: bool = True):
        """Start the WebSocket streaming and optionally block the main thread."""
        self._start_streaming()
        logging.info("%s: WebSocket streaming started", self.__class__.__name__)

        if block:
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                self.end()

    def _start_streaming(self):
        """Initialize and run the WebSocket in a background thread."""
        self.ws_app = websocket.WebSocketApp(
            self.ws_url,
            on_open=lambda ws: self._subscribe(ws),
            on_message=self._on_message
        )
        self.ws_thread = threading.Thread(target=self.ws_app.run_forever, daemon=True)
        self.ws_thread.start()
        logging.info("%s: Started WebSocket streaming", self.__class__.__name__)
        return self.ws_thread

    @abc.abstractmethod
    def _subscribe(self, ws):
        """Abstract method for sending subscription message when WebSocket opens."""
        pass

    @abc.abstractmethod
    def _on_message(self, ws, message):
        """Abstract method to process received WebSocket messages."""
        pass

    def end(self):
        """Clean up WebSocket connection and background thread."""
        logging.info("%s: Stopping WebSocket streaming", self.__class__.__name__)
        if self.ws_app:
            self.ws_app.keep_running = False
            self.ws_app.close()

        if self.ws_thread:
            self.ws_thread.join(timeout=2)


class Publisher(WebsocketHandler, abc.ABC):
    """
    Publishes messages received from a WebSocket to NNG subscribers.
    """

    def __init__(self, ws_url: str, nng_bind_url: str = "tcp://127.0.0.1:5555"):
        super().__init__(ws_url)

        self.socket = pynng.Pub0()
        self.socket.listen(nng_bind_url)
        logging.info("%s: NNG PUB socket bound to %s", self.__class__.__name__, nng_bind_url)

        self.publisher_thread = self.PublisherThread(self.socket)
        self.publisher_thread.start()

    class PublisherThread(threading.Thread):
        """
        Thread that manages a message queue and publishes to an NNG socket asynchronously.
        """

        def __init__(self, socket):
            super().__init__(daemon=True)
            self.socket = socket
            self.queue = queue.Queue()
            self.running = True
            self._last_log_time_ns = 0

        def run(self):
            while self.running:
                try:
                    data = self.queue.get(timeout=1)
                    subject = data["subject"]
                    msg = data["msg"]
                    payload = f"{subject} {dumps(msg)}".encode("utf-8")
                    self.socket.send(payload)

                    now_ns = time.time_ns()
                    if now_ns - self._last_log_time_ns >= SECOND_IN_NANOS:
                        logging.info("Publishing to subject '%s': %s", subject, msg)
                        self._last_log_time_ns = now_ns

                    self.queue.task_done()
                except queue.Empty:
                    continue
                except Exception as e:
                    logging.error("NNG PublisherThread Error: %s", e)

        def publish(self, subject: str, msg: dict):
            self.queue.put({"subject": subject, "msg": msg})

        def stop(self):
            self.running = False

    @abc.abstractmethod
    def subject(self, *args, **kwargs):
        pass

    def end(self):
        logging.info("%s: Stopping publisher", self.__class__.__name__)

        if self.ws_app:
            self.ws_app.keep_running = False
            self.ws_app.close()

        if self.ws_thread:
            self.ws_thread.join(timeout=2)

        self.publisher_thread.stop()

        try:
            self.socket.close()
            logging.info("%s: NNG socket closed", self.__class__.__name__)
        except Exception as e:
            logging.error("%s: Error closing NNG socket: %s", self.__class__.__name__, e)

        logging.info("%s: Publisher stopped", self.__class__.__name__)

    def wait_for_subscriber_handshake(self, control_url="tcp://127.0.0.1:6666"):
        logging.info("Waiting for subscriber handshake on %s...", control_url)
        with pynng.Rep0(listen=control_url) as rep:
            msg = rep.recv().decode()
            logging.info("Received handshake message: %s", msg)
            rep.send(b"ack")
            logging.info("Handshake complete. Proceeding with streaming.")


class MarketDataPublisher(Publisher, abc.ABC):
    """
    Abstract class for transforming and publishing parsed market data to ZeroMQ.
    """

    def __init__(self, ws_url):
        super().__init__(ws_url)

    def _on_message(self, ws, message):
        """Receive and parse incoming WebSocket message as market data."""
        logging.debug("%s: WebSocket message received: %s", self.__class__.__name__, message)
        if not isinstance(message, str):
            return
        try:
            timeReceived = time.time_ns()
            data = loads(message)
            self._parse_market_data(data, timeReceived, max_levels=MAX_LEVELS)
        except Exception as e:
            logging.error("%s: Error processing WebSocket message: %s", self.__class__.__name__, e)

    @abstractmethod
    def _parse_market_data(self, data, timeReceived, max_levels):
        """Parse and extract structured data from raw market feed."""
        pass

    @staticmethod
    def subject(exchange, symbol):
        """Generate ZeroMQ topic string."""
        return f"MarketData_{exchange}_{symbol}"

    def publish_market_data(self, exchange, symbol, side, price, qty, time_exchange, time_received):
        """Publish a parsed order book update message."""
        subject = self.subject(exchange, symbol)

        msg = {
            "exchange": exchange,
            "symbol": symbol,
            "side": side,
            "price": price,
            "qty": qty,
            "timeExchange": time_exchange,
            "timeReceived": time_received,
            "timePublished": time.time_ns(),
        }

        self.publisher_thread.publish(subject, msg)


class CoinbaseMarketDataPublisher(MarketDataPublisher):
    """
    Coinbase-specific implementation for market data publishing using ZeroMQ.
    """

    def __init__(self, ws_url, symbols):
        super().__init__(ws_url)
        self._symbols = symbols

    def _subscribe(self, ws):
        """Send subscription message to Coinbase WebSocket."""
        message = {
            "type": "subscribe",
            "channel": "level2",
            "product_ids": self._symbols,
        }
        ws.send(dumps(message))
        logging.info("%s: Subscribed to Coinbase level2 for symbols: %s", self.__class__.__name__, self._symbols)

    def _parse_market_data(self, data, time_received, max_levels):
        """Parse the Coinbase level2 update into structured market data."""
        time_exchange = datetime_str_to_nanoseconds(data.get("timestamp"), format="%Y-%m-%dT%H:%M:%S.%fZ")

        for event in data.get("events", []):
            symbol = event.get("product_id")
            for upd in event.get("updates", []):
                side = upd.get("side")
                price = upd.get("price_level")
                qty = upd.get("new_quantity")
                if side and price and qty:
                    try:
                        self.publish_market_data(Exchange.COINBASE, symbol, side, float(price), float(qty),
                                                 time_exchange, time_received)
                    except Exception as e:
                        logging.error("%s: Failed to update %s %s: %s", self.__class__.__name__, side, price, e)


if __name__ == "__main__":
    coinbase_market_data_publisher = CoinbaseMarketDataPublisher(
        ws_url="wss://advanced-trade-ws.coinbase.com",
        symbols=["BTC-USD"]
    )

    coinbase_market_data_publisher.wait_for_subscriber_handshake(control_url="tcp://127.0.0.1:6666")

    coinbase_thread = threading.Thread(target=coinbase_market_data_publisher.start, kwargs={'block': False})
    coinbase_thread.start()

    time.sleep(60)
    coinbase_market_data_publisher.end()
    coinbase_thread.join()
