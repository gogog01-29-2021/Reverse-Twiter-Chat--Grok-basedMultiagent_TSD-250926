# import abc
# import asyncio
# import logging
# import queue
# import threading
# import time
# from abc import abstractmethod
#
# import websocket
# from nats.aio.client import Client as NATS
#
# from core.types.market.exchange import Exchange
# from core.utils.timeutils import datetime_str_to_nanoseconds, SECOND_IN_NANOS
#
# MAX_LEVELS = 10
#
# logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)
#
# try:
#     import orjson as json_parser
#
#
#     def dumps(obj):
#         return json_parser.dumps(obj).decode("utf-8")
#
#
#     loads = json_parser.loads
# except ImportError:
#     import json as json_parser
#
#     dumps = json_parser.dumps
#     loads = json_parser.loads
#
#
# class WebsocketHandler(abc.ABC):
#     """
#     Abstract base class for managing WebSocket connections and a background logging process.
#     Subclasses must implement specific behaviors for subscription, message handling, and logging.
#     """
#
#     def __init__(self, ws_url: str):
#         self.ws_url = ws_url
#         self.ws_app = None
#         self.ws_thread = None
#         self.logging_running = False
#         self.logging_thread = None
#
#     def start(self, block: bool = True):
#         """
#         Start the full streaming system, including WebSocket and background logging.
#
#         Args:
#             block (bool): If True, blocks the main thread until interrupted.
#         """
#         self._start_streaming()
#         logging.info("%s: WebSocket streaming started", self.__class__.__name__)
#
#         if block:
#             try:
#                 while True:
#                     time.sleep(1)
#             except KeyboardInterrupt:
#                 self.end()
#
#     def _start_streaming(self):
#         """
#         Start the WebSocket connection and message processing in a separate thread.
#         """
#         self.ws_app = websocket.WebSocketApp(
#             self.ws_url,
#             on_open=lambda ws: self._subscribe(ws),
#             on_message=self._on_message
#         )
#         self.ws_thread = threading.Thread(target=self.ws_app.run_forever, daemon=True)
#         self.ws_thread.start()
#         logging.info("%s: Started WebSocket streaming", self.__class__.__name__)
#         return self.ws_thread
#
#     @abc.abstractmethod
#     def _subscribe(self, ws):
#         """
#         Define the WebSocket subscription logic. Invoked upon connection open.
#
#         Args:
#             ws (websocket.WebSocket): Active WebSocket instance.
#         """
#         pass
#
#     @abc.abstractmethod
#     def _on_message(self, ws, message):
#         """
#         Handle incoming messages from the WebSocket.
#
#         Args:
#             ws (websocket.WebSocket): Active WebSocket instance.
#             message (str): Incoming message as a string.
#         """
#         pass
#
#     def end(self):
#         """
#         Stop all components: WebSocket, logging, and clean up threads.
#         """
#         logging.info("%s: Stopping WebSocket streaming", self.__class__.__name__)
#         if self.ws_app:
#             self.ws_app.keep_running = False
#             self.ws_app.close()
#
#         if self.ws_thread:
#             self.ws_thread.join(timeout=2)
#
#
# class Publisher(WebsocketHandler, abc.ABC):
#     """
#     Abstract base class to stream data from WebSocket and publish it to NATS asynchronously.
#     Subclasses must implement the subject determination, subscription, message handling,
#     and logging behavior.
#     """
#
#     def __init__(self, ws_url: str, nats_url: str = "nats://localhost:4222"):
#         super().__init__(ws_url)
#
#         self.nats_client = NATS()
#         self.loop = asyncio.new_event_loop()
#         self.loop_thread = threading.Thread(target=self.run_loop, daemon=True)
#         self.loop_thread.start()
#
#         # Connect to NATS server asynchronously
#         future = asyncio.run_coroutine_threadsafe(
#             self.nats_client.connect(servers=[nats_url]), self.loop
#         )
#         future.result()  # Wait for NATS connection to complete
#         logging.info("%s: Connected to NATS", self.__class__.__name__)
#
#         # Initialize and start the asynchronous publisher thread
#         self.publisher_thread = self.PublisherThread(self.nats_client, self.loop)
#         self.publisher_thread.start()
#
#     class PublisherThread(threading.Thread):
#         """
#         Dedicated thread for asynchronously publishing messages to NATS.
#         Messages are queued and sent without blocking the main processing logic.
#         """
#
#         def __init__(self, nats_client, loop):
#             super().__init__(daemon=True)
#             self.nats_client = nats_client
#             self.loop = loop
#             self.queue = queue.Queue()
#             self.running = True
#             self._last_log_time_ns = 0  # For throttled logging
#
#         def run(self):
#             """
#             Thread loop to process and publish queued messages to NATS.
#             """
#             while self.running:
#                 try:
#                     data = self.queue.get()
#                     subject = data["subject"]
#                     msg = data["msg"]
#                     payload = dumps(msg).encode("utf-8")
#                     asyncio.run_coroutine_threadsafe(
#                         self.nats_client.publish(subject, payload),
#                         self.loop
#                     )
#                     now_ns = time.time_ns()
#                     if now_ns - self._last_log_time_ns >= SECOND_IN_NANOS:  # 1 second in nanoseconds
#                         logging.info("Publishing to subject '%s': %s", subject, msg)
#                         self._last_log_time_ns = now_ns
#
#                     self.queue.task_done()
#                 except queue.Empty:
#                     continue
#
#         def publish(self, subject: str, msg: dict):
#             """
#             Add a message to the publishing queue.
#
#             Args:
#                 subject (str): NATS subject name.
#                 msg (dict): JSON-serializable message payload.
#             """
#             self.queue.put({"subject": subject, "msg": msg})
#
#         def stop(self):
#             """
#             Signal the thread to terminate.
#             """
#             self.running = False
#
#     @abc.abstractmethod
#     def subject(self, *args, **kwargs):
#         """
#         Define the NATS subject/channel to which messages should be published.
#         Must be implemented by subclasses.
#         """
#         pass
#
#     def run_loop(self):
#         """
#         Run the asyncio event loop in its dedicated thread.
#         """
#         asyncio.set_event_loop(self.loop)
#         self.loop.run_forever()
#
#     def end(self):
#         """
#         Stop the WebSocket, logging loop, and NATS publisher thread.
#         Gracefully shuts down all resources.
#         """
#         logging.info("%s: Stopping publisher", self.__class__.__name__)
#
#         if self.ws_app:
#             self.ws_app.keep_running = False
#             self.ws_app.close()
#
#         if self.ws_thread:
#             self.ws_thread.join(timeout=2)
#
#         self.publisher_thread.stop()
#
#         self.loop.call_soon_threadsafe(self.loop.stop)
#         self.loop_thread.join(timeout=2)
#         logging.info("%s: Publisher stopped", self.__class__.__name__)
#
#
# class MarketDataPublisher(Publisher, abc.ABC):
#     def __init__(self, ws_url):
#         super().__init__(ws_url)
#
#     def _on_message(self, ws, message):
#         """
#         Handler for incoming WebSocket messages.
#
#         Parses the message and passes it to the update_order_book method.
#
#         Args:
#             ws: WebSocket connection.
#             message (str): Raw message string from WebSocket.
#         """
#         logging.debug("%s: WebSocket message received: %s", self.__class__.__name__, message)
#         if not isinstance(message, str):
#             return
#         try:
#             timeReceived = time.time_ns()
#             data = loads(message)
#             self._parse_market_data(data, timeReceived, max_levels=MAX_LEVELS)
#         except Exception as e:
#             logging.error("%s: Error processing WebSocket message: %s", self.__class__.__name__, e)
#
#     @abstractmethod
#     def _parse_market_data(self, data, timeReceived, max_levels):
#         pass
#
#     @staticmethod
#     def subject(exchange, symbol):
#         """
#         Construct a subject name for publishing order book updates.
#
#         Args:
#             exchange (str): Exchange name.
#             symbol (str): Trading symbol.
#
#         Returns:
#             str: Formatted subject string.
#         """
#         return f"MarketData_{exchange}_{symbol}"
#
#     def publish_market_data(self, exchange, symbol, side, price, qty, time_exchange, time_received):
#         subject = self.subject(exchange, symbol)
#
#         msg = {
#             "exchange": exchange,
#             "symbol": symbol,
#             "side": side,
#             "price": price,
#             "qty": qty,
#             "timeExchange": time_exchange,
#             "timeReceived": time_received,
#             "timePublished": time.time_ns(),
#         }
#
#         self.publisher_thread.publish(subject, msg)
#
#
# class CoinbaseMarketDataPublisher(MarketDataPublisher):
#     # TODO: event_time
#     def __init__(self, ws_url, symbols):
#         super().__init__(ws_url)
#         self._symbols = symbols
#
#     def _subscribe(self, ws):
#         message = {
#             "type": "subscribe",
#             "channel": "level2",
#             "product_ids": self._symbols,
#         }
#         ws.send(dumps(message))
#         logging.info("%s: Subscribed to Coinbase level2 for symbols: %s", self.__class__.__name__, self._symbols)
#
#     def _parse_market_data(self, data, time_received, max_levels):
#         time_exchange = datetime_str_to_nanoseconds(data.get("timestamp"), format="%Y-%m-%dT%H:%M:%S.%fZ")
#
#         for event in data.get("events", []):
#             symbol = event.get("product_id")
#             for upd in event.get("updates", []):
#                 side = upd.get("side")
#                 price = upd.get("price_level")
#                 qty = upd.get("new_quantity")
#                 if side and price and qty:
#                     try:
#                         self.publish_market_data(Exchange.COINBASE, symbol, side, float(price), float(qty),
#                                                  time_exchange, time_received)
#                     except Exception as e:
#                         logging.error("%s: Failed to update %s %s: %s", self.__class__.__name__, side, price, e)
#
#
# if __name__ == "__main__":
#     coinbase_market_data_publisher = CoinbaseMarketDataPublisher(
#         ws_url="wss://advanced-trade-ws.coinbase.com",
#         symbols=["BTC-USD"]
#     )
#
#     coinbase_thread = threading.Thread(target=coinbase_market_data_publisher.start, kwargs={'block': False})
#
#     coinbase_thread.start()
#
#     time.sleep(20)
#
#     coinbase_market_data_publisher.end()
#
#     coinbase_thread.join()
