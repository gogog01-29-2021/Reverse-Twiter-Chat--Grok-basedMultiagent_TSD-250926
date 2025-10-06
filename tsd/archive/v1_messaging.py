import abc
import asyncio
import logging
import queue
import threading
import time
from typing import List

import websocket
from nats.aio.client import Client as NATS

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


class Streamer(abc.ABC):
    """
    Abstract base class for managing WebSocket connections and a background logging process.
    Subclasses must implement specific behaviors for subscription, message handling, and logging.
    """

    def __init__(self, ws_url: str):
        self.ws_url = ws_url
        self.ws_app = None
        self.ws_thread = None
        self.logging_running = False
        self.logging_thread = None

    @abc.abstractmethod
    def subscribe(self, ws):
        """
        Define the WebSocket subscription logic. Invoked upon connection open.

        Args:
            ws (websocket.WebSocket): Active WebSocket instance.
        """
        pass

    @abc.abstractmethod
    def on_message(self, ws, message):
        """
        Handle incoming messages from the WebSocket.

        Args:
            ws (websocket.WebSocket): Active WebSocket instance.
            message (str): Incoming message as a string.
        """
        pass

    @abc.abstractmethod
    def logging_loop(self):
        """
        Define the logging behavior in a separate background thread.
        """
        pass

    def start_streaming(self):
        """
        Start the WebSocket connection and message processing in a separate thread.
        """
        self.ws_app = websocket.WebSocketApp(
            self.ws_url,
            on_open=lambda ws: self.subscribe(ws),
            on_message=self.on_message
        )
        self.ws_thread = threading.Thread(target=self.ws_app.run_forever, daemon=True)
        self.ws_thread.start()
        logging.info("%s: Started WebSocket streaming", self.__class__.__name__)
        return self.ws_thread

    def start(self, block: bool = True):
        """
        Start the full streaming system, including WebSocket and background logging.

        Args:
            block (bool): If True, blocks the main thread until interrupted.
        """
        self.start_streaming()
        self.start_logging()
        logging.info("%s: Publisher is running", self.__class__.__name__)

        if block:
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                self.end()

    def end(self):
        """
        Stop all components: WebSocket, logging, and clean up threads.
        """
        logging.info("%s: Stopping publisher", self.__class__.__name__)
        self.stop_logging()
        if self.ws_app:
            self.ws_app.keep_running = False
            self.ws_app.close()
        if self.ws_thread:
            self.ws_thread.join(timeout=2)

    def start_logging(self):
        """
        Launch the logging loop in a daemon thread.
        """
        self.logging_running = True
        self.logging_thread = threading.Thread(target=self.logging_loop, daemon=True)
        self.logging_thread.start()

    def stop_logging(self):
        """
        Gracefully stop the logging thread.
        """
        self.logging_running = False
        if self.logging_thread:
            self.logging_thread.join(timeout=2)


class Publisher(Streamer, abc.ABC):
    """
    Abstract base class to stream data from WebSocket and publish it to NATS asynchronously.
    Subclasses must implement the subject determination, subscription, message handling,
    and logging behavior.
    """

    def __init__(self, ws_url: str, nats_url: str = "nats://localhost:4222"):
        super().__init__(ws_url)

        self.nats_client = NATS()
        self.loop = asyncio.new_event_loop()
        self.loop_thread = threading.Thread(target=self.run_loop, daemon=True)
        self.loop_thread.start()

        # Connect to NATS server asynchronously
        future = asyncio.run_coroutine_threadsafe(
            self.nats_client.connect(servers=[nats_url]), self.loop
        )
        future.result()  # Wait for NATS connection to complete
        logging.info("%s: Connected to NATS", self.__class__.__name__)

        # Initialize and start the asynchronous publisher thread
        self.publisher_thread = PublisherThread(self.nats_client, self.loop)
        self.publisher_thread.start()

    @abc.abstractmethod
    def subject(self, *args, **kwargs):
        """
        Define the NATS subject/channel to which messages should be published.
        Must be implemented by subclasses.
        """
        pass

    def run_loop(self):
        """
        Run the asyncio event loop in its dedicated thread.
        """
        asyncio.set_event_loop(self.loop)
        self.loop.run_forever()

    def end(self):
        """
        Stop the WebSocket, logging loop, and NATS publisher thread.
        Gracefully shuts down all resources.
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
        logging.info("%s: Publisher stopped", self.__class__.__name__)


class PublisherThread(threading.Thread):
    """
    Dedicated thread for asynchronously publishing messages to NATS.
    Messages are queued and sent without blocking the main processing logic.
    """

    def __init__(self, nats_client, loop):
        super().__init__(daemon=True)
        self.nats_client = nats_client
        self.loop = loop
        self.queue = queue.Queue()
        self.running = True

    def run(self):
        """
        Thread loop to process and publish queued messages to NATS.
        """
        while self.running:
            try:
                data = self.queue.get()
                payload = dumps(data["msg"]).encode("utf-8")
                asyncio.run_coroutine_threadsafe(
                    self.nats_client.publish(data["subject"], payload),
                    self.loop
                )
                self.queue.task_done()
            except queue.Empty:
                continue

    def publish(self, subject: str, msg: dict):
        """
        Add a message to the publishing queue.

        Args:
            subject (str): NATS subject name.
            msg (dict): JSON-serializable message payload.
        """
        self.queue.put({"subject": subject, "msg": msg})

    def stop(self):
        """
        Signal the thread to terminate.
        """
        self.running = False


class Subscriber(abc.ABC):
    """
    Abstract base class for subscribing to one or more NATS subjects.
    Handles setup and graceful shutdown of the NATS client connection.
    """

    def __init__(self, subjects: List[str], nats_url: str = "nats://localhost:4222"):
        self.subjects = subjects
        self.nats_url = nats_url
        self.nc = NATS()
        self._running = False

    @abc.abstractmethod
    async def _handle_message(self, msg):
        """
        Process messages received from NATS.

        Args:
            msg: The message object received from NATS.
        """
        pass

    async def connect(self):
        """
        Establish a connection to the configured NATS server.
        """
        await self.nc.connect(servers=[self.nats_url])
        logging.info("Connected to NATS at %s", self.nats_url)

    async def subscribe(self):
        """
        Subscribe to all specified NATS subjects with the defined message handler.
        """
        for subject in self.subjects:
            await self.nc.subscribe(subject, cb=self._handle_message)
            logging.info("Subscribed to subject: %s", subject)

    async def run(self):
        """
        Main loop for the subscriber. Continuously listens for messages.
        """
        self._running = True
        await self.connect()
        await self.subscribe()

        logging.info("Listening for messages...")
        while self._running:
            await asyncio.sleep(1)

    async def shutdown(self):
        """
        Gracefully close the NATS connection and stop listening.
        """
        self._running = False
        await self.nc.close()
        logging.info("NATS connection closed.")
