import abc
import json
import logging
import queue
import socket
import threading
import time

import websocket

logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)


class Publisher(abc.ABC):
    """
    Abstract base class to manage WebSocket data streaming and publishing messages over UDP.
    Suitable for ultra-low latency environments, including AWS EC2 with ENA networking.
    """

    def __init__(self, ws_url, udp_host: str, udp_port: int):
        """
        Initialize the Publisher with WebSocket and UDP socket.

        Args:
            ws_url (str): The WebSocket endpoint URL.
            udp_host (str): IP address or hostname of the UDP receiver.
            udp_port (int): UDP port number to send messages to.
        """
        self.ws_url = ws_url

        # Setup UDP socket for low-latency sending
        self.udp_target = (udp_host, udp_port)
        self.udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.udp_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

        # Optional: minimize kernel buffering for latency
        self.udp_socket.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 1 << 20)

        # Initialize publisher thread
        self.publisher_thread = PublisherThread(self.udp_socket, self.udp_target)
        self.publisher_thread.start()

        # Placeholder variables
        self.ws_app = None
        self.ws_thread = None
        self.logging_running = False
        self.logging_thread = None

    @abc.abstractmethod
    def subscribe(self, ws):
        """
        Abstract method to define WebSocket subscription behavior.
        """
        pass

    @abc.abstractmethod
    def websocket_handler(self, ws, message):
        """
        Abstract method to handle incoming WebSocket messages.
        Should call self.publish(...) internally.
        """
        pass

    @abc.abstractmethod
    def logging_loop(self):
        """
        Optional background logging or stats loop.
        """
        pass

    def publish(self, msg: dict):
        """
        Send a message to the UDP publisher queue.

        Args:
            msg (dict): JSON-serializable message.
        """
        self.publisher_thread.publish(msg)

    def start_streaming(self):
        """
        Start the WebSocket client and begin data streaming.
        """
        self.ws_app = websocket.WebSocketApp(
            self.ws_url,
            on_open=lambda ws: self.subscribe(ws),
            on_message=self.websocket_handler
        )
        self.ws_thread = threading.Thread(target=self.ws_app.run_forever)
        self.ws_thread.daemon = True
        self.ws_thread.start()
        logging.info("%s: Started WebSocket streaming", self.__class__.__name__)
        return self.ws_thread

    def start(self, block=True):
        """
        Start the publisher system: WebSocket, UDP, and logging loop.

        Args:
            block (bool): If True, block until interrupted.
        """
        self.start_streaming()
        self.start_logging()
        logging.info("%s: publisher is running", self.__class__.__name__)
        if block:
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                self.end()

    def end(self):
        """
        Clean shutdown of WebSocket and background threads.
        """
        logging.info("%s: Stopping publisher", self.__class__.__name__)
        self.stop_logging()

        if self.ws_app is not None:
            self.ws_app.keep_running = False
            self.ws_app.close()

        self.publisher_thread.stop()

        if self.ws_thread is not None:
            self.ws_thread.join(timeout=2)

        logging.info("%s: publisher stopped", self.__class__.__name__)

    def start_logging(self):
        """
        Start optional background logging loop.
        """
        self.logging_running = True
        self.logging_thread = threading.Thread(target=self.logging_loop)
        self.logging_thread.daemon = True
        self.logging_thread.start()

    def stop_logging(self):
        """
        Stop the logging thread.
        """
        self.logging_running = False
        if self.logging_thread:
            self.logging_thread.join(timeout=2)


class PublisherThread(threading.Thread):
    """
    Background thread for sending messages over UDP in a non-blocking way.
    """

    def __init__(self, udp_socket: socket.socket, target: tuple):
        super().__init__()
        self.udp_socket = udp_socket
        self.target = target
        self.queue = queue.Queue()
        self.running = True
        self.daemon = True

    def run(self):
        """
        Main loop to drain message queue and transmit over UDP.
        """
        MAX_UDP_PAYLOAD_SIZE = 1400  # Safe for MTU, headers

        while self.running:
            try:
                msg = self.queue.get()
                payload = json.dumps(msg, separators=(',', ':')).encode("utf-8")

                if len(payload) > MAX_UDP_PAYLOAD_SIZE:
                    logging.warning("Dropped oversized UDP payload (%d bytes) for %s",
                                    len(payload), msg.get("symbol", "unknown"))
                    self.queue.task_done()
                    continue

                self.udp_socket.sendto(payload, self.target)
                self.queue.task_done()
            except Exception as e:
                logging.error("UDP publish failed: %s", e)

    def publish(self, msg: dict):
        """
        Queue a message for UDP publishing.

        Args:
            msg (dict): JSON-serializable data.
        """
        self.queue.put(msg)

    def stop(self):
        """
        Stop the thread gracefully.
        """
        self.running = False
