import hashlib
import json
import logging
import threading
import time
from abc import ABC, abstractmethod
from collections import deque
from datetime import datetime, timezone
from typing import Callable

import orjson as json_parser
import websocket
import zmq
from csp import PushMode, ts
from csp.impl.adaptermanager import AdapterManagerImpl
from csp.impl.pushadapter import PushInputAdapter
from csp.impl.wiring import py_push_adapter_def


# -----------------------------
# Abstract Base for Transport
# -----------------------------

class BaseTransport(ABC):
    @abstractmethod
    def start(self):
        pass

    @abstractmethod
    def stop(self):
        pass


class PubTransport(BaseTransport):
    @abstractmethod
    def publish(self, topic: str, msg: dict):
        pass


class SubTransport(BaseTransport):
    @abstractmethod
    def subscribe(self, topic: str, msg_handler: Callable[[str, dict], None]):
        pass


# --------------------------------
# Concrete ZMQ Publisher Transport
# --------------------------------

class ZmqPubTransport(PubTransport, ABC):
    """
    ZMQ PUB socket-based transport for publishing serialized messages.

    Features:
        - Asynchronous publishing via thread and queue.
        - Duplicate detection if debug=True.
    """

    def __init__(self, zmq_bind_addr: str, debug: bool = False):
        self._context = zmq.Context()
        self._socket = self._context.socket(zmq.PUB)
        self._socket.setsockopt(zmq.SNDHWM, 0)
        self._socket.bind(zmq_bind_addr)

        self.queue = deque()
        self.condition = threading.Condition()
        self.running = threading.Event()
        self.running.set()

        self.debug = debug
        self._last_hashes = deque(maxlen=50) if debug else None

        self.thread = threading.Thread(target=self.start, daemon=True)
        self.thread.start()

    def publish(self, topic: str, msg: dict):
        with self.condition:
            self.queue.append((topic, msg))
            self.condition.notify()

    def start(self):
        while self.running.is_set():
            with self.condition:
                while not self.queue and self.running.is_set():
                    self.condition.wait()
                if not self.running.is_set():
                    break
                topic, msg = self.queue.popleft()

            try:
                if self.debug:
                    msg_str = json.dumps(msg, sort_keys=True)
                    hash_val = hashlib.md5((topic + msg_str).encode()).hexdigest()
                    if hash_val in self._last_hashes:
                        logging.warning("ZmqPubTransport duplicate detected for topic %s", topic)
                    self._last_hashes.append(hash_val)

                self._socket.send_multipart([topic.encode(), json_parser.dumps(msg)], copy=False)
            except Exception as e:
                logging.error("ZmqPubTransport error: %s", e)

    def stop(self):
        with self.condition:
            self.running.clear()
            self.condition.notify()
        self.thread.join(timeout=2)
        self._socket.close()
        self._context.term()


# -------------------------
# WebSocket Client Mixin
# -------------------------

class WebsocketClient(ABC):
    def __init__(self, ws_url: str):
        self.ws_url = ws_url
        self.ws_app = None
        self.ws_thread = None

    def start(self):
        logging.info("%s: WebSocket streaming started", self.__class__.__name__)
        self.ws_app = websocket.WebSocketApp(
            self.ws_url,
            on_open=self.on_open,
            on_message=self.on_message
        )
        self.ws_thread = threading.Thread(target=self.ws_app.run_forever, daemon=True)
        self.ws_thread.start()

    def stop(self):
        logging.info("%s: Closing WebSocket connection", self.__class__.__name__)
        if self.ws_app:
            self.ws_app.keep_running = False
            self.ws_app.close()
        if self.ws_thread:
            self.ws_thread.join(timeout=2)

    @abstractmethod
    def on_open(self, ws):
        pass

    @abstractmethod
    def on_message(self, ws, message):
        pass


# ----------------------------------
# Combined Publisher Implementation
# ----------------------------------

class WebsocketPublisher(WebsocketClient, ABC):
    """
    Base class to receive WebSocket data and publish to a transport layer.

    Args:
        transport (PubTransport): Underlying transport to publish messages.
        verbose (bool): Log message throughput.
        debug (bool): Detect duplicate messages.
        log_on_change (bool): Log only when new messages differ.
    """

    def __init__(self, ws_url: str, transport: PubTransport, verbose: bool = False,
                 debug: bool = False, log_on_change: bool = False):
        WebsocketClient.__init__(self, ws_url)
        self.transport = transport
        self.verbose = verbose
        self.debug = debug
        self.log_on_change = log_on_change

        self._msg_count = 0
        self._last_msg = None
        self._last_topic = None
        self._lock = threading.Lock()
        self._published_hashes = deque(maxlen=10) if debug else None

        if self.verbose and not self.log_on_change:
            threading.Thread(target=self._log_loop, daemon=True).start()

    def publish(self, topic: str, msg: dict):
        if self.debug:
            msg_str = json.dumps(msg, sort_keys=True)
            hash_val = hashlib.md5((topic + msg_str).encode()).hexdigest()
            if hash_val in self._published_hashes:
                logging.warning("Publisher duplicate detected for topic %s", topic)
            self._published_hashes.append(hash_val)

        with self._lock:
            changed = (self._last_topic != topic or self._last_msg != msg)
            self._msg_count += 1

            if self.verbose and self.log_on_change and changed:
                logging.info("%s: [%s] Topic: %s | Msg: %s",
                             self.__class__.__name__, datetime.now(timezone.utc).isoformat(), topic, msg)

            self._last_topic = topic
            self._last_msg = msg

        self.transport.publish(topic, msg)

    def _log_loop(self):
        has_started = False
        while True:
            time.sleep(1)
            with self._lock:
                count = self._msg_count
                topic = self._last_topic
                msg = self._last_msg
                self._msg_count = 0

            if count > 0:
                has_started = True
            if not has_started:
                continue

            logging.info("%s: [%s] %d msg/sec | Topic: %s | Msg: %s",
                         self.__class__.__name__, datetime.now(timezone.utc).isoformat(),
                         count, topic or "N/A", str(msg) if msg else "N/A")


# ---------------------------------
# ZMQ Subscriber with Dispatch Hook
# ---------------------------------

class ZmqSubTransport(SubTransport, ABC):
    """
    Generic ZeroMQ SUB transport with topic-based dispatch.

    Args:
        zmq_connect_addr (str): The ZeroMQ URL to connect to.
    """

    def __init__(self, zmq_connect_addr: str):
        self._context = zmq.Context()
        self._socket = self._context.socket(zmq.SUB)
        self._socket.connect(zmq_connect_addr)
        self._running = False
        self._thread = None
        self._msg_handlers = {}

    def subscribe(self, topic: str, msg_handler: Callable[[str, dict], None]):
        self._socket.setsockopt_string(zmq.SUBSCRIBE, topic)
        self._msg_handlers[topic] = msg_handler
        logging.info("%s: Subscribed to topic %s", self.__class__.__name__, topic)

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._listen_loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=2)
        self._socket.close()
        self._context.term()

    def _listen_loop(self):
        while self._running:
            try:
                topic, raw_msg = self._socket.recv_multipart()
                msg = json_parser.loads(raw_msg)
                topic_str = topic.decode()
                if topic_str in self._msg_handlers:
                    self._msg_handlers[topic_str](topic_str, msg)
                else:
                    logging.warning("No handler for topic: %s", topic_str)
            except Exception as e:
                logging.error("%s: Error in listen loop: %s", self.__class__.__name__, e)


class Subscriber(ABC):
    def __init__(self, sub_transport: SubTransport):
        self._transport = sub_transport

    def start(self):
        self._transport.start()

    def stop(self):
        self._transport.stop()

    def subscribe(self, topic: str, **kwargs):
        self._transport.subscribe(topic, lambda t, m: self._handle_message(t, m, **kwargs))

    @abstractmethod
    def _handle_message(self, topic: str, msg: dict, **kwargs):
        pass


class PushInputAdapterManager(ABC):
    def __init__(self, subscriber: Subscriber):
        self._subscriber = subscriber

    class PushInputAdapterImpl(PushInputAdapter):
        def __init__(self, manager_impl, topic):
            super().__init__()
            manager_impl.register_input_adapter(self, topic)

    class PushInputAdapterManagerImpl(AdapterManagerImpl):
        def __init__(self, engine, subscriber: Subscriber):
            super().__init__(engine)
            self._subscriber = subscriber
            self._adapters = {}

        def register_input_adapter(self, adapter, topic: str):
            self._adapters[topic] = adapter
            self._subscriber.subscribe(topic, adapter=adapter)

        def start(self, starttime, endtime):
            self._subscriber.start()

        def stop(self):
            self._subscriber.stop()

    def subscribe(self, topic: str, out_type, push_mode: PushMode = PushMode.NON_COLLAPSING):
        push_adapter = py_push_adapter_def(
            self.__class__.__name__,
            self.PushInputAdapterImpl,
            ts[out_type],
            self,
            topic=str
        )
        return push_adapter(self, topic, push_mode=push_mode)

    def _create(self, engine, memo):
        return self.PushInputAdapterManagerImpl(engine, self._subscriber)
