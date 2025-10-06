import hashlib
import json
import logging
import threading
import time
from abc import ABC, abstractmethod
from collections import deque
from datetime import datetime
from typing import Callable, Dict, List, Tuple

import orjson as json_parser
import websocket
import zmq
from csp import PushMode, ts
from csp.impl.adaptermanager import AdapterManagerImpl
from csp.impl.pushadapter import PushInputAdapter
from csp.impl.wiring import py_push_adapter_def

logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)


# ---------- Transport Layer ----------

class PubTransport(ABC):
    @abstractmethod
    def send(self, topic: str, msg: dict):
        pass


class ZmqPubTransport(PubTransport):
    def __init__(self, zmq_bind_addr: str, debug: bool = False):
        self._context = zmq.Context()
        self._socket = self._context.socket(zmq.PUB)
        self._socket.setsockopt(zmq.SNDHWM, 0)
        self._socket.bind(zmq_bind_addr)

        self._queue = deque()
        self._condition = threading.Condition()
        self._running = threading.Event()
        self._running.set()
        self._debug = debug
        self._last_hashes = deque(maxlen=50) if debug else None

        self._thread = threading.Thread(target=self._publish_loop, daemon=True)
        self._thread.start()

    def send(self, topic: str, msg: dict):
        with self._condition:
            self._queue.append((topic, msg))
            self._condition.notify()

    def _publish_loop(self):
        while self._running.is_set():
            with self._condition:
                while not self._queue and self._running.is_set():
                    self._condition.wait()
                if not self._running.is_set():
                    break
                topic, msg = self._queue.popleft()

            try:
                if self._debug:
                    msg_str = json.dumps(msg, sort_keys=True)
                    hash_val = hashlib.md5((topic + msg_str).encode()).hexdigest()
                    if hash_val in self._last_hashes:
                        logging.warning("ZmqPubTransport duplicate detected for topic %s", topic)
                    self._last_hashes.append(hash_val)

                self._socket.send_multipart([topic.encode(), json_parser.dumps(msg)], copy=False)
            except Exception as e:
                logging.error("ZmqPubTransport error: %s", e)

    def stop(self):
        with self._condition:
            self._running.clear()
            self._condition.notify()
        self._thread.join(timeout=2)
        self._socket.close()
        self._context.term()


# ---------- WebSocket Streamer ----------

class WebsocketStreamer(ABC):
    def __init__(self, ws_url: str):
        self._ws_url = ws_url
        self._ws_app = None
        self._thread = None

    def start(self):
        logging.info("%s: WebSocket streaming started", self.__class__.__name__)
        self._ws_app = websocket.WebSocketApp(
            self._ws_url,
            on_open=self.on_open,
            on_message=self.on_message
        )
        self._thread = threading.Thread(target=self._ws_app.run_forever, daemon=True)
        self._thread.start()

    def stop(self):
        if self._ws_app:
            self._ws_app.keep_running = False
            self._ws_app.close()
        if self._thread:
            self._thread.join(timeout=2)

    @abstractmethod
    def on_open(self, ws):
        pass

    @abstractmethod
    def on_message(self, ws, message):
        pass


# ---------- Parser Interface ----------

class MessageParser(ABC):
    @abstractmethod
    def parse(self, raw_message: str, time_received: int) -> List[Tuple[str, dict]]:
        pass


# ---------- Publisher Base ----------

class Publisher(WebsocketStreamer, ABC):
    def __init__(
            self,
            ws_url: str,
            transport: PubTransport,
            parser: MessageParser,
            verbose: bool = False,
            debug: bool = False,
            log_on_change: bool = False
    ):
        super().__init__(ws_url)
        self._transport = transport
        self._parser = parser
        self._verbose = verbose
        self._debug = debug
        self._log_on_change = log_on_change

        self._msg_count = 0
        self._last_topic = None
        self._last_msg = None
        self._hashes = deque(maxlen=10) if debug else None
        self._lock = threading.Lock()

        if verbose and not log_on_change:
            threading.Thread(target=self._log_loop, daemon=True).start()

    def publish(self, topic: str, msg: dict):
        with self._lock:
            changed = (self._last_topic != topic or self._last_msg != msg)
            self._msg_count += 1

            if self._debug:
                msg_str = json.dumps(msg, sort_keys=True)
                h = hashlib.md5((topic + msg_str).encode()).hexdigest()
                if h in self._hashes:
                    logging.warning("Duplicate detected on topic %s", topic)
                self._hashes.append(h)

            if self._verbose and self._log_on_change and changed:
                logging.info("[change] %s | %s", topic, msg)

            self._last_topic = topic
            self._last_msg = msg

        self._transport.send(topic, msg)

    def on_message(self, ws, message):
        if isinstance(message, bytes):
            message = message.decode()
        if not isinstance(message, str):
            return

        time_received = int(datetime.utcnow().timestamp() * 1000)
        for topic, msg in self._parser.parse(message, time_received):
            self.publish(topic, msg)

    def _log_loop(self):
        while True:
            time.sleep(1)
            with self._lock:
                count, topic, msg = self._msg_count, self._last_topic, self._last_msg
                self._msg_count = 0
            if count > 0:
                logging.info("[rate] %d msg/sec | %s | %s", count, topic, msg)

    @abstractmethod
    def topic(self, *args):
        pass


class SubTransport(ABC):
    @abstractmethod
    def subscribe(self, topic: str, handler: Callable[[str, dict], None]):
        pass

    @abstractmethod
    def start(self):
        pass

    @abstractmethod
    def stop(self):
        pass


class ZmqSubTransport(SubTransport):
    def __init__(self, zmq_connect_addr: str):
        self._context = zmq.Context()
        self._socket = self._context.socket(zmq.SUB)
        self._socket.connect(zmq_connect_addr)
        self._running = False
        self._thread = None
        self._handlers: Dict[str, Callable[[str, dict], None]] = {}

    def subscribe(self, topic: str, handler: Callable[[str, dict], None]):
        self._socket.setsockopt_string(zmq.SUBSCRIBE, topic)
        self._handlers[topic] = handler
        logging.info("ZmqSubTransport: Subscribed to topic: %s", topic)

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
                topic_bytes, raw_msg = self._socket.recv_multipart()
                topic = topic_bytes.decode()
                msg = json_parser.loads(raw_msg)
                handler = self._handlers.get(topic)
                if handler:
                    handler(topic, msg)
                else:
                    logging.warning("ZmqSubTransport: Unhandled topic: %s", topic)
            except Exception as e:
                logging.error("ZmqSubTransport error: %s", e)


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
