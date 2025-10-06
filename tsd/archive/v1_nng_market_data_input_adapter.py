import json
import logging
import threading
import time
from datetime import datetime, timedelta

import csp
import pynng
from csp import ts
from csp.impl.adaptermanager import AdapterManagerImpl
from csp.impl.pushadapter import PushInputAdapter
from csp.impl.wiring import py_push_adapter_def

from core.types.exchange import Exchange
from core.types.instrument import Instrument
from core.types.marketdata import OrderBook, MarketOrder
from archive.v1_nng_market_data_publisher import MarketDataPublisher

logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)


class MyAdapterManager:
    def __init__(self, exchange: str):
        logging.info("MyAdapterManager::__init__")
        self._exchange = exchange

    def subscribe(self, symbol, push_mode=csp.PushMode.NON_COLLAPSING):
        return MyPushAdapter(self, symbol, push_mode=push_mode)

    def _create(self, engine, memo):
        return MyAdapterManagerImpl(engine, self._exchange)


class MyAdapterManagerImpl(AdapterManagerImpl):
    def __init__(self, engine, exchange):
        super().__init__(engine)
        logging.info("MyAdapterManagerImpl::__init__")

        self._exchange = exchange
        self._inputs = {}
        self._running = False
        self._nng_thread = None
        self._socket = pynng.Sub0()
        self._subscriptions_initialized = threading.Event()
        self._connected = False

    def start(self, starttime, endtime):
        logging.info("MyAdapterManagerImpl::start")

        self._subscriptions_initialized.wait(timeout=2)

        self._handshake_with_publisher()

        if not self._connected:
            for attempt in range(10):
                try:
                    self._socket.dial("tcp://127.0.0.1:5555", block=True)
                    self._connected = True
                    logging.info("Dialed NNG socket to tcp://localhost:5555")
                    break
                except pynng.exceptions.ConnectionRefused:
                    logging.warning(f"Dial attempt {attempt + 1}/10 failed; retrying in 0.5s...")
                    time.sleep(0.5)
            else:
                logging.error("Failed to dial NNG socket after multiple retries.")
                raise ConnectionError("NNG subscriber could not connect to publisher.")

        self._running = True
        self._nng_thread = threading.Thread(target=self._run_nng_loop, daemon=True)
        self._nng_thread.start()

    def _handshake_with_publisher(self, control_url="tcp://127.0.0.1:6666"):
        logging.info("Sending handshake to publisher at %s...", control_url)
        with pynng.Req0(dial=control_url) as req:
            req.send(b"subscriber_ready")
            ack = req.recv().decode()
            logging.info("Received handshake ack: %s", ack)

    def stop(self):
        logging.info("MyAdapterManagerImpl::stop")
        self._running = False
        if self._nng_thread:
            self._nng_thread.join(timeout=2)
        try:
            self._socket.close()
        except Exception as e:
            logging.error(f"Error closing NNG socket: {e}")

    def _run_nng_loop(self):
        while self._running:
            try:
                msg_bytes = self._socket.recv()
                msg = msg_bytes.decode("utf-8")
                subject, raw_data = msg.split(' ', 1)
                logging.info(f"[NNG RECEIVED] Subject: {subject}: {raw_data}")
                self.process_message({"data": raw_data})
            except Exception as e:
                if self._running:
                    logging.error(f"[NNG ERROR] Exception in NNG loop: {e}")

    def register_input_adapter(self, symbol, adapter):
        if symbol not in self._inputs:
            self._inputs[symbol] = []

            topic = MarketDataPublisher.subject(self._exchange, symbol)
            self._socket.subscribe(topic.encode())
            logging.info(f"Subscribed to NNG topic: {topic}")

        self._inputs[symbol].append(adapter)
        logging.info(f"Registered adapter for: {symbol}")

        if len(self._inputs) > 0:
            self._subscriptions_initialized.set()

    def process_next_sim_timeslice(self, now):
        return None

    def process_message(self, msg):
        logging.debug(f"[PROCESS_MESSAGE] Raw msg: {msg}")

        if not isinstance(msg, dict):
            logging.error("[PROCESS_MESSAGE] Invalid message format (not a dict).")
            return

        raw_data = msg.get("data")
        try:
            parsed_data = json.loads(raw_data) if isinstance(raw_data, str) else raw_data
        except json.JSONDecodeError:
            logging.error(f"[PROCESS_MESSAGE] JSON decode failed: {raw_data}")
            return

        if "data" in parsed_data and isinstance(parsed_data["data"], dict):
            parsed_data = parsed_data["data"]

        required_keys = [
            "symbol", "side", "price", "qty",
            "timeExchange", "timeReceived", "timePublished"
        ]
        missing_keys = [key for key in required_keys if key not in parsed_data]
        if missing_keys:
            logging.warning(f"[PROCESS_MESSAGE] Missing keys: {missing_keys}")
            return

        symbol = parsed_data["symbol"]
        if symbol not in self._inputs:
            logging.warning(f"[PROCESS_MESSAGE] Unsubscribed symbol: {symbol}")
            return

        exchange = parsed_data["exchange"]
        side = parsed_data["side"]

        try:
            instr = Instrument(name=symbol)
            order = MarketOrder(
                instr=instr,
                exchange=Exchange(exchange),
                price=parsed_data["price"],
                qty=parsed_data["qty"],
                time_exchange=parsed_data["timeExchange"],
                time_received=parsed_data["timePublished"]
            )
            my_data = OrderBook(
                instr=instr,
                bids=[order] if side == "bid" else [],
                asks=[order] if side == "offer" else [],
                time_exchange=parsed_data["timeExchange"],
                time_received=parsed_data["timePublished"]
            )
        except Exception as e:
            logging.error(f"[PROCESS_MESSAGE] Failed to construct MyData: {e}")
            return

        for adapter in self._inputs[symbol]:
            adapter.push_tick(my_data)


class MyPushAdapterImpl(PushInputAdapter):
    def __init__(self, manager_impl, symbol):
        logging.info(f"MyPushAdapterImpl::__init__ {symbol}")
        manager_impl.register_input_adapter(symbol, self)
        super().__init__()


MyPushAdapter = py_push_adapter_def(
    "MyPushAdapter",
    MyPushAdapterImpl,
    ts[OrderBook],
    MyAdapterManager,
    symbol=str
)


@csp.node
def log_mid_price(data: ts[OrderBook]) -> ts[float]:
    with csp.alarms():
        alarm = csp.alarm(bool)

    with csp.state():
        s_mid = 0.0

    with csp.start():
        csp.schedule_alarm(alarm, timedelta(seconds=1), True)

    if csp.ticked(data):
        if data.bids and data.asks:
            s_mid = (data.bids[0].price + data.asks[0].price) / 2.0

    if csp.ticked(alarm):
        csp.schedule_alarm(alarm, timedelta(seconds=1), True)
        return s_mid


@csp.graph
def my_graph():
    logging.info("Building CSP graph")
    adapter_manager = MyAdapterManager(exchange="COINBASE")
    symbols = ["BTC-USD"]

    for symbol in symbols:
        data = adapter_manager.subscribe(symbol, csp.PushMode.NON_COLLAPSING)
        mid_price = log_mid_price(data)
        csp.print(f"{symbol} mid price:", mid_price)

    logging.info("Graph building complete")


def run_my_graph():
    csp.run(
        my_graph,
        starttime=datetime.utcnow(),
        endtime=timedelta(seconds=10),
        realtime=True
    )


if __name__ == "__main__":
    run_my_graph()
