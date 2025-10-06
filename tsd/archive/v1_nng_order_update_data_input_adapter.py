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

logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)


class OrderUpdateAdapterManager:
    def __init__(self, exchange: str):
        self._exchange = exchange

    def subscribe(self, symbol, push_mode=csp.PushMode.NON_COLLAPSING):
        return OrderUpdatePushAdapter(self, symbol, push_mode=push_mode)

    def _create(self, engine, memo):
        return OrderUpdateAdapterManagerImpl(engine, self._exchange)


class OrderUpdateAdapterManagerImpl(AdapterManagerImpl):
    def __init__(self, engine, exchange):
        super().__init__(engine)
        self._exchange = exchange
        self._inputs = {}
        self._running = False
        self._nng_thread = None
        self._socket = pynng.Sub0()
        self._subscriptions_initialized = threading.Event()
        self._connected = False

    def start(self, starttime, endtime):
        self._subscriptions_initialized.wait(timeout=2)
        self._handshake_with_publisher()

        if not self._connected:
            for attempt in range(10):
                try:
                    self._socket.dial("tcp://127.0.0.1:5555", block=True)
                    self._connected = True
                    logging.info("Connected to publisher via tcp://127.0.0.1:5555")
                    break
                except pynng.exceptions.ConnectionRefused:
                    logging.warning(f"Dial attempt {attempt + 1}/10 failed; retrying...")
                    time.sleep(0.5)
            else:
                raise ConnectionError("Unable to connect to publisher")

        self._running = True
        self._nng_thread = threading.Thread(target=self._nng_loop, daemon=True)
        self._nng_thread.start()

    def _handshake_with_publisher(self, control_url="tcp://127.0.0.1:6666"):
        with pynng.Req0(dial=control_url) as req:
            req.send(b"subscriber_ready")
            ack = req.recv().decode()
            logging.info("Handshake ack received: %s", ack)

    def stop(self):
        self._running = False
        if self._nng_thread:
            self._nng_thread.join(timeout=2)
        self._socket.close()

    def register_input_adapter(self, symbol, adapter):
        if symbol not in self._inputs:
            self._inputs[symbol] = []
            topic = f"OrderUpdate_{self._exchange}_{symbol}"
            self._socket.subscribe(topic.encode())
            logging.info(f"Subscribed to topic: {topic}")
        self._inputs[symbol].append(adapter)
        if self._inputs:
            self._subscriptions_initialized.set()

    def _nng_loop(self):
        while self._running:
            try:
                msg_bytes = self._socket.recv()
                msg = msg_bytes.decode("utf-8")
                subject, raw_data = msg.split(' ', 1)
                self._process_message(subject, raw_data)
            except Exception as e:
                logging.error(f"NNG loop error: {e}")

    def _process_message(self, subject, raw_data):
        try:
            data = json.loads(raw_data)
        except json.JSONDecodeError as e:
            logging.warning(f"Invalid JSON: {e}")
            return

        symbol = data.get("product_id")
        if not symbol or symbol not in self._inputs:
            logging.warning(f"Unregistered or missing symbol: {symbol}")
            return

        # You can add minimal field validation or transformation here if needed
        order_update = {
            "exchange": data.get("exchange"),
            "client_order_id": data.get("client_order_id"),
            "order_id": data.get("order_id"),
            "retail_portfolio_id": data.get("retail_portfolio_id"),
            "symbol": data.get("product_id"),
            "product_type": data.get("product_type"),
            "order_side": data.get("order_side"),
            "order_type": data.get("order_type"),
            "status": data.get("status"),
            "time_in_force": data.get("time_in_force"),
            "avg_price": data.get("avg_price"),
            "filled_value": data.get("filled_value"),
            "limit_price": data.get("limit_price"),
            "stop_price": data.get("stop_price"),
            "cumulative_quantity": data.get("cumulative_quantity"),
            "leaves_quantity": data.get("leaves_quantity"),
            "number_of_fills": data.get("number_of_fills"),
            "total_fees": data.get("total_fees"),
            "cancel_reason": data.get("cancel_reason"),
            "reject_reason": data.get("reject_Reason"),
            "time_exchange": data.get("timeExchange"),
            "time_received": data.get("timeReceived"),
            "time_published": data.get("timePublished")
        }

        for adapter in self._inputs[symbol]:
            adapter.push_tick(order_update)


class OrderUpdatePushAdapterImpl(PushInputAdapter):
    def __init__(self, manager_impl, symbol):
        manager_impl.register_input_adapter(symbol, self)
        super().__init__()


OrderUpdatePushAdapter = py_push_adapter_def(
    "OrderUpdatePushAdapter",
    OrderUpdatePushAdapterImpl,
    ts[dict],  # <--- use dict instead of a typed class
    OrderUpdateAdapterManager,
    symbol=str
)

@csp.node
def log_order_status(update: ts[dict]) -> ts[str]:
    if csp.ticked(update):
        return f"{update.get('symbol')}: {update.get('status')} | Filled: {update.get('cumulative_quantity')}"


@csp.graph
def order_update_graph():
    manager = OrderUpdateAdapterManager(exchange="COINBASE")
    symbols = ["BTC-USD"]
    for symbol in symbols:
        updates = manager.subscribe(symbol)
        summary = log_order_status(updates)
        csp.print(f"{symbol} order update:", summary)


if __name__ == "__main__":
    csp.run(
        order_update_graph,
        starttime=datetime.utcnow(),
        endtime=timedelta(seconds=30),
        realtime=True
    )
