import logging
import threading
from datetime import datetime, timedelta

import csp
import zmq
from csp import ts
from csp.impl.adaptermanager import AdapterManagerImpl
from csp.impl.pushadapter import PushInputAdapter
from csp.impl.wiring import py_push_adapter_def

from core.types.instrument import Instrument, Currency, Spot
from core.types.exchange import Exchange
from core.types.marketdata import OrderBook, MarketOrder, Side
from order_book_publisher import OrderBookPublisher
from order_book_publisher import loads

logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)


class ZMQAdapterManagerImplBase(AdapterManagerImpl):
    def __init__(self, engine, zmq_url):
        super().__init__(engine)
        self._inputs = {}
        self._running = False
        self._zmq_thread = None
        self._context = zmq.Context()
        self._socket = self._context.socket(zmq.SUB)
        self._socket.connect(zmq_url)
        self._subscriptions_initialized = threading.Event()

    def start(self, starttime, endtime):
        logging.info(f"{self.__class__.__name__}::start")
        self._running = True
        self._subscriptions_initialized.wait(timeout=2)
        self._zmq_thread = threading.Thread(target=self._run_zmq_loop, daemon=True)
        self._zmq_thread.start()

    def stop(self):
        logging.info(f"{self.__class__.__name__}::stop")
        self._running = False
        if self._zmq_thread:
            self._zmq_thread.join(timeout=2)
        self._socket.close()
        self._context.term()

    def register_input_adapter(self, symbol, adapter, topic_fn):
        if symbol not in self._inputs:
            self._inputs[symbol] = []
            topic = topic_fn(symbol)
            print(topic)
            self._socket.setsockopt_string(zmq.SUBSCRIBE, topic)
            logging.info(f"Subscribed to ZMQ topic: {topic}")

        self._inputs[symbol].append(adapter)
        logging.info(f"Registered adapter for: {symbol}")

        if len(self._inputs) > 0:
            self._subscriptions_initialized.set()

    def _run_zmq_loop(self):
        raise NotImplementedError


class GenericPushAdapterImpl(PushInputAdapter):
    def __init__(self, manager_impl, symbol):
        logging.info(f"GenericPushAdapterImpl::__init__ {symbol}")
        manager_impl.register_input_adapter(symbol, self, manager_impl.topic_fn)
        super().__init__()


class OrderBookAdapterManager:
    def __init__(self, exchange: str):
        logging.info("MyAdapterManager::__init__")
        self._exchange = exchange

    def subscribe(self, symbol, push_mode=None):
        from csp import PushMode
        if push_mode is None:
            push_mode = PushMode.NON_COLLAPSING
        return MyPushAdapter(self, symbol, push_mode=push_mode)

    def _create(self, engine, memo):
        return MyAdapterManagerImpl(engine, self._exchange)


class MyAdapterManagerImpl(ZMQAdapterManagerImplBase):
    def __init__(self, engine, exchange):
        super().__init__(engine, zmq_url="tcp://localhost:5556")
        self._exchange = exchange

    def _run_zmq_loop(self):
        while self._running:
            try:
                msg_parts = self._socket.recv_multipart()
                if len(msg_parts) != 2:
                    logging.warning(f"[ZMQ WARNING] Skipping malformed message: {msg_parts}")
                    continue

                topic, raw_msg = msg_parts
                if not topic.decode().startswith("OrderBook_"):
                    logging.debug(f"[ZMQ DEBUG] Ignored unrelated topic: {topic}")
                    continue
                data = loads(raw_msg)
                self.process_message(data)
            except Exception as e:
                logging.error(f"[ZMQ ERROR] Exception in ZMQ loop: {e}")

    def process_message(self, msg):
        raw_symbol = msg["symbol"]
        # Convert back to Spot and re-get normalized name
        symbol = raw_symbol  # e.g., "BTC-USD"

        wrapped_msg = {
            "exchange": msg["exchange"],
            "symbol": symbol,
            "bids": msg["bids"],
            "asks": msg["asks"],
            "timeExchange": msg["time_exchange"],
            "timeReceived": msg["time_received"],
        }

        for adapter in self._inputs.get(symbol, []):
            adapter.push_tick(wrapped_msg)

    def topic_fn(self, symbol):
        return OrderBookPublisher.topic(self._exchange, Spot(base=Currency.BTC, term=Currency.USD))


MyPushAdapter = py_push_adapter_def(
    "MyPushAdapter",
    GenericPushAdapterImpl,
    ts[dict],
    OrderBookAdapterManager,
    symbol=str,
)


def parse_iso8601_utc(ts: str) -> datetime:
    """
    Parses a timestamp in the format 'YYYY-MM-DDTHH:MM:SS.ssssss+00:00'
    into a timezone-aware datetime object (UTC).
    """
    return datetime.fromisoformat(ts)


@csp.node
def order_book_node(order_book: ts[dict]) -> ts[OrderBook]:
    exchange = Exchange(order_book["exchange"])
    instrument = Instrument(symbol=order_book["symbol"])
    time_exchange = parse_iso8601_utc(order_book["timeExchange"])
    time_received = parse_iso8601_utc(order_book["timeReceived"])

    # Directly generate MarketOrder lists (no need for SortedDicts)
    bids = [
        MarketOrder(
            instr=instrument,
            exchange=exchange,
            side=Side.BID,
            price=price,
            qty=qty,
            time_exchange=time_exchange,
            time_received=time_received
        )
        for price, qty in order_book["bids"]
    ]

    asks = [
        MarketOrder(
            instr=instrument,
            exchange=exchange,
            side=Side.ASK,
            price=price,
            qty=qty,
            time_exchange=time_exchange,
            time_received=time_received
        )
        for price, qty in order_book["asks"]
    ]

    return OrderBook(
        instr=instrument,
        bids=bids,
        asks=asks,
        time_exchange=time_exchange,
        time_received=time_received
    )


@csp.graph
def my_graph():
    logging.info("Building CSP graph")
    order_book_adapter_manager = OrderBookAdapterManager(exchange=Exchange.COINBASE)
    symbols = ["BTCUSD"]

    for symbol in symbols:
        order_book_data = order_book_adapter_manager.subscribe(symbol, csp.PushMode.NON_COLLAPSING)
        order_book = order_book_node(order_book=order_book_data)
        csp.print(f"{symbol} order book:", order_book)

    logging.info("Graph building complete")


def run_my_graph():
    csp.run(
        my_graph,
        starttime=datetime.utcnow(),
        endtime=timedelta(seconds=5),
        realtime=True
    )


if __name__ == "__main__":
    run_my_graph()
