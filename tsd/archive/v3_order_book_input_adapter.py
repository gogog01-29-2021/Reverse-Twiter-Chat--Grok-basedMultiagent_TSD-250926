import logging
import threading
from abc import ABC, abstractmethod
from datetime import datetime, timedelta, timezone

import csp
import orjson as json_parser
import zmq
from csp import PushMode
from csp import ts
from csp.impl.adaptermanager import AdapterManagerImpl
from csp.impl.pushadapter import PushInputAdapter
from csp.impl.wiring import py_push_adapter_def

from core.types.exchange import Exchange
from core.types.instrument import Spot, Currency
from core.types.marketdata import OrderBook, MarketOrder, Side
from dsm.utils.conversion_utils import epoch_ms_to_datetime, symbol_to_instrument
from archive.v3_order_book_publisher import CoinbaseOrderBookPublisher, ZmqSocketHandler

logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)


class ZmqSubscriber(ZmqSocketHandler, AdapterManagerImpl, ABC):
    def __init__(self, engine, zmq_url: str):
        AdapterManagerImpl.__init__(self, engine)
        ZmqSocketHandler.__init__(self)
        self._socket = self._context.socket(zmq.SUB)
        self._socket.connect(zmq_url)

        self._adapter = None
        self._running = False
        self._thread = None
        self._subscriptions_initialized = threading.Event()

    def start(self, starttime, endtime):
        logging.info(f"{self.__class__.__name__}::start")
        self._running = True
        self._subscriptions_initialized.wait(timeout=2)
        self._thread = threading.Thread(target=self._listen_loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=2)
        super().stop()

    def register_input_adapter(self, adapter, topic: str):
        self._socket.setsockopt_string(zmq.SUBSCRIBE, topic)
        logging.info(f"{self.__class__.__name__}: Subscribed to topic {topic}")
        self._adapter = adapter
        self._subscriptions_initialized.set()

    def _listen_loop(self):
        while self._running:
            try:
                topic, raw_msg = self._socket.recv_multipart()
                msg = json_parser.loads(raw_msg)
                self.handle_message(topic.decode(), msg)
            except Exception as e:
                logging.error(f"{self.__class__.__name__}: Error in _listen_loop - {e}")

    @abstractmethod
    def handle_message(self, topic: str, msg: dict):
        pass


class OrderBookSubscriber(ZmqSubscriber):
    def __init__(self, engine, zmq_url: str, exchange: Exchange):
        super().__init__(engine, zmq_url)
        self._exchange = exchange

    def handle_message(self, topic: str, msg: dict):
        exchange = Exchange(msg["exchange"])
        instrument = symbol_to_instrument(msg["symbol"])
        time_exchange = epoch_ms_to_datetime(msg["time_exchange"])
        time_received = epoch_ms_to_datetime(msg["time_published"])

        bids = [
            MarketOrder(
                instr=instrument,
                exchange=exchange,
                side=Side.BID,
                price=price,
                qty=qty,
                time_exchange=time_exchange,
                time_received=time_received
            ) for price, qty in msg["bids"]
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
            ) for price, qty in msg["asks"]
        ]

        order_book = OrderBook(
            instr=instrument,
            bids=bids,
            asks=asks,
            time_exchange=time_exchange,
            time_received=time_received
        )

        self._adapter.push_tick(order_book)


class OrderBookAdapterManager:
    def __init__(self, zmq_url, exchange):
        self._zmq_url = zmq_url
        self._exchange = exchange

    def subscribe(self, topic: str, push_mode=PushMode.NON_COLLAPSING):
        return OrderBookPushAdapter(self, topic, push_mode=push_mode)

    def _create(self, engine, memo):
        return OrderBookSubscriber(engine, zmq_url=self._zmq_url, exchange=self._exchange)


class OrderBookPushAdapterImpl(PushInputAdapter):
    def __init__(self, manager_impl, topic):
        super().__init__()
        manager_impl.register_input_adapter(self, topic)
        logging.info(f"OrderBookPushAdapterImpl::__init__ {topic}")


OrderBookPushAdapter = py_push_adapter_def(
    "OrderBookPushAdapter",
    OrderBookPushAdapterImpl,
    ts[OrderBook],
    OrderBookAdapterManager,
    topic=str,
)


@csp.graph
def my_graph():
    logging.info("Building CSP graph")
    order_book_adapter_manager = OrderBookAdapterManager(zmq_url="tcp://localhost:5556", exchange=Exchange.COINBASE)

    for instrument in [Spot(base=Currency.BTC, term=Currency.USD)]:
        topic = CoinbaseOrderBookPublisher.subject(Exchange.COINBASE, instrument)
        book = order_book_adapter_manager.subscribe(topic)
        csp.print(f"{instrument} order book:", book)


def run_my_graph():
    csp.run(
        my_graph,
        starttime=datetime.now(timezone.utc),
        endtime=timedelta(seconds=5),
        realtime=True
    )


if __name__ == "__main__":
    run_my_graph()
