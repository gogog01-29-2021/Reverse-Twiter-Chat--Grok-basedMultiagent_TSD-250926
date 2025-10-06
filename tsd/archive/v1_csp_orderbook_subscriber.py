import asyncio
import json
import logging
import sys
import threading
from datetime import datetime, timedelta

import csp
from csp import ts
from csp.impl.adaptermanager import AdapterManagerImpl
from csp.impl.pushadapter import PushInputAdapter
from csp.impl.wiring import py_push_adapter_def
from nats.aio.client import Client as NATS

from core.types.exchange import Exchange
from core.types.instrument import Instrument
from core.types.marketdata import OrderBook, MarketOrder

# Initialize logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO
)


class MyAdapterManager:
    """
    Adapter manager responsible for handling subscriptions to symbols.
    """

    def __init__(self, exchange: str):
        logging.info("MyAdapterManager::__init__")
        self._exchange = exchange

    def subscribe(self, symbol, push_mode=csp.PushMode.NON_COLLAPSING):
        """
        Public API to subscribe to a symbol's data stream.
        """
        return MyPushAdapter(self, symbol, push_mode=push_mode)

    def _create(self, engine, memo):
        """
        Called by CSP engine to create the runtime manager implementation.
        """
        logging.info("MyAdapterManager::_create")
        return MyAdapterManagerImpl(engine, self._exchange)


class MyAdapterManagerImpl(AdapterManagerImpl):
    """
    Runtime implementation for MyAdapterManager. Handles NATS subscriptions,
    message processing, and pushing data into the CSP engine.
    """

    def __init__(self, engine, exchange):
        super().__init__(engine)
        logging.info("MyAdapterManagerImpl::__init__")

        self._exchange = exchange
        self._inputs = {}

        self._running = False
        self._nats_loop = asyncio.new_event_loop()
        self._nats_client = NATS()
        self._nats_thread = None
        self._subscriptions_initialized = threading.Event()

    def start(self, starttime, endtime):
        """
        Start the NATS listener thread.
        """
        logging.info("MyAdapterManagerImpl::start")
        self._running = True
        self._subscriptions_initialized.wait(timeout=2)

        self._nats_thread = threading.Thread(target=self._run_nats_loop, daemon=True)
        self._nats_thread.start()

    def stop(self):
        """
        Cleanly shut down the NATS listener and event loop.
        """
        logging.info("MyAdapterManagerImpl::stop")
        if not self._running:
            return
        self._running = False

        async def _shutdown():
            try:
                if self._nats_client.is_connected:
                    logging.info("Draining NATS client...")
                    await self._nats_client.drain()
                    await self._nats_client.close()
                    logging.info("NATS client closed.")
            except Exception as e:
                logging.error(f"Error during NATS client shutdown: {e}")

        future = asyncio.run_coroutine_threadsafe(_shutdown(), self._nats_loop)
        try:
            future.result(timeout=2)
        except Exception as e:
            logging.error(f"NATS shutdown future exception: {e}")

        self._nats_loop.call_soon_threadsafe(self._nats_loop.stop)
        self._nats_thread.join(timeout=2)

    def _run_nats_loop(self):
        """
        Entrypoint for running the NATS event loop in a background thread.
        """
        asyncio.set_event_loop(self._nats_loop)
        self._nats_loop.run_until_complete(self._nats_main())
        self._nats_loop.run_forever()

    async def _nats_main(self):
        """
        Main NATS client setup and subscription handler.
        """
        await self._nats_client.connect(servers=["nats://localhost:4222"])
        logging.info("Connected to NATS")

        async def message_handler(msg):
            try:
                raw_data = msg.data.decode()
                logging.debug(f"[NATS RECEIVED] Subject: {msg.topic}, Data: {raw_data}")
                self.process_message({"data": raw_data})
            except Exception as e:
                logging.error(f"[NATS ERROR] Exception in message_handler: {e}")

        for symbol in self._inputs:
            subject = f"ORDERBOOK_{self._exchange}_{symbol}"
            await self._nats_client.subscribe(subject, cb=message_handler)
            logging.info(f"Subscribed to NATS subject: {subject}")

    def register_input_adapter(self, symbol, adapter):
        """
        Register a symbol-specific input adapter with the manager.
        """
        if symbol not in self._inputs:
            self._inputs[symbol] = []
        self._inputs[symbol].append(adapter)
        logging.info(f"Registered adapter for: {symbol}")

        if len(self._inputs) > 0:
            self._subscriptions_initialized.set()

    def process_next_sim_timeslice(self, now):
        """
        Required by CSP engine, returns None for realtime.
        """
        return None

    def process_message(self, msg):
        """
        Process incoming messages and dispatch valid ticks to adapters.
        """
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
            "symbol", "bidPrices", "bidSizes", "askPrices", "askSizes",
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

        if not parsed_data["bidPrices"] or not parsed_data["askPrices"]:
            logging.warning(f"[PROCESS_MESSAGE] Empty bid/ask for {symbol}")
            return

        try:
            instr = Instrument(name=symbol)
            my_data = OrderBook(
                instr=instr,
                bids=[MarketOrder(instr=instr, exchange=Exchange(exchange), price=price, qty=qty, time_exchange=int(parsed_data["timeExchange"]), time_received=parsed_data["timePublished"])
                       for price, qty in zip(parsed_data["bidPrices"], parsed_data["bidSizes"])],
                asks=[MarketOrder(instr=instr, exchange=Exchange(exchange), price=price, qty=qty,
                                  time_exchange=parsed_data["timeExchange"], time_received=parsed_data["timePublished"])
                      for price, qty in zip(parsed_data["askPrices"], parsed_data["askSizes"])],
                time_exchange= parsed_data["timeExchange"],
                time_received=parsed_data["timePublished"]
            )
        except Exception as e:
            logging.error(f"[PROCESS_MESSAGE] Failed to construct MyData: {e}")
            return

        for adapter in self._inputs[symbol]:
            adapter.push_tick(my_data)


class MyPushAdapterImpl(PushInputAdapter):
    """
    Adapter for receiving pushed ticks and registering with the manager.
    """

    def __init__(self, manager_impl, symbol):
        logging.info(f"MyPushAdapterImpl::__init__ {symbol}")
        manager_impl.register_input_adapter(symbol, self)
        super().__init__()


# Define adapter wiring
MyPushAdapter = py_push_adapter_def(
    "MyPushAdapter",
    MyPushAdapterImpl,
    ts[OrderBook],
    MyAdapterManager,
    symbol=str
)


@csp.node
def log_mid_price(data: ts[OrderBook]) -> ts[float]:
    """
    Node that computes and logs the mid-price of bid/ask data.
    """
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
    """
    CSP graph that subscribes to a symbol and logs its mid-price.
    """
    logging.info("Building CSP graph")
    adapter_manager = MyAdapterManager(exchange="COINBASE")
    symbols = ["BTC-USD"]

    for symbol in symbols:
        data = adapter_manager.subscribe(symbol, csp.PushMode.NON_COLLAPSING)
        mid_price = log_mid_price(data)
        csp.print(f"{symbol} mid price:", mid_price)

    logging.info("Graph building complete")


def run_my_graph():
    """
    Entry point to run the CSP graph in realtime.
    """
    csp.run(
        my_graph,
        starttime=datetime.utcnow(),
        endtime=timedelta(seconds=5),
        realtime=True
    )


if __name__ == "__main__":
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    run_my_graph()