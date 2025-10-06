"""
This example demonstrates an AdapterManager for real-time market data.
The manager simulates price feeds for multiple symbols and pushes price updates into the CSP engine.
"""

import random
import threading
import time
from datetime import datetime, timedelta

import csp
from csp import ts
from csp.impl.adaptermanager import AdapterManagerImpl
from csp.impl.pushadapter import PushInputAdapter
from csp.impl.wiring import py_push_adapter_def


class PriceData(csp.Struct):
    symbol: str
    price: float


class MarketDataAdapterManager:
    def __init__(self, interval: timedelta):
        print("MarketDataAdapterManager::__init__")
        self._interval = interval

    def subscribe(self, symbol, push_mode=csp.PushMode.NON_COLLAPSING):
        return MarketDataPushAdapter(self, symbol, push_mode=push_mode)

    def _create(self, engine, memo):
        print("MarketDataAdapterManager::_create")
        return MarketDataAdapterManagerImpl(engine, self._interval)


class MarketDataAdapterManagerImpl(AdapterManagerImpl):
    def __init__(self, engine, interval):
        print("MarketDataAdapterManagerImpl::__init__")
        super().__init__(engine)

        self._interval = interval
        self._last_price = {}
        self._inputs = {}

        self._running = False
        self._thread = None

    def start(self, starttime, endtime):
        print("MarketDataAdapterManagerImpl::start")
        # Initialize price for each symbol
        for symbol in self._inputs:
            self._last_price[symbol] = 100.0
        self._running = True
        self._thread = threading.Thread(target=self._run)
        self._thread.start()

    def stop(self):
        print("MarketDataAdapterManagerImpl::stop")
        if self._running:
            self._running = False
            self._thread.join()

    def register_input_adapter(self, symbol, adapter):
        if symbol not in self._inputs:
            self._inputs[symbol] = []
        self._inputs[symbol].append(adapter)

    def process_next_sim_timeslice(self, now):
        """Not used for realtime adapter."""
        return None

    def _run(self):
        symbols = list(self._inputs.keys())
        while self._running:
            symbol = random.choice(symbols)
            adapters = self._inputs[symbol]

            # Simulate random walk price change
            price_change = random.uniform(-1.0, 1.0)
            self._last_price[symbol] += price_change
            price = self._last_price[symbol]

            data = PriceData(symbol=symbol, price=price)

            for adapter in adapters:
                adapter.push_tick(data)

            time.sleep(self._interval.total_seconds())


class MarketDataPushAdapterImpl(PushInputAdapter):
    def __init__(self, manager_impl, symbol):
        print(f"MarketDataPushAdapterImpl::__init__ {symbol}")
        manager_impl.register_input_adapter(symbol, self)
        super().__init__()


MarketDataPushAdapter = py_push_adapter_def(
    "MarketDataPushAdapter", MarketDataPushAdapterImpl, ts[PriceData], MarketDataAdapterManager, symbol=str
)


@csp.graph
def market_data_graph():
    print("Start of market_data_graph building")

    adapter_manager = MarketDataAdapterManager(timedelta(seconds=0.75))
    symbols = ["AAPL", "IBM", "TSLA", "GS", "JPM"]

    for symbol in symbols:
        data_last = adapter_manager.subscribe(symbol, csp.PushMode.LAST_VALUE)
        csp.print(f"{symbol} LAST_VALUE", data_last)

        data_burst = adapter_manager.subscribe(symbol, csp.PushMode.BURST)
        csp.print(f"{symbol} BURST", data_burst)

        data_non_collapsing = adapter_manager.subscribe(symbol, csp.PushMode.NON_COLLAPSING)
        csp.print(f"{symbol} NON_COLLAPSING", data_non_collapsing)

    print("End of market_data_graph building")


def main():
    csp.run(market_data_graph, starttime=datetime.utcnow(), endtime=timedelta(seconds=10), realtime=True)


if __name__ == "__main__":
    main()
