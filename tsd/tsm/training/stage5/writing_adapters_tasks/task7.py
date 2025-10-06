"""
This example demonstrates an AdapterManager managing both input and output adapters
for a real-time market data feed simulation.
"""

import random
import threading
import time
import typing
from datetime import datetime, timedelta

import csp
from csp import ts
from csp.impl.adaptermanager import AdapterManagerImpl
from csp.impl.outputadapter import OutputAdapter
from csp.impl.pushadapter import PushInputAdapter
from csp.impl.wiring import py_output_adapter_def, py_push_adapter_def

T = typing.TypeVar("T")


class PriceData(csp.Struct):
    symbol: str
    price: float


class MarketDataAdapterManager(AdapterManagerImpl):
    def __init__(self, interval: timedelta):
        self._interval = interval
        self._last_price = {}
        self._subscriptions = {}
        self._publications = {}
        self._running = False
        self._thread = None

    def subscribe(self, symbol):
        return _market_input_adapter(self, symbol, push_mode=csp.PushMode.NON_COLLAPSING)

    def publish(self, data: ts["T"], symbol: str):
        return _market_output_adapter(self, data, symbol)

    def _create(self, engine, memo):
        super().__init__(engine)
        return self

    def start(self, starttime, endtime):
        # Initialize starting price for each subscribed symbol
        for symbol in self._subscriptions:
            self._last_price[symbol] = 100.0
        self._running = True
        self._thread = threading.Thread(target=self._run)
        self._thread.start()

    def stop(self):
        if self._running:
            self._running = False
            self._thread.join()
        for name in self._publications.values():
            print(f"closing publication {name}")

    def register_subscription(self, symbol, adapter):
        if symbol not in self._subscriptions:
            self._subscriptions[symbol] = []
        self._subscriptions[symbol].append(adapter)

    def register_publication(self, symbol):
        if symbol not in self._publications:
            self._publications[symbol] = f"publication_{symbol}"

    def _run(self):
        symbols = list(self._subscriptions.keys())
        while self._running:
            symbol = random.choice(symbols)
            # Simulate random price movement
            price_change = random.uniform(-1.0, 1.0)
            self._last_price[symbol] += price_change
            price = self._last_price[symbol]
            data = PriceData(symbol=symbol, price=price)

            for adapter in self._subscriptions[symbol]:
                adapter.push_tick(data)

            time.sleep(self._interval.total_seconds())

    def _on_tick(self, symbol, value):
        print(f"{self._publications[symbol]}: {value.price:.2f}")


class MarketInputAdapterImpl(PushInputAdapter):
    def __init__(self, manager, symbol):
        manager.register_subscription(symbol, self)
        super().__init__()


class MarketOutputAdapterImpl(OutputAdapter):
    def __init__(self, manager, symbol):
        manager.register_publication(symbol)
        self._manager = manager
        self._symbol = symbol
        super().__init__()

    def on_tick(self, time, value):
        self._manager._on_tick(self._symbol, value)


_market_input_adapter = py_push_adapter_def(
    name="MarketInputAdapter",
    adapterimpl=MarketInputAdapterImpl,
    out_type=ts[PriceData],
    manager_type=MarketDataAdapterManager,
    symbol=str,
)

_market_output_adapter = py_output_adapter_def(
    name="MarketOutputAdapter",
    adapterimpl=MarketOutputAdapterImpl,
    manager_type=MarketDataAdapterManager,
    input=ts[T],
    symbol=str,
)


@csp.graph
def market_data_graph():
    adapter_manager = MarketDataAdapterManager(timedelta(seconds=0.75))

    data_aapl = adapter_manager.subscribe("AAPL")
    data_ibm = adapter_manager.subscribe("IBM")
    data_tsla = adapter_manager.subscribe("TSLA")

    csp.print("AAPL price", data_aapl)
    csp.print("IBM price", data_ibm)
    csp.print("TSLA price", data_tsla)

    # publish AAPL and IBM to the same publication
    adapter_manager.publish(data_aapl, "AAPL")
    adapter_manager.publish(data_ibm, "AAPL")

    # publish TSLA to its own publication
    adapter_manager.publish(data_tsla, "TSLA")


def main():
    csp.run(market_data_graph, starttime=datetime.utcnow(), endtime=timedelta(seconds=10), realtime=True)


if __name__ == "__main__":
    main()
