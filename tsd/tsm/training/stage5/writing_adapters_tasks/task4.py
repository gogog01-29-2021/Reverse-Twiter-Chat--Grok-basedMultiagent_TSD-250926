"""
PricePushAdapter simulates a real-time market data feed for a single symbol.
It pushes simulated price updates from a separate thread into the CSP engine.
"""

import threading
import time
import random
from datetime import datetime, timedelta

import csp
from csp import ts
from csp.impl.pushadapter import PushInputAdapter
from csp.impl.wiring import py_push_adapter_def


class PricePushAdapterImpl(PushInputAdapter):
    def __init__(self, interval: timedelta, start_price: float = 100.0):
        print("PricePushAdapterImpl::__init__")
        self._interval = interval
        self._thread = None
        self._running = False
        self._price = start_price

    def start(self, starttime, endtime):
        """Starts the real-time price feed simulation thread."""
        print("PricePushAdapterImpl::start")
        self._running = True
        self._thread = threading.Thread(target=self._run)
        self._thread.start()

    def stop(self):
        """Stops the simulation thread and cleans up resources."""
        print("PricePushAdapterImpl::stop")
        if self._running:
            self._running = False
            self._thread.join()

    def _run(self):
        while self._running:
            # Simulate a random walk for price changes
            price_change = random.uniform(-0.5, 0.5)
            self._price += price_change

            # Push the simulated price to the CSP engine
            self.push_tick(self._price)

            time.sleep(self._interval.total_seconds())


# Define the graph-time adapter
PricePushAdapter = py_push_adapter_def(
    "PricePushAdapter", PricePushAdapterImpl, ts[float], interval=timedelta, start_price=float
)


@csp.graph
def price_graph():
    print("Start of price_graph building")
    price_data = PricePushAdapter(timedelta(seconds=1), start_price=100.0)
    csp.print("Price", price_data)
    print("End of price_graph building")


def main():
    # Run for 10 seconds in real-time
    csp.run(price_graph, realtime=True, starttime=datetime.utcnow(), endtime=timedelta(seconds=10))


if __name__ == "__main__":
    main()
