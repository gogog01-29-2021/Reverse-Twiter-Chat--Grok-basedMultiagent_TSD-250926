"""
PricePullAdapter is a simple PullInputAdapter to simulate historical price data.
Each instance simulates a single price timeseries using a random walk, returning a new price at each interval.
This can be useful for backtesting or testing processing pipelines with synthetic data.
"""

from datetime import datetime, timedelta
import random

import csp
from csp import ts
from csp.impl.pulladapter import PullInputAdapter
from csp.impl.wiring import py_pull_adapter_def


class PricePullAdapterImpl(PullInputAdapter):
    def __init__(self, interval: timedelta, num_ticks: int, start_price: float = 100.0):
        print("PricePullAdapterImpl::__init__")
        self._interval = interval
        self._num_ticks = num_ticks
        self._counter = 0
        self._next_time = None
        self._price = start_price
        super().__init__()

    def start(self, start_time, end_time):
        """Initialize the price stream at engine start."""
        print("PricePullAdapterImpl::start")
        super().start(start_time, end_time)
        self._next_time = start_time

    def stop(self):
        """Cleanup resources at engine stop."""
        print("PricePullAdapterImpl::stop")

    def next(self):
        """Return the next timestamp and simulated price, or None if done."""
        if self._counter < self._num_ticks:
            self._counter += 1
            time = self._next_time
            self._next_time += self._interval

            # Simulate random price movement
            price_change = random.uniform(-0.5, 0.5)
            self._price += price_change

            return time, self._price

        return None


PricePullAdapter = py_pull_adapter_def(
    "PricePullAdapter",
    PricePullAdapterImpl,
    ts[float],  # Output a timeseries of float prices
    interval=timedelta,
    num_ticks=int,
    start_price=float
)


@csp.graph
def price_graph():
    print("Building price_graph...")
    price_data = PricePullAdapter(timedelta(seconds=1), num_ticks=10, start_price=100.0)
    csp.print("Price", price_data)
    print("Finished building price_graph.")


def main():
    csp.run(price_graph, starttime=datetime(2020, 12, 28))


if __name__ == "__main__":
    main()
