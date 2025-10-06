"""
csp.GenericPushAdapter is used here to simulate a real-time market data feed,
pushing price updates from a non-csp thread into the csp engine for processing.
"""

import threading
import time
import random
from datetime import datetime, timedelta

import csp


class MarketDataDriver:
    def __init__(self, adapter: csp.GenericPushAdapter):
        self._adapter = adapter
        self._active = False
        self._thread = None
        self._price = 100.0  # Starting price

    def start(self):
        self._active = True
        self._thread = threading.Thread(target=self._run)
        self._thread.start()

    def stop(self):
        if self._active:
            self._active = False
            self._thread.join()

    def _run(self):
        print("Market data driver thread started")
        self._adapter.wait_for_start()

        while self._active and not self._adapter.stopped():
            # Simulate a random walk for the price
            price_change = random.uniform(-0.5, 0.5)
            self._price += price_change

            # Push the simulated price into the CSP engine
            self._adapter.push_tick(self._price)

            time.sleep(1)  # Simulate 1-second intervals between price updates


@csp.graph
def market_data_graph():
    adapter = csp.GenericPushAdapter(float)  # We are pushing float prices
    driver = MarketDataDriver(adapter)
    driver.start()

    # Stop the driver thread when the CSP engine stops
    csp.schedule_on_engine_stop(driver.stop)

    # Print the incoming price data
    csp.print("Price Update", adapter.out())


def main():
    # Run the graph in real-time mode for 10 seconds
    csp.run(market_data_graph, realtime=True, starttime=datetime.utcnow(), endtime=timedelta(seconds=10))


if __name__ == "__main__":
    main()
