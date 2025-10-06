import csp
from csp.impl.pushadapter import PushInputAdapter
from csp.impl.wiring import py_push_adapter_def

from core.types.exchange import Exchange
from core.types.instrument import Spot, Currency
from core.types.exchange.utils import instrument_to_symbol
from core.types.marketdata import OrderBook, MarketOrder
from core.types.trade import OrderSide
from archive.v1_messaging import loads
import asyncio
import threading
import queue
import logging
from nats.aio.client import Client as NATS


class NATSOrderBookSubscriber:
    """
    Subscribes to a specific NATS subject and forwards incoming messages to a callback.
    Runs its own asyncio event loop in a background thread.
    """
    def __init__(self, subject: str, message_callback, nats_url="nats://localhost:4222"):
        """
        Initialize the subscriber.

        Args:
            subject (str): NATS subject to subscribe to.
            message_callback (Callable): Function to call when a message is received.
            nats_url (str): URL for the NATS server.
        """
        self._subject = subject
        self._callback = message_callback
        self._nats_url = nats_url
        self._nats = NATS()
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._running = threading.Event()

    def start(self):
        """Start the subscriber in a background thread."""
        self._running.set()
        self._thread.start()

    def stop(self):
        """Stop the subscriber and gracefully shutdown the event loop."""
        self._running.clear()
        if self._nats.is_connected:
            future = asyncio.run_coroutine_threadsafe(self._nats.drain(), self._loop)
            try:
                future.result(timeout=3)
            except Exception as e:
                logging.error("Error draining NATS: %s", e)

        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join()

    def _run_loop(self):
        """Run the asyncio loop and handle subscription."""
        asyncio.set_event_loop(self._loop)
        self._loop.run_until_complete(self._initialize())
        self._loop.run_forever()

        # Cleanup pending tasks
        pending = asyncio.all_tasks(loop=self._loop)
        for task in pending:
            task.cancel()
        self._loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
        self._loop.run_until_complete(self._nats.close())
        self._loop.close()

    async def _initialize(self):
        """Async initialization to connect and subscribe to NATS."""
        await self._nats.connect(servers=[self._nats_url])

        async def handler(msg):
            try:
                self._callback(msg.data)
            except Exception as e:
                logging.exception("Failed to handle message: %s", e)

        await self._nats.subscribe(self._subject, cb=handler)
        logging.info("Subscribed to %s", self._subject)


class OrderBookAdapterImpl(PushInputAdapter):
    """
    CSP PushInputAdapter that receives OrderBook updates from NATS and pushes them into the CSP engine.
    """
    def __init__(self, exchange: Exchange, instrument: Spot):
        """
        Initialize the adapter.

        Args:
            exchange (Exchange): Exchange enum.
            instrument (Spot): Spot instrument (e.g., BTC/USD).
        """
        super().__init__()
        self._exchange = exchange
        self._instrument = instrument
        self._symbol = instrument_to_symbol(exchange, instrument)
        self._subject = f"ORDERBOOK_{exchange}_{self._symbol}"

        self._queue = queue.Queue()
        self._subscriber = NATSOrderBookSubscriber(self._subject, self._enqueue_message)
        self._thread = None
        self._running = False

    def _enqueue_message(self, data_bytes):
        """Decode and enqueue incoming NATS messages."""
        try:
            data = loads(data_bytes)
            self._queue.put(data)
        except Exception as e:
            logging.error("Failed to decode message: %s", e)

    def start(self, start_time, end_time):
        """Start the adapter: initiate subscriber and processing thread."""
        self._running = True
        self._subscriber.start()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        """Stop the adapter and subscriber."""
        self._running = False
        self._subscriber.stop()
        if self._thread:
            self._thread.join()

    def _run(self):
        """
        Background thread that consumes the message queue and pushes OrderBook data into CSP.
        """
        while self._running:
            try:
                data = self._queue.get(timeout=1)
            except queue.Empty:
                continue

            try:
                exchange = self._exchange
                instr = self._instrument

                bids = [
                    MarketOrder(
                        instr=instr,
                        exchange=exchange,
                        order_side=OrderSide.BUY,
                        price=price,
                        qty=qty,
                        time_exchange=data["timeExchange"],
                        time_received=data["timeReceived"]
                    )
                    for price, qty in zip(data["bidPrices"], data["bidSizes"])
                ]

                asks = [
                    MarketOrder(
                        instr=instr,
                        exchange=exchange,
                        order_side=OrderSide.SELL,
                        price=price,
                        qty=qty,
                        time_exchange=data["timeExchange"],
                        time_received=data["timeReceived"]
                    )
                    for price, qty in zip(data["askPrices"], data["askSizes"])
                ]

                ob = OrderBook(
                    instr=instr,
                    bids=bids,
                    asks=asks,
                    time_exchange=data["timeExchange"],
                    time_received=data["timeReceived"]
                )

                self.push_tick(ob)
            except Exception as e:
                logging.exception("Error processing message in adapter: %s", e)
                self.shutdown_engine(e)


# Register CSP push input adapter
OrderBookAdapter = py_push_adapter_def(
    name="OrderBookAdapter",
    adapterimpl=OrderBookAdapterImpl,
    out_type=csp.ts[OrderBook],
    exchange=Exchange,
    instrument=Spot
)


@csp.graph
def orderbook_graph():
    """Example CSP graph subscribing to BTC/USD and ETH/USD order books from Coinbase."""
    btc = OrderBookAdapter(exchange=Exchange.COINBASE, instrument=Spot(base=Currency.BTC, term=Currency.USD))
    eth = OrderBookAdapter(exchange=Exchange.COINBASE, instrument=Spot(base=Currency.ETH, term=Currency.USD))
    csp.print("BTC OrderBook", btc)
    csp.print("ETH OrderBook", eth)


if __name__ == "__main__":
    from datetime import datetime, timedelta
    csp.run(orderbook_graph, starttime=datetime.utcnow(), endtime=timedelta(seconds=5), realtime=True)
