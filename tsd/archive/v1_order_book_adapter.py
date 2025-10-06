"""
This example introduces the concept of an AdapterManager for realtime data. AdapterManagers are constructs that are used
when you have a shared input or output resources (ie single CSV / Parquet file, some pub/sub session, etc)
that you want to connect to once, but provide data to/from many input/output adapters (aka time series)
"""
import datetime
import logging
import threading
import time
from collections import defaultdict

import csp
from csp import ts
from csp.impl.adaptermanager import AdapterManagerImpl
from csp.impl.pushadapter import PushInputAdapter
from csp.impl.wiring import py_push_adapter_def
from sortedcontainers import SortedDict

from config import EXCHANGE_CONFIG
from archive.v1_messaging import Streamer, loads
from archive.v1_messaging import dumps
from core.types.exchange import Exchange
from core.types.instrument import Instrument
from core.types.marketdata import OrderBook, MarketOrder
from core.types.trade import OrderSide
from core.utils.timeutils import datetime_str_to_nanoseconds


class OrderBookAdapterManager:
    def __init__(self):
        pass

    def subscribe(self, symbol, push_mode=csp.PushMode.NON_COLLAPSING):
        return OrderBookPushAdapter(self, symbol, push_mode=push_mode)

    def _create(self, engine, memo):
        adapter = OrderBookAdapterManagerImpl(engine)
        OrderBookAdapterManager.adapter = adapter
        return adapter


class OrderBookAdapterManagerImpl(AdapterManagerImpl):
    def __init__(self, engine):
        super().__init__(engine)
        self._inputs = {}

    def register_input_adapter(self, symbol, adapter):
        if symbol not in self._inputs:
            self._inputs[symbol] = []
        self._inputs[symbol].append(adapter)

    def start(self, starttime, endtime):
        pass

    def stop(self):
        pass

    def process_next_sim_timeslice(self, now):
        return None

    def push_order_book(self, order_book: OrderBook):
        for adapter in self._inputs.get(order_book.instr.symbol, []):
            adapter.push_tick(order_book)


class OrderBookPushAdapterImpl(PushInputAdapter):
    def __init__(self, manager_impl, symbol):
        manager_impl.register_input_adapter(symbol, self)
        super().__init__()


OrderBookPushAdapter = py_push_adapter_def(
    "OrderBookPushAdapter", OrderBookPushAdapterImpl, ts[OrderBook], OrderBookAdapterManager, symbol=str)


class CoinbaseOrderBookStreamer(Streamer):
    def __init__(self, ws_url: str, exchange, symbols, csp_adapter):
        super().__init__(ws_url)
        self.exchange = exchange
        self.symbols = symbols
        self.csp_adapter = csp_adapter

        self.order_book = defaultdict(lambda: self.OrderBook())

    class OrderBook:
        """
        Represents an in-memory order book for a single trading symbol.

        Attributes:
            bids (SortedDict): Bid side of the book, sorted in descending order.
            asks (SortedDict): Ask side of the book, sorted in ascending order.
        """

        def __init__(self):
            self.bids = SortedDict(lambda price: -price)
            self.asks = SortedDict()

        def update_order(self, price: float, quantity: float, side: str):
            """
            Insert or update a price level in the order book, or remove it if quantity is 0.

            Args:
                price (float): Price level.
                quantity (float): Quantity at the given price level.
                side (str): 'bid' or 'ask'.
            """
            book = self.bids if side.lower() == "bid" else self.asks
            if quantity == 0.0:
                book.pop(price, None)  # Remove price level if it exists
                logging.debug("Removed %s at price %s", side, price)
            else:
                book[price] = quantity
                logging.debug("Set %s at price %s to quantity %s", side, price, quantity)

    def on_message(self, ws, message):
        logging.debug("%s: WebSocket message received: %s", self.__class__.__name__, message)
        try:
            time_received = time.time_ns()
            data = loads(message)
            self.update_order_book(data, time_received)
        except Exception as e:
            logging.error("%s: Error processing WebSocket message: %s", self.exchange, e)

    def logging_loop(self):
        pass

    def subscribe(self, ws):
        """
        Authenticate and subscribe to level2 channels for given symbols.
        """
        message = {
            "type": "subscribe",
            "channel": "level2",
            "product_ids": self.symbols
        }
        ws.send(dumps(message))
        logging.info("%s: Sent subscription message on channel %s", self.__class__.__name__, "level2")

    def update_order_book(self, data, time_received):
        time_exchange = datetime_str_to_nanoseconds(data.get("timestamp"), format="%Y-%m-%dT%H:%M:%S.%fZ")
        for event in data.get("events", []):
            symbol = event.get("product_id")
            ob = self.order_book[symbol]
            if event.get("type") == "snapshot":
                ob.bids.clear()
                ob.asks.clear()
            for upd in event.get("updates", []):
                side = upd.get("side")
                price = upd.get("price_level")
                quantity = upd.get("new_quantity")
                if side and price and quantity:
                    try:
                        ob.update_order(float(price), float(quantity), side)
                    except Exception as e:
                        logging.error("%s: Failed to update %s %s: %s", self.__class__.__name__, side, price, e)
                instr = Instrument(name=symbol)
                self.csp_adapter.push_order_book(OrderBook(
                    instr=instr,
                    bids=[
                        MarketOrder(instr=instr, exchange=self.exchange, order_side=OrderSide.BUY, price=price, qty=qty,
                                    time_exchange=time_exchange, time_received=time.time_ns()) for price, qty in
                        ob.bids.items()],
                    asks=[MarketOrder(instr=instr, exchange=self.exchange, order_side=OrderSide.SELL, price=price,
                                      qty=qty, time_exchange=time_exchange, time_received=time.time_ns()) for price, qty
                          in ob.asks.items()],
                    time_exchange=time_exchange,
                    time_received=time_received
                ))


@csp.graph
def print_order_book(symbol: str):
    manager = OrderBookAdapterManager()
    order_book = manager.subscribe(symbol, push_mode=csp.PushMode.LAST_VALUE)
    csp.print(f"Order Book: {symbol}", order_book)


def main():
    RUN_TIME = 10  # in seconds

    symbol = "BTC-USD"

    def run_engine():
        csp.run(print_order_book, symbol=symbol, starttime=datetime.datetime.utcnow(),
                endtime=datetime.timedelta(seconds=RUN_TIME),
                realtime=True)

    engine_thread = threading.Thread(target=run_engine)
    engine_thread.start()

    order_book_streamer = CoinbaseOrderBookStreamer(
        ws_url=EXCHANGE_CONFIG["coinbase"]["ws_url"],
        exchange=Exchange.COINBASE,
        symbols=[symbol],
        csp_adapter=OrderBookAdapterManager.adapter)

    order_book_streamer.start()
    time.sleep(RUN_TIME)
    order_book_streamer.end()
    engine_thread.join()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
