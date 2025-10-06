import logging
import time
from abc import ABC, abstractmethod
from typing import Callable
from typing import Dict, List, Tuple

import orjson as json_parser
from sortedcontainers import SortedDict

from config.routing_config import get_instruments_by_port, TransportAddressBuilder
from core.types.exchange import Exchange
from core.types.instrument import Instrument
from core.types.marketdata import Side
from dsm.core.pubsub_base import Publisher, PubTransport, MessageParser, ZmqPubTransport
from dsm.utils.conversion_utils import iso_to_epoch_ms, now_epoch_ms, instrument_to_exchange_symbol, \
    exchange_symbol_to_instrument_symbol

logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)


class OrderBookData:
    """
    Maintains a local view of an order book with capped depth for bids and asks.

    This class is used to manage and apply updates from market data feeds,
    ensuring the book always maintains up to `max_levels` entries per side.

    Attributes:
        max_levels (int): Maximum number of levels to retain for bids and asks.
        bids (SortedDict): Descending price-sorted map of bid prices to quantities.
        asks (SortedDict): Ascending price-sorted map of ask prices to quantities.

    Notes:
        - Assumes updates arrive in sorted order or are quickly trimmed.
        - This structure is not thread-safe.
    """

    def __init__(self, max_levels: int):
        self.max_levels = max_levels
        self.bids = SortedDict(lambda x: -x)
        self.asks = SortedDict()

    def apply_snapshot(self, updates: List[Tuple[float, float, Side]]):
        """
        Apply a full snapshot to reset the order book.

        Args:
            updates (List[Tuple[float, float, Side]]):
                A list of (price, quantity, side) tuples representing the book.

        Warning:
            Entries with qty=0 are ignored. Bids/asks capped to `max_levels`.
        """
        self.bids.clear()
        self.asks.clear()
        for price, qty, side in updates:
            if qty == 0:
                continue
            book = self.bids if side == Side.BID else self.asks
            if len(book) < self.max_levels:
                book[price] = qty
            if len(self.bids) >= self.max_levels and len(self.asks) >= self.max_levels:
                break

    def apply_update_one(self, price: float, qty: float, side: Side) -> bool:
        """
        Apply a single level update to the order book.

        Args:
            price (float): The price level being updated.
            qty (float): The new quantity at that level. If 0, the level is removed.
            side (Side): BID or ASK side.

        Returns:
            bool: False if the update violates the level limit, else True.
        """
        book = self.bids if side == Side.BID else self.asks
        if qty == 0:
            book.pop(price, None)
        else:
            book[price] = qty

        if len(book) > self.max_levels:
            worst_price = next(reversed(book))
            if (side == Side.BID and price < worst_price) or (side == Side.ASK and price > worst_price):
                return False
        return True

    def trim(self):
        """
        Ensure bids and asks contain no more than max_levels entries each.
        """
        while len(self.bids) > self.max_levels:
            self.bids.popitem()
        while len(self.asks) > self.max_levels:
            self.asks.popitem()

    def get_book(self) -> Dict[str, List[Dict[str, float]]]:
        """
        Export the order book as a dictionary with sorted bid/ask levels.

        Returns:
            dict: {"bids": [...], "asks": [...]} sorted by price.
        """
        return {
            "bids": [{"price": p, "qty": self.bids[p]} for p in self.bids],
            "asks": [{"price": p, "qty": self.asks[p]} for p in self.asks],
        }


class OrderBookParser(MessageParser, ABC):
    def __init__(self, exchange: Exchange, instruments: List[Instrument], max_levels: int):
        self.exchange = exchange
        self.symbols = [instrument_to_exchange_symbol(exchange, instrument) for instrument in instruments]
        self.max_levels = max_levels
        self._books = {symbol: {"bids": {}, "asks": {}} for symbol in self.symbols}

    @abstractmethod
    def parse(self, raw_message: str, time_received: int) -> List[Tuple[str, dict]]:
        pass


class OrderBookPublisher(Publisher, ABC):
    def __init__(
            self,
            ws_url: str,
            exchange: Exchange,
            instruments: List[Instrument],
            max_levels: int,
            transport: PubTransport,
            parser_cls: Callable[[Exchange, List[Instrument], int], MessageParser],
            verbose: bool = False,
            debug: bool = False,
            log_on_change: bool = False
    ):
        parser = parser_cls(exchange, instruments, max_levels)
        super().__init__(ws_url, transport, parser, verbose, debug, log_on_change)
        self._exchange = exchange
        self._symbols = [instrument_to_exchange_symbol(exchange, instrument) for instrument in instruments]
        self._max_levels = max_levels

    @staticmethod
    def topic(exchange: Exchange, instrument: Instrument) -> str:
        """
        Compose a ZMQ subject string for the given exchange and instrument.

        Args:
            exchange (Exchange): Exchange identifier.
            instrument (Instrument): Market instrument (e.g., spot pair).

        Returns:
            str: ZMQ topic string.
        """
        return f"OrderBook_{exchange.value}_{instrument.symbol}"

    @abstractmethod
    def on_open(self, ws):
        pass


# ---------- Coinbase Implementation ----------

class CoinbaseOrderBookParser(OrderBookParser):
    def __init__(self, exchange: Exchange, instruments: List[Instrument], max_levels: int):
        super().__init__(exchange, instruments, max_levels)
        self._books = {symbol: OrderBookData(max_levels) for symbol in self.symbols}

    def parse(self, raw_message: str, time_received: int) -> List[Tuple[str, dict]]:
        messages = []
        data = json_parser.loads(raw_message)
        if data.get("channel") != "l2_data":
            return []

        time_exchange = iso_to_epoch_ms(data.get("timestamp", ""))
        for event in data.get("events", []):
            symbol = event.get("product_id", "")
            book = self._books.get(symbol)
            if not book:
                continue

            updates = event.get("updates", [])
            if event["type"] == "snapshot":
                parsed = []
                for u in updates:
                    try:
                        price = float(u["price_level"])
                        qty = float(u["new_quantity"])
                        side = Side.BID if u["side"] == "bid" else Side.ASK
                        parsed.append((price, qty, side))
                    except Exception:
                        continue
                book.apply_snapshot(parsed)
            elif event["type"] == "update":
                for u in updates:
                    try:
                        price = float(u["price_level"])
                        qty = float(u["new_quantity"])
                        side = Side.BID if u["side"] == "bid" else Side.ASK
                        keep = book.apply_update_one(price, qty, side)
                        if not keep:
                            break
                    except Exception:
                        continue
                book.trim()

            topic = f"OrderBook_{self.exchange}_{symbol.replace('-', '')}"
            snapshot = book.get_book()
            msg = {
                "exchange": self.exchange.value,
                "symbol": exchange_symbol_to_instrument_symbol(self.exchange, symbol),
                "bids": [[p["price"], p["qty"]] for p in snapshot["bids"]],
                "asks": [[p["price"], p["qty"]] for p in snapshot["asks"]],
                "time_exchange": time_exchange,
                "time_received": time_received,
                "time_published": now_epoch_ms(),
            }
            messages.append((topic, msg))
        return messages


class CoinbaseOrderBookPublisher(OrderBookPublisher):
    def __init__(
            self,
            instruments: List[Instrument],
            max_levels: int,
            transport: PubTransport,
            verbose: bool = False,
            debug: bool = False,
            log_on_change: bool = False
    ):
        super().__init__(
            ws_url="wss://advanced-trade-ws.coinbase.com",
            exchange=Exchange.COINBASE,
            instruments=instruments,
            max_levels=max_levels,
            transport=transport,
            parser_cls=CoinbaseOrderBookParser,
            verbose=verbose,
            debug=debug,
            log_on_change=log_on_change
        )

    def on_open(self, ws):
        ws.send(json_parser.dumps({
            "type": "subscribe",
            "channel": "level2",
            "product_ids": self._symbols
        }))
        logging.info("Subscribed to Coinbase level2 for %s", self._symbols)


if __name__ == "__main__":
    port = 5000
    exchange_to_connect = Exchange.COINBASE
    instruments_to_publish = get_instruments_by_port(exchange_to_connect, port)
    zmq_bind_addr = TransportAddressBuilder.order_book(exchange_to_connect, instruments_to_publish, "tcp://0.0.0.0")
    zmq_pub_transport = ZmqPubTransport(zmq_bind_addr, debug=True)
    publisher = CoinbaseOrderBookPublisher(
        instruments=instruments_to_publish,
        max_levels=10,
        transport=zmq_pub_transport,
        verbose=True,
        debug=True,
        log_on_change=False
    )

    try:
        publisher.start()
        time.sleep(600)
    finally:
        publisher.stop()
        zmq_pub_transport.stop()
