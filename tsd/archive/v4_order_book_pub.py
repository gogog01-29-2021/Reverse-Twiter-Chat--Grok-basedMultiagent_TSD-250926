import logging
import threading
import time
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple

import orjson as json_parser
from sortedcontainers import SortedDict

from config import TransportAddressBuilder, InstrumentGroup
from core.types.exchange import Exchange
from core.types.instrument import Instrument
from core.types.marketdata import Side
from dsm.utils.conversion_utils import symbol_to_instrument, instrument_to_exchange_symbol, now_epoch_ms, \
    iso_to_epoch_ms
from archive.v4_streaming import WebsocketPublisher, ZmqPubTransport, PubTransport

logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)


class OrderBookPublisher(WebsocketPublisher, ABC):
    """
    Generic OrderBookPublisher that defines how to publish order book messages
    to a topic via the WebsocketPublisher base.
    """

    @staticmethod
    def subject(exchange: Exchange, instrument: Instrument) -> str:
        """
        Compose a ZMQ subject string for the given exchange and instrument.

        Args:
            exchange (Exchange): Exchange identifier.
            instrument (Instrument): Market instrument (e.g., spot pair).

        Returns:
            str: ZMQ topic string.
        """
        return f"OrderBook_{exchange.value}_{instrument.symbol}"

    def publish_order_book(
            self,
            exchange: Exchange,
            instrument: Instrument,
            bids: List[List[float]],
            asks: List[List[float]],
            time_exchange: int,
            time_received: int,
            time_published: int
    ):
        msg = {
            "exchange": exchange.value,
            "symbol": instrument.symbol,
            "bids": bids,
            "asks": asks,
            "time_exchange": time_exchange,
            "time_received": time_received,
            "time_published": time_published,
        }
        self.publish(self.subject(exchange, instrument), msg)

    def on_message(self, ws, message):
        """
        Handle incoming WebSocket message.

        Args:
            ws: Active WebSocketApp connection.
            message (str): Raw JSON-encoded message.
        """
        if not isinstance(message, str):
            return
        try:
            time_received = now_epoch_ms()
            market_order_updates = json_parser.loads(message)
            self._handle_market_order_updates(market_order_updates, time_received)
        except Exception as e:
            logging.error("%s: Error processing message: %s", self.__class__.__name__, e)

    @abstractmethod
    def _handle_market_order_updates(self, market_order_updates: Dict, time_received: int):
        pass


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


class CoinbaseOrderBookPublisher(OrderBookPublisher):
    """
    Concrete implementation for publishing Coinbase order book snapshots.
    """

    EXCHANGE = Exchange.COINBASE
    WS_URL = TransportAddressBuilder.websocket(EXCHANGE)

    def __init__(self, instruments: List[Instrument], max_levels: int,
                 transport: PubTransport, verbose: bool = False, debug: bool = False):
        super().__init__(self.WS_URL, transport, verbose=verbose, debug=debug)
        self.instruments = instruments
        self.symbols = [instrument_to_exchange_symbol(self.EXCHANGE, inst) for inst in instruments]
        self.max_levels = max_levels
        self._order_books = {symbol: OrderBookData(self.max_levels) for symbol in self.symbols}

    def on_open(self, ws):
        ws.send(json_parser.dumps({
            "type": "subscribe",
            "channel": "level2",
            "product_ids": self.symbols
        }))
        logging.info("%s: Subscribed to Coinbase level2 for %s", self.__class__.__name__, self.symbols)

    def _handle_market_order_updates(self, market_order_updates: Dict, time_received: int):
        """
        Parse and apply order book snapshot or updates.

        Args:
            market_data (dict): Parsed WebSocket message.
            time_received (int): Local epoch milliseconds of receipt.
        """
        if market_order_updates.get("channel") != "l2_data":
            return

        time_exchange = iso_to_epoch_ms(market_order_updates.get("timestamp", ""))

        for event in market_order_updates.get("events", []):
            symbol = event.get("product_id")
            book = self._order_books.get(symbol)
            if not book:
                continue

            updates = event.get("updates", [])
            if event["type"] == "snapshot":
                parsed = [
                    (float(u["price_level"]), float(u["new_quantity"]),
                     Side.BID if u["side"] == "bid" else Side.ASK)
                    for u in self._filter_snapshot(updates)
                    if "price_level" in u and "new_quantity" in u
                ]
                book.apply_snapshot(parsed)
            elif event["type"] == "update":
                for u in updates:
                    try:
                        price = float(u["price_level"])
                        qty = float(u["new_quantity"])
                        side = Side.BID if u["side"] == "bid" else Side.ASK
                        keep_going = book.apply_update_one(price, qty, side)
                        if not keep_going:
                            break
                    except Exception:
                        continue
                book.trim()

            book_data = book.get_book()
            self.publish_order_book(
                self.EXCHANGE,
                symbol_to_instrument(symbol),
                [[p["price"], p["qty"]] for p in book_data["bids"]],
                [[p["price"], p["qty"]] for p in book_data["asks"]],
                time_exchange,
                time_received,
                now_epoch_ms()
            )

    def _filter_snapshot(self, updates: List[Dict]) -> List[Dict]:
        """
        Trim the snapshot to the top N levels per side.

        Args:
            updates (List[Dict]): Raw snapshot update messages.

        Returns:
            List[Dict]: Filtered updates preserving top bid/ask levels.
        """
        filtered, bids, asks = [], 0, 0
        for update in updates:
            side = update.get("side")
            if side == "bid" and bids < self.max_levels:
                bids += 1
                filtered.append(update)
            elif side == "offer" and asks < self.max_levels:
                asks += 1
                filtered.append(update)
            if bids >= self.max_levels and asks >= self.max_levels:
                break
        return filtered


if __name__ == "__main__":
    instrument_group = InstrumentGroup.GROUP_1
    address = TransportAddressBuilder.order_book(Exchange.COINBASE, instrument_group, "tcp://0.0.0.0")
    print(instrument_group.to_list(), address)

    coinbase = CoinbaseOrderBookPublisher(
        instruments=instrument_group.to_list(),
        max_levels=10,
        transport=ZmqPubTransport(
            zmq_bind_addr=address,
            debug=False),
        verbose=True,
        debug=False
    )

    publishers = [
        coinbase,
        # binance,
        # BybitMarketDataPublisher("wss://stream.bybit.com/realtime", ["BTCUSD"], shared_publisher),
        # UpbitMarketDataPublisher("wss://api.upbit.com/websocket/v1", ["KRW-BTC"], shared_publisher),
        # OKXMarketDataPublisher("wss://ws.okx.com:8443/ws/v5/public", ["BTC-USDT"], shared_publisher),
        # BitgetMarketDataPublisher("wss://ws.bitget.com/spot/v1/stream", ["btcusdt"], shared_publisher),
        # MEXCMarketDataPublisher("wss://wbs.mexc.com/ws", ["btcusdt"], shared_publisher),
        # CryptoComMarketDataPublisher("wss://stream.crypto.com/v2/market", ["BTC_USDT"], shared_publisher),
        # HTXMarketDataPublisher("wss://api.huobi.pro/ws", ["btcusdt"], shared_publisher),
        # GateIOMarketDataPublisher("wss://api.gate.io/ws/v4/", ["BTC_USDT"], shared_publisher)
    ]

    threads = [
        threading.Thread(target=p.start)
        for p in publishers
    ]

    for t in threads:
        t.start()

    try:
        time.sleep(600)
    finally:
        for p in publishers:
            p.stop()
        for t in threads:
            t.join()
