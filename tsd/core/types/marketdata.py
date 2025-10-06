from datetime import datetime
from enum import StrEnum
from typing import List

import csp

from core.types.exchange import Exchange
from core.types.instrument import Instrument


class Side(StrEnum):
    BID = "BID"
    ASK = "ASK"


class TwoWayPrice(csp.Struct):
    """
    Represents a snapshot of the best bid and ask prices and quantities
    for a given instrument at a specific time.

    Attributes:
        bid_price (float): Best bid price available.
        ask_price (float): Best ask price available.
        bid_qty (float): Quantity available at the best bid price.
        ask_qty (float): Quantity available at the best ask price.
        time_exchange (int): Time (in nanos) when the quote was recorded at the exchange.
        time_received (int): Time (in nanos) when the quote was received by the system.
    """
    bid_price: float
    ask_price: float
    bid_qty: float
    ask_qty: float
    time_exchange: int
    time_received: int


class MarketOrder(csp.Struct):
    """
    Represents a single market order received from an exchange.

    Attributes:
        instr (Instrument): Instrument the order relates to.
        exchange (Exchange): Exchange from which the order originated.
        side (Side): Indicates whether the order is a bid or an ask.
        price (float): Price level of the order.
        qty (float): Quantity specified in the order.
        time_exchange (int): Time (in nanos) when the order was registered at the exchange.
        time_received (int): Time (in nanos) when the order was received by the system.
    """
    instr: Instrument
    exchange: Exchange
    side: Side
    price: float
    qty: float
    time_exchange: datetime
    time_received: datetime

    def __repr__(self):
        return f"MarketOrder( instr={self.instr.symbol}, exchange={self.exchange.value}, side={self.side}, price={self.price}, qty={self.qty}, time_exchange={self.time_exchange}, time_received={self.time_received} )"

    def __str__(self):
        return f"MarketOrder( instr={self.instr.symbol}, exchange={self.exchange.value}, side={self.side}, price={self.price}, qty={self.qty}, time_exchange={self.time_exchange}, time_received={self.time_received} )"


class OrderBook(csp.Struct):
    """
    Represents the full depth of market for a given instrument at a specific time,
    containing both bid and ask orders.

    Attributes:
        instr (Instrument): Instrument to which this order book pertains.
        bids (List[MarketOrder]): List of bid-side orders sorted by price and/or time priority.
        asks (List[MarketOrder]): List of ask-side orders sorted by price and/or time priority.
        time_exchange (int): Time (in nanos) when the order book snapshot was recorded at the exchange.
        time_received (int): Time (in nanos) when the snapshot was received by the system.
    """
    instr: Instrument
    bids: List[MarketOrder]
    asks: List[MarketOrder]
    time_exchange: datetime
    time_received: datetime

    def __repr__(self):
        return f"OrderBook( instr={self.instr.symbol}, bids={self.bids}, asks={self.asks}, time_exchange={self.time_exchange}, time_received={self.time_received} )"

    def __str__(self):
        return f"OrderBook( instr={self.instr.symbol}, bids={self.bids}, asks={self.asks}, time_exchange={self.time_exchange}, time_received={self.time_received} )"
