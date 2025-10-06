import uuid
from enum import StrEnum
from typing import Optional

import csp

from core.types.exchange import Exchange
from core.types.instrument import Instrument
from config.portfolio_config import Portfolio

MAX_CLIENT_ORDER_ID_LEN = 36
UUID_LEN = 8
DELIMITER_LEN = 1
MAX_PORTFOLIO_ID_LEN = MAX_CLIENT_ORDER_ID_LEN - UUID_LEN - DELIMITER_LEN


class TimeInForce(StrEnum):
    IOC = "IOC"
    GTC = "GTC"
    DAY = "DAY"


class OrderSide(StrEnum):
    BUY = "BUY"
    SELL = "SELL"


class OrderType(StrEnum):
    LIMIT = "LIMIT"
    MARKET = "MARKET"


class ExternalOrder(csp.Struct):
    instr: Instrument
    exchange: Exchange
    order_side: OrderSide
    order_type: OrderType

    price: float
    qty: Optional[float] = None  # optional: base 기반 주문용 BTCUSDT -> 0.1 BTC
    quote_order_qty: Optional[float] = None  # optional: quote 기반 주문용 BTCUSDT -> 1000 USDT

    time_in_force: TimeInForce
    time_sent: int
    portfolio: Portfolio

    # optional parameters
    # stop_price: float = 0.0                  # optional: stop order trigger price
    # post_only: bool = False                  # optional: maker only
    # reduce_only: bool = False                # optional: 포지션 감소 전용
    # margin_type: str = "cash"                # optional: spot은 'cash', margin은 cross/isolated
    # leverage: int = 1                        # optional: spot은 1배

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.client_order_id = generate_client_order_id(self.portfolio)


class Trade(csp.Struct):
    instr: Instrument
    price: float
    qty: float
    side: OrderSide


class ExecutionReport(csp.Struct):
    test: str


def generate_client_order_id(portfolio: Portfolio) -> str:
    """
    Generate a client_order_id with only portfolio metadata.
    Format: <portfolio>_<8char UUID>
    """
    validate_portfolio_id(portfolio)
    uid = uuid.uuid4().hex[:8]
    return f"{portfolio}_{uid}"


def validate_portfolio_id(portfolio: Portfolio):
    if len(portfolio) > MAX_PORTFOLIO_ID_LEN:
        raise ValueError(f"portfolio too long (max {MAX_PORTFOLIO_ID_LEN} characters): {portfolio}")
