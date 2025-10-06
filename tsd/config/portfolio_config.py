from enum import StrEnum
from typing import List

from core.types.exchange import Exchange


class Portfolio(StrEnum):
    DEFAULT = "DEFAULT"


def supported_exchanges(portfolio: Portfolio) -> List[Exchange]:
    if portfolio == Portfolio.DEFAULT:
        return [Exchange.COINBASE, Exchange.BINANCE]
    else:
        raise ValueError(f"No portfolio for {portfolio}")

def parse_client_order_id(client_order_id: str) -> str:
    """
    Extract portfolio_id from client_order_id.
    Assumes format: <portfolio_id>_<uuid>
    """
    parts = client_order_id.split("_", 1)
    if len(parts) == 2:
        portfolio_str = parts[0]
        if portfolio_str in Portfolio:
            return Portfolio(portfolio_str)
    raise ValueError(f"Unknown portfolio for client_order_id {client_order_id}")