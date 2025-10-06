from typing import Any, Dict
from core.types.trade import ExternalOrder, OrderSide, OrderType
from dsm.utils.conversion_utils import instrument_to_exchange_symbol


def external_order_to_bybit_payload(order: ExternalOrder) -> Dict[str, Any]:
    payload = {
        "category": "spot",
        "symbol": instrument_to_exchange_symbol(order.exchange, order.instr),
        "side": "Buy" if order.order_side == OrderSide.BUY else "Sell",
        "orderType": "Market" if order.order_type == OrderType.MARKET else "Limit",
        "qty": str(order.qty),
        "price": str(order.price) if order.order_type == OrderType.LIMIT else None,
        "timeInForce": order.time_in_force.value if order.order_type == OrderType.LIMIT else None,
        "orderLinkId": order.client_order_id if order.client_order_id else None,
        # "reduceOnly": order.reduce_only if order.reduce_only else None,
    }
    return {k: v for k, v in payload.items() if v is not None}