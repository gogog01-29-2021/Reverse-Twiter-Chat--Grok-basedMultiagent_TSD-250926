from typing import Any, Dict
from core.types.trade import ExternalOrder, OrderType
from dsm.utils.conversion_utils import instrument_to_exchange_symbol


def external_order_to_binance_payload(order: ExternalOrder) -> Dict[str, Any]:
    payload = {
        "symbol": instrument_to_exchange_symbol(order.exchange, order.instr),
        "side": order.order_side.value,
        "type": order.order_type.value,
        "timeInForce": order.time_in_force.value if order.order_type == OrderType.LIMIT else None,
        "quantity": str(order.qty) if order.qty else None,
        "quoteOrderQty": str(order.quote_order_qty) if order.quote_order_qty else None,
        "price": str(order.price) if order.price else None,
        "newClientOrderId": order.client_order_id if order.client_order_id else None,
        # "stopPrice": str(order.stop_price) if order.stop_price else None,
        # "postOnly": order.post_only if order.post_only else None,
        "recvWindow": 5000,
        "timestamp": order.time_sent,
    }
    return {k: v for k, v in payload.items() if v is not None}