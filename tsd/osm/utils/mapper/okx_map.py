from typing import Any, Dict
from core.types.trade import ExternalOrder, OrderSide, OrderType, TimeInForce
from dsm.utils.conversion_utils import instrument_to_exchange_symbol


def external_order_to_okx_payload(order: ExternalOrder) -> Dict[str, Any]:
    ordType = "market" if order.order_type == OrderType.MARKET else "limit"
    if order.post_only:
        ordType = "post_only"
    elif order.time_in_force == "FOK":
        ordType = "fok"
    elif order.time_in_force == TimeInForce.IOC:
        ordType = "ioc"

    payload = {
        "instId": instrument_to_exchange_symbol(order.exchange, order.instr),
        # "tdMode": order.margin_type if order.margin_type else "cash",
        "tdMode": "cash",
        "side": "buy" if order.order_side == OrderSide.BUY else "sell",
        "ordType": ordType,
        "sz": str(order.qty),
        "px": str(order.price) if order.order_type == OrderType.LIMIT else None,
        "clOrdId": order.client_order_id if order.client_order_id else None,
        # "reduceOnly": order.reduce_only if order.reduce_only else None,
    }
    return {k: v for k, v in payload.items() if v is not None}