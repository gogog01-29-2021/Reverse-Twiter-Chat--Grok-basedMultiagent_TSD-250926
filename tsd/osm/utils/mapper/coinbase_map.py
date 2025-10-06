from typing import Any, Dict
from core.types.trade import ExternalOrder, OrderType
from dsm.utils.conversion_utils import instrument_to_exchange_symbol


def external_order_to_coinbase_payload(order: ExternalOrder) -> Dict[str, Any]:
    order_configuration: Dict[str, Any] = {}
    if order.order_type == OrderType.MARKET:
        order_configuration["market_market_ioc"] = {
            "base_size": str(order.qty),
            # "quote_size": str(order.quote_order_qty) if order.quote_order_qty else "0",
        }
    elif order.order_type == OrderType.LIMIT:
        order_configuration["limit_limit_gtc"] = {
            "base_size": str(order.qty),
            "limit_price": str(order.price),
            # "post_only": order.post_only,
        }
    
    payload = {
        "client_order_id": order.client_order_id,
        "product_id": instrument_to_exchange_symbol(order.exchange, order.instr),
        "side": order.order_side.value.lower(),
        "type": order.order_type.value.lower(),
        "price": str(order.price) if order.price else None,
        "size": str(order.qty) if order.qty else None,
        "order_configuration": order_configuration,
        "time_in_force": order.time_in_force.value,
    }
    return payload
