import uuid
from config import EXCHANGE_CONFIG
from core.types.trade import ExternalOrder
from osm.core.base import AuthenticatedOrderManager
from osm.utils.auth import get_bybit_headers
from osm.utils.latency import measure_latency

class BybitOrderManager(AuthenticatedOrderManager):
    def __init__(self, api_key: str, secret: str):
        self.api_key = api_key
        self.secret = secret
        self.api_url = EXCHANGE_CONFIG["bybit"]['url']  

    def _build_payload(self, order: ExternalOrder, client_order_id: str):
        return {
            "category": "linear",
            "symbol": order.instr.symbol,
            "side": order.order_side.lower(),
            "orderType": order.order_type.lower(),
            "price": str(order.price),
            "qty": str(order.qty),
            "timeInForce": order.time_in_force,
            "orderLinkId": client_order_id,
        }

    @measure_latency("Bybit Order")
    async def send_order(self, order: ExternalOrder):
        client_order_id = f"byb-{uuid.uuid4().hex[:12]}"
        payload = self._build_payload(order, client_order_id)
        headers = get_bybit_headers(self.api_key, self.secret, payload)
        res = await self.send_http_request(self.api_url, payload, headers)
        self.handle_response(res, "Bybit")
        return {"exchange": "BYBIT", "client_order_id": client_order_id}