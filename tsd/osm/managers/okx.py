import uuid, json
import httpx
from core.types.trade import ExternalOrder
from osm.core.base import AuthenticatedOrderManager
from osm.utils.auth import get_okx_headers
from osm.utils.latency import measure_latency

class OkxOrderManager(AuthenticatedOrderManager):
    def __init__(self, api_key: str, secret_key: str, passphrase: str):
        self.api_key = api_key
        self.secret_key = secret_key
        self.passphrase = passphrase
        self.api_url = "https://www.okx.com/api/v5/trade/order"
        self.request_path = "/api/v5/trade/order"

    def _build_payload(self, order: ExternalOrder, client_order_id: str):
        return {
            "instId": order.instr.symbol,
            "tdMode": "cash",
            "side": order.order_side.lower(),
            "ordType": order.order_type.lower(),
            "px": str(order.price),
            "sz": str(order.qty),
            "clOrdId": client_order_id
        }

    @measure_latency("OKX Order")
    async def send_order(self, order: ExternalOrder):
        client_order_id = f"okx-{uuid.uuid4().hex[:12]}"
        payload = self._build_payload(order, client_order_id)
        body = json.dumps(payload)
        headers = get_okx_headers(
            self.api_key, self.secret_key, self.passphrase,
            body=body, method="POST", request_path=self.request_path
        )
        async with httpx.AsyncClient(timeout=1.0) as client:
            response = await client.post(self.api_url, data=body, headers=headers)
        self.handle_response(response, "OKX")
        return {"exchange": "OKX", "client_order_id": client_order_id}