import uuid
from config import EXCHANGE_CONFIG
from core.types.trade import ExternalOrder
from osm.core.base import AuthenticatedOrderManager
from osm.utils.auth import get_binance_headers
from osm.utils.latency import measure_latency
from osm.utils.mapper.binance_map import external_order_to_binance_payload


class BinanceOrderManager(AuthenticatedOrderManager):
    def __init__(self, api_key: str, secret: str):
        self.api_key = api_key
        self.secret = secret
        self.api_url = EXCHANGE_CONFIG["binance"]['url']

    def _build_payload(self, order: ExternalOrder):
        return external_order_to_binance_payload(order)

    @measure_latency("Binance REST Order")
    async def send_order(self, order: ExternalOrder):
        
        payload = self._build_payload(order)
        headers, query = get_binance_headers(self.api_key, self.secret, payload)
        url = f"{self.api_url}?{query}"

        response = await self.send_http_request(url, None, headers)
        self.handle_response(response, "Binance")
        return {"exchange": "BINANCE", "client_order_id": order.client_order_id, **response.json()}