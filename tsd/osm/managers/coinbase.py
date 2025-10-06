import uuid
import json
import httpx
from config import EXCHANGE_CONFIG
from core.types.trade import ExternalOrder
from osm.core.base import AuthenticatedOrderManager
from osm.utils.auth import get_coinbase_headers
from osm.utils.latency import measure_latency
from osm.utils.mapper.coinbase_map import external_order_to_coinbase_payload


class CoinbaseOrderManager(AuthenticatedOrderManager):
    def __init__(self, api_key: str, secret: str, passphrase: str):
        self.api_key = api_key
        self.secret = secret
        self.passphrase = passphrase
        self.api_url = EXCHANGE_CONFIG["coinbase"]['url']   
        self.request_path = "/orders"

    @staticmethod
    def _build_payload(order: ExternalOrder):
        return external_order_to_coinbase_payload(order)

    @measure_latency("Coinbase REST Order")
    async def send_order(self, order: ExternalOrder):
        payload = self._build_payload(order)
        body = json.dumps(payload)

        headers = get_coinbase_headers(
            self.api_key, self.secret, self.passphrase,
            method="POST",
            request_path=self.request_path,
            body=body
        )

        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.post(self.api_url, json=payload, headers=headers)

        self.handle_response(response, "Coinbase")
        return {"exchange": "COINBASE", "client_order_id": order.client_order_id, **response.json()}