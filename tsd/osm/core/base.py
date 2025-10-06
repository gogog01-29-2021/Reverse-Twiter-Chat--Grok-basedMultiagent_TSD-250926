from abc import ABC, abstractmethod

import httpx


class BaseOrderManager(ABC):
    @abstractmethod
    async def send_order(self, order) -> None:
        ...


class AuthenticatedOrderManager(BaseOrderManager, ABC):
    @staticmethod
    async def send_http_request(self, url: str, payload: dict, headers: dict) -> httpx.Response:
        async with httpx.AsyncClient(timeout=1.0) as client:
            return await client.post(url, json=payload, headers=headers)

    @staticmethod
    def handle_response(self, response: httpx.Response, label: str):
        if response.status_code != 200:
            print(f"[{label}] Order failed: {response.text}")
        else:
            print(f"[{label}] Order successful: {response.json()}")
