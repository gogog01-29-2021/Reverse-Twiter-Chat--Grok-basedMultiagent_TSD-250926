from __future__ import annotations

import os
from types import SimpleNamespace
from typing import Any, AsyncIterator

import httpx

from .runtime import AgentProvider, AgentRuntime


class DummySessionService:
    async def create_session(self, *_, **__):  # type: ignore[no-untyped-def]
        return None


class LocalAgentEvent:
    def __init__(self, text: str):
        part = SimpleNamespace(text=text)
        self.content = SimpleNamespace(parts=[part])
        self._function_calls: list[Any] = []
        self._function_responses: list[Any] = []
        self.actions: Any = None

    def get_function_calls(self) -> list[Any]:
        return self._function_calls

    def get_function_responses(self) -> list[Any]:
        return self._function_responses


class LocalAgentRunner:
    def __init__(self, endpoint: str, timeout: float):
        self._endpoint = endpoint
        self._timeout = timeout

    async def run_async(  # type: ignore[override]
        self,
        user_id: str,
        session_id: str,
        new_message: Any,
    ) -> AsyncIterator[LocalAgentEvent]:
        prompt = _extract_prompt(new_message)
        payload = {
            "user_id": user_id,
            "session_id": session_id,
            "prompt": prompt,
        }

        async with httpx.AsyncClient(timeout=self._timeout) as client:
            response = await client.post(self._endpoint, json=payload)
            response.raise_for_status()
            data = response.json()

        text = _extract_text_from_response(data)
        yield LocalAgentEvent(text=text)


def _extract_prompt(message: Any) -> str:
    if hasattr(message, "parts") and message.parts:
        part = message.parts[0]
        if hasattr(part, "text") and isinstance(part.text, str):
            return part.text
    if isinstance(message, str):
        return message
    return str(message)


def _extract_text_from_response(data: Any) -> str:
    if isinstance(data, dict):
        for key in ("text", "response", "output", "content"):
            value = data.get(key)
            if isinstance(value, str) and value.strip():
                return value
    return ""


def create_runtime(provider: AgentProvider) -> AgentRuntime:
    endpoint = os.getenv("LOCAL_AGENT_ENDPOINT", "http://localhost:8001/generate")
    timeout = float(os.getenv("LOCAL_AGENT_TIMEOUT", "60"))

    runner = LocalAgentRunner(endpoint=endpoint, timeout=timeout)
    session_service = DummySessionService()

    return AgentRuntime(
        provider=provider,
        runner=runner,
        session_service=session_service,
    )
