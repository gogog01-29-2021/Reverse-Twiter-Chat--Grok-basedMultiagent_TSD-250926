from __future__ import annotations

import os
from dataclasses import dataclass
from enum import Enum
from functools import lru_cache
from typing import Any


class AgentProvider(str, Enum):
    API = "api"
    LOCAL = "local"


@dataclass
class AgentRuntime:
    provider: AgentProvider
    runner: Any
    session_service: Any


def _resolve_provider(value: str | None) -> AgentProvider:
    provider_name = (value or AgentProvider.API.value).strip().lower()
    try:
        return AgentProvider(provider_name)
    except ValueError as exc:
        raise ValueError(
            f"Unsupported AGENT_PROVIDER '{value}'. Valid options: {[p.value for p in AgentProvider]}"
        ) from exc


@lru_cache(maxsize=1)
def get_agent_runtime() -> AgentRuntime:
    provider = _resolve_provider(os.getenv("AGENT_PROVIDER"))

    if provider == AgentProvider.API:
        from .google_runtime import create_runtime
    else:
        from .local_runtime import create_runtime

    return create_runtime(provider)
