from __future__ import annotations

from google.adk.memory import InMemoryMemoryService
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService

from .google_agents import host_agent
from .runtime import AgentProvider, AgentRuntime


def create_runtime(provider: AgentProvider) -> AgentRuntime:
    session_service = InMemorySessionService()
    memory_service = InMemoryMemoryService()

    runner = Runner(
        app_name="mek-multi-agent",
        agent=host_agent,
        session_service=session_service,
        memory_service=memory_service,
    )

    return AgentRuntime(
        provider=provider,
        runner=runner,
        session_service=session_service,
    )
