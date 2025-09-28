import asyncio
import uuid
from typing import Any

from ..agents import AgentProvider, get_agent_runtime
from . import models
from .repository import ConversationRepository
from .agent_action_mapper import get_agent_action_type

# ===============================================================================
#  Agent Configuration
# ===============================================================================
_agent_runtime = get_agent_runtime()
session_service = _agent_runtime.session_service
host_agent_runner = _agent_runtime.runner
_agent_provider = _agent_runtime.provider


# ===============================================================================
#  Background Message Processing
# ===============================================================================
async def process_message_async(conversation_id: uuid.UUID, user_message: models.Message):
    repository = ConversationRepository()
    await asyncio.sleep(1)

    try:
        new_message = _build_model_input(_agent_provider, user_message)
        async for event in host_agent_runner.run_async(
            user_id="mek-user",
            session_id=str(conversation_id),
            new_message=new_message,
        ):
            message: models.Message | None = None
            if getattr(event, "content", None) and getattr(event.content, "parts", None):
                if hasattr(event, "get_function_calls") and event.get_function_calls():
                    print("  Type: Tool Call Request")
                elif hasattr(event, "get_function_responses") and event.get_function_responses():
                    function_response = event.get_function_responses()[0]
                    message = models.Message(
                        id=uuid.uuid4(),
                        role="agent",
                        parts=[models.ActionPart(
                            action=get_agent_action_type(function_response.name),
                            data=function_response.response,
                        )],
                    )
                elif event.content.parts[0].text:
                    message = models.Message(
                        id=uuid.uuid4(),
                        role="agent",
                        parts=[models.TextPart(text=event.content.parts[0].text)],
                    )
            elif getattr(event, "actions", None):
                if getattr(event.actions, "transfer_to_agent", None):
                    message = models.Message(
                        id=uuid.uuid4(),
                        role="agent",
                        parts=[models.ActionPart(
                            action="transfer_to_agent",
                            data={"agent_name": event.actions.transfer_to_agent},
                        )],
                    )
            if message:
                repository.add_message_to_conversation(conversation_id, message)

    except Exception as e:
        error_message = models.Message(
            id=uuid.uuid4(),
            role="agent",
            parts=[models.TextPart(text=f"오류가 발생했습니다: {e}")],
        )
        conversation = repository.get_conversation(conversation_id)
        if conversation:
            repository.add_message_to_conversation(conversation_id, error_message)

    finally:
        repository.remove_pending_message(user_message.id)


# ===============================================================================
#  Service Layer
# ===============================================================================

async def create_new_conversation_async() -> models.Conversation:
    """새 대화를 생성하고 DB에 저장합니다."""
    repository = ConversationRepository()
    new_conversation = models.Conversation(id=uuid.uuid4())

    create_session = getattr(session_service, "create_session", None)
    if callable(create_session):
        await create_session(
            app_name="mek-multi-agent", user_id="mek-user", session_id=str(new_conversation.id)
        )

    repository.add_conversation(new_conversation)
    return new_conversation


def get_all_conversations() -> list[models.Conversation]:
    """모든 대화 목록을 반환합니다."""
    repository = ConversationRepository()
    return repository.get_all_conversations()


def get_conversation_by_id(conversation_id: uuid.UUID) -> models.Conversation | None:
    """ID로 특정 대화를 조회합니다."""
    repository = ConversationRepository()
    return repository.get_conversation(conversation_id)


def send_message(conversation_id: uuid.UUID, role: str, parts: list[models.Part]) -> models.Message | None:
    """
    대화에 사용자 메시지를 추가하고, 에이전트 응답을 위한 백그라운드 작업을 시작합니다.
    """
    repository = ConversationRepository()
    conversation = repository.get_conversation(conversation_id)
    if not conversation:
        return None

    user_message = models.Message(id=uuid.uuid4(), role=role, parts=parts)
    repository.add_message_to_conversation(conversation_id, user_message)
    repository.add_pending_message(user_message.id, conversation_id)

    # 에이전트의 응답을 생성하는 비동기 작업을 백그라운드에서 실행, Non-blocking
    asyncio.create_task(process_message_async(conversation_id, user_message))

    return user_message


def get_pending_messages_list() -> list[tuple[uuid.UUID, uuid.UUID]]:
    """처리 중인 모든 메시지 목록을 반환합니다."""
    repository = ConversationRepository()
    return repository.get_pending_messages()


def _build_model_input(provider: AgentProvider, user_message: models.Message) -> Any:
    if provider == AgentProvider.API:
        from google.genai import types as genai_types

        return genai_types.Content(
            role="user",
            parts=[genai_types.Part.from_text(text=user_message.parts[0].text)],
        )

    return user_message.parts[0].text if user_message.parts else ""
