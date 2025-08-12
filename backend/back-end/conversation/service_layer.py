import asyncio
import uuid
from google.genai import types as genai_types
from google.adk.sessions import InMemorySessionService
from google.adk.memory import InMemoryMemoryService
from google.adk.runners import Runner
from ..agents import agents
from . import models
from .repository import ConversationRepository
from .agent_action_mapper import get_agent_action_type

# ===============================================================================
#  Agent Configuration
# ===============================================================================
session_service = InMemorySessionService()
memory_service = InMemoryMemoryService()

host_agent_runner: Runner = Runner(
    app_name="mek-multi-agent",
    agent=agents.host_agent,
    session_service=session_service,
    memory_service=memory_service
)


# ===============================================================================
#  Background Message Processing
# ===============================================================================
async def process_message_async(conversation_id: uuid.UUID, user_message: models.Message):
    repository = ConversationRepository()
    await asyncio.sleep(1)

    try:
        async for event in host_agent_runner.run_async(
            user_id="mek-user",
            session_id=str(conversation_id),
            new_message=genai_types.Content(
                role='user', parts=[genai_types.Part.from_text(text=user_message.parts[0].text)]
            ),
        ):
            message: models.Message | None = None
            if event.content and event.content.parts:
                if event.get_function_calls():
                    print("  Type: Tool Call Request")
                elif event.get_function_responses():
                    function_response = event.get_function_responses()[0]
                    message = models.Message(
                        id=uuid.uuid4(),
                        role='agent',
                        parts=[models.ActionPart(
                            action=get_agent_action_type(function_response.name),
                            data=function_response.response
                        )]
                    )
                elif event.content.parts[0].text:
                    message = models.Message(
                        id=uuid.uuid4(),
                        role='agent',
                        parts=[models.TextPart(text=event.content.parts[0].text)]
                    )
            elif event.actions:
                if event.actions.transfer_to_agent:
                    message = models.Message(
                        id=uuid.uuid4(),
                        role='agent',
                        parts=[models.ActionPart(
                            action="transfer_to_agent",
                            data={"agent_name": event.actions.transfer_to_agent}
                        )]
                    )
            if message:
                repository.add_message_to_conversation(conversation_id, message)

    except Exception as e:
        error_message = models.Message(
            id=uuid.uuid4(),
            role='agent',
            parts=[models.TextPart(text=f"오류가 발생했습니다: {e}")]
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
    await session_service.create_session(
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
