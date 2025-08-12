from fastapi import status, APIRouter, HTTPException
import uuid

from . import models
from . import service_layer as service
from . import schemas


router = APIRouter(prefix="/api/v1", tags=["Interact with Agents"])

@router.post(
    "/conversations",
    response_model=schemas.ConversationSummary,
    status_code=status.HTTP_201_CREATED,
    summary="새 대화 세션 생성",
    description="새로운 대화 세션을 생성하고 생성된 대화 객체를 반환합니다."
)
async def create_conversation():
    """새로운 대화 세션을 생성합니다."""
    new_conv: models.Conversation = await service.create_new_conversation_async()
    return schemas.ConversationSummary.from_conversation(new_conv)


@router.get(
    "/conversations",
    response_model=schemas.ListConversations,
    summary="모든 대화 목록 조회",
    description="서버에 저장된 모든 대화의 목록을 가져옵니다."
)
async def list_conversations():
    """모든 대화 목록을 반환합니다."""
    all_convs: list[models.Conversation] = service.get_all_conversations()
    return schemas.ListConversations.from_conversations(all_convs)


@router.post(
    "/conversations/{conversation_id}/messages",
    response_model=schemas.MessageInfo,
    status_code=status.HTTP_202_ACCEPTED,
    summary="대화에 메시지 전송",
    description="""
### 언제 (When):  
사용자가 메시지 전송 버튼을 클릭했을 때 호출합니다.  

### 핵심 응답 (Key Response):  
이 API는 즉시 `202 Accepted`와 함께 요청한 메시지의 고유 `message_id`를 반환합니다.

### **그 다음은 (What's Next):**  
1.  응답으로 받은 **`message_id`를 프론트엔드 상태에 저장**하세요.
2.  UI의 로딩 상태를 활성화합니다.
3.  아래 설명된 `GET /messages/pending` API를 **주기적으로 폴링(polling)**하세요.
    """,
)
async def send_message(conversation_id: uuid.UUID, request: schemas.CreateMessage):
    """특정 대화에 메시지를 전송하고, 처리 작업을 'pending' 상태로 등록합니다."""
    if service.get_conversation_by_id(conversation_id) is None:
        raise HTTPException(status_code=404, detail="Conversation not found")
    
    user_message = service.send_message(
        conversation_id=conversation_id,
        role=request.role,
        parts=request.parts
    )

    if user_message is None:
        raise HTTPException(status_code=500, detail="Failed to add message to conversation")

    return schemas.MessageInfo(
        conversation_id=conversation_id,
        message_id=user_message.id,
    )


@router.get(
    "/conversations/{conversation_id}/messages",
    response_model=schemas.ListMessages,
    summary="대화의 전체 메세지 히스토리 조회",
    description="""
### 언제 (When):
**두 가지 상황**에서 사용됩니다.
1.  **초기 로드:** 대화방에 처음 진입하여 이전 대화 내역 전체를 불러올 때.
2.  **최종 갱신:** `/messages/pending` 폴링이 끝난 후, AI의 답변이 포함된 최종 결과를 가져올 때.

### **주의 (Caution):**
이 API를 반복적으로 호출하면 전체 메시지를 매번 전송하여 네트워크 비효율성을 유발합니다.
상태 확인 폴링에는 반드시 가벼운 `GET /messages/pending` API를 사용해는 것을 추천합니다.
    """
)
async def list_messages(conversation_id: uuid.UUID):
    """특정 대화의 모든 메시지 목록을 반환합니다."""
    conversation = service.get_conversation_by_id(conversation_id)
    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")
    
    return schemas.ListMessages.from_messages(conversation.messages)


@router.get(
    "/messages/pending",
    response_model=schemas.ListPendingMessages,
    summary="처리 중인 메세지 목록 조회",
    description="""
### 언제 (When):
메시지 전송 직후, AI의 응답이 완료되었는지 확인하기 위해 `setInterval` 등으로 **반복 호출**합니다.
이 API는 네트워크 부하가 매우 적어 반복 호출에 적합합니다.

### 핵심 로직 (Core Logic):
이 API가 반환하는 `pending_messages` 목록에서, 내가 기다리던 `message_id`가 **사라지면** AI의 처리가 완료된 것입니다.

### **그 다음은 (What's Next):**
1.  **폴링을 중단** (`clearInterval`) 하세요.
2.  UI의 로딩 상태를 비활성화합니다.
3.  `GET /conversations/{conversation_id}/messages`를 **단 한 번 호출**하여 최종 결과를 화면에 갱신하세요.
    """,
)
async def list_pending_messages():
    """처리 중인 모든 메시지 목록을 반환합니다."""
    pending_list = service.get_pending_messages_list()
    pending_messages = [
        schemas.PendingMessageInfo(message_id=msg_id, conversation_id=conv_id)
        for msg_id, conv_id in pending_list
    ]
    return schemas.ListPendingMessages(pending_messages=pending_messages)

