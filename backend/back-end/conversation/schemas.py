import uuid
from pydantic import BaseModel
from typing import Literal
from datetime import datetime
from . import models

class RequestModel(BaseModel):
    pass

class ResponseModel(BaseModel):
    pass

# --- Message 관련 스키마 ---
class CreateMessage(RequestModel):
    """메시지 생성을 위한 요청 모델"""
    role: Literal['user', 'agent']
    parts: list[models.Part]

    class Config:
        json_schema_extra = {
            "example": {
                "role": "user",
                "parts": [
                    {"type": "text", "text": "시뮬레이션 해야... 겠지?"}
                ]
            }
        }

class MessageInfo(ResponseModel):
    """메시지 전송 후 수신하는 확인 정보 모델"""
    conversation_id: uuid.UUID
    message_id: uuid.UUID
    status: str = "accepted"

    class Config:
        json_schema_extra = {
            "example": {
                "conversation_id": "conv-1234567890abcdef",
                "message_id": "msg-abcdef1234567890",
                "status": "accepted"
            }
        }


class MessageSummary(ResponseModel):
    """메시지 요약 정보"""
    id: uuid.UUID
    role: Literal['user', 'agent']
    parts: list[dict]

    class Config:
        json_schema_extra = {
            "example": {
                "id": "msg-abcdef1234567890",
                "role": "user",
                "parts": [
                    {"type": "text", "text": "안녕, AI야!"}
                ]
            }
        }

    @classmethod
    def from_message(cls, message: models.Message) -> "MessageSummary":
        parts: list[dict] = []
        for part in message.parts:
            if isinstance(part, models.ActionPart):
                parts.append({"type": part.action} | part.data)
            else:
                parts.append(part.model_dump())
        return cls(
            id=message.id,
            role=message.role,
            parts=parts
        )

class ListMessages(ResponseModel):
    """대화의 메시지 목록 응답 모델"""
    messages: list[MessageSummary]

    class Config:
        json_schema_extra = {
            "example": {
                "messages": [
                    {
                        "id": "msg-abcdef1234567890",
                        "role": "user",
                        "parts": [
                            {"type": "text", "text": "안녕, AI야!"}
                        ]
                    },
                    {
                        "id": "msg-1234567890abcdef",
                        "role": "agent",
                        "parts": [
                            {"type": "text", "text": "안녕하세요! 무엇을 도와드릴까요?"}
                        ]
                    },
                ]
            }
        }

    @classmethod
    def from_messages(cls, messages: list[models.Message]) -> "ListMessages":
        return cls(
            messages=[MessageSummary.from_message(message) for message in messages]
        )

# --- Conversation 관련 스키마 ---
class ConversationSummary(ResponseModel):
    id: uuid.UUID
    name: str
    created_at: datetime

    class Config:
        json_schema_extra = {
            "example": {
                "id": "conv-1234567890abcdef",
                "name": "새 대화",
                "created_at": "2024-06-01T12:34:56.789000"
            }
        }

    @classmethod
    def from_conversation(cls, conversation: models.Conversation) -> "ConversationSummary":
        return cls(
            id=conversation.id,
            name=conversation.name,
            created_at=conversation.created_at
        )

class ListConversations(ResponseModel):
    """전체 대화 목록 응답 모델"""
    conversations: list[ConversationSummary]

    class Config:
        json_schema_extra = {
            "example": {
                "conversations": [
                    {
                        "id": "conv-1234567890abcdef",
                        "name": "나쁜 강아지는 없다",
                        "created_at": "2024-06-01T12:34:56.789000"
                    },
                    {
                        "id": "conv-abcdef1234567890",
                        "name": "뽀로로 vs 피카츄",
                        "created_at": "2024-06-02T09:00:00.000000"
                    }
                ]
            }
        }

    @classmethod
    def from_conversations(cls, conversations: list[models.Conversation]) -> "ListConversations":
        return cls(
            conversations=[ConversationSummary.from_conversation(conv) for conv in conversations]
        )

# --- Pending Message 관련 스키마 ---
class PendingMessageInfo(ResponseModel):
    """처리 중인 단일 메시지 정보"""
    conversation_id: uuid.UUID
    message_id: uuid.UUID

    class Config:
        json_schema_extra = {
            "example": {
                "conversation_id": "conv-1234567890abcdef",
                "message_id": "msg-abcdef1234567890"
            }
        }

class ListPendingMessages(ResponseModel):
    """처리 중인 모든 메시지 목록 응답 모델"""
    pending_messages: list[PendingMessageInfo]

    class Config:
        json_schema_extra = {
            "example": {
                "pending_messages": [
                    {
                        "conversation_id": "conv-1234567890abcdef",
                        "message_id": "msg-abcdef1234567890"
                    },
                    {
                        "conversation_id": "conv-abcdef1234567890",
                        "message_id": "msg-1234567890abcdef"
                    }
                ]
            }
        } 