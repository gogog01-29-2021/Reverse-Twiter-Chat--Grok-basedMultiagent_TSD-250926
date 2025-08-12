import uuid
from typing import Literal, Annotated
from pydantic import BaseModel, Field
from datetime import datetime

class TextPart(BaseModel):
    """텍스트 콘텐츠를 나타내는 파트"""
    type: Literal['text'] = 'text'
    text: str

class FilePart(BaseModel):
    """파일 콘텐츠를 나타내는 파트. 여기서는 URI 형태만 사용합니다."""
    type: Literal['file'] = 'file'
    uri: str
    mime_type: str | None = None

class ActionPart(BaseModel):
    type: Literal['action'] = 'action'
    action: str
    data: dict

Part = Annotated[
    TextPart | FilePart | ActionPart,
    Field(discriminator='type')
]

class Message(BaseModel):
    """하나의 메시지를 나타내는 모델"""
    id: uuid.UUID
    role: Literal['user', 'agent']
    parts: list[Part]

class Conversation(BaseModel):
    """하나의 대화 세션을 나타내는 모델"""
    id: uuid.UUID
    name: str = "새 대화"
    created_at: datetime = Field(default_factory=datetime.now)
    messages: list[Message] = Field(default_factory=list) 