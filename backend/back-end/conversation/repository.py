import uuid
from . import models


in_memory_conversations_db: dict[uuid.UUID, models.Conversation] = {}  # {conversation_id: Conversation}
in_memory_pending_messages_db: dict[uuid.UUID, uuid.UUID] = {}  # {message_id: conversation_id}

class ConversationRepository:
    def __init__(self):
        self.conversations = in_memory_conversations_db
        self.pending_messages = in_memory_pending_messages_db
    
    def add_conversation(self, conversation: models.Conversation) -> models.Conversation:
        """새 대화를 저장합니다."""
        self.conversations[conversation.id] = conversation
        return conversation
    
    def get_conversation(self, conversation_id: uuid.UUID) -> models.Conversation | None:
        """ID로 특정 대화를 조회합니다."""
        return self.conversations.get(conversation_id)
    
    def get_all_conversations(self) -> list[models.Conversation]:
        """모든 대화 목록을 반환합니다."""
        return list(self.conversations.values())
    
    def add_message_to_conversation(self, conversation_id: uuid.UUID, message: models.Message) -> models.Message | None:
        """대화에 메시지를 추가합니다."""
        conversation = self.get_conversation(conversation_id)
        if not conversation:
            return None
        
        conversation.messages.append(message)
        return message
    
    def add_pending_message(self, message_id: uuid.UUID, conversation_id: uuid.UUID) -> None:
        """처리 중인 메시지를 등록합니다."""
        self.pending_messages[message_id] = conversation_id
        print(f"+_+::add_pending_message::Panding messages: {self.pending_messages}")
    
    def remove_pending_message(self, message_id: uuid.UUID) -> None:
        """처리 중인 메시지를 제거합니다."""
        if message_id in self.pending_messages:
            del self.pending_messages[message_id]
            print(f"+_+::remove_pending_message::Panding messages: {self.pending_messages}")
    
    def get_pending_messages(self) -> list[tuple[uuid.UUID, uuid.UUID]]:
        """처리 중인 모든 메시지 목록을 반환합니다."""
        return list(self.pending_messages.items()) 