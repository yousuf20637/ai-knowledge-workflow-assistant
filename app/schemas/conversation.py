import uuid
from datetime import datetime

from pydantic import BaseModel


class MessageRead(BaseModel):
    id: uuid.UUID
    role: str
    content: str
    model: str | None
    created_at: datetime


class ConversationSummary(BaseModel):
    id: uuid.UUID
    title: str | None
    created_at: datetime
    updated_at: datetime
    message_count: int


class ConversationRead(ConversationSummary):
    messages: list[MessageRead]
