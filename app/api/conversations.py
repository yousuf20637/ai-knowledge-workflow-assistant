import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.orm import Session, selectinload

from app.db.session import get_db
from app.models.conversation import Conversation, Message
from app.schemas.conversation import ConversationRead, ConversationSummary, MessageRead

router = APIRouter(prefix="/conversations", tags=["conversations"])


def to_message_read(message: Message) -> MessageRead:
    return MessageRead(
        id=message.id,
        role=message.role.value,
        content=message.content,
        model=message.model,
        created_at=message.created_at,
    )


def to_conversation_summary(conversation: Conversation) -> ConversationSummary:
    return ConversationSummary(
        id=conversation.id,
        title=conversation.title,
        created_at=conversation.created_at,
        updated_at=conversation.updated_at,
        message_count=len(conversation.messages),
    )


@router.get("", response_model=list[ConversationSummary])
def list_conversations(
    db: Annotated[Session, Depends(get_db)],
) -> list[ConversationSummary]:
    conversations = db.scalars(
        select(Conversation)
        .options(selectinload(Conversation.messages))
        .order_by(Conversation.updated_at.desc(), Conversation.created_at.desc())
    ).all()

    return [to_conversation_summary(conversation) for conversation in conversations]


@router.get("/{conversation_id}", response_model=ConversationRead)
def get_conversation(
    conversation_id: uuid.UUID,
    db: Annotated[Session, Depends(get_db)],
) -> ConversationRead:
    conversation = db.scalar(
        select(Conversation)
        .where(Conversation.id == conversation_id)
        .options(selectinload(Conversation.messages))
    )

    if conversation is None:
        raise HTTPException(status_code=404, detail="Conversation not found")

    messages = sorted(conversation.messages, key=lambda message: message.created_at)
    summary = to_conversation_summary(conversation)
    return ConversationRead(
        **summary.model_dump(),
        messages=[to_message_read(message) for message in messages],
    )
