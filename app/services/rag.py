from dataclasses import dataclass

from sqlalchemy.orm import Session

from app.models.conversation import Conversation, Message, MessageRole
from app.services.answer_providers import AnswerProvider
from app.services.vector_store import VectorSearchResult, VectorStore


@dataclass(frozen=True)
class Citation:
    result: VectorSearchResult

    @property
    def label(self) -> str:
        return f"{self.result.filename}#chunk-{self.result.chunk_index}"


@dataclass(frozen=True)
class RagAnswer:
    conversation: Conversation
    question: str
    answer: str
    citations: list[Citation]


def answer_question(
    db: Session,
    vector_store: VectorStore,
    answer_provider: AnswerProvider,
    question: str,
    limit: int = 4,
) -> RagAnswer:
    results = vector_store.search(question, limit=limit)
    answer = answer_provider.generate_answer(question, results)

    conversation = Conversation(title=question[:255])
    db.add(conversation)
    db.flush()

    db.add_all(
        [
            Message(
                conversation_id=conversation.id,
                role=MessageRole.USER,
                content=question,
            ),
            Message(
                conversation_id=conversation.id,
                role=MessageRole.ASSISTANT,
                content=answer,
                model=answer_provider.model_name,
            ),
        ]
    )
    db.commit()
    db.refresh(conversation)

    return RagAnswer(
        conversation=conversation,
        question=question,
        answer=answer,
        citations=[Citation(result=result) for result in results],
    )
