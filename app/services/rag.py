from dataclasses import dataclass

from sqlalchemy.orm import Session

from app.models.conversation import Conversation, Message, MessageRole
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


def build_local_answer(question: str, results: list[VectorSearchResult]) -> str:
    if not results:
        return (
            "I could not find relevant document chunks for that question yet. "
            "Upload more source material or try a more specific question."
        )

    lines = [
        "Based on the indexed documents, the most relevant context is:",
        "",
    ]

    for index, result in enumerate(results, start=1):
        citation = f"[{index}] {result.filename}#chunk-{result.chunk_index}"
        lines.append(f"{citation}: {result.content}")

    lines.extend(
        [
            "",
            "This is a retrieval-grounded draft answer. The next milestone will replace this",
            "formatter with an OpenAI-generated answer while preserving these citations.",
        ]
    )
    return "\n".join(lines)


def answer_question(
    db: Session,
    vector_store: VectorStore,
    question: str,
    limit: int = 4,
) -> RagAnswer:
    results = vector_store.search(question, limit=limit)
    answer = build_local_answer(question, results)

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
                model="local-retrieval-formatter",
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
