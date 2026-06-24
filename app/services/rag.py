from dataclasses import dataclass

from sqlalchemy.orm import Session

from app.models.conversation import Conversation
from app.services.answer_providers import AnswerProvider
from app.services.rag_graph import build_rag_graph
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
    graph = build_rag_graph(
        db=db,
        vector_store=vector_store,
        answer_provider=answer_provider,
    )
    state = graph.invoke(
        {
            "question": question,
            "limit": limit,
            "results": [],
            "answer": "",
            "conversation": None,
        }
    )
    conversation = state["conversation"]
    assert conversation is not None
    results = state["results"]

    return RagAnswer(
        conversation=conversation,
        question=question,
        answer=state["answer"],
        citations=[Citation(result=result) for result in results],
    )
