import uuid

from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.db.base import Base
from app.models.conversation import Message, MessageRole
from app.services.answer_providers import LocalAnswerProvider
from app.services.rag import answer_question
from app.services.vector_store import VectorSearchResult


engine = create_engine(
    "sqlite+pysqlite:///:memory:",
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)
TestingSessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)


class FakeVectorStore:
    def __init__(self, results: list[VectorSearchResult]) -> None:
        self.results = results
        self.search_count = 0

    def search(self, query: str, limit: int = 5) -> list[VectorSearchResult]:
        self.search_count += 1
        return self.results[:limit]

    def upsert_document_chunks(self, document, chunks) -> None:
        pass


def setup_function() -> None:
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)


def test_rag_graph_generates_answer_from_retrieved_context() -> None:
    result = VectorSearchResult(
        chunk_id=uuid.uuid4(),
        document_id=uuid.uuid4(),
        filename="graph.md",
        chunk_index=0,
        content="LangGraph coordinates retrieval and answer generation.",
        distance=0.1,
    )

    with TestingSessionLocal() as db:
        rag_answer = answer_question(
            db=db,
            vector_store=FakeVectorStore([result]),
            answer_provider=LocalAnswerProvider(),
            question="What coordinates the workflow?",
            limit=1,
        )
        messages = db.scalars(select(Message).order_by(Message.created_at)).all()

    assert "graph.md#chunk-0" in rag_answer.answer
    assert len(rag_answer.citations) == 1
    assert len(messages) == 2
    assert messages[0].role == MessageRole.USER
    assert messages[1].role == MessageRole.ASSISTANT


def test_rag_graph_routes_to_fallback_when_no_context_is_found() -> None:
    with TestingSessionLocal() as db:
        rag_answer = answer_question(
            db=db,
            vector_store=FakeVectorStore([]),
            answer_provider=LocalAnswerProvider(),
            question="What is missing?",
            limit=1,
        )

    assert rag_answer.citations == []
    assert "could not find relevant document chunks" in rag_answer.answer


def test_rag_graph_answers_small_talk_without_retrieval() -> None:
    vector_store = FakeVectorStore(
        [
            VectorSearchResult(
                chunk_id=uuid.uuid4(),
                document_id=uuid.uuid4(),
                filename="readme.md",
                chunk_index=0,
                content="This should not be retrieved for greetings.",
                distance=0.1,
            )
        ]
    )

    with TestingSessionLocal() as db:
        rag_answer = answer_question(
            db=db,
            vector_store=vector_store,
            answer_provider=LocalAnswerProvider(),
            question="hello there",
            limit=1,
        )

    assert vector_store.search_count == 0
    assert rag_answer.citations == []
    assert "Upload a document or ask a question" in rag_answer.answer
