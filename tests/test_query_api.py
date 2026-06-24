import uuid
from collections.abc import Generator

from fastapi.testclient import TestClient
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app.api.dependencies import get_answer_provider, get_vector_store
from app.db.base import Base
from app.db.session import get_db
from app.main import app
from app.models.conversation import Conversation, Message, MessageRole
from app.services.answer_providers import LocalAnswerProvider
from app.services.vector_store import VectorSearchResult


engine = create_engine(
    "sqlite+pysqlite:///:memory:",
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)
TestingSessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)


class FakeVectorStore:
    def search(self, query: str, limit: int = 5) -> list[VectorSearchResult]:
        return [
            VectorSearchResult(
                chunk_id=uuid.uuid4(),
                document_id=uuid.uuid4(),
                filename="rag-notes.md",
                chunk_index=2,
                content="RAG retrieves relevant chunks before generating an answer.",
                distance=0.12,
            )
        ][:limit]

    def upsert_document_chunks(self, document, chunks) -> None:
        pass


def override_get_db() -> Generator[Session, None, None]:
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()


client = TestClient(app)


def setup_function() -> None:
    app.dependency_overrides[get_db] = override_get_db
    app.dependency_overrides[get_vector_store] = lambda: FakeVectorStore()
    app.dependency_overrides[get_answer_provider] = lambda: LocalAnswerProvider()
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)


def teardown_function() -> None:
    app.dependency_overrides.clear()


def test_query_documents_returns_answer_with_citations_and_persists_messages() -> None:
    response = client.post("/query", json={"question": "How does RAG work?", "limit": 1})

    assert response.status_code == 200
    body = response.json()
    assert body["question"] == "How does RAG work?"
    assert "rag-notes.md#chunk-2" in body["answer"]
    assert body["citations"][0]["filename"] == "rag-notes.md"

    with TestingSessionLocal() as db:
        conversation = db.scalar(select(Conversation))
        messages = db.scalars(select(Message).order_by(Message.created_at)).all()

    assert conversation is not None
    assert len(messages) == 2
    assert messages[0].role == MessageRole.USER
    assert messages[1].role == MessageRole.ASSISTANT
    assert messages[1].model == "local-retrieval-formatter"
