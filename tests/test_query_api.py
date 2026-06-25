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


def test_conversation_history_lists_and_returns_messages() -> None:
    query_response = client.post(
        "/query",
        json={"question": "What does the assistant retrieve?", "limit": 1},
    )
    conversation_id = query_response.json()["conversation_id"]

    list_response = client.get("/conversations")

    assert list_response.status_code == 200
    conversations = list_response.json()
    assert len(conversations) == 1
    assert conversations[0]["id"] == conversation_id
    assert conversations[0]["message_count"] == 2
    assert conversations[0]["title"] == "What does the assistant retrieve?"

    detail_response = client.get(f"/conversations/{conversation_id}")

    assert detail_response.status_code == 200
    detail = detail_response.json()
    assert detail["id"] == conversation_id
    assert [message["role"] for message in detail["messages"]] == ["user", "assistant"]
    assert detail["messages"][0]["content"] == "What does the assistant retrieve?"
    assert "rag-notes.md#chunk-2" in detail["messages"][1]["content"]


def test_get_conversation_returns_404_for_unknown_id() -> None:
    missing_id = uuid.uuid4()

    response = client.get(f"/conversations/{missing_id}")

    assert response.status_code == 404
    assert response.json()["detail"] == "Conversation not found"
