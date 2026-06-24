from collections.abc import Generator

from fastapi.testclient import TestClient
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app.api.dependencies import get_vector_store
from app.db.base import Base
from app.db.session import get_db
from app.main import app
from app.models.document import Document, DocumentChunk
from app.services.vector_store import VectorSearchResult


engine = create_engine(
    "sqlite+pysqlite:///:memory:",
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)
TestingSessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)


class FakeVectorStore:
    def __init__(self) -> None:
        self.indexed_chunks: list[DocumentChunk] = []

    def upsert_document_chunks(self, document: Document, chunks: list[DocumentChunk]) -> None:
        self.indexed_chunks.extend(chunks)

    def search(self, query: str, limit: int = 5) -> list[VectorSearchResult]:
        return [
            VectorSearchResult(
                chunk_id=self.indexed_chunks[0].id,
                document_id=self.indexed_chunks[0].document_id,
                filename="notes.md",
                chunk_index=self.indexed_chunks[0].chunk_index,
                content=self.indexed_chunks[0].content,
                distance=0.25,
            )
        ][:limit]


fake_vector_store = FakeVectorStore()


def override_get_db() -> Generator[Session, None, None]:
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()


app.dependency_overrides[get_db] = override_get_db
app.dependency_overrides[get_vector_store] = lambda: fake_vector_store
client = TestClient(app)


def setup_function() -> None:
    fake_vector_store.indexed_chunks.clear()
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)


def teardown_module() -> None:
    app.dependency_overrides.clear()


def test_upload_document_stores_document_and_chunks() -> None:
    content = "This is a small knowledge base note. " * 80

    response = client.post(
        "/documents",
        files={"file": ("notes.md", content.encode("utf-8"), "text/markdown")},
    )

    assert response.status_code == 201
    body = response.json()
    assert body["filename"] == "notes.md"
    assert body["content_type"] == "text/markdown"
    assert body["source"] == "upload"
    assert body["chunk_count"] > 0
    assert len(fake_vector_store.indexed_chunks) == body["chunk_count"]

    with TestingSessionLocal() as db:
        assert db.scalar(select(Document).where(Document.filename == "notes.md")) is not None
        assert db.scalar(select(DocumentChunk)) is not None


def test_upload_document_rejects_empty_file() -> None:
    response = client.post(
        "/documents",
        files={"file": ("empty.txt", b"", "text/plain")},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "Upload cannot be empty."


def test_search_documents_returns_vector_results() -> None:
    client.post(
        "/documents",
        files={"file": ("notes.md", b"retrieval augmented generation notes", "text/markdown")},
    )

    response = client.post("/documents/search", json={"query": "retrieval", "limit": 1})

    assert response.status_code == 200
    body = response.json()
    assert len(body) == 1
    assert body[0]["filename"] == "notes.md"
    assert body[0]["distance"] == 0.25
