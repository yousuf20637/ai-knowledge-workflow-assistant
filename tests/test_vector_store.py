import uuid

from app.models.document import Document, DocumentChunk
from app.services.vector_store import ChromaVectorStore


def test_chroma_vector_store_indexes_and_searches_chunks(tmp_path) -> None:
    document = Document(
        id=uuid.uuid4(),
        filename="retrieval.md",
        content_type="text/markdown",
        source="test",
    )
    chunk = DocumentChunk(
        id=uuid.uuid4(),
        document_id=document.id,
        chunk_index=0,
        content="retrieval augmented generation uses vector search",
        token_count=6,
        vector_id=f"{document.id}:0",
    )

    vector_store = ChromaVectorStore(
        persist_path=str(tmp_path / "chroma"),
        collection_name="test_document_chunks",
    )
    vector_store.upsert_document_chunks(document, [chunk])

    results = vector_store.search("vector retrieval", limit=1)

    assert len(results) == 1
    assert results[0].chunk_id == chunk.id
    assert results[0].document_id == document.id
    assert results[0].filename == "retrieval.md"
