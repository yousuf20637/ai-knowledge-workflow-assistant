import uuid
from dataclasses import dataclass
from typing import Protocol

import chromadb

from app.models.document import Document, DocumentChunk
from app.services.embeddings import EmbeddingProvider, HashEmbeddingProvider


@dataclass(frozen=True)
class VectorSearchResult:
    chunk_id: uuid.UUID
    document_id: uuid.UUID
    filename: str
    chunk_index: int
    content: str
    distance: float | None


class VectorStore(Protocol):
    def upsert_document_chunks(self, document: Document, chunks: list[DocumentChunk]) -> None:
        pass

    def search(self, query: str, limit: int = 5) -> list[VectorSearchResult]:
        pass


class ChromaVectorStore:
    def __init__(
        self,
        persist_path: str,
        collection_name: str,
        embedding_provider: EmbeddingProvider | None = None,
    ) -> None:
        self.embedding_provider = embedding_provider or HashEmbeddingProvider()
        self.client = chromadb.PersistentClient(path=persist_path)
        self.collection = self.client.get_or_create_collection(name=collection_name)

    def upsert_document_chunks(self, document: Document, chunks: list[DocumentChunk]) -> None:
        if not chunks:
            return

        embeddings = self.embedding_provider.embed_texts([chunk.content for chunk in chunks])
        self.collection.upsert(
            ids=[str(chunk.vector_id) for chunk in chunks],
            documents=[chunk.content for chunk in chunks],
            embeddings=embeddings,
            metadatas=[
                {
                    "chunk_id": str(chunk.id),
                    "document_id": str(document.id),
                    "filename": document.filename,
                    "chunk_index": chunk.chunk_index,
                }
                for chunk in chunks
            ],
        )

    def search(self, query: str, limit: int = 5) -> list[VectorSearchResult]:
        query_embedding = self.embedding_provider.embed_texts([query])[0]
        raw_results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=limit,
            include=["documents", "distances", "metadatas"],
        )

        documents = raw_results.get("documents", [[]])[0]
        distances = raw_results.get("distances", [[]])[0]
        metadatas = raw_results.get("metadatas", [[]])[0]

        results: list[VectorSearchResult] = []
        for content, distance, metadata in zip(documents, distances, metadatas, strict=False):
            results.append(
                VectorSearchResult(
                    chunk_id=uuid.UUID(str(metadata["chunk_id"])),
                    document_id=uuid.UUID(str(metadata["document_id"])),
                    filename=str(metadata["filename"]),
                    chunk_index=int(metadata["chunk_index"]),
                    content=content,
                    distance=distance,
                )
            )

        return results
