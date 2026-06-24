from functools import lru_cache

from app.core.config import settings
from app.services.vector_store import ChromaVectorStore, VectorStore


@lru_cache
def get_vector_store() -> VectorStore:
    return ChromaVectorStore(
        persist_path=settings.chroma_path,
        collection_name=settings.chroma_collection_name,
    )
