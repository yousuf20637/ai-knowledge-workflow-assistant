from functools import lru_cache

from app.core.config import settings
from app.services.answer_providers import AnswerProvider, build_answer_provider
from app.services.vector_store import ChromaVectorStore, VectorStore


@lru_cache
def get_vector_store() -> VectorStore:
    return ChromaVectorStore(
        persist_path=settings.chroma_path,
        collection_name=settings.chroma_collection_name,
    )


@lru_cache
def get_answer_provider() -> AnswerProvider:
    return build_answer_provider(
        provider_name=settings.answer_provider,
        openai_api_key=settings.openai_api_key,
        openai_model=settings.openai_answer_model,
    )
