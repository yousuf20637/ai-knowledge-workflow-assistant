import uuid

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    question: str = Field(min_length=1)
    limit: int = Field(default=4, ge=1, le=10)


class QueryCitation(BaseModel):
    document_id: uuid.UUID
    chunk_id: uuid.UUID
    filename: str
    chunk_index: int
    distance: float | None


class QueryResponse(BaseModel):
    conversation_id: uuid.UUID
    question: str
    answer: str
    citations: list[QueryCitation]
