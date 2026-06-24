import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict


class DocumentChunkRead(BaseModel):
    id: uuid.UUID
    chunk_index: int
    content: str
    token_count: int | None
    vector_id: str | None
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class DocumentRead(BaseModel):
    id: uuid.UUID
    filename: str
    content_type: str | None
    source: str | None
    created_at: datetime
    chunk_count: int

    model_config = ConfigDict(from_attributes=True)


class DocumentSearchRequest(BaseModel):
    query: str
    limit: int = 5


class DocumentSearchResult(BaseModel):
    chunk_id: uuid.UUID
    document_id: uuid.UUID
    filename: str
    chunk_index: int
    content: str
    distance: float | None
