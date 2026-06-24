from typing import Annotated

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.api.dependencies import get_vector_store
from app.db.session import get_db
from app.schemas.query import QueryCitation, QueryRequest, QueryResponse
from app.services.rag import answer_question
from app.services.vector_store import VectorStore

router = APIRouter(prefix="/query", tags=["query"])


@router.post("", response_model=QueryResponse)
def query_documents(
    request: QueryRequest,
    db: Annotated[Session, Depends(get_db)],
    vector_store: Annotated[VectorStore, Depends(get_vector_store)],
) -> QueryResponse:
    rag_answer = answer_question(
        db=db,
        vector_store=vector_store,
        question=request.question,
        limit=request.limit,
    )

    return QueryResponse(
        conversation_id=rag_answer.conversation.id,
        question=rag_answer.question,
        answer=rag_answer.answer,
        citations=[
            QueryCitation(
                document_id=citation.result.document_id,
                chunk_id=citation.result.chunk_id,
                filename=citation.result.filename,
                chunk_index=citation.result.chunk_index,
                distance=citation.result.distance,
            )
            for citation in rag_answer.citations
        ],
    )
