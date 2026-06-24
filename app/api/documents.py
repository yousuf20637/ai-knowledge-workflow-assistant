from typing import Annotated

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.models.document import Document
from app.schemas.document import DocumentRead
from app.services.documents import create_document_with_chunks, parse_text_upload

router = APIRouter(prefix="/documents", tags=["documents"])


def to_document_read(document: Document) -> DocumentRead:
    return DocumentRead(
        id=document.id,
        filename=document.filename,
        content_type=document.content_type,
        source=document.source,
        created_at=document.created_at,
        chunk_count=len(document.chunks),
    )


@router.post("", response_model=DocumentRead, status_code=status.HTTP_201_CREATED)
async def upload_document(
    file: Annotated[UploadFile, File()],
    db: Annotated[Session, Depends(get_db)],
) -> DocumentRead:
    raw_content = await file.read()

    try:
        parsed_upload = parse_text_upload(file.filename, file.content_type, raw_content)
        document = create_document_with_chunks(db, parsed_upload)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

    return to_document_read(document)


@router.get("", response_model=list[DocumentRead])
def list_documents(db: Annotated[Session, Depends(get_db)]) -> list[DocumentRead]:
    documents = db.scalars(select(Document).order_by(Document.created_at.desc())).unique().all()
    return [to_document_read(document) for document in documents]
