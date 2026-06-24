from dataclasses import dataclass

from sqlalchemy.orm import Session

from app.models.document import Document, DocumentChunk
from app.services.chunking import chunk_text

SUPPORTED_TEXT_TYPES = {
    "text/markdown",
    "text/plain",
    "application/octet-stream",
}


@dataclass(frozen=True)
class ParsedUpload:
    filename: str
    content_type: str | None
    text: str


def parse_text_upload(filename: str, content_type: str | None, raw_content: bytes) -> ParsedUpload:
    if content_type not in SUPPORTED_TEXT_TYPES:
        raise ValueError("Only text and Markdown uploads are supported right now.")

    try:
        text = raw_content.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("Upload must be UTF-8 encoded text.") from exc

    if not text.strip():
        raise ValueError("Upload cannot be empty.")

    return ParsedUpload(filename=filename, content_type=content_type, text=text)


def create_document_with_chunks(db: Session, upload: ParsedUpload) -> Document:
    chunks = chunk_text(upload.text)
    if not chunks:
        raise ValueError("Upload did not contain enough text to index.")

    document = Document(
        filename=upload.filename,
        content_type=upload.content_type,
        source="upload",
    )
    db.add(document)
    db.flush()

    for chunk in chunks:
        db.add(
            DocumentChunk(
                document_id=document.id,
                chunk_index=chunk.index,
                content=chunk.content,
                token_count=chunk.token_count,
                vector_id=f"{document.id}:{chunk.index}",
            )
        )

    db.commit()
    db.refresh(document)
    return document
