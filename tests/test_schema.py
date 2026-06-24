from app import models  # noqa: F401
from app.db.base import Base


def test_database_metadata_includes_core_tables() -> None:
    assert set(Base.metadata.tables) >= {
        "conversations",
        "document_chunks",
        "documents",
        "messages",
    }
