from dataclasses import dataclass


@dataclass(frozen=True)
class TextChunk:
    index: int
    content: str
    token_count: int


def estimate_token_count(text: str) -> int:
    return max(1, len(text.split()))


def chunk_text(text: str, max_words: int = 220, overlap_words: int = 40) -> list[TextChunk]:
    words = text.split()
    if not words:
        return []

    chunks: list[TextChunk] = []
    start = 0

    while start < len(words):
        end = min(start + max_words, len(words))
        content = " ".join(words[start:end])
        chunks.append(
            TextChunk(
                index=len(chunks),
                content=content,
                token_count=estimate_token_count(content),
            )
        )

        if end == len(words):
            break

        start = max(end - overlap_words, start + 1)

    return chunks
