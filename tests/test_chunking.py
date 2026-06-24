from app.services.chunking import chunk_text


def test_chunk_text_splits_long_text_with_overlap() -> None:
    text = " ".join(f"word-{index}" for index in range(500))

    chunks = chunk_text(text, max_words=100, overlap_words=20)

    assert len(chunks) == 6
    assert chunks[0].index == 0
    assert chunks[1].content.startswith("word-80")
    assert chunks[-1].token_count == 100
