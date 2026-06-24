import uuid

from app.services.answer_providers import (
    LocalAnswerProvider,
    OpenAIAnswerProvider,
    build_answer_provider,
)
from app.services.vector_store import VectorSearchResult


def make_result() -> VectorSearchResult:
    return VectorSearchResult(
        chunk_id=uuid.uuid4(),
        document_id=uuid.uuid4(),
        filename="source.md",
        chunk_index=1,
        content="OpenAI can generate a grounded answer from retrieved context.",
        distance=0.2,
    )


def test_build_answer_provider_falls_back_to_local_without_api_key() -> None:
    provider = build_answer_provider(
        provider_name="openai",
        openai_api_key="",
        openai_model="gpt-5.5-mini",
    )

    assert isinstance(provider, LocalAnswerProvider)


def test_openai_answer_provider_builds_cited_prompt(monkeypatch) -> None:
    captured: dict[str, str] = {}

    class FakeResponses:
        def create(self, model: str, input: str):
            captured["model"] = model
            captured["input"] = input

            class FakeResponse:
                output_text = "A grounded answer [1]."

            return FakeResponse()

    class FakeClient:
        responses = FakeResponses()

    monkeypatch.setattr(
        "app.services.answer_providers.OpenAI",
        lambda api_key: FakeClient(),
    )

    provider = OpenAIAnswerProvider(api_key="test-key", model_name="gpt-5.5-mini")
    answer = provider.generate_answer("What does OpenAI do here?", [make_result()])

    assert answer == "A grounded answer [1]."
    assert captured["model"] == "gpt-5.5-mini"
    assert "Cite sources inline" in captured["input"]
    assert "source.md#chunk-1" in captured["input"]
