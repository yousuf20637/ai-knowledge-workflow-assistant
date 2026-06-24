from typing import Protocol

from openai import OpenAI

from app.services.vector_store import VectorSearchResult


class AnswerProvider(Protocol):
    model_name: str

    def generate_answer(self, question: str, results: list[VectorSearchResult]) -> str:
        pass


class LocalAnswerProvider:
    model_name = "local-retrieval-formatter"

    def generate_answer(self, question: str, results: list[VectorSearchResult]) -> str:
        if not results:
            return (
                "I could not find relevant document chunks for that question yet. "
                "Upload more source material or try a more specific question."
            )

        lines = [
            "Based on the indexed documents, the most relevant context is:",
            "",
        ]

        for index, result in enumerate(results, start=1):
            citation = f"[{index}] {result.filename}#chunk-{result.chunk_index}"
            lines.append(f"{citation}: {result.content}")

        lines.extend(
            [
                "",
                "This is a retrieval-grounded draft answer. Set ANSWER_PROVIDER=openai",
                "to generate a natural-language answer while preserving these citations.",
            ]
        )
        return "\n".join(lines)


class OpenAIAnswerProvider:
    def __init__(self, api_key: str, model_name: str) -> None:
        self.model_name = model_name
        self.client = OpenAI(api_key=api_key)

    def generate_answer(self, question: str, results: list[VectorSearchResult]) -> str:
        if not results:
            return LocalAnswerProvider().generate_answer(question, results)

        context = "\n\n".join(
            f"[{index}] {result.filename}#chunk-{result.chunk_index}\n{result.content}"
            for index, result in enumerate(results, start=1)
        )
        prompt = (
            "Answer the user's question using only the provided context. "
            "Cite sources inline using bracket numbers like [1]. "
            "If the context does not answer the question, say what is missing.\n\n"
            f"Question:\n{question}\n\n"
            f"Context:\n{context}"
        )

        response = self.client.responses.create(
            model=self.model_name,
            input=prompt,
        )
        return response.output_text


def build_answer_provider(
    provider_name: str,
    openai_api_key: str,
    openai_model: str,
) -> AnswerProvider:
    if provider_name.lower() == "openai" and openai_api_key:
        return OpenAIAnswerProvider(api_key=openai_api_key, model_name=openai_model)

    return LocalAnswerProvider()
