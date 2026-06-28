from typing import Literal, TypedDict

from langgraph.graph import END, START, StateGraph
from sqlalchemy.orm import Session

from app.models.conversation import Conversation, Message, MessageRole
from app.services.answer_providers import AnswerProvider, LocalAnswerProvider
from app.services.tracing import trace_rag_node
from app.services.vector_store import VectorSearchResult, VectorStore


class RagGraphState(TypedDict):
    question: str
    limit: int
    results: list[VectorSearchResult]
    answer: str
    conversation: Conversation | None


def build_rag_graph(
    db: Session,
    vector_store: VectorStore,
    answer_provider: AnswerProvider,
):
    def retrieve_context(state: RagGraphState) -> dict:
        return {
            "results": vector_store.search(
                state["question"],
                limit=state["limit"],
            )
        }

    def route_after_retrieval(state: RagGraphState) -> Literal["generate_answer", "fallback_answer"]:
        if state["results"]:
            return "generate_answer"

        return "fallback_answer"

    def generate_answer(state: RagGraphState) -> dict:
        return {
            "answer": answer_provider.generate_answer(
                state["question"],
                state["results"],
            )
        }

    def fallback_answer(state: RagGraphState) -> dict:
        return {
            "answer": LocalAnswerProvider().generate_answer(
                state["question"],
                state["results"],
            )
        }

    def persist_conversation(state: RagGraphState) -> dict:
        conversation = Conversation(title=state["question"][:255])
        db.add(conversation)
        db.flush()

        db.add_all(
            [
                Message(
                    conversation_id=conversation.id,
                    role=MessageRole.USER,
                    content=state["question"],
                ),
                Message(
                    conversation_id=conversation.id,
                    role=MessageRole.ASSISTANT,
                    content=state["answer"],
                    model=answer_provider.model_name,
                ),
            ]
        )
        db.commit()
        db.refresh(conversation)

        return {"conversation": conversation}

    graph = StateGraph(RagGraphState)
    graph.add_node(
        "retrieve_context",
        trace_rag_node("retrieve_context", run_type="retriever")(retrieve_context),
    )
    graph.add_node(
        "generate_answer",
        trace_rag_node("generate_answer")(generate_answer),
    )
    graph.add_node(
        "fallback_answer",
        trace_rag_node("fallback_answer")(fallback_answer),
    )
    graph.add_node(
        "persist_conversation",
        trace_rag_node("persist_conversation", run_type="tool")(persist_conversation),
    )

    graph.add_edge(START, "retrieve_context")
    graph.add_conditional_edges(
        "retrieve_context",
        route_after_retrieval,
        {
            "generate_answer": "generate_answer",
            "fallback_answer": "fallback_answer",
        },
    )
    graph.add_edge("generate_answer", "persist_conversation")
    graph.add_edge("fallback_answer", "persist_conversation")
    graph.add_edge("persist_conversation", END)

    return graph.compile()
