from functools import lru_cache
import os
from typing import Any, Callable

from langsmith import Client, traceable

from app.core.config import settings


def langsmith_tracing_enabled() -> bool:
    if os.getenv("PYTEST_CURRENT_TEST"):
        return False

    return settings.langsmith_tracing and bool(settings.langsmith_api_key)


@lru_cache
def _langsmith_client(api_key: str) -> Client:
    return Client(api_key=api_key)


def _summarize_state(state: dict[str, Any]) -> dict[str, Any]:
    results = state.get("results") or []
    answer = state.get("answer") or ""
    return {
        "question": state.get("question"),
        "limit": state.get("limit"),
        "result_count": len(results),
        "has_answer": bool(answer),
        "has_conversation": state.get("conversation") is not None,
    }


def _summarize_inputs(inputs: dict[str, Any]) -> dict[str, Any]:
    state = inputs.get("state")
    if isinstance(state, dict):
        return {"state": _summarize_state(state)}
    return inputs


def _summarize_outputs(outputs: Any) -> Any:
    if not isinstance(outputs, dict):
        return outputs

    summary: dict[str, Any] = {"updated_fields": sorted(outputs)}
    if "results" in outputs:
        summary["result_count"] = len(outputs["results"] or [])
    if "answer" in outputs:
        answer = outputs["answer"] or ""
        summary["answer_preview"] = answer[:240]
    if "conversation" in outputs:
        conversation = outputs["conversation"]
        summary["conversation_id"] = str(getattr(conversation, "id", ""))
    return summary


def trace_rag_node(
    name: str,
    run_type: str = "chain",
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    enabled = langsmith_tracing_enabled()
    client = _langsmith_client(settings.langsmith_api_key) if enabled else None

    return traceable(
        name=name,
        run_type=run_type,
        project_name=settings.langsmith_project,
        metadata={"environment": settings.environment},
        tags=["rag", "langgraph"],
        client=client,
        enabled=enabled,
        process_inputs=_summarize_inputs,
        process_outputs=_summarize_outputs,
    )
