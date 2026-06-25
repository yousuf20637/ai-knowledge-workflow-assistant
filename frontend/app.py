import os
from typing import Any

import requests
import streamlit as st


API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8001")
DEMO_FILENAME = "demo-ai-knowledge-workflow.md"
DEMO_DOCUMENT = """# AI Knowledge Workflow Assistant Demo

This project is a retrieval-augmented generation assistant for private knowledge bases.
It uses FastAPI for the backend API, PostgreSQL for persistent documents and
conversation history, Chroma for vector search, LangGraph for the multi-step RAG
workflow, and Streamlit for the frontend.

The main workflow is:

1. A user uploads a text or Markdown document.
2. The backend chunks the document and stores metadata in PostgreSQL.
3. The vector store indexes each chunk for semantic retrieval.
4. A user asks a question from the Streamlit interface.
5. LangGraph retrieves relevant chunks, decides whether enough context exists,
   generates a grounded answer, and saves the conversation.
6. The UI shows the answer, citations, indexed documents, and saved history.

The project is designed to stay inexpensive. By default it uses local deterministic
embeddings and a local answer formatter. OpenAI can be enabled with environment
variables when higher-quality answer generation is needed.

Resume highlights include API design, Dockerized services, PostgreSQL schema design,
vector search, RAG orchestration, automated tests, and a usable frontend demo.
"""
EXAMPLE_QUESTIONS = [
    "What technologies does this project use?",
    "How does the RAG workflow operate step by step?",
    "How does this project keep costs low?",
]


def api_url(path: str) -> str:
    return f"{API_BASE_URL}{path}"


def get_documents() -> list[dict[str, Any]]:
    response = requests.get(api_url("/documents"), timeout=10)
    response.raise_for_status()
    return response.json()


def get_conversations() -> list[dict[str, Any]]:
    response = requests.get(api_url("/conversations"), timeout=10)
    response.raise_for_status()
    return response.json()


def get_conversation(conversation_id: str) -> dict[str, Any]:
    response = requests.get(api_url(f"/conversations/{conversation_id}"), timeout=10)
    response.raise_for_status()
    return response.json()


def upload_document(uploaded_file) -> dict[str, Any]:
    files = {
        "file": (
            uploaded_file.name,
            uploaded_file.getvalue(),
            uploaded_file.type or "text/plain",
        )
    }
    response = requests.post(api_url("/documents"), files=files, timeout=30)
    response.raise_for_status()
    return response.json()


def upload_demo_document() -> dict[str, Any]:
    files = {
        "file": (
            DEMO_FILENAME,
            DEMO_DOCUMENT.encode("utf-8"),
            "text/markdown",
        )
    }
    response = requests.post(api_url("/documents"), files=files, timeout=30)
    response.raise_for_status()
    return response.json()


def ask_question(question: str, limit: int) -> dict[str, Any]:
    response = requests.post(
        api_url("/query"),
        json={"question": question, "limit": limit},
        timeout=60,
    )
    response.raise_for_status()
    return response.json()


def format_conversation_label(conversation: dict[str, Any]) -> str:
    title = conversation["title"] or "Untitled conversation"
    return f"{title[:70]} ({conversation['message_count']} messages)"


def render_answer(answer: dict[str, Any]) -> None:
    st.markdown("#### Answer")
    st.markdown(answer["answer"])

    st.markdown("#### Citations")
    if answer["citations"]:
        for citation in answer["citations"]:
            distance = citation.get("distance")
            distance_text = "n/a" if distance is None else f"{distance:.3f}"
            st.caption(
                f"{citation['filename']}#chunk-{citation['chunk_index']} "
                f"(distance: {distance_text})"
            )
    else:
        st.caption("No citations returned.")


def render_conversation(conversation: dict[str, Any]) -> None:
    st.markdown("#### Saved Conversation")
    for message in conversation["messages"]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])


st.set_page_config(
    page_title="AI Knowledge Workflow Assistant",
    layout="wide",
)

st.title("AI Knowledge Workflow Assistant")

selected_conversation_id = st.session_state.get("selected_conversation_id")

with st.sidebar:
    st.subheader("Demo")
    if st.button("Load sample knowledge base", type="primary"):
        try:
            result = upload_demo_document()
            st.success(f"Indexed {result['filename']} into {result['chunk_count']} chunk(s).")
        except requests.HTTPError as exc:
            detail = exc.response.text if exc.response is not None else str(exc)
            st.error(f"Demo upload failed: {detail}")
        except requests.RequestException as exc:
            st.error(f"API request failed: {exc}")

    st.divider()
    st.subheader("Upload")
    uploaded_file = st.file_uploader(
        "Text or Markdown",
        type=["txt", "md"],
        label_visibility="collapsed",
    )

    if st.button("Upload document", type="primary", disabled=uploaded_file is None):
        try:
            result = upload_document(uploaded_file)
            st.success(f"Indexed {result['filename']} into {result['chunk_count']} chunk(s).")
        except requests.HTTPError as exc:
            detail = exc.response.text if exc.response is not None else str(exc)
            st.error(f"Upload failed: {detail}")
        except requests.RequestException as exc:
            st.error(f"API request failed: {exc}")

    st.divider()
    st.subheader("History")
    try:
        conversations = get_conversations()
    except requests.RequestException as exc:
        conversations = []
        st.error(f"Could not load history: {exc}")

    conversation_options = [conversation["id"] for conversation in conversations]
    conversation_labels = {
        conversation["id"]: format_conversation_label(conversation)
        for conversation in conversations
    }

    if conversation_options:
        if selected_conversation_id not in conversation_options:
            selected_conversation_id = conversation_options[0]

        selected_conversation_id = st.selectbox(
            "Recent conversations",
            conversation_options,
            index=conversation_options.index(selected_conversation_id),
            format_func=lambda conversation_id: conversation_labels[conversation_id],
            label_visibility="collapsed",
        )
        st.session_state["selected_conversation_id"] = selected_conversation_id
    else:
        st.caption("Ask a question to start a conversation.")

    st.divider()
    st.caption(f"API: {API_BASE_URL}")

left, right = st.columns([1, 2], gap="large")

with left:
    st.subheader("Documents")
    try:
        documents = get_documents()
    except requests.RequestException as exc:
        documents = []
        st.error(f"Could not load documents: {exc}")

    if documents:
        for document in documents:
            st.markdown(f"**{document['filename']}**")
            st.caption(f"{document['chunk_count']} chunk(s) · {document['source']}")
    else:
        st.info("No documents indexed yet.")

with right:
    st.subheader("Ask")
    question_value = st.session_state.pop("question_template", "")

    example_columns = st.columns(len(EXAMPLE_QUESTIONS))
    for column, example_question in zip(example_columns, EXAMPLE_QUESTIONS, strict=True):
        if column.button(example_question):
            st.session_state["question_template"] = example_question
            st.rerun()

    with st.form("query-form"):
        question = st.text_area(
            "Question",
            value=question_value,
            placeholder="What does this project use for database migrations?",
            height=110,
        )
        limit = st.slider("Retrieved chunks", min_value=1, max_value=10, value=4)
        submitted = st.form_submit_button("Ask question", type="primary")

    if submitted:
        if not question.strip():
            st.warning("Enter a question first.")
        else:
            try:
                answer = ask_question(question.strip(), limit)
                st.session_state["selected_conversation_id"] = answer["conversation_id"]
                render_answer(answer)
            except requests.HTTPError as exc:
                detail = exc.response.text if exc.response is not None else str(exc)
                st.error(f"Query failed: {detail}")
            except requests.RequestException as exc:
                st.error(f"API request failed: {exc}")

    if st.session_state.get("selected_conversation_id"):
        try:
            conversation = get_conversation(st.session_state["selected_conversation_id"])
            render_conversation(conversation)
        except requests.RequestException as exc:
            st.error(f"Could not load conversation: {exc}")
