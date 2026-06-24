# AI Knowledge Workflow Assistant

A resume-oriented AI application that will combine FastAPI, LangGraph, LangChain,
PostgreSQL, Chroma, Docker, and the OpenAI API to answer questions over uploaded
documents with citations and persistent conversation history.

## Current Status

- FastAPI backend scaffolded
- Health check endpoint available at `/health`
- Local Python virtual environment configured
- Dockerfile added for the API service
- Docker Compose added for API + PostgreSQL

## Run Locally

```bash
source .venv/bin/activate
uvicorn app.main:app --reload
```

Then open:

```text
http://127.0.0.1:8000/health
```

Expected response:

```json
{"status":"ok"}
```

## Run With Docker

Create a local `.env` file from the example values:

```bash
cp .env.example .env
```

Start the API and PostgreSQL:

```bash
docker compose up --build
```

Then open:

```text
http://127.0.0.1:8001/health
```

Stop the stack:

```bash
docker compose down
```

## Development Checks

Run tests:

```bash
pytest
```

Apply database migrations:

```bash
alembic upgrade head
```

Upload a text or Markdown document:

```bash
curl -X POST http://127.0.0.1:8001/documents \
  -F "file=@README.md;type=text/markdown"
```

List uploaded documents:

```bash
curl http://127.0.0.1:8001/documents
```

Search indexed document chunks:

```bash
curl -X POST http://127.0.0.1:8001/documents/search \
  -H "Content-Type: application/json" \
  -d '{"query":"database migrations", "limit": 3}'
```

Ask a retrieval-grounded question:

```bash
curl -X POST http://127.0.0.1:8001/query \
  -H "Content-Type: application/json" \
  -d '{"question":"What does this project use for database migrations?", "limit": 3}'
```

By default, answers use a free local formatter. To use OpenAI for answer
generation, set these values in `.env`:

```bash
ANSWER_PROVIDER="openai"
OPENAI_API_KEY="your-api-key"
OPENAI_ANSWER_MODEL="gpt-5.5-mini"
```
