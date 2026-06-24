from fastapi import FastAPI

from app.api.documents import router as documents_router

app = FastAPI(
    title="AI Knowledge Workflow Assistant",
    version="0.1.0",
)

app.include_router(documents_router)


@app.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "ok"}
