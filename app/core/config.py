from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "AI Knowledge Workflow Assistant"
    environment: str = "development"
    database_url: str = "postgresql+psycopg://postgres:postgres@localhost:5432/ai_knowledge"
    chroma_path: str = ".chroma"
    chroma_collection_name: str = "document_chunks"
    answer_provider: str = "local"
    openai_api_key: str = ""
    openai_answer_model: str = "gpt-5.5-mini"
    langsmith_tracing: bool = False
    langsmith_api_key: str = ""
    langsmith_project: str = "ai-knowledge-workflow-assistant"

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")


settings = Settings()
