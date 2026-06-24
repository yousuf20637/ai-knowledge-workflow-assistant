from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "AI Knowledge Workflow Assistant"
    environment: str = "development"
    database_url: str = "postgresql://postgres:postgres@localhost:5432/ai_knowledge"

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")


settings = Settings()
