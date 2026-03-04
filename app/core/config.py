import os
from pathlib import Path
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    PROJECT_NAME: str = "RAG Anything API"
    API_V1_STR: str = "/api/v1"
    
    # Storage
    BASE_DIR: Path = Path(__file__).resolve().parent.parent.parent
    UPLOADS_DIR: Path = BASE_DIR / "uploads"
    BASE_RAG_DIR: Path = BASE_DIR / "rag_storage"
    
    # OpenAI
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    EMBEDDING_MODEL: str = "text-embedding-3-small"
    LLM_MODEL: str = "gpt-4o-mini"

    # Security
    ALLOWED_EXTENSIONS: set = {".pdf", ".txt", ".docx", ".png", ".jpg", ".jpeg", ".md"}
    MAX_FILE_SIZE: int = 10 * 1024 * 1024  # 10MB

    class Config:
        env_file = ".env"
        extra = "ignore"

settings = Settings()

# Ensure directories exist
settings.UPLOADS_DIR.mkdir(exist_ok=True)
