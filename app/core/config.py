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
    LLM_MODEL: str = "gpt-4o"



    # ============================================================
    # Local Reranker Configuration (BGE — free, runs on GPU)
    # Đặt RERANK_ENABLE=true trong .env để bật re-ranking
    # Model mặc định: BAAI/bge-reranker-v2-m3 (~560MB VRAM, đa ngôn ngữ)
    # Lựa chọn nhẹ hơn: BAAI/bge-reranker-base (~278MB VRAM)
    # ============================================================
    RERANK_ENABLE: bool = os.getenv("RERANK_ENABLE", "false").lower() == "true"
    RERANK_MODEL: str = os.getenv("RERANK_MODEL", "BAAI/bge-reranker-v2-m3")
    # ============================================================

    # Security
    ALLOWED_EXTENSIONS: set = {".pdf", ".txt", ".docx", ".png", ".jpg", ".jpeg", ".md"}
    MAX_FILE_SIZE: int = 10 * 1024 * 1024  # 10MB



    class Config:
        env_file = ".env"
        extra = "ignore"

settings = Settings()

# Ensure directories exist
settings.UPLOADS_DIR.mkdir(exist_ok=True)
