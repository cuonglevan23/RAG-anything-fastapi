from fastapi import FastAPI
from contextlib import asynccontextmanager
from app.core.config import settings
from app.api.routes import rag
from app.services.rag_service import rag_service
from loguru import logger

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info(f"Starting {settings.PROJECT_NAME}...")
    yield
    # Shutdown
    await rag_service.finalize()

app = FastAPI(
    title=settings.PROJECT_NAME,
    lifespan=lifespan
)

# Include Routers
app.include_router(rag.router, prefix=settings.API_V1_STR)

@app.get("/")
async def root():
    return {
        "message": f"Welcome to {settings.PROJECT_NAME}",
        "version": "1.0.0",
        "api_v1_base": settings.API_V1_STR
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
