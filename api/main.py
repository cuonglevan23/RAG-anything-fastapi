import os
import shutil
from pathlib import Path
from typing import Optional, List, Dict, Any
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from pydantic import BaseModel
from contextlib import asynccontextmanager
from loguru import logger

from raganything import RAGAnything, RAGAnythingConfig
from lightrag.llm.openai import openai_complete_if_cache, openai_embed
from lightrag.utils import EmbeddingFunc

# --- Configuration & Singleton ---
UPLOADS_DIR = Path("uploads")
UPLOADS_DIR.mkdir(exist_ok=True)
RAG_STORAGE_DIR = "rag_storage"

class RAGState:
    rag: Optional[RAGAnything] = None

rag_state = RAGState()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize RAGAnything on startup
    logger.info("Initializing RAGAnything singleton...")
    
    # Check for API Key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        logger.warning("OPENAI_API_KEY not found in environment. RAG functions may fail.")
    
    config = RAGAnythingConfig(
        working_dir=RAG_STORAGE_DIR,
    )

    def llm_model_func(prompt, system_prompt=None, history_messages=[], **kwargs):
        return openai_complete_if_cache(
            "gpt-4o-mini", prompt, system_prompt=system_prompt,
            history_messages=history_messages, api_key=api_key, **kwargs,
        )

    embedding_func = EmbeddingFunc(
        embedding_dim=1536, max_token_size=8192,
        func=lambda texts: openai_embed(texts, model="text-embedding-3-small", api_key=api_key),
    )

    rag_state.rag = RAGAnything(
        config=config,
        llm_model_func=llm_model_func,
        embedding_func=embedding_func,
    )
    
    # Ensure LightRAG is ready
    await rag_state.rag._ensure_lightrag_initialized()
    logger.info("RAGAnything initialized successfully.")
    
    yield
    
    # Cleanup on shutdown
    if rag_state.rag:
        await rag_state.rag.finalize_storages()
        logger.info("RAGAnything storages finalized.")

app = FastAPI(title="RAG Anything API", lifespan=lifespan)

# --- Models ---
class RagRequest(BaseModel):
    query: str
    mode: str = "hybrid"  # local, global, hybrid, naive, mix
    context_id: Optional[str] = None

class QueryResponse(BaseModel):
    query: str
    response: str
    context_id: Optional[str] = None

# --- Endpoints ---
@app.get("/")
async def root():
    status = "online" if rag_state.rag else "initializing"
    return {"message": "Welcome to RAG Anything API", "status": status}

@app.post("/upload-document")
async def upload_document(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    """
    Endpoint to upload and process a document for RAG.
    """
    if not rag_state.rag:
        raise HTTPException(status_code=503, detail="RAG system is not initialized")
    
    file_path = UPLOADS_DIR / file.filename
    try:
        with file_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        logger.info(f"File saved to {file_path}. Starting processing...")
        
        # Start processing in background to avoid blocking the request
        background_tasks.add_task(
            rag_state.rag.process_document_complete, 
            file_path=str(file_path)
        )
        
        return {
            "filename": file.filename,
            "status": "processing",
            "message": "File uploaded successfully. Indexing started in background."
        }
    except Exception as e:
        logger.error(f"Error handling file upload: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/rag-anything", response_model=QueryResponse)
async def rag_anything(request: RagRequest):
    """
    Endpoint to run RAG anything query.
    """
    if not rag_state.rag:
        raise HTTPException(status_code=503, detail="RAG system is not initialized")
    
    try:
        logger.info(f"Processing RAG query: {request.query} (mode: {request.mode})")
        
        # Execute query
        response = await rag_state.rag.aquery(
            query=request.query,
            mode=request.mode
        )
        
        return QueryResponse(
            query=request.query,
            response=response,
            context_id=request.context_id
        )
    except Exception as e:
        logger.error(f"Error executing RAG query: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
