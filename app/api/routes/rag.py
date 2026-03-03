import shutil
from pathlib import Path
from fastapi import APIRouter, UploadFile, File, BackgroundTasks, HTTPException
from app.models.schema import RagQueryRequest, QueryResponse, UploadResponse, ProcessingStatus
from app.services.rag_service import rag_service
from app.core.config import settings
from loguru import logger

router = APIRouter(prefix="/rag", tags=["RAG"])

@router.post("/upload", response_model=UploadResponse)
async def upload_document(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    """Upload and process a document"""
    # 1. Validate Extension
    extension = Path(file.filename).suffix.lower()
    if extension not in settings.ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400, 
            detail=f"Unsupported file type: {extension}. Allowed: {settings.ALLOWED_EXTENSIONS}"
        )
    
    # 2. Validate Size
    if file.size > settings.MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413, 
            detail=f"File too large. Max allowed: {settings.MAX_FILE_SIZE / (1024*1024)}MB"
        )

    file_path = settings.UPLOADS_DIR / file.filename
    try:
        with file_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        task_id = rag_service.create_task(file.filename)
        
        background_tasks.add_task(
            rag_service.process_document,
            task_id=task_id,
            file_path=str(file_path),
            filename=file.filename
        )
        
        return UploadResponse(
            task_id=task_id,
            filename=file.filename,
            status="pending",
            message="File uploaded. Processing started in background."
        )
    except Exception as e:
        logger.error(f"Upload failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/status/{task_id}", response_model=ProcessingStatus)
async def get_status(task_id: str):
    """Check task status"""
    status = rag_service.get_task_status(task_id)
    if not status:
        raise HTTPException(status_code=404, detail="Task not found")
    return status

@router.post("/query", response_model=QueryResponse)
async def query_rag(request: RagQueryRequest):
    """Query the RAG system"""
    try:
        response = await rag_service.query(request.query, mode=request.mode)
        return QueryResponse(
            query=request.query,
            response=response,
            context_id=request.context_id
        )
    except Exception as e:
        logger.error(f"Query failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
