import shutil
from pathlib import Path
from fastapi import APIRouter, UploadFile, File, Form, BackgroundTasks, HTTPException
from typing import List
from app.models.schema import (
    QueryRequest, QueryResponse, UploadResponse, ProcessingStatus,
    ProjectListResponse, EvalQueryRequest, EvalQueryResponse, BatchUploadResponse
)
from app.services.rag_service import rag_service
from app.core.config import settings
from loguru import logger

router = APIRouter(prefix="/rag", tags=["RAG"])

@router.post("/upload", response_model=UploadResponse)
async def upload_document(
    background_tasks: BackgroundTasks, 
    file: UploadFile = File(...),
    project_id: str = Form(...)
):
    """Upload and process a document into a specific workspace"""
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
        
        task_id = rag_service.create_task(file.filename, project_id)
        
        background_tasks.add_task(
            rag_service.process_document,
            task_id=task_id,
            file_path=str(file_path),
            filename=file.filename,
            project_id=project_id
        )
        
        return UploadResponse(
            task_id=task_id,
            filename=file.filename,
            project_id=project_id,
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
async def query_rag(request: QueryRequest):
    """Query the RAG system within a specific workspace"""
    try:
        response = await rag_service.query(
            project_id=request.project_id,
            query=request.query,
            mode=request.mode,
            top_k=request.top_k,
            response_type=request.response_type,
        )
        return QueryResponse(
            query=request.query,
            response=response,
            context_id=request.context_id
        )
    except Exception as e:
        logger.error(f"Query failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/projects", response_model=ProjectListResponse)
async def list_projects():
    """List all available workspaces/projects"""
    projects = await rag_service.list_projects()
    return ProjectListResponse(projects=projects)

@router.post("/query_eval", response_model=EvalQueryResponse)
async def query_eval(request: EvalQueryRequest):
    """
    Query the RAG system and return BOTH the synthesized answer AND raw retrieved contexts.
    Used by the RAGAS Evaluation tab in Streamlit.
    - answer   = hybrid/mix mode response (best quality)
    - contexts = naive mode response split into paragraphs (raw retrieved chunks)
    """
    try:
        result = await rag_service.query_with_context(
            project_id=request.project_id,
            query=request.query,
            mode=request.mode,
            top_k=request.top_k,
            response_type=request.response_type,
        )
        return EvalQueryResponse(
            query=result["query"],
            answer=result["answer"],
            contexts=result["contexts"],
        )
    except Exception as e:
        logger.error(f"Eval query failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/upload_batch", response_model=BatchUploadResponse)
async def upload_batch(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(...),
    project_id: str = Form(...),
):
    """
    Upload nhiều file cùng lúc (từ 1 folder) vào 1 workspace.
    Mỗi file được xử lý độc lập theo background task riêng.
    Trả về danh sách task_id để poll tiến trình từng file.
    """
    results = []

    for file in files:
        extension = Path(file.filename).suffix.lower()

        # Validate extension per file — skip unsupported, report lỗi
        if extension not in settings.ALLOWED_EXTENSIONS:
            logger.warning(f"Batch upload: skipping unsupported file '{file.filename}'")
            results.append({
                "filename": file.filename,
                "task_id": None,
                "status": "skipped",
                "message": f"Unsupported file type: {extension}",
            })
            continue

        # Validate size per file
        contents = await file.read()
        if len(contents) > settings.MAX_FILE_SIZE:
            results.append({
                "filename": file.filename,
                "task_id": None,
                "status": "skipped",
                "message": f"File too large: {len(contents) / 1024 / 1024:.1f}MB > {settings.MAX_FILE_SIZE / 1024 / 1024}MB limit",
            })
            continue

        # Save to disk
        file_path = settings.UPLOADS_DIR / file.filename
        file_path.write_bytes(contents)

        # Create task and fire background processing
        task_id = rag_service.create_task(file.filename, project_id)
        background_tasks.add_task(
            rag_service.process_document,
            task_id=task_id,
            file_path=str(file_path),
            filename=file.filename,
            project_id=project_id,
        )

        results.append({
            "filename": file.filename,
            "task_id": task_id,
            "status": "pending",
            "message": "Queued for processing.",
        })
        logger.info(f"Batch upload: '{file.filename}' queued → task {task_id}")

    accepted = sum(1 for r in results if r["status"] == "pending")
    logger.info(f"Batch upload: {accepted}/{len(files)} files accepted for workspace '{project_id}'")

    return BatchUploadResponse(
        project_id=project_id,
        total=len(files),
        results=results,
    )
