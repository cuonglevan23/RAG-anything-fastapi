from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
from uuid import UUID

class QueryRequest(BaseModel):
    query: str
    project_id: str
    mode: str = "hybrid"  # local, global, hybrid, naive, mix
    top_k: int = 100
    response_type: str = "Structured List"  # e.g. "Structured List", "Single Paragraph", "Multiple Paragraphs"
    context_id: Optional[str] = None

class EvalQueryRequest(BaseModel):
    """Request for evaluation query - returns answer + retrieved contexts for RAGAS"""
    query: str
    project_id: str
    mode: str = "hybrid"
    top_k: int = 100
    response_type: str = "Structured List"

class EvalQueryResponse(BaseModel):
    """Response for evaluation - includes both answer and raw retrieved contexts"""
    query: str
    answer: str
    contexts: List[str]  # Raw retrieved text passages (for RAGAS context metrics)

class QueryResponse(BaseModel):
    query: str
    response: str
    context_id: Optional[str] = None

class ProcessingStatus(BaseModel):
    task_id: str
    filename: str
    status: str  # pending, parsing, indexing, completed, failed
    message: str
    percentage: float = 0.0
    logs: List[str] = Field(default_factory=list)
    project_id: str
    error: Optional[str] = None

class UploadResponse(BaseModel):
    task_id: str
    filename: str
    project_id: str
    status: str
    message: str

class ProjectListResponse(BaseModel):
    projects: List[str]

class BatchUploadResponse(BaseModel):
    """Response for batch upload — one entry per file"""
    project_id: str
    total: int
    results: List[Dict[str, Any]]   # [{filename, task_id, status, message}]
