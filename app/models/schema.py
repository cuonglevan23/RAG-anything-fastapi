from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
from uuid import UUID

class QueryRequest(BaseModel):
    query: str
    project_id: str
    mode: str = "hybrid"  # local, global, hybrid, naive, mix
    context_id: Optional[str] = None

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
