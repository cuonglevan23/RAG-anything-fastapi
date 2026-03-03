from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
from uuid import UUID

class RagQueryRequest(BaseModel):
    query: str
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
    error: Optional[str] = None

class UploadResponse(BaseModel):
    task_id: str
    filename: str
    status: str
    message: str
