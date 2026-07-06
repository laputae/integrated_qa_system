"""Pydantic request/response models for the API."""
from typing import Optional, Dict, List, Any
from pydantic import BaseModel


class QueryRequest(BaseModel):
    query: str
    source_filter: Optional[str] = None
    session_id: Optional[str] = None


class QueryResponse(BaseModel):
    answer: str
    is_streaming: bool
    session_id: str
    processing_time: float


class RegisterRequest(BaseModel):
    username: str
    password: str
    tenant_name: str = "default"


class LoginRequest(BaseModel):
    username: str
    password: str
    tenant_name: str = "default"


class RefreshRequest(BaseModel):
    refresh_token: str


class EvalRunRequest(BaseModel):
    dataset: Optional[list] = None
    triggered_by: str = "manual"


class ChunkConfigResponse(BaseModel):
    default_strategy: str
    doc_type_strategies: Dict[str, str]
    semantic_model_path: str
    semantic_device: str
    semantic_fallback_strategy: str
    parent_chunk_size: int
    child_chunk_size: int
    chunk_overlap: int


class ChunkConfigUpdate(BaseModel):
    default_strategy: Optional[str] = None
    doc_type_strategies: Optional[Dict[str, str]] = None
    semantic_model_path: Optional[str] = None
    semantic_device: Optional[str] = None
    semantic_fallback_strategy: Optional[str] = None
    parent_chunk_size: Optional[int] = None
    child_chunk_size: Optional[int] = None
    chunk_overlap: Optional[int] = None


class DeleteHistoryRequest(BaseModel):
    session_ids: list[str]
