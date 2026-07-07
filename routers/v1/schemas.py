"""Pydantic request/response models for the API."""
from pydantic import BaseModel


class QueryRequest(BaseModel):
    query: str
    source_filter: str | None = None
    session_id: str | None = None


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
    dataset: list | None = None
    triggered_by: str = "manual"


class ChunkConfigResponse(BaseModel):
    default_strategy: str
    doc_type_strategies: dict[str, str]
    semantic_model_path: str
    semantic_device: str
    semantic_fallback_strategy: str
    parent_chunk_size: int
    child_chunk_size: int
    chunk_overlap: int


class ChunkConfigUpdate(BaseModel):
    default_strategy: str | None = None
    doc_type_strategies: dict[str, str] | None = None
    semantic_model_path: str | None = None
    semantic_device: str | None = None
    semantic_fallback_strategy: str | None = None
    parent_chunk_size: int | None = None
    child_chunk_size: int | None = None
    chunk_overlap: int | None = None


class DeleteHistoryRequest(BaseModel):
    session_ids: list[str]
