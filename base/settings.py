from __future__ import annotations

import json
import os
from typing import Optional

from pydantic import Field, ValidationInfo, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class AppSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # ---- MySQL ----
    mysql_host: str = Field(default="127.0.0.1", validation_alias="MYSQL_HOST")
    mysql_user: str = Field(default="root", validation_alias="MYSQL_USER")
    mysql_password: str = Field(default="", validation_alias="MYSQL_PASSWORD")
    mysql_database: str = Field(default="subjects_kg", validation_alias="MYSQL_DATABASE")

    # ---- Redis ----
    redis_host: str = Field(default="127.0.0.1", validation_alias="REDIS_HOST")
    redis_port: int = Field(default=6379, validation_alias="REDIS_PORT")
    redis_password: str = Field(default="", validation_alias="REDIS_PASSWORD")
    redis_db: int = Field(default=0, validation_alias="REDIS_DB")

    # ---- Milvus ----
    milvus_host: str = Field(default="127.0.0.1", validation_alias="MILVUS_HOST")
    milvus_port: str = Field(default="19530", validation_alias="MILVUS_PORT")
    milvus_database_name: str = Field(default="itcast", validation_alias="MILVUS_DATABASE_NAME")
    milvus_collection_name: str = Field(default="edurag_final", validation_alias="MILVUS_COLLECTION_NAME")
    milvus_timeout: int = Field(default=10, validation_alias="MILVUS_TIMEOUT")

    # ---- LLM ----
    llm_model: str = Field(default="deepseek-v4-pro", validation_alias="DEEPSEEK_MODEL")
    llm_api_key: str = Field(default="", validation_alias="DEEPSEEK_API_KEY")
    llm_base_url: str = Field(default="https://api.deepseek.com", validation_alias="DEEPSEEK_BASE_URL")

    # ---- Auth ----
    jwt_secret_key: str = Field(default="", validation_alias="JWT_SECRET_KEY")
    access_token_expire_minutes: int = Field(default=30, validation_alias="ACCESS_TOKEN_EXPIRE_MINUTES")
    refresh_token_expire_days: int = Field(default=7, validation_alias="REFRESH_TOKEN_EXPIRE_DAYS")
    bcrypt_cost_factor: int = Field(default=12, validation_alias="BCRYPT_COST_FACTOR")

    # ---- CORS ----
    cors_origins: str = Field(
        default="http://localhost:3000,http://127.0.0.1:8000",
        validation_alias="CORS_ORIGINS",
    )

    # ---- Metrics Auth ----
    metrics_auth_user: str = Field(default="", validation_alias="METRICS_AUTH_USER")
    metrics_auth_password: str = Field(default="", validation_alias="METRICS_AUTH_PASSWORD")

    # ---- Security Headers ----
    secure_headers_enabled: bool = Field(default=True, validation_alias="SECURE_HEADERS_ENABLED")

    # ---- Retrieval ----
    retrieval_k: int = Field(default=5, validation_alias="RETRIEVAL_K")
    reranker_score_threshold: float = Field(default=0.3, validation_alias="RERANKER_SCORE_THRESHOLD")

    # ---- Concurrency ----
    max_concurrent_llm_calls: int = Field(default=10, validation_alias="MAX_CONCURRENT_LLM_CALLS")
    thread_pool_workers: int = Field(default=20, validation_alias="THREAD_POOL_WORKERS")

    # ---- Logging ----
    log_level: str = Field(default="INFO", validation_alias="LOG_LEVEL")
    log_file: str = Field(default="logs/app.log", validation_alias="LOG_FILE")
    log_format: str = Field(default="json", validation_alias="LOG_FORMAT")

    # ---- Eval ----
    eval_llm_model: Optional[str] = Field(default=None, validation_alias="EVAL_LLM_MODEL")
    eval_llm_base_url: Optional[str] = Field(default=None, validation_alias="EVAL_LLM_BASE_URL")
    eval_embedding_model: str = Field(default="mxbai-embed-large", validation_alias="EVAL_EMBEDDING_MODEL")
    eval_embedding_base_url: str = Field(default="http://localhost:11434", validation_alias="EVAL_EMBEDDING_BASE_URL")

    # ---- Health Check ----
    health_check_timeout: float = Field(default=5.0, validation_alias="HEALTH_CHECK_TIMEOUT")

    # ---- Customer Service ----
    customer_service_phone: str = Field(default="", validation_alias="CUSTOMER_SERVICE_PHONE")

    # ---- Valid Sources (JSON array string) ----
    valid_sources: str = Field(
        default='["ai", "java", "test", "ops", "bigdata"]',
        validation_alias="VALID_SOURCES",
    )

    # ---- Sensitive Config Warnings ----
    security_mode: str = Field(default="dev", validation_alias="SECURITY_MODE")

    @field_validator("jwt_secret_key")
    @classmethod
    def jwt_secret_must_not_be_empty_in_prod(cls, v: str) -> str:
        security_mode = os.environ.get("SECURITY_MODE", "dev")
        if security_mode == "prod" and not v:
            raise ValueError(
                "JWT_SECRET_KEY must be set via environment variable in production mode "
                "(SECURITY_MODE=prod). Generate with: openssl rand -hex 64"
            )
        return v

    @field_validator("mysql_password", "redis_password")
    @classmethod
    def password_should_be_set_in_prod(cls, v: str, info: ValidationInfo) -> str:
        security_mode = os.environ.get("SECURITY_MODE", "dev")
        if security_mode == "prod" and not v:
            raise ValueError(
                f"{info.field_name} must be set via environment variable in production mode."
            )
        return v

    @field_validator("valid_sources")
    @classmethod
    def valid_sources_must_be_json_array(cls, v: str) -> str:
        try:
            parsed = json.loads(v)
            if not isinstance(parsed, list):
                raise ValueError("VALID_SOURCES must be a JSON array")
        except json.JSONDecodeError:
            raise ValueError(f"VALID_SOURCES is not valid JSON: {v}")
        return v


def validate_config() -> AppSettings:
    """Validate configuration at startup. Raises ValidationError on failure."""
    settings = AppSettings()
    if settings.jwt_secret_key:
        return settings
    if os.environ.get("SECURITY_MODE", "dev") != "dev":
        raise ValueError("JWT_SECRET_KEY is required and not set via environment variable.")
    return settings
