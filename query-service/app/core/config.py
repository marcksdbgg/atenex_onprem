import logging
import os
import sys
from pathlib import Path 
from typing import Optional, List, Any, Dict
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import AnyHttpUrl, SecretStr, Field, field_validator, ValidationInfo, ValidationError

# --- Default Values ---
POSTGRES_K8S_HOST_DEFAULT = "postgresql-service.nyro-develop.svc.cluster.local"
POSTGRES_K8S_PORT_DEFAULT = 5432
POSTGRES_K8S_DB_DEFAULT = "atenex"
POSTGRES_K8S_USER_DEFAULT = "postgres"

ZILLIZ_ENDPOINT_DEFAULT = "https://in03-0afab716eb46d7f.serverless.gcp-us-west1.cloud.zilliz.com"
MILVUS_DEFAULT_COLLECTION = "atenex_collection"
MILVUS_DEFAULT_EMBEDDING_FIELD = "embedding"
MILVUS_DEFAULT_CONTENT_FIELD = "content"
MILVUS_DEFAULT_COMPANY_ID_FIELD = "company_id"
MILVUS_DEFAULT_DOCUMENT_ID_FIELD = "document_id"
MILVUS_DEFAULT_FILENAME_FIELD = "file_name"
MILVUS_DEFAULT_METADATA_FIELDS = ["company_id", "document_id", "file_name", "page", "title"]
MILVUS_DEFAULT_GRPC_TIMEOUT = 15
MILVUS_DEFAULT_SEARCH_PARAMS = {"metric_type": "IP", "params": {"nprobe": 10}}

EMBEDDING_SERVICE_K8S_URL_DEFAULT = "http://embedding-service.nyro-develop.svc.cluster.local:80" 
SPARSE_SEARCH_SERVICE_K8S_URL_DEFAULT = "http://sparse-search-service.nyro-develop.svc.cluster.local:80" 

# --- Prompts (Renamed to generic/granite) ---
PROMPT_DIR = Path(__file__).resolve().parent.parent / "prompts"
DEFAULT_RAG_PROMPT_TEMPLATE_PATH = str(PROMPT_DIR / "rag_template_granite.txt")
DEFAULT_GENERAL_PROMPT_TEMPLATE_PATH = str(PROMPT_DIR / "general_template_granite.txt")
DEFAULT_MAP_PROMPT_TEMPLATE_PATH = str(PROMPT_DIR / "map_prompt_template.txt")
DEFAULT_REDUCE_PROMPT_TEMPLATE_PATH = str(PROMPT_DIR / "reduce_prompt_template_v2.txt")

# Models defaults for Granite 3.2 2b
DEFAULT_EMBEDDING_DIMENSION = 1536
LLM_API_BASE_URL_DEFAULT = "http://192.168.1.43:9090"
LLM_MODEL_NAME_DEFAULT = "granite-3.2-2b-instruct-q4_k_m.gguf"
LLM_MAX_OUTPUT_TOKENS_DEFAULT = 2048 

# RAG Pipeline Parameters Optimized for Small Model
DEFAULT_RETRIEVER_TOP_K = 40 
DEFAULT_BM25_ENABLED = True
DEFAULT_DIVERSITY_FILTER_ENABLED = False 
DEFAULT_MAX_CONTEXT_CHUNKS = 10 
DEFAULT_HYBRID_ALPHA = 0.5
DEFAULT_DIVERSITY_LAMBDA = 0.5
DEFAULT_MAX_CHAT_HISTORY_MESSAGES = 6 
DEFAULT_NUM_SOURCES_TO_SHOW = 5
DEFAULT_MAX_TOKENS_PER_CHUNK = 800
DEFAULT_MAX_CHARS_PER_CHUNK = 3500
DEFAULT_MAPREDUCE_ENABLED = True
DEFAULT_MAPREDUCE_CHUNK_BATCH_SIZE = 3
DEFAULT_TIKTOKEN_ENCODING_NAME = "cl100k_base"

# Budgeting
DEFAULT_LLM_CONTEXT_WINDOW_TOKENS = 16000 
DEFAULT_DIRECT_RAG_TOKEN_LIMIT = 8000 
DEFAULT_HTTP_CLIENT_TIMEOUT = 30
DEFAULT_HTTP_CLIENT_MAX_RETRIES = 2
DEFAULT_HTTP_CLIENT_BACKOFF_FACTOR = 2.0

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="QUERY_", case_sensitive=True, extra="ignore")

    PROJECT_NAME: str = "Atenex Query Service"
    API_V1_STR: str = "/api/v1"
    LOG_LEVEL: str = "INFO"

    # --- Database ---
    POSTGRES_USER: str = Field(default=POSTGRES_K8S_USER_DEFAULT)
    POSTGRES_PASSWORD: SecretStr
    POSTGRES_SERVER: str = Field(default=POSTGRES_K8S_HOST_DEFAULT)
    POSTGRES_PORT: int = Field(default=POSTGRES_K8S_PORT_DEFAULT)
    POSTGRES_DB: str = Field(default=POSTGRES_K8S_DB_DEFAULT)

    # --- Vector Store ---
    ZILLIZ_API_KEY: SecretStr = Field(description="API Key for Zilliz Cloud connection.")
    MILVUS_URI: AnyHttpUrl = Field(default_factory=lambda: AnyHttpUrl(ZILLIZ_ENDPOINT_DEFAULT))
    MILVUS_COLLECTION_NAME: str = Field(default=MILVUS_DEFAULT_COLLECTION)
    MILVUS_EMBEDDING_FIELD: str = Field(default=MILVUS_DEFAULT_EMBEDDING_FIELD)
    MILVUS_CONTENT_FIELD: str = Field(default=MILVUS_DEFAULT_CONTENT_FIELD)
    MILVUS_COMPANY_ID_FIELD: str = Field(default=MILVUS_DEFAULT_COMPANY_ID_FIELD)
    MILVUS_DOCUMENT_ID_FIELD: str = Field(default=MILVUS_DEFAULT_DOCUMENT_ID_FIELD)
    MILVUS_FILENAME_FIELD: str = Field(default=MILVUS_DEFAULT_FILENAME_FIELD)
    MILVUS_METADATA_FIELDS: List[str] = Field(default=MILVUS_DEFAULT_METADATA_FIELDS)
    MILVUS_GRPC_TIMEOUT: int = Field(default=MILVUS_DEFAULT_GRPC_TIMEOUT)
    MILVUS_SEARCH_PARAMS: Dict[str, Any] = Field(default_factory=lambda: MILVUS_DEFAULT_SEARCH_PARAMS.copy())

    # --- Embedding ---
    EMBEDDING_DIMENSION: int = Field(default=DEFAULT_EMBEDDING_DIMENSION)
    EMBEDDING_SERVICE_URL: AnyHttpUrl = Field(default_factory=lambda: AnyHttpUrl(EMBEDDING_SERVICE_K8S_URL_DEFAULT))
    EMBEDDING_CLIENT_TIMEOUT: int = Field(default=30)

    # --- LLM (Local via LlamaCpp) ---
    LLM_API_BASE_URL: AnyHttpUrl = Field(default=LLM_API_BASE_URL_DEFAULT)
    LLM_MODEL_NAME: str = Field(default=LLM_MODEL_NAME_DEFAULT)
    LLM_MAX_OUTPUT_TOKENS: int = Field(default=LLM_MAX_OUTPUT_TOKENS_DEFAULT)

    # --- External Services ---
    BM25_ENABLED: bool = Field(default=DEFAULT_BM25_ENABLED)
    SPARSE_SEARCH_SERVICE_URL: AnyHttpUrl = Field(default_factory=lambda: AnyHttpUrl(SPARSE_SEARCH_SERVICE_K8S_URL_DEFAULT))
    SPARSE_SEARCH_CLIENT_TIMEOUT: int = Field(default=30)

    # --- Pipeline Config ---
    DIVERSITY_FILTER_ENABLED: bool = Field(default=DEFAULT_DIVERSITY_FILTER_ENABLED)
    QUERY_DIVERSITY_LAMBDA: float = Field(default=DEFAULT_DIVERSITY_LAMBDA)
    
    RETRIEVER_TOP_K: int = Field(default=DEFAULT_RETRIEVER_TOP_K)
    HYBRID_FUSION_ALPHA: float = Field(default=DEFAULT_HYBRID_ALPHA)
    
    MAX_CONTEXT_CHUNKS: int = Field(default=DEFAULT_MAX_CONTEXT_CHUNKS)
    MAX_TOKENS_PER_CHUNK: int = Field(default=DEFAULT_MAX_TOKENS_PER_CHUNK)
    MAX_CHARS_PER_CHUNK: int = Field(default=DEFAULT_MAX_CHARS_PER_CHUNK)
    
    MAX_CHAT_HISTORY_MESSAGES: int = Field(default=DEFAULT_MAX_CHAT_HISTORY_MESSAGES)
    NUM_SOURCES_TO_SHOW: int = Field(default=DEFAULT_NUM_SOURCES_TO_SHOW)
    
    MAPREDUCE_ENABLED: bool = Field(default=DEFAULT_MAPREDUCE_ENABLED)
    MAPREDUCE_CHUNK_BATCH_SIZE: int = Field(default=DEFAULT_MAPREDUCE_CHUNK_BATCH_SIZE)
    
    # --- Budgeting ---
    LLM_CONTEXT_WINDOW_TOKENS: int = Field(default=DEFAULT_LLM_CONTEXT_WINDOW_TOKENS)
    DIRECT_RAG_TOKEN_LIMIT: int = Field(default=DEFAULT_DIRECT_RAG_TOKEN_LIMIT)
    HTTP_CLIENT_TIMEOUT: int = Field(default=DEFAULT_HTTP_CLIENT_TIMEOUT)
    HTTP_CLIENT_MAX_RETRIES: int = Field(default=DEFAULT_HTTP_CLIENT_MAX_RETRIES)
    HTTP_CLIENT_BACKOFF_FACTOR: float = Field(default=DEFAULT_HTTP_CLIENT_BACKOFF_FACTOR)
    
    TIKTOKEN_ENCODING_NAME: str = Field(default=DEFAULT_TIKTOKEN_ENCODING_NAME)

    # --- Prompts ---
    RAG_PROMPT_TEMPLATE_PATH: str = Field(default=DEFAULT_RAG_PROMPT_TEMPLATE_PATH)
    GENERAL_PROMPT_TEMPLATE_PATH: str = Field(default=DEFAULT_GENERAL_PROMPT_TEMPLATE_PATH)
    MAP_PROMPT_TEMPLATE_PATH: str = Field(default=DEFAULT_MAP_PROMPT_TEMPLATE_PATH)
    REDUCE_PROMPT_TEMPLATE_PATH: str = Field(default=DEFAULT_REDUCE_PROMPT_TEMPLATE_PATH)

    @field_validator('LOG_LEVEL')
    @classmethod
    def check_log_level(cls, v: str) -> str:
        if v.upper() not in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]: raise ValueError("Invalid LOG_LEVEL")
        return v.upper()

    @field_validator('POSTGRES_PASSWORD', 'ZILLIZ_API_KEY', mode='before')
    @classmethod
    def check_secrets(cls, v: Any) -> Any:
        if isinstance(v, SecretStr) and not v.get_secret_value(): raise ValueError("Secret cannot be empty.")
        if not v: raise ValueError("Secret cannot be empty.")
        return v

try:
    settings = Settings()
except ValidationError as e:
    print(f"FATAL: Configuration Validation Error: {e}")
    sys.exit(1)