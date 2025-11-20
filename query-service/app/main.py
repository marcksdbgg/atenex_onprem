from fastapi import FastAPI, HTTPException, status as fastapi_status, Request, Depends
from fastapi.exceptions import RequestValidationError, ResponseValidationError
from fastapi.responses import JSONResponse, Response, PlainTextResponse
import structlog
import uvicorn
import logging
import sys
import asyncio
import json
import uuid
import os
from contextlib import asynccontextmanager
from typing import Annotated, Optional
import httpx

from app.core.config import settings
from app.core.logging_config import setup_logging
setup_logging()
log = structlog.get_logger("query_service.main")

from app.api.v1.endpoints import query as query_router_module
from app.api.v1.endpoints import chat as chat_router_module

# Ports & Adapters
from app.application.ports import (
    ChatRepositoryPort, LogRepositoryPort, VectorStorePort, LLMPort,
    SparseRetrieverPort, DiversityFilterPort, ChunkContentRepositoryPort,
    EmbeddingPort 
)
from app.infrastructure.persistence.postgres_repositories import (
    PostgresChatRepository, PostgresLogRepository, PostgresChunkContentRepository
)
from app.infrastructure.vectorstores.milvus_adapter import MilvusAdapter

from app.infrastructure.llms.llama_cpp_adapter import LlamaCppAdapter

from app.infrastructure.clients.sparse_search_service_client import SparseSearchServiceClient
from app.infrastructure.retrievers.remote_sparse_retriever_adapter import RemoteSparseRetrieverAdapter
from app.infrastructure.filters.diversity_filter import MMRDiversityFilter, StubDiversityFilter
from app.infrastructure.clients.embedding_service_client import EmbeddingServiceClient
from app.infrastructure.embedding.remote_embedding_adapter import RemoteEmbeddingAdapter
from app.application.use_cases.ask_query_use_case import AskQueryUseCase
from app.dependencies import set_ask_query_use_case_instance
from app.infrastructure.persistence import postgres_connector

# Global state & instances
SERVICE_READY = False
chat_repo_instance: Optional[ChatRepositoryPort] = None
log_repo_instance: Optional[LogRepositoryPort] = None
chunk_content_repo_instance: Optional[ChunkContentRepositoryPort] = None
vector_store_instance: Optional[VectorStorePort] = None
llm_instance: Optional[LLMPort] = None
sparse_retriever_instance: Optional[SparseRetrieverPort] = None
sparse_search_service_client_instance: Optional[SparseSearchServiceClient] = None 
diversity_filter_instance: Optional[DiversityFilterPort] = None
embedding_service_client_instance: Optional[EmbeddingServiceClient] = None
embedding_adapter_instance: Optional[EmbeddingPort] = None
ask_query_use_case_instance: Optional[AskQueryUseCase] = None
http_client_instance: Optional[httpx.AsyncClient] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global SERVICE_READY, chat_repo_instance, log_repo_instance, chunk_content_repo_instance, \
           vector_store_instance, llm_instance, sparse_retriever_instance, \
           sparse_search_service_client_instance, \
           diversity_filter_instance, ask_query_use_case_instance, \
           embedding_service_client_instance, embedding_adapter_instance, http_client_instance

    SERVICE_READY = False
    log.info(f"Starting up {settings.PROJECT_NAME}...")
    dependencies_ok = True
    critical_failure_message = ""

    # 0. HTTP Client
    try:
        http_client_instance = httpx.AsyncClient(
            timeout=settings.HTTP_CLIENT_TIMEOUT,
            limits=httpx.Limits(max_connections=100, max_keepalive_connections=20) 
        )
    except Exception as e:
        critical_failure_message = "Failed to initialize global HTTP client."
        log.critical(f"CRITICAL: {critical_failure_message}", error=str(e))
        dependencies_ok = False

    # 1. DB Pool
    if dependencies_ok:
        try:
            await postgres_connector.get_db_pool()
            if await postgres_connector.check_db_connection():
                chat_repo_instance = PostgresChatRepository()
                log_repo_instance = PostgresLogRepository()
                chunk_content_repo_instance = PostgresChunkContentRepository() 
                log.info("DB connected.")
            else:
                raise ConnectionError("DB Check Failed")
        except Exception as e:
            critical_failure_message = "Failed PostgreSQL initialization."
            log.critical(f"CRITICAL: {critical_failure_message}", error=str(e))
            dependencies_ok = False

    # 2. Embedding
    if dependencies_ok:
        try:
            embedding_service_client_instance = EmbeddingServiceClient(
                base_url=str(settings.EMBEDDING_SERVICE_URL),
                timeout=settings.EMBEDDING_CLIENT_TIMEOUT
            )
            embedding_adapter_instance = RemoteEmbeddingAdapter(client=embedding_service_client_instance)
            await embedding_adapter_instance.initialize()
            if not await embedding_adapter_instance.health_check():
                 log.warning("Embedding Service check failed, startup continues but errors may occur.")
        except Exception as e:
            log.error("Embedding Service initialization failed.", error=str(e))

    # 2.B. Sparse Search
    if dependencies_ok and settings.BM25_ENABLED: 
        try:
            sparse_search_service_client_instance = SparseSearchServiceClient(
                base_url=str(settings.SPARSE_SEARCH_SERVICE_URL),
                timeout=settings.SPARSE_SEARCH_CLIENT_TIMEOUT
            )
            sparse_retriever_instance = RemoteSparseRetrieverAdapter(client=sparse_search_service_client_instance)
            await sparse_search_service_client_instance.check_health()
        except Exception as e:
            log.error("Sparse Search initialization failed", error=str(e))
            sparse_retriever_instance = None 

    # 3. Milvus
    if dependencies_ok:
        try:
            vector_store_instance = MilvusAdapter()
            await vector_store_instance.connect()
        except Exception as e:
            critical_failure_message = "Failed to initialize Milvus."
            log.critical(f"CRITICAL: {critical_failure_message}", error=str(e))
            dependencies_ok = False

    # 4. LLM (Strictly Local)
    if dependencies_ok:
        try:
            llm_instance = LlamaCppAdapter(
                base_url=str(settings.LLM_API_BASE_URL),
                model_name=settings.LLM_MODEL_NAME,
                timeout=settings.HTTP_CLIENT_TIMEOUT, 
                max_output_tokens=settings.LLM_MAX_OUTPUT_TOKENS,
            )
            if await llm_instance.health_check():
                log.info("LlamaCppAdapter initialized.", model=settings.LLM_MODEL_NAME)
            else:
                critical_failure_message = "LLM Adapter health check failed (llama.cpp unreachable)."
                log.critical(critical_failure_message)
                dependencies_ok = False
        except Exception as e:
            critical_failure_message = "Failed to initialize LlamaCppAdapter."
            log.critical(critical_failure_message, error=str(e))
            dependencies_ok = False
    
    # 5. Filters
    if dependencies_ok:
        if settings.DIVERSITY_FILTER_ENABLED and embedding_adapter_instance:
            diversity_filter_instance = MMRDiversityFilter(lambda_mult=settings.QUERY_DIVERSITY_LAMBDA)
        else:
            diversity_filter_instance = StubDiversityFilter()

    # 6. Use Case
    if dependencies_ok:
         try:
             ask_query_use_case_instance = AskQueryUseCase(
                 chat_repo=chat_repo_instance,
                 log_repo=log_repo_instance,
                 chunk_content_repo=chunk_content_repo_instance,
                 vector_store=vector_store_instance,
                 llm=llm_instance,
                 embedding_adapter=embedding_adapter_instance,
                 http_client=http_client_instance,
                 sparse_retriever=sparse_retriever_instance,
                 diversity_filter=diversity_filter_instance
             )
             SERVICE_READY = True 
             set_ask_query_use_case_instance(ask_query_use_case_instance, SERVICE_READY)
             log.info(f"{settings.PROJECT_NAME} READY.")
         except Exception as e:
              critical_failure_message = "Failed to instantiate AskQueryUseCase."
              log.critical(f"CRITICAL: {critical_failure_message}", error=str(e))
              SERVICE_READY = False

    if not SERVICE_READY:
        log.critical(f"Startup finished with ERRORS. Service NOT Ready. {critical_failure_message}")

    yield 

    log.info("Shutting down...")
    if http_client_instance: await http_client_instance.aclose()
    await postgres_connector.close_db_pool()
    if vector_store_instance: await vector_store_instance.disconnect()
    if embedding_service_client_instance: await embedding_service_client_instance.close()
    if sparse_search_service_client_instance: await sparse_search_service_client_instance.close()
    if llm_instance: await llm_instance.close()

app = FastAPI(
    title=settings.PROJECT_NAME,
    openapi_url=f"{settings.API_V1_STR}/openapi.json",
    version="0.4.0", 
    lifespan=lifespan
)

@app.middleware("http")
async def add_request_id_timing_logging(request: Request, call_next):
    request_id = request.headers.get("x-request-id", str(uuid.uuid4()))
    structlog.contextvars.bind_contextvars(request_id=request_id)
    response = await call_next(request)
    return response

app.include_router(query_router_module.router, prefix=settings.API_V1_STR, tags=["Query Interaction"])
app.include_router(chat_router_module.router, prefix=settings.API_V1_STR, tags=["Chat Management"])

@app.get("/health", tags=["Health Check"])
async def health():
    if not SERVICE_READY: raise HTTPException(status_code=503, detail="Not Ready")
    return {"status": "ok"}

if __name__ == "__main__":
    port = int(os.getenv("PORT", "8001")) 
    uvicorn.run("app.main:app", host="0.0.0.0", port=port, reload=True, log_level="info")