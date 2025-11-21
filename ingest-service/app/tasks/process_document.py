# ingest-service/app/tasks/process_document.py
import os
import tempfile
import uuid
import sys
import pathlib
import time
import json
from typing import Optional, Dict, Any, List

import structlog
import httpx 
from celery import Task, states
from celery.exceptions import Ignore, Reject, MaxRetriesExceededError, Retry
from celery.signals import worker_process_init
from sqlalchemy import Engine
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type, before_sleep_log
import logging 

logging.basicConfig(
    stream=sys.stdout, 
    level=logging.DEBUG, 
    format='%(asctime)s - %(name)s - %(levelname)s - [StdLib] - %(message)s',
    force=True 
)
stdlib_task_logger = logging.getLogger("app.tasks.process_document.stdlib")


def normalize_filename(filename: str) -> str:
    return " ".join(filename.strip().split())


from app.core.config import settings
from app.db.postgres_client import get_sync_engine, set_status_sync, bulk_insert_chunks_sync
from app.models.domain import DocumentStatus
from app.services.minio_client import MinIOClient, MinIOClientError
from app.services.ingest_pipeline import (
    index_chunks_in_milvus_and_prepare_for_pg,
    delete_milvus_chunks
)
from app.services.text_processor import TextProcessor 
from app.tasks.celery_app import celery_app

task_struct_log = structlog.get_logger(__name__) 
IS_WORKER = "worker" in sys.argv

sync_engine: Optional[Engine] = None
minio_client_global: Optional[MinIOClient] = None
text_processor_global: Optional[TextProcessor] = None

sync_http_retry_strategy = retry(
    stop=stop_after_attempt(settings.HTTP_CLIENT_MAX_RETRIES +1),
    wait=wait_exponential(multiplier=settings.HTTP_CLIENT_BACKOFF_FACTOR, min=1, max=10),
    retry=retry_if_exception_type((httpx.RequestError, httpx.HTTPStatusError)),
    before_sleep=before_sleep_log(task_struct_log, logging.WARNING), 
    reraise=True
)


@worker_process_init.connect(weak=False)
def init_worker_resources(**kwargs):
    global sync_engine, minio_client_global, text_processor_global
    
    init_log_struct = structlog.get_logger("app.tasks.worker_init.struct")
    init_log_std = logging.getLogger("app.tasks.worker_init.std")
    
    init_log_struct.info("Worker process initializing resources (structlog)...", signal="worker_process_init")
    init_log_std.info("Worker process initializing resources (std)...")
    try:
        if sync_engine is None:
            sync_engine = get_sync_engine()
            init_log_struct.info("Synchronous DB engine initialized for worker.")
        
        if minio_client_global is None:
            minio_client_global = MinIOClient() 
            init_log_struct.info("MinIO client initialized for worker.")

        if text_processor_global is None:
            text_processor_global = TextProcessor()
            init_log_struct.info("TextProcessor initialized for worker.")
        
        init_log_std.info("Worker resources initialization complete.")

    except Exception as e:
        init_log_struct.critical("CRITICAL FAILURE during worker resource initialization!", error=str(e), exc_info=True)
        print(f"CRITICAL WORKER INIT FAILURE (print): {e}", file=sys.stderr, flush=True)
        sync_engine = None
        minio_client_global = None
        text_processor_global = None


@celery_app.task(
    bind=True,
    name="ingest.process_document",
    autoretry_for=(
        httpx.RequestError, 
        httpx.HTTPStatusError, 
        MinIOClientError, 
        ConnectionRefusedError,
        Exception 
    ),
    exclude=(
        Reject, Ignore, ValueError, ConnectionError, RuntimeError, TypeError,
    ),
    retry_backoff=True,
    retry_backoff_max=600, 
    retry_jitter=True,
    max_retries=5, 
    acks_late=True
)
def process_document_standalone(self: Task, *args, **kwargs) -> Dict[str, Any]:
    sys.stdout.flush()
    sys.stderr.flush()

    early_task_id = str(self.request.id or uuid.uuid4()) 
    stdlib_task_logger.info(f"--- TASK ENTRY ID: {early_task_id} --- RAW KWARGS: {kwargs}")


    document_id_str = kwargs.get('document_id')
    company_id_str = kwargs.get('company_id')
    filename = kwargs.get('filename')
    content_type = kwargs.get('content_type')
    
    attempt = self.request.retries + 1
    max_attempts = (self.max_retries or 0) + 1
    
    log_context = {
        "task_id": early_task_id, "attempt": f"{attempt}/{max_attempts}", "doc_id": document_id_str,
        "company_id": company_id_str, "filename": filename, "content_type": content_type
    }
    log = structlog.get_logger("app.tasks.process_document.task_exec").bind(**log_context)
    
    log.info("Starting document processing task.")


    if not IS_WORKER:
         err_msg_not_worker = "Task function called outside of a worker context! Rejecting."
         log.critical(err_msg_not_worker)
         raise Reject(err_msg_not_worker, requeue=False)

    if not all([document_id_str, company_id_str, filename, content_type]):
        err_msg_args = "Missing required arguments (doc_id, company_id, filename, content_type)"
        log.error(err_msg_args, payload_kwargs=kwargs)
        raise Reject(err_msg_args, requeue=False)

    global_resources_check = {"Sync DB Engine": sync_engine, "MinIO Client": minio_client_global, "TextProcessor": text_processor_global}
    for name, resource in global_resources_check.items():
        if not resource:
            error_msg_resource = f"Worker resource '{name}' is not initialized. Task {early_task_id} cannot proceed."
            log.critical(error_msg_resource)
            if name != "Sync DB Engine" and sync_engine and document_id_str:
                try: 
                    doc_uuid_for_error_res = uuid.UUID(document_id_str)
                    set_status_sync(engine=sync_engine, document_id=doc_uuid_for_error_res, status=DocumentStatus.ERROR, error_message=error_msg_resource)
                except Exception: 
                    pass
            raise Reject(error_msg_resource, requeue=False)

    doc_uuid: uuid.UUID
    try:
        doc_uuid = uuid.UUID(document_id_str)
    except ValueError:
         err_msg_uuid = "Invalid document_id format."
         log.error(err_msg_uuid, received_doc_id=document_id_str)
         raise Reject(err_msg_uuid, requeue=False)

    if content_type not in settings.SUPPORTED_CONTENT_TYPES:
        error_msg_content = f"Unsupported content type by ingest-service: {content_type}"
        log.error(error_msg_content)
        if sync_engine: 
             set_status_sync(engine=sync_engine, document_id=doc_uuid, status=DocumentStatus.ERROR, error_message=error_msg_content)
        raise Reject(error_msg_content, requeue=False)

    normalized_filename = normalize_filename(filename) 
    object_name = f"{company_id_str}/{document_id_str}/{normalized_filename}"
    file_bytes: Optional[bytes] = None
    
    try:
        log.info("Attempting to set status to PROCESSING in DB.") 
        status_updated = set_status_sync(
            engine=sync_engine, document_id=doc_uuid, status=DocumentStatus.PROCESSING, error_message=None
        )
        if not status_updated:
            log.warning("Failed to update status to PROCESSING. Document might be deleted. Ignoring task.")
            raise Ignore()

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir_path = pathlib.Path(temp_dir)
            temp_file_path_obj = temp_dir_path / normalized_filename
            
            log.info(f"Attempting MinIO download: {object_name}")
            minio_client_global.download_file_sync(object_name, str(temp_file_path_obj))
            log.info("File downloaded successfully.")
            
            file_bytes = temp_file_path_obj.read_bytes()

        if file_bytes is None:
            raise RuntimeError("File_bytes is None after MinIO download.")

        log.info("Calling Document Processing Service (synchronous)...")
        docproc_url = str(settings.DOCPROC_SERVICE_URL)
        files_payload = {'file': (normalized_filename, file_bytes, content_type)}
        data_payload = {
            'original_filename': normalized_filename, 
            'content_type': content_type,
            'document_id': document_id_str,
            'company_id': company_id_str
        }
        try:
            with httpx.Client(timeout=settings.HTTP_CLIENT_TIMEOUT) as client:
                @sync_http_retry_strategy
                def call_docproc():
                    return client.post(docproc_url, files=files_payload, data=data_payload)
                
                response = call_docproc()
                response.raise_for_status() 
                docproc_response_data = response.json()

            if "data" not in docproc_response_data or "chunks" not in docproc_response_data.get("data", {}):
                raise ValueError("Invalid response format from DocProc Service.")
            
            processed_chunks_from_docproc = docproc_response_data.get("data", {}).get("chunks", [])
            log.info(f"DocProc returned {len(processed_chunks_from_docproc)} raw chunks.")

        except httpx.HTTPStatusError as hse:
            error_msg = f"DocProc Error ({hse.response.status_code}): {hse.response.text[:200]}"
            log.error("DocProc HTTP Error", error=error_msg)
            if sync_engine: set_status_sync(engine=sync_engine, document_id=doc_uuid, status=DocumentStatus.ERROR, error_message=error_msg)
            if 500 <= hse.response.status_code < 600: raise 
            else: raise Reject(error_msg, requeue=False) from hse
        except Exception as e: 
            log.error("DocProc Request Error", error=str(e))
            if sync_engine: set_status_sync(engine=sync_engine, document_id=doc_uuid, status=DocumentStatus.ERROR, error_message=str(e))
            raise

        if not processed_chunks_from_docproc:
            log.warning("DocProc returned no chunks. Finishing.")
            if sync_engine: set_status_sync(engine=sync_engine, document_id=doc_uuid, status=DocumentStatus.PROCESSED, chunk_count=0, error_message=None)
            return {"status": "processed", "chunks": 0}

        # --- REFINEMENT PHASE ---
        log.info("Refining chunks for high-density ingestion...")
        refined_chunks = text_processor_global.refine_chunks(processed_chunks_from_docproc, normalized_filename)
        log.info(f"Refinement complete. Expanded {len(processed_chunks_from_docproc)} raw chunks into {len(refined_chunks)} high-density chunks.")

        if not refined_chunks:
            log.warning("Refinement resulted in 0 chunks (empty texts?). Finishing.")
            if sync_engine: set_status_sync(engine=sync_engine, document_id=doc_uuid, status=DocumentStatus.PROCESSED, chunk_count=0, error_message=None)
            return {"status": "processed", "chunks": 0}
        
        texts_to_embed = [c['text'] for c in refined_chunks]

        log.info(f"Calling Embedding Service for {len(texts_to_embed)} refined texts...")
        embedding_service_url = str(settings.EMBEDDING_SERVICE_URL)
        
        embedding_request_payload = {
            "texts": texts_to_embed,
            "text_type": "passage"
        }
        embeddings: List[List[float]] = []
        try:
            with httpx.Client(timeout=settings.HTTP_CLIENT_TIMEOUT) as client:
                @sync_http_retry_strategy
                def call_embedding_svc():
                    return client.post(embedding_service_url, json=embedding_request_payload)

                response_embed = call_embedding_svc()
                response_embed.raise_for_status()
                embedding_response_data = response_embed.json()

            embeddings = embedding_response_data.get("embeddings", [])
            model_info = embedding_response_data.get("model_info", {})

            if len(embeddings) != len(texts_to_embed):
                raise RuntimeError(f"Embedding count mismatch. Expected {len(texts_to_embed)}, got {len(embeddings)}.")
            
            if embeddings and model_info.get("dimension") != settings.EMBEDDING_DIMENSION:
                 err_dim = f"Embedding dimension mismatch: Service={model_info.get('dimension')}, Config={settings.EMBEDDING_DIMENSION}"
                 log.error(err_dim)
                 if sync_engine: set_status_sync(engine=sync_engine, document_id=doc_uuid, status=DocumentStatus.ERROR, error_message=err_dim)
                 raise Reject(err_dim, requeue=False)

        except httpx.HTTPStatusError as hse_embed:
            err_emb = f"Embedding Service Error ({hse_embed.response.status_code}): {hse_embed.response.text[:200]}"
            log.error("Embedding Service HTTP Error", error=err_emb)
            if sync_engine: set_status_sync(engine=sync_engine, document_id=doc_uuid, status=DocumentStatus.ERROR, error_message=err_emb)
            if 500 <= hse_embed.response.status_code < 600: raise
            else: raise Reject(err_emb, requeue=False) from hse_embed
        except Exception as e:
            log.error("Embedding Service Request Error", error=str(e))
            if sync_engine: set_status_sync(engine=sync_engine, document_id=doc_uuid, status=DocumentStatus.ERROR, error_message=str(e))
            raise

        log.info("Indexing refined chunks in Milvus and preparing PG data...")
        inserted_milvus_count, milvus_pks, chunks_for_pg_insert = index_chunks_in_milvus_and_prepare_for_pg(
            chunks_to_index=refined_chunks,
            embeddings=embeddings,
            filename=normalized_filename,
            company_id_str=company_id_str,
            document_id_str=document_id_str,
            delete_existing_milvus_chunks=True 
        )
        log.info(f"Milvus indexing complete. Inserted: {inserted_milvus_count}")

        if inserted_milvus_count == 0:
            log.warning("No chunks inserted in Milvus.")
            if sync_engine: set_status_sync(engine=sync_engine, document_id=doc_uuid, status=DocumentStatus.PROCESSED, chunk_count=0, error_message=None)
            return {"status": "processed", "chunks": 0}

        log.info(f"Bulk inserting {len(chunks_for_pg_insert)} chunks into PostgreSQL.") 
        try:
            inserted_pg_count = bulk_insert_chunks_sync(engine=sync_engine, chunks_data=chunks_for_pg_insert)
            if inserted_pg_count != len(chunks_for_pg_insert):
                 log.warning("PG Insert count mismatch", prepared=len(chunks_for_pg_insert), inserted=inserted_pg_count)
        except Exception as pg_err:
             log.critical("PG Insert Failed after Milvus success!", error=str(pg_err))
             try: delete_milvus_chunks(company_id=company_id_str, document_id=document_id_str)
             except: pass
             if sync_engine: set_status_sync(engine=sync_engine, document_id=doc_uuid, status=DocumentStatus.ERROR, error_message=f"DB Insert Failed: {str(pg_err)[:200]}")
             raise Reject(f"PG Insert Failed: {pg_err}", requeue=False) from pg_err

        log.info("Setting final status to PROCESSED.") 
        if sync_engine: set_status_sync(engine=sync_engine, document_id=doc_uuid, status=DocumentStatus.PROCESSED, chunk_count=inserted_pg_count, error_message=None)
        
        return {"status": "processed", "chunks": inserted_pg_count, "document_id": document_id_str}

    except MinIOClientError as mc_err:
        log.error("MinIO Error", error=str(mc_err))
        if sync_engine and 'doc_uuid' in locals(): 
             set_status_sync(engine=sync_engine, document_id=doc_uuid, status=DocumentStatus.ERROR, error_message=f"Storage Error: {str(mc_err)}")
        if "Object not found" in str(mc_err): raise Reject(f"Object missing: {object_name}", requeue=False) from mc_err
        raise 
    except Reject: raise 
    except Ignore: raise 
    except Exception as exc: 
        log.exception("Unexpected error in task")
        if sync_engine and 'doc_uuid' in locals(): 
            set_status_sync(engine=sync_engine, document_id=doc_uuid, status=DocumentStatus.PROCESSING, error_message=f"Error: {str(exc)[:200]}") 
        raise