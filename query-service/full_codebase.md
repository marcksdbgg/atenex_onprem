# Estructura de la Codebase

```
app/
├── api
│   └── v1
│       ├── __init__.py
│       ├── endpoints
│       │   ├── __init__.py
│       │   ├── chat.py
│       │   └── query.py
│       ├── mappers.py
│       └── schemas.py
├── application
│   ├── __init__.py
│   ├── ports
│   │   ├── __init__.py
│   │   ├── embedding_port.py
│   │   ├── llm_port.py
│   │   ├── repository_ports.py
│   │   ├── retrieval_ports.py
│   │   └── vector_store_port.py
│   └── use_cases
│       ├── __init__.py
│       ├── ask_query
│       │   ├── __init__.py
│       │   ├── config_types.py
│       │   ├── pipeline.py
│       │   ├── prompt_service.py
│       │   ├── steps.py
│       │   ├── token_accountant.py
│       │   └── types.py
│       └── ask_query_use_case.py
├── core
│   ├── __init__.py
│   ├── config.py
│   └── logging_config.py
├── dependencies.py
├── domain
│   ├── __init__.py
│   └── models.py
├── infrastructure
│   ├── __init__.py
│   ├── clients
│   │   ├── __init__.py
│   │   ├── embedding_service_client.py
│   │   └── sparse_search_service_client.py
│   ├── embedding
│   │   ├── __init__.py
│   │   └── remote_embedding_adapter.py
│   ├── filters
│   │   ├── __init__.py
│   │   └── diversity_filter.py
│   ├── llms
│   │   ├── __init__.py
│   │   ├── gemini_adapter.py
│   │   └── llama_cpp_adapter.py
│   ├── persistence
│   │   ├── __init__.py
│   │   ├── postgres_connector.py
│   │   └── postgres_repositories.py
│   ├── retrievers
│   │   ├── __init__.py
│   │   └── remote_sparse_retriever_adapter.py
│   └── vectorstores
│       ├── __init__.py
│       └── milvus_adapter.py
├── main.py
├── models
│   └── __init__.py
├── pipelines
│   └── rag_pipeline.py
├── prompts
│   ├── general_template_gemini_v2.txt
│   ├── map_prompt_template.txt
│   ├── rag_template_gemini_v2.txt
│   └── reduce_prompt_template_v2.txt
└── utils
    ├── __init__.py
    └── helpers.py
```

# Codebase: `app`

## File: `app/api/v1/__init__.py`
```py

```

## File: `app/api/v1/endpoints/__init__.py`
```py

```

## File: `app/api/v1/endpoints/chat.py`
```py
# query-service/app/api/v1/endpoints/chat.py
import uuid
from typing import List, Optional
import structlog
from fastapi import (
    APIRouter, Depends, HTTPException, status, Path, Query, Header, Request,
    Response
)

from app.api.v1 import schemas
# LLM_REFACTOR_STEP_4: Import Repository directly for simple operations
from app.infrastructure.persistence.postgres_repositories import PostgresChatRepository
from app.application.ports.repository_ports import ChatRepositoryPort

log = structlog.get_logger(__name__)

router = APIRouter()

# --- Headers Dependencies (Sin cambios) ---
async def get_current_company_id(x_company_id: Optional[str] = Header(None, alias="X-Company-ID")) -> uuid.UUID:
    # ... (código existente sin cambios)
    if not x_company_id:
        log.warning("Missing required X-Company-ID header")
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="Missing required header: X-Company-ID")
    try:
        return uuid.UUID(x_company_id)
    except ValueError:
        log.warning("Invalid UUID format in X-Company-ID header", header_value=x_company_id)
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="Invalid X-Company-ID header format")

async def get_current_user_id(x_user_id: Optional[str] = Header(None, alias="X-User-ID")) -> uuid.UUID:
    # ... (código existente sin cambios)
    if not x_user_id:
        log.warning("Missing required X-User-ID header")
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="Missing required header: X-User-ID")
    try:
        return uuid.UUID(x_user_id)
    except ValueError:
        log.warning("Invalid UUID format in X-User-ID header", header_value=x_user_id)
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="Invalid X-User-ID header format")

# --- Dependency for Chat Repository ---
# LLM_REFACTOR_STEP_4: Inject repository dependency (simplified)
def get_chat_repository() -> ChatRepositoryPort:
    """Provides an instance of the Chat Repository."""
    # In a real setup, this would come from a DI container configured in main.py
    return PostgresChatRepository()


# --- Endpoints Refactored to use Repository Dependency ---

@router.get(
    "/chats",
    response_model=List[schemas.ChatSummary],
    status_code=status.HTTP_200_OK,
    summary="List User Chats",
    description="Retrieves a list of chat summaries using X-Company-ID and X-User-ID headers.",
)
async def list_chats(
    user_id: uuid.UUID = Depends(get_current_user_id),
    company_id: uuid.UUID = Depends(get_current_company_id),
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    # LLM_REFACTOR_STEP_4: Inject repository
    chat_repo: ChatRepositoryPort = Depends(get_chat_repository),
    request: Request = None
):
    request_id = request.headers.get("x-request-id") if request else str(uuid.uuid4())
    endpoint_log = log.bind(request_id=request_id, user_id=str(user_id), company_id=str(company_id), limit=limit, offset=offset)
    endpoint_log.info("Request received to list chats")

    try:
        # LLM_REFACTOR_STEP_4: Use the injected repository method
        chats_domain = await chat_repo.get_user_chats(
            user_id=user_id, company_id=company_id, limit=limit, offset=offset
        )
        endpoint_log.info("Chats listed successfully", count=len(chats_domain))
        # Map domain to schema if necessary (ChatSummary is compatible for now)
        return [schemas.ChatSummary(**chat.model_dump()) for chat in chats_domain]
    except Exception as e:
        endpoint_log.exception("Error listing chats")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to retrieve chat list.")


@router.get(
    "/chats/{chat_id}/messages",
    response_model=List[schemas.ChatMessage],
    status_code=status.HTTP_200_OK,
    summary="Get Chat Messages",
    description="Retrieves messages for a specific chat using X-Company-ID and X-User-ID headers.",
    responses={403: {"description": "Chat not found or access denied."}} # Changed 404 to 403 as check_ownership returns false
)
async def get_chat_messages_endpoint(
    chat_id: uuid.UUID = Path(..., description="The ID of the chat."),
    user_id: uuid.UUID = Depends(get_current_user_id),
    company_id: uuid.UUID = Depends(get_current_company_id),
    limit: int = Query(default=100, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
    # LLM_REFACTOR_STEP_4: Inject repository
    chat_repo: ChatRepositoryPort = Depends(get_chat_repository),
    request: Request = None
):
    request_id = request.headers.get("x-request-id") if request else str(uuid.uuid4())
    endpoint_log = log.bind(request_id=request_id, user_id=str(user_id), company_id=str(company_id), chat_id=str(chat_id), limit=limit, offset=offset)
    endpoint_log.info("Request received to get chat messages")

    try:
        # LLM_REFACTOR_STEP_4: Use the injected repository method
        # The repo method already includes the ownership check
        messages_domain = await chat_repo.get_chat_messages(
            chat_id=chat_id, user_id=user_id, company_id=company_id, limit=limit, offset=offset
        )
        # If ownership check failed inside repo, it returns empty list, no exception needed here.
        endpoint_log.info("Chat messages retrieved successfully", count=len(messages_domain))
        # Map domain to schema if necessary (ChatMessage is compatible for now)
        return [schemas.ChatMessage(**msg.model_dump()) for msg in messages_domain]
    except Exception as e:
        # Catch potential DB errors from the repository
        endpoint_log.exception("Error getting chat messages")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to retrieve chat messages.")


@router.delete(
    "/chats/{chat_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete Chat",
    description="Deletes a chat and associated data using X-Company-ID and X-User-ID headers.",
    responses={404: {"description": "Chat not found or access denied."}, 204: {}} # 404 if ownership fails
)
async def delete_chat_endpoint(
    chat_id: uuid.UUID = Path(..., description="The ID of the chat to delete."),
    user_id: uuid.UUID = Depends(get_current_user_id),
    company_id: uuid.UUID = Depends(get_current_company_id),
    # LLM_REFACTOR_STEP_4: Inject repository
    chat_repo: ChatRepositoryPort = Depends(get_chat_repository),
    request: Request = None
):
    request_id = request.headers.get("x-request-id") if request else str(uuid.uuid4())
    endpoint_log = log.bind(request_id=request_id, user_id=str(user_id), company_id=str(company_id), chat_id=str(chat_id))
    endpoint_log.info("Request received to delete chat")

    try:
        # LLM_REFACTOR_STEP_4: Use the injected repository method
        # The repo method includes ownership check and returns bool
        deleted = await chat_repo.delete_chat(chat_id=chat_id, user_id=user_id, company_id=company_id)
        if deleted:
            endpoint_log.info("Chat deleted successfully")
            # Return 204 No Content requires a Response object
            return Response(status_code=status.HTTP_204_NO_CONTENT)
        else:
            # If not deleted, it means ownership check failed or chat didn't exist
            endpoint_log.warning("Chat not found or access denied for deletion")
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Chat not found or access denied.")
    except Exception as e:
        # Catch potential DB errors from the repository
        endpoint_log.exception("Error deleting chat")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to delete chat.")
```

## File: `app/api/v1/endpoints/query.py`
```py
# query-service/app/api/v1/endpoints/query.py
import uuid
from typing import Dict, Any, Optional, List
import structlog
import asyncio
import re

from fastapi import APIRouter, Depends, HTTPException, status, Header, Body, Request

from app.api.v1 import schemas
from app.core.config import settings
from app.application.use_cases.ask_query_use_case import AskQueryUseCase
from app.utils.helpers import truncate_text
from .chat import get_current_company_id, get_current_user_id

log = structlog.get_logger(__name__)

router = APIRouter()

GREETING_REGEX = re.compile(r"^\s*(hola|hello|hi|buenos días|buenas tardes|buenas noches|hey|qué tal|hi there)\s*[\.,!?]*\s*$", re.IGNORECASE)


from app.dependencies import get_ask_query_use_case

@router.post(
    "/ask",
    response_model=schemas.QueryResponse,
    status_code=status.HTTP_200_OK,
    summary="Process Query / Manage Chat",
    description="Handles user queries via RAG pipeline or simple greeting, manages chat state.",
)
async def process_query(
    request_body: schemas.QueryRequest = Body(...),
    company_id: uuid.UUID = Depends(get_current_company_id),
    user_id: uuid.UUID = Depends(get_current_user_id),
    use_case: AskQueryUseCase = Depends(get_ask_query_use_case),
    request: Request = None 
):
    request_id = request.headers.get("x-request-id", str(uuid.uuid4())) if request else str(uuid.uuid4())
    endpoint_log = log.bind(
        request_id=request_id,
        company_id=str(company_id),
        user_id=str(user_id),
        query=truncate_text(request_body.query, 100),
        provided_chat_id=str(request_body.chat_id) if request_body.chat_id else "None"
    )
    endpoint_log.info("Processing query request via Use Case")

    try:
        answer, retrieved_chunks_domain, log_id, final_chat_id = await use_case.execute(
            query=request_body.query,
            company_id=company_id,
            user_id=user_id,
            chat_id=request_body.chat_id,
            top_k=request_body.retriever_top_k
        )

        retrieved_docs_api = []
        if retrieved_chunks_domain: # Solo mapear si hay chunks
            retrieved_docs_api = [
                schemas.RetrievedDocument(
                    id=chunk.id,
                    score=chunk.score,
                    content_preview=truncate_text(chunk.content, 150) if chunk.content else None,
                    content=chunk.content, # Pasar contenido completo
                    metadata=chunk.metadata,
                    document_id=chunk.document_id,
                    file_name=chunk.file_name,
                    cita_tag=chunk.cita_tag # Pasar cita_tag
                ) for chunk in retrieved_chunks_domain
            ]

        endpoint_log.info("Use case executed successfully, returning response", num_retrieved=len(retrieved_docs_api))
        return schemas.QueryResponse(
            answer=answer,
            retrieved_documents=retrieved_docs_api,
            query_log_id=log_id,
            chat_id=final_chat_id
        )

    except HTTPException as http_exc:
        raise http_exc
    except ConnectionError as ce:
        endpoint_log.error("Dependency connection error reported by Use Case", error=str(ce), exc_info=True)
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=f"A required service is unavailable.")
    except Exception as e:
        endpoint_log.exception("Unhandled exception during use case execution")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"An internal error occurred.")
```

## File: `app/api/v1/mappers.py`
```py
# query-service/app/api/v1/mappers.py
# This file will contain mapping functions between API DTOs (schemas)
# and Domain objects, if needed in later steps.
```

## File: `app/api/v1/schemas.py`
```py
# ./app/api/v1/schemas.py
import uuid
from pydantic import BaseModel, Field, validator
from typing import Optional, List, Dict, Any, Union
from datetime import datetime

# --- Chat Schemas ---

class ChatSummary(BaseModel):
    id: uuid.UUID = Field(..., description="Unique ID of the chat.")
    title: Optional[str] = Field(None, description="Title of the chat (may be null).")
    updated_at: datetime = Field(..., description="Timestamp of the last update (last message or creation).")

class ChatMessage(BaseModel):
    id: uuid.UUID = Field(..., description="Unique ID of the message.")
    role: str = Field(..., description="Role of the sender ('user' or 'assistant').")
    content: str = Field(..., description="Content of the message.")
    sources: Optional[List[Dict[str, Any]]] = Field(None, description="List of source documents cited by the assistant, if any.")
    created_at: datetime = Field(..., description="Timestamp when the message was created.")

class CreateMessageRequest(BaseModel):
    role: str = Field(..., description="Role of the sender ('user' or 'assistant'). Currently only 'user' expected from client.")
    content: str = Field(..., min_length=1, description="Content of the message.")

    @validator('role')
    def role_must_be_user_or_assistant(cls, v):
        if v not in ['user', 'assistant']:
            raise ValueError("Role must be either 'user' or 'assistant'")
        return v

# --- Request Schemas ---

class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, description="The user's query in natural language.")
    retriever_top_k: Optional[int] = Field(None, gt=0, le=20, description="Number of documents to retrieve (overrides server default).")
    chat_id: Optional[uuid.UUID] = Field(None, description="ID of the existing chat to continue, or null/omitted to start a new chat.")
    stream: bool = Field(default=False, description="If true, response will be streamed using Server-Sent Events.")


    @validator('query')
    def query_must_not_be_empty(cls, v):
        if not v.strip():
            raise ValueError('Query cannot be empty or just whitespace')
        return v

# --- Response Schemas ---

class RetrievedDocument(BaseModel):
    id: str = Field(..., description="The unique ID of the retrieved document chunk (usually from Milvus).")
    score: Optional[float] = Field(None, description="Relevance score assigned by the retriever (e.g., cosine similarity).")
    content_preview: Optional[str] = Field(None, description="A short preview of the document chunk's content.")
    content: Optional[str] = Field(None, description="The full content of the document chunk. Provided to enable detailed view in UI.")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Metadata associated with the document chunk.")
    document_id: Optional[str] = Field(None, description="ID of the original source document.")
    file_name: Optional[str] = Field(None, description="Name of the original source file.")
    # REFACTOR_5_1: Add cita_tag for frontend mapping
    cita_tag: Optional[str] = Field(None, description="The citation tag (e.g., '[Doc 1]') used by the LLM for this document.")


    @classmethod
    def from_haystack_doc(cls, doc: Any): # Usar Any para evitar dependencia directa de Haystack aquí
        meta = doc.meta or {}
        return cls(
            id=doc.id,
            score=doc.score,
            content_preview=(doc.content[:150] + '...') if doc.content and len(doc.content) > 150 else doc.content,
            content=doc.content, 
            metadata=meta,
            document_id=meta.get("document_id"),
            file_name=meta.get("file_name")
        )


class QueryResponse(BaseModel):
    answer: str = Field(..., description="The final answer generated by the LLM based on the query and retrieved documents.")
    retrieved_documents: List[RetrievedDocument] = Field(default_factory=list, description="List of documents retrieved and used as context for the answer.")
    query_log_id: Optional[uuid.UUID] = Field(None, description="The unique ID of the logged interaction in the database (if logging was successful).")
    chat_id: uuid.UUID = Field(..., description="The ID of the chat (either existing or newly created) this interaction belongs to.")

# --- SSE Stream Event Schemas ---
class SSETextChunk(BaseModel):
    type: str = Field("text_chunk", literal=True)
    token: str

class SSESourceChunk(BaseModel): # Para enviar información de fuentes una vez resueltas
    type: str = Field("source_chunk", literal=True)
    sources: List[RetrievedDocument]

class SSEErrorChunk(BaseModel):
    type: str = Field("error_chunk", literal=True)
    detail: str

class SSEEndOfStreamChunk(BaseModel):
    type: str = Field("end_of_stream_chunk", literal=True)
    log_id: Optional[uuid.UUID]
    chat_id: uuid.UUID
    # REFACTOR_5_1: Incluir resumen ejecutivo y siguiente pregunta sugerida aquí
    # ya que se generan al final después de la respuesta detallada.
    resumen_ejecutivo: Optional[str] = None
    siguiente_pregunta_sugerida: Optional[str] = None


class HealthCheckDependency(BaseModel):
    status: str = Field(..., description="Status of the dependency ('ok', 'error', 'pending')")
    details: Optional[str] = Field(None, description="Additional details in case of error.")

class HealthCheckResponse(BaseModel):
    status: str = Field(default="ok", description="Overall status of the service.")
    service: str = Field(..., description="Name of the service.")
    ready: bool = Field(..., description="Indicates if the service is ready to serve requests.")
    dependencies: Dict[str, str]
```

## File: `app/application/__init__.py`
```py
# query-service/app/application/use_cases/__init__.py
```

## File: `app/application/ports/__init__.py`
```py
# query-service/app/application/ports/__init__.py
from .llm_port import LLMPort
from .vector_store_port import VectorStorePort
from .repository_ports import ChatRepositoryPort, LogRepositoryPort, ChunkContentRepositoryPort
from .retrieval_ports import SparseRetrieverPort, RerankerPort, DiversityFilterPort # RerankerPort se mantiene si se quiere usar como interfaz para el cliente remoto, o se elimina si se usa httpx directamente en el use case. Por ahora lo dejo.
from .embedding_port import EmbeddingPort

__all__ = [
    "LLMPort",
    "VectorStorePort",
    "ChatRepositoryPort",
    "LogRepositoryPort",
    "ChunkContentRepositoryPort",
    "SparseRetrieverPort",
    "RerankerPort",
    "DiversityFilterPort",
    "EmbeddingPort",
]
```

## File: `app/application/ports/embedding_port.py`
```py
# query-service/app/application/ports/embedding_port.py
import abc
from typing import List

class EmbeddingPort(abc.ABC):
    """
    Puerto abstracto para la generación de embeddings.
    """

    @abc.abstractmethod
    async def embed_query(self, query_text: str) -> List[float]:
        """
        Genera el embedding para un único texto de consulta.

        Args:
            query_text: El texto de la consulta.

        Returns:
            Una lista de floats representando el embedding.

        Raises:
            ConnectionError: Si hay problemas de comunicación con el servicio de embedding.
            ValueError: Si la respuesta del servicio de embedding es inválida.
        """
        raise NotImplementedError

    @abc.abstractmethod
    async def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        Genera embeddings para una lista de textos.

        Args:
            texts: Una lista de textos.

        Returns:
            Una lista de embeddings, donde cada embedding es una lista de floats.

        Raises:
            ConnectionError: Si hay problemas de comunicación con el servicio de embedding.
            ValueError: Si la respuesta del servicio de embedding es inválida.
        """
        raise NotImplementedError

    @abc.abstractmethod
    async def get_embedding_dimension(self) -> int:
        """
        Devuelve la dimensión de los embeddings generados por el modelo subyacente.
        """
        raise NotImplementedError

    @abc.abstractmethod
    async def health_check(self) -> bool:
        """
        Verifica la salud del servicio de embedding subyacente.
        """
        raise NotImplementedError
```

## File: `app/application/ports/llm_port.py`
```py
# query-service/app/application/ports/llm_port.py
import abc
from typing import Optional, Type

from pydantic import BaseModel

class LLMPort(abc.ABC):
    """Puerto abstracto para interactuar con un Large Language Model."""

    @abc.abstractmethod
    async def generate(
        self,
        prompt: str,
        response_pydantic_schema: Optional[Type[BaseModel]] = None,
    ) -> str:
        """
        Genera texto basado en el prompt proporcionado.

        Args:
            prompt: El prompt a enviar al LLM.
            response_pydantic_schema: Esquema opcional para indicar que la respuesta debería estar en JSON.

        Returns:
            La respuesta generada por el LLM.

        Raises:
            ConnectionError: Si falla la comunicación con el servicio LLM.
            Exception: Para otros errores inesperados.
        """
        raise NotImplementedError
```

## File: `app/application/ports/repository_ports.py`
```py
# query-service/app/application/ports/repository_ports.py
import abc
import uuid
from typing import List, Optional, Dict, Any, Tuple # Añadir Tuple
# LLM_REFACTOR_STEP_2: Importar modelos de dominio
from app.domain.models import ChatMessage, ChatSummary, QueryLog

class ChatRepositoryPort(abc.ABC):
    """Puerto abstracto para operaciones de persistencia de Chats y Mensajes."""

    @abc.abstractmethod
    async def create_chat(self, user_id: uuid.UUID, company_id: uuid.UUID, title: Optional[str] = None) -> uuid.UUID:
        raise NotImplementedError

    @abc.abstractmethod
    async def get_user_chats(self, user_id: uuid.UUID, company_id: uuid.UUID, limit: int = 50, offset: int = 0) -> List[ChatSummary]:
        raise NotImplementedError

    @abc.abstractmethod
    async def check_chat_ownership(self, chat_id: uuid.UUID, user_id: uuid.UUID, company_id: uuid.UUID) -> bool:
        raise NotImplementedError

    @abc.abstractmethod
    async def get_chat_messages(self, chat_id: uuid.UUID, user_id: uuid.UUID, company_id: uuid.UUID, limit: int = 100, offset: int = 0) -> List[ChatMessage]:
        raise NotImplementedError

    @abc.abstractmethod
    async def save_message(self, chat_id: uuid.UUID, role: str, content: str, sources: Optional[List[Dict[str, Any]]] = None) -> uuid.UUID:
        raise NotImplementedError

    @abc.abstractmethod
    async def delete_chat(self, chat_id: uuid.UUID, user_id: uuid.UUID, company_id: uuid.UUID) -> bool:
        raise NotImplementedError


class LogRepositoryPort(abc.ABC):
    """Puerto abstracto para operaciones de persistencia de Logs de Consultas."""

    @abc.abstractmethod
    async def log_query_interaction(
        self,
        user_id: Optional[uuid.UUID],
        company_id: uuid.UUID,
        query: str,
        answer: str,
        retrieved_documents_data: List[Dict[str, Any]], 
        metadata: Optional[Dict[str, Any]] = None,
        chat_id: Optional[uuid.UUID] = None,
    ) -> uuid.UUID:
        raise NotImplementedError


class ChunkContentRepositoryPort(abc.ABC):
    """Puerto abstracto para obtener contenido textual y metadatos de chunks desde la persistencia."""

    @abc.abstractmethod
    async def get_chunk_contents_by_company(self, company_id: uuid.UUID) -> Dict[str, Dict[str, Any]]:
        """
        Obtiene un diccionario de {chunk_id: {'content': str, 'document_id': str, 'file_name': str}} para una compañía.
        """
        raise NotImplementedError

    @abc.abstractmethod
    async def get_chunk_contents_by_ids(self, chunk_ids: List[str]) -> Dict[str, Dict[str, Any]]:
        """
        Obtiene un diccionario de {chunk_id: {'content': str, 'document_id': str, 'file_name': str}} para una lista de IDs.
        Los chunk_ids aquí son los embedding_id de Milvus.
        """
        raise NotImplementedError
```

## File: `app/application/ports/retrieval_ports.py`
```py
# query-service/app/application/ports/retrieval_ports.py
import abc
import uuid # Importar uuid
from typing import List, Tuple
# LLM_REFACTOR_STEP_3: Importar modelo de dominio
from app.domain.models import RetrievedChunk

# Puerto para Retrievers dispersos (como BM25)
class SparseRetrieverPort(abc.ABC):
    """Puerto abstracto para recuperar chunks usando métodos dispersos (keyword-based)."""

    @abc.abstractmethod
    async def search(self, query: str, company_id: uuid.UUID, top_k: int) -> List[Tuple[str, float]]: # MODIFICADO: company_id es uuid.UUID
        """
        Busca chunks relevantes basados en la consulta textual y filtra por compañía.

        Args:
            query: La consulta del usuario.
            company_id: El ID de la compañía para filtrar (UUID).
            top_k: El número máximo de IDs de chunks a devolver.

        Returns:
            Una lista de tuplas (chunk_id: str, score: float).
            Nota: Este puerto solo devuelve IDs y scores, el contenido se recupera por separado.
        """
        raise NotImplementedError

# Puerto para Rerankers
class RerankerPort(abc.ABC):
    """Puerto abstracto para reordenar chunks recuperados."""

    @abc.abstractmethod
    async def rerank(self, query: str, chunks: List[RetrievedChunk]) -> List[RetrievedChunk]:
        """
        Reordena una lista de chunks basada en la consulta.

        Args:
            query: La consulta original del usuario.
            chunks: Lista de objetos RetrievedChunk (deben tener contenido).

        Returns:
            Una lista reordenada de RetrievedChunk.
        """
        raise NotImplementedError

# Puerto para Filtros de Diversidad
class DiversityFilterPort(abc.ABC):
    """Puerto abstracto para aplicar filtros de diversidad a los chunks."""

    @abc.abstractmethod
    async def filter(self, chunks: List[RetrievedChunk], k_final: int) -> List[RetrievedChunk]:
        """
        Filtra una lista de chunks para maximizar diversidad y relevancia.

        Args:
            chunks: Lista de RetrievedChunk (generalmente ya reordenados).
            k_final: El número deseado de chunks finales.

        Returns:
            Una sublista filtrada de RetrievedChunk.
        """
        raise NotImplementedError
```

## File: `app/application/ports/vector_store_port.py`
```py
# query-service/app/application/ports/vector_store_port.py
import abc
from typing import List
# LLM_REFACTOR_STEP_2: Importar el objeto de dominio
from app.domain.models import RetrievedChunk

class VectorStorePort(abc.ABC):
    """Puerto abstracto para interactuar con una base de datos vectorial."""

    @abc.abstractmethod
    async def search(self, embedding: List[float], company_id: str, top_k: int) -> List[RetrievedChunk]:
        """
        Busca chunks relevantes basados en un embedding y filtra por compañía.

        Args:
            embedding: El vector de embedding de la consulta.
            company_id: El ID de la compañía para filtrar los resultados.
            top_k: El número máximo de chunks a devolver.

        Returns:
            Una lista de objetos RetrievedChunk relevantes.

        Raises:
            ConnectionError: Si falla la comunicación con la base de datos vectorial.
            Exception: Para otros errores inesperados.
        """
        raise NotImplementedError
```

## File: `app/application/use_cases/__init__.py`
```py

```

## File: `app/application/use_cases/ask_query/__init__.py`
```py
"""Utilities supporting the AskQuery use case."""

```

## File: `app/application/use_cases/ask_query/config_types.py`
```py
from dataclasses import dataclass

@dataclass
class PromptBudgetConfig:
    llm_context_window: int
    direct_rag_token_limit: int
    map_prompt_ratio: float
    reduce_prompt_ratio: float

@dataclass
class MapReduceConfig:
    enabled: bool
    chunk_batch_size: int
    tiktoken_encoding: str

@dataclass
class RetrievalConfig:
    top_k: int
    bm25_enabled: bool
    reranker_enabled: bool
    diversity_enabled: bool
    hybrid_alpha: float
    diversity_lambda: float
    max_context_chunks: int
```

## File: `app/application/use_cases/ask_query/pipeline.py`
```py
from abc import ABC, abstractmethod
from typing import Any, Dict, List
import structlog
import time

log = structlog.get_logger()

class PipelineStep(ABC):
    """Abstract base class for a step in the RAG pipeline."""
    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    async def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Executes the pipeline step. Reads/writes to context."""
        pass

class RAGPipeline:
    """Orchestrates the execution of a sequence of PipelineSteps."""
    def __init__(self, steps: List[PipelineStep]):
        self.steps = steps

    async def run(self, initial_context: Dict[str, Any]) -> Dict[str, Any]:
        context = initial_context.copy()
        req_id = context.get("request_id", "unknown")
        pipeline_log = log.bind(pipeline_exec_id=req_id)
        
        total_start = time.perf_counter()
        
        for step in self.steps:
            step_start = time.perf_counter()
            try:
                # pipeline_log.debug(f"Starting step: {step.name}")
                context = await step.execute(context)
                duration = (time.perf_counter() - step_start) * 1000
                # pipeline_log.debug(f"Completed step: {step.name}", duration_ms=duration)
            except Exception as e:
                pipeline_log.error(f"Error in step {step.name}: {str(e)}")
                raise e
        
        total_duration = (time.perf_counter() - total_start) * 1000
        pipeline_log.info("Pipeline execution completed", total_duration_ms=total_duration)
        return context
```

## File: `app/application/use_cases/ask_query/prompt_service.py`
```py
import asyncio
import os
from enum import Enum
from typing import Dict, List, Optional, Any
import structlog
from haystack.components.builders.prompt_builder import PromptBuilder
from haystack import Document
from app.core.config import settings
from app.domain.models import RetrievedChunk

log = structlog.get_logger(__name__)

class PromptType(Enum):
    RAG = "rag"
    GENERAL = "general" # Still kept as requested in refactor notes check but unused logic could be deprecated
    MAP = "map"
    REDUCE = "reduce"

class PromptService:
    def __init__(self) -> None:
        self._builders: Dict[PromptType, PromptBuilder] = {
            PromptType.RAG: self._load_builder(settings.RAG_PROMPT_TEMPLATE_PATH),
            PromptType.GENERAL: self._load_builder(settings.GENERAL_PROMPT_TEMPLATE_PATH),
            PromptType.MAP: self._load_builder(settings.MAP_PROMPT_TEMPLATE_PATH),
            PromptType.REDUCE: self._load_builder(settings.REDUCE_PROMPT_TEMPLATE_PATH),
        }

    @staticmethod
    def _load_builder(template_path: str) -> PromptBuilder:
        if not os.path.exists(template_path):
            log.error(f"Prompt template missing at {template_path}")
            raise FileNotFoundError(f"Prompt template file not found at {template_path}")
        
        with open(template_path, "r", encoding="utf-8") as f:
            content = f.read()
        
        if not content.strip():
            raise ValueError(f"Prompt template is empty: {template_path}")
            
        return PromptBuilder(template=content)

    async def build_rag_prompt(self, query: str, chunks: List[RetrievedChunk], chat_history: str) -> str:
        return await self._run_builder(PromptType.RAG, query=query, documents=self._to_haystack_docs(chunks), chat_history=chat_history)

    async def build_map_prompt(self, query: str, chunks: List[RetrievedChunk], index_offset: int, total: int) -> str:
        # Override dict structure matching the Map template requirements
        data = {
            "original_query": query,
            "documents": self._to_haystack_docs(chunks),
            "document_index": index_offset,
            "total_documents": total
        }
        builder = self._builders[PromptType.MAP]
        result = await asyncio.to_thread(builder.run, **data)
        return result.get("prompt")

    async def build_reduce_prompt(self, query: str, map_results: str, original_chunks: List[RetrievedChunk], chat_history: str) -> str:
        data = {
            "original_query": query,
            "mapped_responses": map_results,
            "original_documents_for_citation": self._to_haystack_docs(original_chunks),
            "chat_history": chat_history
        }
        builder = self._builders[PromptType.REDUCE]
        result = await asyncio.to_thread(builder.run, **data)
        return result.get("prompt")

    async def _run_builder(self, prompt_type: PromptType, **kwargs) -> str:
        builder = self._builders[prompt_type]
        result = await asyncio.to_thread(builder.run, **kwargs)
        return result.get("prompt")

    def _to_haystack_docs(self, chunks: List[RetrievedChunk]) -> List[Document]:
        docs = []
        for chunk in chunks:
            meta = dict(chunk.metadata or {})
            meta.update({
                "file_name": chunk.file_name,
                "document_id": chunk.document_id,
                "company_id": chunk.company_id,
                "title": chunk.metadata.get("title") if chunk.metadata else None,
                "page": chunk.metadata.get("page") if chunk.metadata else None
            })
            docs.append(Document(id=chunk.id, content=chunk.content or "", score=chunk.score, meta=meta))
        return docs
```

## File: `app/application/use_cases/ask_query/steps.py`
```py
import asyncio
import uuid
import structlog
from typing import Any, Dict, List, Optional, Tuple

from app.application.use_cases.ask_query.pipeline import PipelineStep
from app.application.use_cases.ask_query.config_types import RetrievalConfig, MapReduceConfig, PromptBudgetConfig
from app.application.use_cases.ask_query.token_accountant import TokenAccountant
from app.application.use_cases.ask_query.prompt_service import PromptService
from app.domain.models import RetrievedChunk
from app.application.ports import (
    VectorStorePort, SparseRetrieverPort, RerankerPort, 
    LLMPort, EmbeddingPort, ChunkContentRepositoryPort, DiversityFilterPort
)

log = structlog.get_logger()

class EmbeddingStep(PipelineStep):
    def __init__(self, embedding_adapter: EmbeddingPort):
        super().__init__("EmbeddingStep")
        self.embedding = embedding_adapter

    async def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        query = context["query"]
        context["pipeline_stages_used"].append("query_embedding")
        context["query_embedding"] = await self.embedding.embed_query(query)
        return context

class RetrievalStep(PipelineStep):
    def __init__(self, vector_store: VectorStorePort, sparse_retriever: Optional[SparseRetrieverPort], config: RetrievalConfig):
        super().__init__("RetrievalStep")
        self.vector = vector_store
        self.sparse = sparse_retriever
        self.config = config

    async def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        query, company_id = context["query"], context["company_id"]
        embedding = context["query_embedding"]
        top_k = context.get("top_k") or self.config.top_k
        
        context["pipeline_stages_used"].append("dense_retrieval")
        dense_task = self.vector.search(embedding, str(company_id), top_k)
        
        # Create a task that returns an empty list immediately if sparse is disabled
        # This uses a lambda to match the expected return signature of a task
        sparse_task = asyncio.create_task(self._noop_sparse())

        if self.config.bm25_enabled and self.sparse:
            context["pipeline_stages_used"].append("sparse_retrieval")
            sparse_task = self.sparse.search(query, company_id, top_k)
            
        # Use return_exceptions=True to prevent one failure from killing the other
        results = await asyncio.gather(dense_task, sparse_task, return_exceptions=True)
        
        dense_res = results[0]
        sparse_res = results[1]

        # Error handling for Dense
        if isinstance(dense_res, Exception):
            log.error("Dense retrieval failed", error=str(dense_res))
            dense_res = []
        
        # Error handling for Sparse
        if isinstance(sparse_res, Exception):
            log.error("Sparse retrieval failed", error=str(sparse_res))
            sparse_res = []
            
        context["dense_chunks"] = dense_res
        context["sparse_results"] = sparse_res # List[Tuple[str, float]]
        return context
    
    async def _noop_sparse(self):
        return []

class FusionStep(PipelineStep):
    def __init__(self):
        super().__init__("FusionStep")

    async def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        dense: List[RetrievedChunk] = context["dense_chunks"]
        sparse: List[Tuple[str, float]] = context["sparse_results"]
        
        # Map chunks by ID
        merged: Dict[str, RetrievedChunk] = {c.id: c for c in dense}
        
        # Merge sparse (creating placeholders if needed)
        for doc_id, score in sparse:
            if doc_id not in merged:
                # Crear un placeholder. Necesitamos asegurar que document_id y company_id tengan tipos correctos
                # SparseSearchServiceClient devuelve chunk_id que es el embedding_id.
                merged[doc_id] = RetrievedChunk(
                    id=doc_id, 
                    content=None, # Needs fetch in next step
                    score=score, 
                    metadata={"retrieval_source": "sparse_only"},
                    company_id=str(context["company_id"])
                )
        
        context["fused_chunks"] = list(merged.values())
        return context

class ContentFetchStep(PipelineStep):
    def __init__(self, repo: ChunkContentRepositoryPort):
        super().__init__("ContentFetchStep")
        self.repo = repo

    async def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        chunks: List[RetrievedChunk] = context["fused_chunks"]
        missing_ids = [c.id for c in chunks if not c.content]
        
        if missing_ids:
            try:
                fetched_map = await self.repo.get_chunk_contents_by_ids(missing_ids)
                valid_chunks = []
                for c in chunks:
                    if c.content:
                        valid_chunks.append(c)
                    elif c.id in fetched_map:
                        data = fetched_map[c.id]
                        c.content = data["content"]
                        c.document_id = data.get("document_id")
                        c.file_name = data.get("file_name")
                        if "metadata" in data:
                             c.metadata.update(data["metadata"] or {})
                        valid_chunks.append(c)
                    else:
                        log.warning(f"Content not found for chunk {c.id}, dropping from results.")
                context["fused_chunks"] = valid_chunks
            except Exception as e:
                log.error("Content fetch failed", error=str(e))
                # If fetch fails, we must drop chunks without content to avoid LLM errors
                context["fused_chunks"] = [c for c in chunks if c.content]
        
        return context

class RerankStep(PipelineStep):
    def __init__(self, reranker: Optional[RerankerPort], config: RetrievalConfig):
        super().__init__("RerankStep")
        self.reranker = reranker
        self.config = config

    async def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        chunks = context["fused_chunks"]
        if self.config.reranker_enabled and self.reranker and chunks:
            context["pipeline_stages_used"].append("reranking")
            try:
                chunks = await self.reranker.rerank(context["query"], chunks)
            except Exception as e:
                log.error(f"Reranking failed: {e}. Falling back to fusion results.")
        context["reranked_chunks"] = chunks
        return context

class FilterStep(PipelineStep):
    def __init__(self, diver_filter: Optional[DiversityFilterPort], config: RetrievalConfig):
        super().__init__("FilterStep")
        self.diver_filter = diver_filter
        self.config = config

    async def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        chunks = context["reranked_chunks"]
        limit = self.config.max_context_chunks
        
        if self.config.diversity_enabled and self.diver_filter:
            context["pipeline_stages_used"].append("mmr_filter")
            chunks = await self.diver_filter.filter(chunks, limit)
        else:
            chunks = chunks[:limit]
            
        context["final_chunks"] = chunks
        return context

class DirectGenerationStep(PipelineStep):
    def __init__(self, llm: LLMPort, prompt_service: PromptService, token_accountant: TokenAccountant, config: PromptBudgetConfig):
        super().__init__("DirectGenerationStep")
        self.llm = llm
        self.prompts = prompt_service
        self.accountant = token_accountant
        self.config = config

    async def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        chunks = context["final_chunks"]
        query = context["query"]
        history = context.get("chat_history", "")
        
        context["pipeline_stages_used"].append("direct_rag")
        
        # Truncation logic to fit budget for small model
        valid_chunks = []
        # Base overhead including system prompt + query + history
        current_tokens = 300 + self.accountant.count_tokens_for_text(query + history)
        
        for chunk in chunks:
            ct = self.accountant.count_tokens_for_text(chunk.content)
            if current_tokens + ct > self.config.direct_rag_token_limit:
                break
            valid_chunks.append(chunk)
            current_tokens += ct
            
        context["final_used_chunks"] = valid_chunks
        prompt = await self.prompts.build_rag_prompt(query, valid_chunks, history)
        
        # Granite model needs schema guidance, provided via prompt text mostly, but adapter enforces json_object
        context["llm_response_raw"] = await self.llm.generate(prompt, response_pydantic_schema=None) 
        # Passing None schema here because prompts already have strict JSON structure and Adapter adds json_object format. 
        # If we pass schema, Adapter will try to clean it and inject it, which is good but Granite is safer with text instruction + json mode.
        
        context["generation_mode"] = "direct_rag"
        return context

class MapReduceGenerationStep(PipelineStep):
    def __init__(self, llm: LLMPort, prompt_service: PromptService, map_config: MapReduceConfig):
        super().__init__("MapReduceGenerationStep")
        self.llm = llm
        self.prompts = prompt_service
        self.config = map_config
        self._sem = asyncio.Semaphore(self.config.chunk_batch_size)

    async def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        chunks = context["final_chunks"]
        query = context["query"]
        
        context["pipeline_stages_used"].append("map_reduce")
        
        # Batching
        batch_size = 3 
        batches = [chunks[i:i + batch_size] for i in range(0, len(chunks), batch_size)]
        
        total_docs = len(chunks)
        doc_idx = 0
        
        async def _process_batch(b_chunks, current_idx):
            async with self._sem: 
                p = await self.prompts.build_map_prompt(query, b_chunks, current_idx, total_docs)
                return await self.llm.generate(p)

        tasks = []
        for b in batches:
            tasks.append(_process_batch(b, doc_idx))
            doc_idx += len(b)
            
        raw_map_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        valid_maps = []
        for res in raw_map_results:
            if isinstance(res, str):
                # Check for the specific rejection string from our optimized prompt
                if "NO_RELEVANT_INFO" not in res and len(res.strip()) > 5:
                    valid_maps.append(res)
            elif isinstance(res, Exception):
                log.error("Map batch failed", error=str(res))
        
        if not valid_maps:
             # If all map phases failed or found no info, fall back to direct RAG with top 2 chunks just to give an answer
             log.warning("MapReduce found no relevant info. Falling back to direct generation with minimal context.")
             context["generation_mode"] = "map_reduce_fallback"
             # Re-use chunks for Direct RAG flow but minimal
             context["final_chunks"] = chunks[:2] 
             # We cannot easily jump steps here, so we handle empty maps in Reduce prompt or return default
             combined_map = "No se encontró información relevante detallada en los fragmentos analizados."
        else:
             combined_map = "\n".join(valid_maps)

        reduce_prompt = await self.prompts.build_reduce_prompt(query, combined_map, chunks, context.get("chat_history", ""))
        
        context["llm_response_raw"] = await self.llm.generate(reduce_prompt)
        context["generation_mode"] = "map_reduce"
        context["final_used_chunks"] = chunks 
        return context

class AdaptiveGenerationStep(PipelineStep):
    """Decides between Direct and MapReduce based on token budget."""
    def __init__(self, direct: DirectGenerationStep, mapred: MapReduceGenerationStep, accountant: TokenAccountant, budget: PromptBudgetConfig, map_config: MapReduceConfig):
        super().__init__("AdaptiveGenerationStep")
        self.direct = direct
        self.mapred = mapred
        self.accountant = accountant
        self.budget = budget
        self.map_config = map_config

    async def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        chunks = context["final_chunks"]
        # Calculate based on retrieved chunks
        analysis = self.accountant.calculate_token_usage(chunks)
        total_tokens = analysis.total_tokens
        
        log.info(f"Token analysis: {total_tokens} tokens in {len(chunks)} chunks. Limit: {self.budget.direct_rag_token_limit}")

        if self.map_config.enabled and total_tokens > self.budget.direct_rag_token_limit:
            return await self.mapred.execute(context)
        else:
            return await self.direct.execute(context)
```

## File: `app/application/use_cases/ask_query/token_accountant.py`
```py
from __future__ import annotations
import hashlib
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple, Union
import structlog
import tiktoken
from app.core.config import settings
from app.domain.models import RetrievedChunk
from .types import TokenAnalysis

log = structlog.get_logger(__name__)

class _BoundedCache(OrderedDict):
    def __init__(self, max_size: int) -> None:
        super().__init__()
        self._max_size = max_size

    def __setitem__(self, key, value):
        if key not in self and len(self) >= self._max_size:
            self.popitem(last=False)
        super().__setitem__(key, value)
        self.move_to_end(key)

class TokenAccountant:
    """Centralized token counting logic."""
    def __init__(self, cache_max_size: int = 1000) -> None:
        self._encoding: Optional[tiktoken.Encoding] = None
        self._token_cache: _BoundedCache = _BoundedCache(cache_max_size)
        self._encoding_name = settings.TIKTOKEN_ENCODING_NAME
        self._cache_max_size = cache_max_size

    def _get_encoding(self) -> tiktoken.Encoding:
        if self._encoding is None:
            try:
                self._encoding = tiktoken.get_encoding(self._encoding_name)
            except Exception as exc:
                log.error("Failed to load tiktoken encoding; fallback to gpt2", error=str(exc))
                self._encoding = tiktoken.get_encoding("gpt2")
        return self._encoding

    def count_tokens_for_chunks(self, chunks: List[RetrievedChunk]) -> Tuple[int, List[int]]:
        if not chunks: return 0, []
        encoding = self._get_encoding()
        per_chunk = []
        for chunk in chunks:
            content = chunk.content or ""
            if not content.strip():
                per_chunk.append(0)
                continue
            chash = hashlib.md5(content.encode("utf-8")).hexdigest()
            cnt = self._token_cache.get(chash)
            if cnt is None:
                cnt = len(encoding.encode(content))
                self._token_cache[chash] = cnt
            per_chunk.append(cnt)
        return sum(per_chunk), per_chunk

    def count_tokens_for_text(self, text: Optional[str]) -> int:
        if not text: return 0
        return len(self._get_encoding().encode(text))

    def calculate_token_usage(self, chunks: List[RetrievedChunk]) -> TokenAnalysis:
        total, per_chunk = self.count_tokens_for_chunks(chunks)
        return TokenAnalysis(
            total_tokens=total,
            per_chunk_tokens=per_chunk,
            cache_info={"size": len(self._token_cache)}
        )
```

## File: `app/application/use_cases/ask_query/types.py`
```py
from __future__ import annotations
import uuid
from dataclasses import dataclass
from typing import Dict, List, Optional, Union, Any
from app.domain.models import ChatMessage, RetrievedChunk

@dataclass
class TokenAnalysis:
    total_tokens: int
    per_chunk_tokens: List[int]
    cache_info: Dict[str, Union[int, float]]

@dataclass
class RetrievalOutcome:
    chunks: List[RetrievedChunk]
    stages: List[str]
```

## File: `app/application/use_cases/ask_query_use_case.py`
```py
import uuid
import structlog
import json
from typing import List, Tuple, Optional
from fastapi import HTTPException

from app.core.config import settings
from app.domain.models import ChatMessage, RetrievedChunk, RespuestaEstructurada
from app.application.ports import ChatRepositoryPort, LogRepositoryPort, ChunkContentRepositoryPort, LLMPort, VectorStorePort, SparseRetrieverPort, EmbeddingPort, RerankerPort, DiversityFilterPort

from app.application.use_cases.ask_query.config_types import PromptBudgetConfig, MapReduceConfig, RetrievalConfig
from app.application.use_cases.ask_query.token_accountant import TokenAccountant
from app.application.use_cases.ask_query.prompt_service import PromptService
from app.application.use_cases.ask_query.pipeline import RAGPipeline
from app.application.use_cases.ask_query.steps import (
    EmbeddingStep, RetrievalStep, FusionStep, ContentFetchStep, RerankStep, FilterStep,
    DirectGenerationStep, MapReduceGenerationStep, AdaptiveGenerationStep
)

log = structlog.get_logger(__name__)

class AskQueryUseCase:
    def __init__(
        self,
        chat_repo: ChatRepositoryPort,
        log_repo: LogRepositoryPort,
        chunk_content_repo: ChunkContentRepositoryPort,
        vector_store: VectorStorePort,
        sparse_retriever: Optional[SparseRetrieverPort],
        reranker: Optional[RerankerPort],
        embedding_adapter: EmbeddingPort,
        diversity_filter: Optional[DiversityFilterPort],
        llm: LLMPort,
        http_client: Any = None # Deprecated but kept for signature compat if needed, steps use injected ports
    ):
        self.chat_repo = chat_repo
        self.log_repo = log_repo
        
        # Helper Services
        self.token_accountant = TokenAccountant()
        self.prompt_service = PromptService()
        
        # Config Objects from Global Settings
        self.budget_config = PromptBudgetConfig(
            llm_context_window=settings.LLM_CONTEXT_WINDOW_TOKENS,
            direct_rag_token_limit=settings.DIRECT_RAG_TOKEN_LIMIT,
            map_prompt_ratio=0.7, reduce_prompt_ratio=0.8
        )
        self.map_config = MapReduceConfig(
            enabled=settings.MAPREDUCE_ENABLED,
            chunk_batch_size=settings.MAPREDUCE_CHUNK_BATCH_SIZE,
            tiktoken_encoding=settings.TIKTOKEN_ENCODING_NAME
        )
        self.retrieval_config = RetrievalConfig(
            top_k=settings.RETRIEVER_TOP_K,
            bm25_enabled=settings.BM25_ENABLED,
            reranker_enabled=settings.RERANKER_ENABLED,
            diversity_enabled=settings.DIVERSITY_FILTER_ENABLED,
            hybrid_alpha=settings.HYBRID_FUSION_ALPHA,
            diversity_lambda=settings.QUERY_DIVERSITY_LAMBDA,
            max_context_chunks=settings.MAX_CONTEXT_CHUNKS
        )
        
        # Initialize Pipeline Steps
        self.embed_step = EmbeddingStep(embedding_adapter)
        self.retrieval_step = RetrievalStep(vector_store, sparse_retriever, self.retrieval_config)
        self.fusion_step = FusionStep()
        self.fetch_step = ContentFetchStep(chunk_content_repo)
        self.rerank_step = RerankStep(reranker, self.retrieval_config)
        self.filter_step = FilterStep(diversity_filter, self.retrieval_config)
        
        # Generation Strategies
        self.direct_gen = DirectGenerationStep(llm, self.prompt_service, self.token_accountant, self.budget_config)
        self.mapred_gen = MapReduceGenerationStep(llm, self.prompt_service, self.map_config)
        self.adaptive_gen = AdaptiveGenerationStep(self.direct_gen, self.mapred_gen, self.token_accountant, self.budget_config, self.map_config)

    async def execute(
        self, query: str, company_id: uuid.UUID, user_id: uuid.UUID,
        chat_id: Optional[uuid.UUID] = None, top_k: Optional[int] = None
    ) -> Tuple[str, List[RetrievedChunk], Optional[uuid.UUID], uuid.UUID]:
        
        # 1. Chat Initialization
        final_chat_id, chat_history_str = await self._init_chat(chat_id, user_id, company_id, query)
        
        # 2. Handle Greeting (Optimization)
        if self._is_greeting(query):
            return await self._handle_greeting(query, final_chat_id, user_id, company_id)
        
        # 3. Build Context
        context = {
            "query": query,
            "company_id": company_id,
            "user_id": user_id,
            "chat_history": chat_history_str,
            "top_k": top_k,
            "request_id": str(uuid.uuid4()),
            "pipeline_stages_used": []
        }
        
        # 4. Run Pipeline
        pipeline = RAGPipeline([
            self.embed_step,
            self.retrieval_step,
            self.fusion_step,
            self.fetch_step,
            self.rerank_step,
            self.filter_step,
            self.adaptive_gen
        ])
        
        try:
            result_context = await pipeline.run(context)
        except Exception as e:
            log.error("Pipeline execution failed", error=str(e))
            # Fallback or generic error handling could go here
            raise HTTPException(status_code=500, detail="Error generating response.")

        # 5. Post-Process Response (Parse JSON, Log, Save DB)
        # Keep complex persistence logic here to keep pipeline pure
        raw_json = result_context.get("llm_response_raw", "")
        used_chunks = result_context.get("final_used_chunks", [])
        
        answer, chunks_for_api, log_id = await self._process_and_save_response(
            raw_json, query, company_id, user_id, final_chat_id, used_chunks, 
            result_context.get("pipeline_stages_used")
        )
        
        return answer, chunks_for_api, log_id, final_chat_id

    # --- Helpers (kept private to keep UseCase clean) ---
    
    async def _init_chat(self, chat_id, user_id, company_id, query) -> Tuple[uuid.UUID, str]:
        history_str = ""
        if chat_id:
            if not await self.chat_repo.check_chat_ownership(chat_id, user_id, company_id):
                raise HTTPException(status_code=403, detail="Chat access denied.")
            final_id = chat_id
            msgs = await self.chat_repo.get_chat_messages(chat_id, user_id, company_id, limit=settings.MAX_CHAT_HISTORY_MESSAGES)
            history_str = self._format_history(msgs)
        else:
            final_id = await self.chat_repo.create_chat(user_id, company_id, title=f"Chat: {query[:30]}...")
            
        await self.chat_repo.save_message(final_id, 'user', content=query)
        return final_id, history_str

    def _is_greeting(self, query: str) -> bool:
        import re
        return bool(re.match(r"^\s*(hola|hello|hi|buenos días)\s*[\.,!?]*\s*$", query, re.IGNORECASE))

    async def _handle_greeting(self, query, chat_id, user_id, company_id):
        answer = "¡Hola! ¿En qué puedo ayudarte hoy con tus documentos?"
        await self.chat_repo.save_message(chat_id, 'assistant', content=answer)
        lid = await self.log_repo.log_query_interaction(user_id, company_id, query, answer, [], chat_id=chat_id)
        return answer, [], lid, chat_id

    def _format_history(self, msgs: List[ChatMessage]) -> str:
        lines = []
        for m in reversed(msgs):
            role = "Usuario" if m.role == 'user' else "Atenex"
            lines.append(f"{role}: {m.content}")
        return "\n".join(lines)

    async def _process_and_save_response(self, raw_json, query, company_id, user_id, chat_id, original_chunks, stages):
        # Parsing logic similar to previous implementation, handling clean JSON from LlamaCpp
        try:
            # Cleanup JSON string if LlamaCpp output some artifacts
            clean_json_str = raw_json.strip()
            if clean_json_str.startswith("```json"): clean_json_str = clean_json_str[7:-3]
            
            struct_resp = RespuestaEstructurada.model_validate_json(clean_json_str)
            answer = struct_resp.respuesta_detallada
            
            # Map citations
            api_chunks = []
            chunk_map = {c.id: c for c in original_chunks}
            sources_for_db = []
            
            for cit in struct_resp.fuentes_citadas:
                if cit.id_documento and cit.id_documento in chunk_map:
                    c = chunk_map[cit.id_documento]
                    c.cita_tag = cit.cita_tag
                    api_chunks.append(c)
                    sources_for_db.append(cit.model_dump())
            
            if not api_chunks and original_chunks:
                 # Fallback if model didn't cite but we used chunks
                 api_chunks = original_chunks[:settings.NUM_SOURCES_TO_SHOW]

            await self.chat_repo.save_message(chat_id, 'assistant', answer, sources=sources_for_db)
            
            log_meta = {"pipeline_stages": stages, "model": settings.LLM_MODEL_NAME}
            lid = await self.log_repo.log_query_interaction(
                user_id, company_id, query, answer, 
                [c.model_dump() for c in api_chunks], metadata=log_meta, chat_id=chat_id
            )
            return answer, api_chunks, lid

        except Exception as e:
            log.error("Failed to parse LLM response", raw=raw_json, error=str(e))
            fallback = "Lo siento, hubo un error procesando la respuesta del asistente."
            await self.chat_repo.save_message(chat_id, 'assistant', fallback)
            return fallback, [], None
```

## File: `app/core/__init__.py`
```py

```

## File: `app/core/config.py`
```py
# query-service/app/core/config.py
import logging
import os
import sys
from pathlib import Path 
from typing import Optional, List, Any, Dict
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import AnyHttpUrl, SecretStr, Field, field_validator, ValidationInfo, ValidationError

# ... (Defaults previos sin cambios hasta LLM) ...
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
RERANKER_SERVICE_K8S_URL_DEFAULT = "http://reranker-service.nyro-develop.svc.cluster.local:80"
SPARSE_SEARCH_SERVICE_K8S_URL_DEFAULT = "http://sparse-search-service.nyro-develop.svc.cluster.local:80"

PROMPT_DIR = Path(__file__).resolve().parent.parent / "prompts"
DEFAULT_RAG_PROMPT_TEMPLATE_PATH = str(PROMPT_DIR / "rag_template_gemini_v2.txt")
DEFAULT_GENERAL_PROMPT_TEMPLATE_PATH = str(PROMPT_DIR / "general_template_gemini_v2.txt")
DEFAULT_MAP_PROMPT_TEMPLATE_PATH = str(PROMPT_DIR / "map_prompt_template.txt")
DEFAULT_REDUCE_PROMPT_TEMPLATE_PATH = str(PROMPT_DIR / "reduce_prompt_template_v2.txt")

DEFAULT_EMBEDDING_DIMENSION = 1536
LLM_API_BASE_URL_DEFAULT = "http://192.168.1.43:9090" # Host external IP
LLM_MODEL_NAME_DEFAULT = "granite-3.2-2b-instruct-q4_k_m.gguf"
LLM_MAX_OUTPUT_TOKENS_DEFAULT = 1024

# Pipeline defaults tuned for stability with 2B model
DEFAULT_RETRIEVER_TOP_K = 50
DEFAULT_BM25_ENABLED = True
DEFAULT_RERANKER_ENABLED = True
DEFAULT_DIVERSITY_FILTER_ENABLED = False
DEFAULT_MAX_CONTEXT_CHUNKS = 10 # Conservative
DEFAULT_HYBRID_ALPHA = 0.5
DEFAULT_DIVERSITY_LAMBDA = 0.5
DEFAULT_MAX_CHAT_HISTORY_MESSAGES = 6
DEFAULT_NUM_SOURCES_TO_SHOW = 5
DEFAULT_MAX_TOKENS_PER_CHUNK = 800
DEFAULT_MAX_CHARS_PER_CHUNK = 3500
DEFAULT_MAPREDUCE_ENABLED = True
DEFAULT_MAPREDUCE_CHUNK_BATCH_SIZE = 3
DEFAULT_TIKTOKEN_ENCODING_NAME = "cl100k_base"

DEFAULT_LLM_CONTEXT_WINDOW_TOKENS = 16000
DEFAULT_DIRECT_RAG_TOKEN_LIMIT = 8000 # Conservative switch to MapReduce

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="QUERY_", case_sensitive=True, extra="ignore")
    
    PROJECT_NAME: str = "Atenex Query Service"
    API_V1_STR: str = "/api/v1"
    LOG_LEVEL: str = "INFO"

    # DB
    POSTGRES_USER: str = Field(default=POSTGRES_K8S_USER_DEFAULT)
    POSTGRES_PASSWORD: SecretStr
    POSTGRES_SERVER: str = Field(default=POSTGRES_K8S_HOST_DEFAULT)
    POSTGRES_PORT: int = Field(default=POSTGRES_K8S_PORT_DEFAULT)
    POSTGRES_DB: str = Field(default=POSTGRES_K8S_DB_DEFAULT)

    # Milvus
    ZILLIZ_API_KEY: SecretStr
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

    # External Services
    EMBEDDING_DIMENSION: int = Field(default=DEFAULT_EMBEDDING_DIMENSION)
    EMBEDDING_SERVICE_URL: AnyHttpUrl = Field(default_factory=lambda: AnyHttpUrl(EMBEDDING_SERVICE_K8S_URL_DEFAULT))
    EMBEDDING_CLIENT_TIMEOUT: int = Field(default=30)

    LLM_API_BASE_URL: AnyHttpUrl = Field(default=LLM_API_BASE_URL_DEFAULT)
    LLM_MODEL_NAME: str = Field(default=LLM_MODEL_NAME_DEFAULT)
    LLM_MAX_OUTPUT_TOKENS: int = Field(default=LLM_MAX_OUTPUT_TOKENS_DEFAULT)
    GEMINI_API_KEY: Optional[SecretStr] = None

    RERANKER_ENABLED: bool = Field(default=DEFAULT_RERANKER_ENABLED)
    RERANKER_SERVICE_URL: AnyHttpUrl = Field(default_factory=lambda: AnyHttpUrl(RERANKER_SERVICE_K8S_URL_DEFAULT))
    RERANKER_CLIENT_TIMEOUT: int = Field(default=30)
    
    BM25_ENABLED: bool = Field(default=DEFAULT_BM25_ENABLED)
    SPARSE_SEARCH_SERVICE_URL: AnyHttpUrl = Field(default_factory=lambda: AnyHttpUrl(SPARSE_SEARCH_SERVICE_K8S_URL_DEFAULT))
    SPARSE_SEARCH_CLIENT_TIMEOUT: int = Field(default=30)

    # Pipeline Logic
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
    LLM_CONTEXT_WINDOW_TOKENS: int = Field(default=DEFAULT_LLM_CONTEXT_WINDOW_TOKENS)
    DIRECT_RAG_TOKEN_LIMIT: int = Field(default=DEFAULT_DIRECT_RAG_TOKEN_LIMIT)
    TIKTOKEN_ENCODING_NAME: str = Field(default=DEFAULT_TIKTOKEN_ENCODING_NAME)

    # Paths
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
        if not v: raise ValueError("Secret cannot be empty.")
        return v

try:
    settings = Settings()
except ValidationError as e:
    print(f"FATAL: Configuration Validation Error: {e}")
    sys.exit(1)
```

## File: `app/core/logging_config.py`
```py
# ./app/core/logging_config.py
import logging
import sys
import structlog
from app.core.config import settings
import os

def setup_logging():
    """Configura el logging estructurado con structlog."""

    # Determine if running inside a known runner like Gunicorn/Uvicorn if needed
    # is_gunicorn_worker = "gunicorn" in sys.argv[0]

    shared_processors = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
    ]

    # Add caller info only in debug mode for performance
    if settings.LOG_LEVEL == "DEBUG":
         shared_processors.append(structlog.processors.CallsiteParameterAdder(
             {
                 structlog.processors.CallsiteParameter.FILENAME,
                 structlog.processors.CallsiteParameter.LINENO,
             }
         ))

    # Configure structlog processors for eventual output
    structlog.configure(
        processors=shared_processors + [
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    # Configure the formatter for stdlib logging handler
    formatter = structlog.stdlib.ProcessorFormatter(
        # These run ONCE per log event before formatting
        foreign_pre_chain=shared_processors,
        # These run on the final structured dict before rendering
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            structlog.processors.JSONRenderer(), # Render logs as JSON
        ],
    )

    # Configure root logger handler (usually StreamHandler to stdout)
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    root_logger = logging.getLogger()

    # Avoid adding handler multiple times if already configured (e.g., by Uvicorn/Gunicorn)
    # Check if a handler with our specific formatter already exists
    handler_exists = any(isinstance(h, logging.StreamHandler) and isinstance(h.formatter, structlog.stdlib.ProcessorFormatter) for h in root_logger.handlers)
    if not handler_exists:
        # Clear existing handlers if we are sure we want to replace them
        # Be cautious with this in production if other libraries add handlers
        # root_logger.handlers.clear()
        root_logger.addHandler(handler)
        root_logger.setLevel(settings.LOG_LEVEL.upper())
    else:
        # Update level of existing handler if necessary
        root_logger.setLevel(settings.LOG_LEVEL.upper())


    # Silence verbose libraries
    logging.getLogger("uvicorn").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("gunicorn").setLevel(logging.INFO)
    logging.getLogger("asyncpg").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("google.generativeai").setLevel(logging.INFO)
    logging.getLogger("haystack").setLevel(logging.INFO) # Or DEBUG for more Haystack details
    logging.getLogger("milvus_haystack").setLevel(logging.INFO)

    log = structlog.get_logger("query_service") # Specific logger name
    log.info("Logging configured", log_level=settings.LOG_LEVEL)
```

## File: `app/dependencies.py`
```py
# query-service/app/dependencies.py
"""
Centralized dependency functions to avoid circular imports.
"""
from fastapi import HTTPException
from app.application.use_cases.ask_query_use_case import AskQueryUseCase

# These will be set by main.py at startup
ask_query_use_case_instance = None
SERVICE_READY = False

def set_ask_query_use_case_instance(instance, ready_flag):
    global ask_query_use_case_instance, SERVICE_READY
    ask_query_use_case_instance = instance
    SERVICE_READY = ready_flag

def get_ask_query_use_case() -> AskQueryUseCase:
    if not SERVICE_READY or not ask_query_use_case_instance:
        raise HTTPException(status_code=503, detail="Query processing service is not ready. Check startup logs.")
    return ask_query_use_case_instance

```

## File: `app/domain/__init__.py`
```py
# query-service/app/domain/__init__.py
# This package will contain the core business entities and value objects.
# e.g., Chat, Message, RetrievedChunk classes without external dependencies.
```

## File: `app/domain/models.py`
```py
# query-service/app/domain/models.py
import uuid
from pydantic import BaseModel, Field, ConfigDict
from typing import Optional, List, Dict, Any
from datetime import datetime

# Usaremos Pydantic por conveniencia, pero estas son conceptualmente entidades de dominio.

class Chat(BaseModel):
    id: uuid.UUID
    user_id: uuid.UUID
    company_id: uuid.UUID
    title: Optional[str]
    created_at: datetime
    updated_at: datetime

class ChatSummary(BaseModel):
    id: uuid.UUID
    title: Optional[str]
    updated_at: datetime

class ChatMessage(BaseModel):
    id: uuid.UUID
    chat_id: uuid.UUID
    role: str # 'user' or 'assistant'
    content: str
    sources: Optional[List[Dict[str, Any]]] = None 
    created_at: datetime

class RetrievedChunk(BaseModel):
    """Representa un chunk recuperado de una fuente (ej: Milvus)."""
    id: str 
    content: Optional[str] = None 
    score: Optional[float] = None 
    metadata: Dict[str, Any] = Field(default_factory=dict)
    embedding: Optional[List[float]] = None 
    
    document_id: Optional[str] = Field(None, alias="document_id") 
    file_name: Optional[str] = Field(None, alias="file_name")
    company_id: Optional[str] = Field(None, alias="company_id")
    # REFACTOR_5_1: Add cita_tag
    cita_tag: Optional[str] = Field(None, description="La etiqueta de cita [Doc N] usada por el LLM para este chunk.")


    model_config = ConfigDict(populate_by_name=True, arbitrary_types_allowed=True)


    @classmethod
    def from_haystack_document(cls, doc: Any):
        """Convierte un Documento Haystack a un RetrievedChunk."""
        doc_meta = doc.meta or {}
        doc_id_str = str(doc_meta.get("document_id")) if doc_meta.get("document_id") else None
        company_id_str = str(doc_meta.get("company_id")) if doc_meta.get("company_id") else None
        embedding_vector = getattr(doc, 'embedding', None)

        return cls(
            id=str(doc.id),
            content=doc.content,
            score=doc.score,
            metadata=doc_meta,
            embedding=embedding_vector, 
            document_id=doc_id_str,
            file_name=doc_meta.get("file_name"),
            company_id=company_id_str
        )

class QueryLog(BaseModel):
    id: uuid.UUID
    user_id: Optional[uuid.UUID]
    company_id: uuid.UUID
    query: str
    response: str
    metadata: Dict[str, Any]
    chat_id: Optional[uuid.UUID]
    created_at: datetime

# --- Nuevos modelos para Respuesta Estructurada ---
class FuenteCitada(BaseModel):
    id_documento: Optional[str] = None
    nombre_archivo: str = Field(..., description="Nombre del archivo fuente.")
    pagina: Optional[str] = None
    score: Optional[float] = None
    cita_tag: str = Field(..., description="La etiqueta de cita usada en el texto, ej: '[Doc 1]'.")
    
    model_config = ConfigDict(extra='ignore')


class RespuestaEstructurada(BaseModel):
    resumen_ejecutivo: Optional[str] = None
    respuesta_detallada: str = Field(..., description="La respuesta completa y elaborada, incluyendo citas [Doc N] donde corresponda.")
    fuentes_citadas: List[FuenteCitada] # El LLM debe devolverla, incluso vacía
    siguiente_pregunta_sugerida: Optional[str] = None
    
    model_config = ConfigDict(extra='ignore') 

class SparseSearchResultItem(BaseModel):
    """
    Representa un ítem de resultado de búsqueda dispersa devuelto por el sparse-search-service.
    Este modelo se utiliza en el query-service para tipar la respuesta del cliente
    de dicho servicio.
    """
    chunk_id: str = Field(..., description="El ID del chunk (usualmente el embedding_id de la tabla document_chunks).")
    score: float = Field(..., description="El score BM25 asignado al chunk.")
```

## File: `app/infrastructure/__init__.py`
```py
# query-service/app/infrastructure/__init__.py# query-service/app/infrastructure/vectorstores/__init__.py
```

## File: `app/infrastructure/clients/__init__.py`
```py
# query-service/app/infrastructure/clients/__init__.py
from .embedding_service_client import EmbeddingServiceClient

__all__ = ["EmbeddingServiceClient"]
```

## File: `app/infrastructure/clients/embedding_service_client.py`
```py
# query-service/app/infrastructure/clients/embedding_service_client.py
import httpx
import structlog
from typing import List, Dict, Any, Optional
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import json

from app.core.config import settings

log = structlog.get_logger(__name__)

class EmbeddingServiceClient:
    """
    Cliente HTTP para interactuar con el Atenex Embedding Service.
    """
    def __init__(self, base_url: str, timeout: int = settings.HTTP_CLIENT_TIMEOUT):
        self.base_url = base_url.rstrip('/')
        
        # Determinar el endpoint correcto para /embed y /health
        parsed_base_url = httpx.URL(self.base_url)
        
        if "/api/v1/embed" in parsed_base_url.path: 
            self.embed_endpoint = str(parsed_base_url)
            self.health_endpoint = f"{parsed_base_url.scheme}://{parsed_base_url.netloc.decode()}/health"
        elif "/api/v1" in parsed_base_url.path: 
            self.embed_endpoint = f"{self.base_url}/embed" # Asume que /api/v1 está en base_url
            self.health_endpoint = f"{parsed_base_url.scheme}://{parsed_base_url.netloc.decode()}/health"
        else: 
            self.embed_endpoint = f"{self.base_url}/api/v1/embed"
            self.health_endpoint = f"{self.base_url}/health"
            
        self._client = httpx.AsyncClient(timeout=timeout)
        log.info("EmbeddingServiceClient initialized", 
                 base_url_configured=self.base_url, 
                 resolved_embed_endpoint=self.embed_endpoint,
                 resolved_health_endpoint=self.health_endpoint)

    @retry(
        stop=stop_after_attempt(settings.HTTP_CLIENT_MAX_RETRIES + 1),
        wait=wait_exponential(multiplier=settings.HTTP_CLIENT_BACKOFF_FACTOR, min=1, max=10),
        retry=retry_if_exception_type((httpx.TimeoutException, httpx.NetworkError, httpx.ConnectError)),
        reraise=True 
    )
    async def generate_embeddings(self, texts: List[str], text_type: str = "passage") -> List[List[float]]:
        """
        Solicita embeddings para una lista de textos al servicio de embedding.
        Args:
            texts: Lista de textos a embeber.
            text_type: El tipo de texto ('query' o 'passage'). Por defecto 'passage'.
        """
        client_log = log.bind(action="generate_embeddings", num_texts=len(texts), text_type=text_type, target_service="embedding-service")
        if not texts:
            client_log.warning("No texts provided to generate_embeddings.")
            return []

        payload = {
            "texts": texts,
            "text_type": text_type 
        }
        try:
            client_log.debug("Sending request to embedding service", endpoint=self.embed_endpoint, payload_preview=str(payload)[:200])
            response = await self._client.post(self.embed_endpoint, json=payload)
            response.raise_for_status() 

            data = response.json()
            if "embeddings" not in data or not isinstance(data["embeddings"], list):
                client_log.error("Invalid response format from embedding service: 'embeddings' field missing or not a list.", response_data=data)
                raise ValueError("Invalid response format from embedding service: 'embeddings' field.")

            client_log.info("Embeddings received successfully from service", num_embeddings=len(data["embeddings"]))
            return data["embeddings"]

        except httpx.HTTPStatusError as e:
            client_log.error("HTTP error from embedding service", status_code=e.response.status_code, response_body=e.response.text)
            raise ConnectionError(f"Embedding service returned error {e.response.status_code}: {e.response.text}") from e
        except httpx.RequestError as e:
            client_log.error("Request error while contacting embedding service", error=str(e))
            raise ConnectionError(f"Could not connect to embedding service: {e}") from e
        except json.JSONDecodeError as e_json: 
            client_log.error("Error parsing JSON response from embedding service", error=str(e_json), raw_response=response.text if 'response' in locals() else "N/A")
            raise ValueError(f"Invalid JSON response from embedding service: {e_json}") from e_json
        except (ValueError, TypeError) as e: 
            client_log.error("Error processing response from embedding service (ValueError/TypeError)", error=str(e))
            raise ValueError(f"Invalid response data from embedding service: {e}") from e


    async def get_model_info(self) -> Optional[Dict[str, Any]]:
        client_log = log.bind(action="get_model_info_via_embed", target_service="embedding-service")
        try:
            # Realizar una petición con un texto de prueba para obtener model_info
            # Se usa "query" text_type ya que no afecta a OpenAI y es un caso de uso para E5.
            response = await self._client.post(self.embed_endpoint, json={"texts": ["test"], "text_type": "query"}) 
            response.raise_for_status()
            data = response.json()
            if "model_info" in data and isinstance(data["model_info"], dict):
                client_log.info("Model info retrieved from embedding service", model_info=data["model_info"])
                return data["model_info"]
            client_log.warning("Model info not found in embedding service response.", response_data=data)
            return None
        except json.JSONDecodeError as e_json: 
            client_log.error("Failed to parse JSON for get_model_info from embedding service", error=str(e_json), raw_response=response.text if 'response' in locals() else "N/A")
            return None
        except Exception as e:
            client_log.error("Failed to get model_info from embedding service via /embed", error=str(e), exc_info=True)
            return None

    @retry(
        stop=stop_after_attempt(settings.HTTP_CLIENT_MAX_RETRIES + 1),
        wait=wait_exponential(multiplier=settings.HTTP_CLIENT_BACKOFF_FACTOR, min=1, max=5), 
        retry=retry_if_exception_type((httpx.TimeoutException, httpx.NetworkError, httpx.ConnectError, ConnectionError)),
        reraise=True,
        before_sleep=lambda retry_state: log.warning(
            "Retrying EmbeddingServiceClient.check_health",
            attempt=retry_state.attempt_number,
            wait_time=f"{retry_state.next_action.sleep:.2f}s", 
            error_type=type(retry_state.outcome.exception()).__name__ if retry_state.outcome else "N/A", 
            error_message=str(retry_state.outcome.exception()) if retry_state.outcome else "N/A" 
        )
    )
    async def check_health(self) -> bool:
        client_log = log.bind(action="check_health_with_retry", target_service="embedding-service")
        try:
            client_log.debug("Attempting health check...", health_endpoint=self.health_endpoint)
            response = await self._client.get(self.health_endpoint)
            response.raise_for_status() 

            data = response.json()
            model_is_ready = data.get("model_status") in ["client_ready", "loaded"]
            if data.get("status") == "ok" and model_is_ready:
                client_log.info("Embedding service health check successful.", health_data=data)
                return True
            else:
                client_log.warning("Embedding service health check returned ok status but model not fully ready or unexpected payload.", health_data=data)
                raise ConnectionError(f"Embedding service not fully ready: status={data.get('status')}, model_status={data.get('model_status')}")
        except httpx.HTTPStatusError as e:
            client_log.warning("HTTP error during embedding service health check (will be retried or reraised).", status_code=e.response.status_code, response_text=e.response.text)
            raise ConnectionError(f"HTTP error from embedding service: {e.response.status_code}") from e 
        except httpx.RequestError as e:
            client_log.error("Request error during embedding service health check (will be retried or reraised).", error=str(e))
            raise ConnectionError(f"Request error connecting to embedding service: {e}") from e
        except json.JSONDecodeError as e_json: 
            client_log.error("Failed to parse JSON response from embedding service health check.", error=str(e_json), raw_response=response.text if 'response' in locals() else "N/A")
            raise ConnectionError(f"Invalid JSON response from embedding service health: {e_json}") from e_json
        except Exception as e: 
            client_log.error("Unexpected error during embedding service health check (will be retried or reraised).", error=str(e))
            raise ConnectionError(f"Unexpected error during health check: {e}") from e

    async def close(self):
        await self._client.aclose()
        log.info("EmbeddingServiceClient closed.")
```

## File: `app/infrastructure/clients/sparse_search_service_client.py`
```py
# query-service/app/infrastructure/clients/sparse_search_service_client.py
import httpx
import structlog
import uuid
from typing import List, Dict, Any, Tuple, Optional
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from app.core.config import settings # Para timeouts y URL base
from app.domain.models import SparseSearchResultItem # Para el tipo de retorno del servicio

log = structlog.get_logger(__name__)

class SparseSearchServiceClient:
    """
    Cliente HTTP para interactuar con el Atenex Sparse Search Service.
    """
    def __init__(self, base_url: str, timeout: int = settings.HTTP_CLIENT_TIMEOUT):
        # Asegurar que la URL base no tenga /api/v1 al final si el endpoint ya lo incluye
        self.base_url = base_url.rstrip('/')
        # El endpoint del sparse-search-service es /api/v1/search
        self.search_endpoint = f"{self.base_url}/api/v1/search"
        self.health_endpoint = f"{self.base_url}/health"

        # Validar que la URL base no incluya /api/v1 si el endpoint ya lo hace.
        # Ejemplo: si base_url es http://service/api/v1, endpoint es http://service/api/v1/search
        # Ejemplo: si base_url es http://service, endpoint es http://service/api/v1/search
        if self.base_url.endswith("/api/v1"):
            self.search_endpoint = f"{self.base_url.rsplit('/api/v1', 1)[0]}/api/v1/search"
        elif self.base_url.endswith("/api"):
             self.search_endpoint = f"{self.base_url.rsplit('/api', 1)[0]}/api/v1/search"

        self._client = httpx.AsyncClient(timeout=timeout)
        log.info("SparseSearchServiceClient initialized",
                 base_url=self.base_url,
                 search_endpoint=self.search_endpoint,
                 health_endpoint=self.health_endpoint)

    @retry(
        stop=stop_after_attempt(settings.HTTP_CLIENT_MAX_RETRIES + 1),
        wait=wait_exponential(multiplier=settings.HTTP_CLIENT_BACKOFF_FACTOR, min=1, max=10),
        retry=retry_if_exception_type((httpx.TimeoutException, httpx.NetworkError, httpx.ConnectError)),
        reraise=True
    )
    async def search(self, query_text: str, company_id: uuid.UUID, top_k: int) -> List[SparseSearchResultItem]:
        """
        Solicita una búsqueda dispersa al sparse-search-service.
        Devuelve una lista de SparseSearchResultItem del dominio.
        """
        client_log = log.bind(action="sparse_search_remote",
                              company_id=str(company_id),
                              query_preview=query_text[:50]+"...",
                              top_k=top_k,
                              target_service="sparse-search-service")
        if not query_text:
            client_log.warning("No query text provided for sparse search.")
            return []

        payload = {
            "query": query_text,
            "company_id": str(company_id), # El servicio espera un UUID string en JSON
            "top_k": top_k
        }
        try:
            client_log.debug("Sending request to sparse search service")
            response = await self._client.post(self.search_endpoint, json=payload)
            response.raise_for_status()

            data = response.json()
            
            if "results" not in data or not isinstance(data["results"], list):
                client_log.error("Invalid response format from sparse search service: 'results' field missing or not a list.", response_data=data)
                raise ValueError("Invalid response format from sparse search service: 'results' field.")

            # Mapear a SparseSearchResultItem del dominio
            domain_results = []
            for item_data in data["results"]:
                # El servicio sparse-search ya devuelve items que coinciden con SparseSearchResultItem
                # así que podemos instanciarlos directamente si el schema coincide.
                # Asumimos que 'chunk_id' y 'score' están presentes.
                domain_results.append(SparseSearchResultItem(**item_data))
            
            client_log.info("Sparse search results received successfully from service", num_results=len(domain_results))
            return domain_results

        except httpx.HTTPStatusError as e:
            client_log.error("HTTP error from sparse search service", status_code=e.response.status_code, response_body=e.response.text)
            raise ConnectionError(f"Sparse search service returned error {e.response.status_code}: {e.response.text}") from e
        except httpx.RequestError as e:
            client_log.error("Request error while contacting sparse search service", error=str(e))
            raise ConnectionError(f"Could not connect to sparse search service: {e}") from e
        except (ValueError, TypeError, AttributeError) as e: # Errores de parsing JSON o validación de Pydantic
            client_log.error("Error processing response from sparse search service", error=str(e))
            raise ValueError(f"Invalid response or data from sparse search service: {e}") from e

    async def check_health(self) -> bool:
        client_log = log.bind(action="check_health_sparse_search", target_service="sparse-search-service")
        try:
            response = await self._client.get(self.health_endpoint, timeout=5) # Shorter timeout for health
            if response.status_code == 200:
                data = response.json()
                # El health check del sparse-search-service devuelve un JSON con `status` y `ready`
                if data.get("status") == "ok" and data.get("ready") is True:
                    client_log.info("Sparse search service health check successful.", health_data=data)
                    return True
                else:
                    client_log.warning("Sparse search service health check returned ok status but service/dependencies not ready.", health_data=data)
                    return False
            else:
                client_log.warning("Sparse search service health check failed.", status_code=response.status_code, response_text=response.text)
                return False
        except httpx.RequestError as e:
            client_log.error("Error connecting to sparse search service for health check.", error=str(e))
            return False
        except Exception as e:
            client_log.error("Unexpected error during sparse search service health check.", error=str(e))
            return False

    async def close(self):
        await self._client.aclose()
        log.info("SparseSearchServiceClient closed.")
```

## File: `app/infrastructure/embedding/__init__.py`
```py
# query-service/app/infrastructure/embedding/__init__.py
from .remote_embedding_adapter import RemoteEmbeddingAdapter

__all__ = ["RemoteEmbeddingAdapter"]
```

## File: `app/infrastructure/embedding/remote_embedding_adapter.py`
```py
# query-service/app/infrastructure/embedding/remote_embedding_adapter.py
import structlog
from typing import List, Optional

from app.application.ports.embedding_port import EmbeddingPort
from app.infrastructure.clients.embedding_service_client import EmbeddingServiceClient
from app.core.config import settings 

log = structlog.get_logger(__name__)

class RemoteEmbeddingAdapter(EmbeddingPort):
    """
    Adaptador que utiliza EmbeddingServiceClient para generar embeddings
    llamando al servicio de embedding externo.
    """
    def __init__(self, client: EmbeddingServiceClient):
        self.client = client
        self._embedding_dimension: Optional[int] = None 
        self._expected_dimension = settings.EMBEDDING_DIMENSION 
        log.info("RemoteEmbeddingAdapter initialized", expected_dimension=self._expected_dimension)

    async def initialize(self):
        init_log = log.bind(adapter="RemoteEmbeddingAdapter", action="initialize")
        try:
            model_info = await self.client.get_model_info()
            if model_info and "dimension" in model_info:
                self._embedding_dimension = model_info["dimension"]
                init_log.info("Successfully retrieved embedding dimension from service.",
                              service_dimension=self._embedding_dimension,
                              service_model_name=model_info.get("model_name"))
                if self._embedding_dimension != self._expected_dimension:
                    init_log.warning("Embedding dimension mismatch!",
                                     configured_dimension=self._expected_dimension,
                                     service_dimension=self._embedding_dimension,
                                     message="Query service configured dimension does not match dimension reported by embedding service. "
                                             "This may cause issues with Milvus or other components. Ensure configurations are aligned.")
            else:
                init_log.warning("Could not retrieve embedding dimension from service. Will use configured dimension.",
                                 configured_dimension=self._expected_dimension)
        except Exception as e:
            init_log.error("Failed to retrieve embedding dimension during initialization.", error=str(e), exc_info=True)

    async def embed_query(self, query_text: str) -> List[float]:
        adapter_log = log.bind(adapter="RemoteEmbeddingAdapter", action="embed_query")
        if not query_text:
            adapter_log.warning("Empty query text provided.")
            raise ValueError("Query text cannot be empty.")
        try:
            # Especificar text_type="query" para las consultas de usuario
            embeddings = await self.client.generate_embeddings(texts=[query_text], text_type="query")
            if not embeddings or len(embeddings) != 1:
                adapter_log.error("Embedding service did not return a valid embedding for the query.", received_embeddings=embeddings)
                raise ValueError("Failed to get a valid embedding for the query.")

            embedding_vector = embeddings[0]
            
            service_dim = self.get_embedding_dimension() 
            if len(embedding_vector) != service_dim:
                adapter_log.error("Embedding dimension mismatch for query embedding.",
                                  expected_dim=service_dim,
                                  received_dim=len(embedding_vector))
                raise ValueError(f"Embedding dimension mismatch: expected {service_dim}, got {len(embedding_vector)}")

            adapter_log.debug("Query embedded successfully via remote service.")
            return embedding_vector
        except ConnectionError as e:
            adapter_log.error("Connection error while embedding query.", error=str(e))
            raise 
        except ValueError as e:
            adapter_log.error("Value error while embedding query.", error=str(e))
            raise 

    async def embed_texts(self, texts: List[str]) -> List[List[float]]:
        adapter_log = log.bind(adapter="RemoteEmbeddingAdapter", action="embed_texts")
        if not texts:
            adapter_log.warning("No texts provided to embed_texts.")
            return []
        try:
            # Para lotes de texto genéricos (ej. passages), usar "passage" por defecto
            embeddings = await self.client.generate_embeddings(texts=texts, text_type="passage")
            if len(embeddings) != len(texts):
                adapter_log.error("Number of embeddings received does not match number of texts sent.",
                                  num_texts=len(texts), num_embeddings=len(embeddings))
                raise ValueError("Mismatch in number of embeddings received from service.")

            service_dim = self.get_embedding_dimension()
            if embeddings and len(embeddings[0]) != service_dim:
                 adapter_log.error("Embedding dimension mismatch for batch texts.",
                                   expected_dim=service_dim,
                                   received_dim=len(embeddings[0]))
                 raise ValueError(f"Embedding dimension mismatch: expected {service_dim}, got {len(embeddings[0])}")

            adapter_log.debug(f"Successfully embedded {len(texts)} texts via remote service.")
            return embeddings
        except ConnectionError as e:
            adapter_log.error("Connection error while embedding texts.", error=str(e))
            raise
        except ValueError as e:
            adapter_log.error("Value error while embedding texts.", error=str(e))
            raise

    def get_embedding_dimension(self) -> int:
        if self._embedding_dimension is not None:
            return self._embedding_dimension
        return self._expected_dimension

    async def health_check(self) -> bool:
        return await self.client.check_health()
```

## File: `app/infrastructure/filters/__init__.py`
```py
# query-service/app/infrastructure/filters/__init__.py
```

## File: `app/infrastructure/filters/diversity_filter.py`
```py
# query-service/app/infrastructure/filters/diversity_filter.py
import structlog
import asyncio
from typing import List, Optional, Tuple
import numpy as np

from app.application.ports.retrieval_ports import DiversityFilterPort
from app.domain.models import RetrievedChunk
from app.core.config import settings

log = structlog.get_logger(__name__)

def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """Calcula la similitud coseno entre dos vectores."""
    if not vec1 or not vec2:
        return 0.0
    v1 = np.array(vec1)
    v2 = np.array(vec2)
    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    if norm_v1 == 0 or norm_v2 == 0:
        return 0.0
    return dot_product / (norm_v1 * norm_v2)

class MMRDiversityFilter(DiversityFilterPort):
    """
    Filtro de diversidad usando Maximal Marginal Relevance (MMR).
    Selecciona chunks que son relevantes para la consulta pero diversos entre sí.
    """

    def __init__(self, lambda_mult: float = settings.QUERY_DIVERSITY_LAMBDA):
        """
        Inicializa el filtro MMR.
        Args:
            lambda_mult: Factor de balance entre relevancia y diversidad (0 a 1).
                         Alto (e.g., 0.7) prioriza relevancia.
                         Bajo (e.g., 0.3) prioriza diversidad.
        """
        if not (0.0 <= lambda_mult <= 1.0):
            raise ValueError("lambda_mult must be between 0.0 and 1.0")
        self.lambda_mult = lambda_mult
        log.info("MMRDiversityFilter initialized", lambda_mult=self.lambda_mult, adapter="MMRDiversityFilter")

    async def filter(self, chunks: List[RetrievedChunk], k_final: int) -> List[RetrievedChunk]:
        """
        Aplica el filtro MMR a la lista de chunks.
        Requiere que los chunks tengan embeddings.
        """
        filter_log = log.bind(adapter="MMRDiversityFilter", action="filter", k_final=k_final, lambda_mult=self.lambda_mult, input_count=len(chunks))

        if not chunks or k_final <= 0:
            filter_log.debug("No chunks to filter or k_final <= 0.")
            return []

        # Filtrar chunks que no tengan embedding
        chunks_with_embeddings = [c for c in chunks if c.embedding is not None]
        if not chunks_with_embeddings:
            filter_log.warning("No chunks with embeddings found. Returning original top-k chunks (or fewer).")
            # Devuelve los primeros k_final chunks originales (aunque no tengan embedding)
            return chunks[:k_final]

        num_chunks_with_embeddings = len(chunks_with_embeddings)
        if k_final >= num_chunks_with_embeddings:
            filter_log.debug(f"k_final ({k_final}) >= number of chunks with embeddings ({num_chunks_with_embeddings}). Returning all chunks with embeddings.")
            return chunks_with_embeddings # Devolver todos los que tienen embedding si k es mayor o igual

        # El primer chunk seleccionado es siempre el más relevante (asume que la lista está ordenada por relevancia)
        selected_indices = {0}
        selected_chunks = [chunks_with_embeddings[0]]

        remaining_indices = set(range(1, num_chunks_with_embeddings))

        while len(selected_chunks) < k_final and remaining_indices:
            mmr_scores = {}
            # Calcular la similitud máxima de cada candidato con los ya seleccionados
            for candidate_idx in remaining_indices:
                candidate_chunk = chunks_with_embeddings[candidate_idx]
                max_similarity = 0.0
                for selected_idx in selected_indices:
                    similarity = cosine_similarity(candidate_chunk.embedding, chunks_with_embeddings[selected_idx].embedding)
                    max_similarity = max(max_similarity, similarity)

                # Calcular score MMR
                # Usamos el score original del chunk como medida de relevancia (podría ser similitud con query si la tuviéramos)
                relevance_score = candidate_chunk.score or 0.0 # Usar 0 si no hay score
                mmr_score = self.lambda_mult * relevance_score - (1 - self.lambda_mult) * max_similarity
                mmr_scores[candidate_idx] = mmr_score

            # Encontrar el mejor candidato según MMR
            if not mmr_scores: break # Salir si no hay más candidatos con score
            best_candidate_idx = max(mmr_scores, key=mmr_scores.get)

            # Añadir el mejor candidato y moverlo de conjuntos
            selected_indices.add(best_candidate_idx)
            selected_chunks.append(chunks_with_embeddings[best_candidate_idx])
            remaining_indices.remove(best_candidate_idx)

        filter_log.info(f"MMR filtering complete. Selected {len(selected_chunks)} diverse chunks.")
        return selected_chunks

class StubDiversityFilter(DiversityFilterPort):
    """Implementación Stub (Fallback si MMR falla o está deshabilitado)."""
    def __init__(self):
        log.warning("Using StubDiversityFilter. No diversity logic is applied.", adapter="StubDiversityFilter")

    async def filter(self, chunks: List[RetrievedChunk], k_final: int) -> List[RetrievedChunk]:
        filter_log = log.bind(adapter="StubDiversityFilter", action="filter", k_final=k_final, input_count=len(chunks))
        if not chunks:
            filter_log.debug("No chunks to filter.")
            return []
        filtered_chunks = chunks[:k_final]
        filter_log.debug(f"Returning top {len(filtered_chunks)} chunks without diversity filtering.")
        return filtered_chunks
```

## File: `app/infrastructure/llms/__init__.py`
```py
# query-service/app/infrastructure/llms/__init__.py

from .llama_cpp_adapter import LlamaCppAdapter

__all__ = ["LlamaCppAdapter"]
```

## File: `app/infrastructure/llms/gemini_adapter.py`
```py
# query-service/app/infrastructure/llms/gemini_adapter.py
import google.generativeai as genai
from google.generativeai import types as genai_types
from google.api_core import exceptions as google_api_exceptions 

import structlog
from typing import Optional, List, Type, Any, Dict, AsyncGenerator
from pydantic import BaseModel
import json
import logging # Para before_sleep_log

from app.core.config import settings
from app.application.ports.llm_port import LLMPort
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type, before_sleep_log
from app.utils.helpers import truncate_text


log = structlog.get_logger(__name__)

def _clean_pydantic_schema_for_gemini_response(pydantic_schema: Dict[str, Any]) -> Dict[str, Any]:
    definitions = pydantic_schema.get("$defs", {})

    schema_copy = {
        k: v
        for k, v in pydantic_schema.items()
        if k not in {"$defs", "title", "description", "$schema"} 
    }

    def resolve_ref(ref_path: str) -> Dict[str, Any]:
        if not ref_path.startswith("#/$defs/"):
            log.warning("Encountered non-internal JSON schema reference, falling back to OBJECT.", ref_path=ref_path)
            return {"type": "OBJECT"} 
        
        def_key = ref_path.split("/")[-1]
        if def_key in definitions:
            return _transform_node(definitions[def_key])
        else:
            log.warning(f"Broken JSON schema reference found and could not be resolved: {ref_path}")
            return {"type": "OBJECT"} 

    def _transform_node(node: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(node, dict):
            return node

        transformed_node = {}
        for key, value in node.items():
            if key in {"default", "examples", "example", "const", "title", "description"}:
                continue
            
            if key == "$ref" and isinstance(value, str):
                 return resolve_ref(value)

            elif key == "anyOf" and isinstance(value, list):
                is_optional_pattern = False
                if len(value) == 2:
                    type_def_item = next((item for item in value if isinstance(item, dict) and item.get("type") != "null"), None)
                    null_def_item = next((item for item in value if isinstance(item, dict) and item.get("type") == "null"), None)
                    
                    if type_def_item and null_def_item:
                        is_optional_pattern = True
                        transformed_type_def = _transform_node(type_def_item)
                        for k_type, v_type in transformed_type_def.items():
                             transformed_node[k_type] = v_type
                        transformed_node["nullable"] = True 
                
                if not is_optional_pattern: 
                    if value:
                        first_type = _transform_node(value[0])
                        for k_first, v_first in first_type.items():
                            transformed_node[k_first] = v_first
                        log.warning("Complex 'anyOf' in Pydantic schema for Gemini response_schema, took first option.",
                                    original_anyof=value, chosen_type=first_type)
                    else:
                        log.warning("Empty 'anyOf' in Pydantic schema for Gemini response_schema.", original_anyof=value)
                continue 

            elif isinstance(value, dict):
                transformed_node[key] = _transform_node(value)
            elif isinstance(value, list) and key not in ["enum", "required"]: 
                transformed_node[key] = [_transform_node(item) if isinstance(item, dict) else item for item in value]
            else:
                transformed_node[key] = value
        
        if "type" in transformed_node:
            json_type = transformed_node["type"]
            if isinstance(json_type, list): 
                if "null" in json_type:
                    transformed_node["nullable"] = True 
                actual_type = next((t for t in json_type if t != "null"), "OBJECT") 
                if isinstance(actual_type, str):
                    transformed_node["type"] = actual_type.upper()
                else: 
                    transformed_node["type"] = _transform_node(actual_type).get("type", "OBJECT")

            elif isinstance(json_type, str):
                transformed_node["type"] = json_type.upper()
            
            if transformed_node["type"] == "LIST": 
                transformed_node["type"] = "ARRAY"

        if transformed_node.get("type") == "ARRAY" and "items" not in transformed_node:
            log.warning("Schema for ARRAY type missing 'items' definition for Gemini. Adding generic object item.", node_details=transformed_node)
            transformed_node["items"] = {"type": "OBJECT"} 

        return transformed_node

    final_schema = _transform_node(schema_copy)
    
    log.debug("Cleaned Pydantic JSON Schema for Gemini response_schema", original_schema_preview=str(pydantic_schema)[:200], cleaned_schema_preview=str(final_schema)[:200])
    return final_schema


class GeminiAdapter(LLMPort):
    _api_key: Optional[str]
    _model_name: str
    _model: Optional[genai.GenerativeModel] = None 
    _safety_settings: List[Dict[str, str]]

    def __init__(self):
        self._api_key = (
            settings.GEMINI_API_KEY.get_secret_value()
            if settings.GEMINI_API_KEY
            else None
        )
        self._model_name = settings.GEMINI_MODEL_NAME
        self._max_output_tokens = settings.GEMINI_MAX_OUTPUT_TOKENS
        # Configuraciones de seguridad más permisivas para evitar bloqueos
        self._safety_settings = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_ONLY_HIGH"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_ONLY_HIGH"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_ONLY_HIGH"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_ONLY_HIGH"},
        ]
        self._configure_client()

    def _configure_client(self):
        try:
            if self._api_key:
                genai.configure(api_key=self._api_key)
                self._model = genai.GenerativeModel(
                    self._model_name,
                    safety_settings=self._safety_settings
                )
                log.info("Gemini client configured successfully using GenerativeModel",
                         model_name=self._model_name,
                         safety_settings=self._safety_settings)
            else:
                log.warning("Gemini API key is missing. Client not configured.")
        except Exception as e:
            log.error("Failed to configure Gemini client (GenerativeModel)", error=str(e), exc_info=True)
            self._model = None
    
    def _create_error_json_response(self, error_message: str, detailed_message: str) -> str:
        """Helper para crear un JSON de error estructurado."""
        return json.dumps({
            "error_message": error_message,
            "respuesta_detallada": detailed_message,
            "fuentes_citadas": [],
            "resumen_ejecutivo": None,
            "siguiente_pregunta_sugerida": None
        })

    @retry(
        stop=stop_after_attempt(settings.HTTP_CLIENT_MAX_RETRIES + 1),
        wait=wait_exponential(multiplier=settings.HTTP_CLIENT_BACKOFF_FACTOR, min=2, max=10),
        retry=retry_if_exception_type((
            TimeoutError,
        )),
        reraise=True,
        before_sleep=before_sleep_log(log, logging.WARNING) 
    )
    async def generate(self, prompt: str,
                       response_pydantic_schema: Optional[Type[BaseModel]] = None
                      ) -> str:
        if not self._model:
            log.error("Gemini client (GenerativeModel) not initialized. Cannot generate answer.")
            raise ConnectionError("Gemini client is not properly configured (missing API key or init failed).")

        generate_log = log.bind(
            adapter="GeminiAdapter",
            model_name=self._model_name,
            prompt_length=len(prompt),
            expecting_json=bool(response_pydantic_schema)
        )

        generation_config_parts: Dict[str, Any] = {
            "temperature": 0.6, 
            "top_p": 0.9,
        }
        if settings.GEMINI_MAX_OUTPUT_TOKENS:
            generation_config_parts["max_output_tokens"] = settings.GEMINI_MAX_OUTPUT_TOKENS
        if self._max_output_tokens:
            generation_config_parts["max_output_tokens"] = self._max_output_tokens
        
        if response_pydantic_schema:
            generation_config_parts["response_mime_type"] = "application/json"
            
            pydantic_schema_json = response_pydantic_schema.model_json_schema()
            cleaned_schema_for_gemini = _clean_pydantic_schema_for_gemini_response(pydantic_schema_json)
            generation_config_parts["response_schema"] = cleaned_schema_for_gemini
            generate_log.debug("Configured Gemini for JSON output using cleaned response_schema.", 
                               schema_name=response_pydantic_schema.__name__,
                               max_output_tokens=self._max_output_tokens)
        
        final_generation_config = genai_types.GenerationConfig(**generation_config_parts)
        
        try:
            call_kwargs: Dict[str, Any] = {"generation_config": final_generation_config}
            
            prompt_size_for_log = len(prompt)
            if prompt_size_for_log > 500: 
                prompt_preview = truncate_text(prompt, 500)
                generate_log.debug("Sending request to Gemini API...", prompt_preview=prompt_preview, prompt_total_length=prompt_size_for_log)
            else:
                generate_log.debug("Sending request to Gemini API...", prompt_text=prompt)


            response = await self._model.generate_content_async(prompt, **call_kwargs)
            
            generate_log.info("Gemini API response received.", 
                              num_candidates=len(response.candidates), 
                              prompt_feedback_block_reason=getattr(response.prompt_feedback, 'block_reason', 'N/A'),
                              usage_prompt_tokens=getattr(response.usage_metadata, 'prompt_token_count', 'N/A'),
                              usage_candidates_tokens=getattr(response.usage_metadata, 'candidates_token_count', 'N/A'),
                              usage_total_tokens=getattr(response.usage_metadata, 'total_token_count', 'N/A')
            )
            generated_text = ""

            try:
                usage_metadata = response.usage_metadata if hasattr(response, 'usage_metadata') else None
                prompt_token_count = getattr(usage_metadata, 'prompt_token_count', 'N/A')
                candidates_token_count = getattr(usage_metadata, 'candidates_token_count', 'N/A')
                total_token_count = getattr(usage_metadata, 'total_token_count', 'N/A')
                
                generate_log.info("Gemini API response received.", 
                                  num_candidates=len(response.candidates) if response.candidates else 0,
                                  prompt_feedback_block_reason=str(getattr(response.prompt_feedback, 'block_reason', 'N/A')),
                                  usage_prompt_tokens=prompt_token_count,
                                  usage_candidates_tokens=candidates_token_count,
                                  usage_total_tokens=total_token_count)
            except Exception as e_log_resp:
                generate_log.warning("Could not fully log Gemini response details", error_logging_response=str(e_log_resp))


            if not response.candidates:
                 finish_reason_str = getattr(response.prompt_feedback, 'block_reason', "UNKNOWN_REASON").name if hasattr(getattr(response.prompt_feedback, 'block_reason', None) , 'name') else str(getattr(response.prompt_feedback, 'block_reason', "UNKNOWN_REASON"))
                 safety_ratings_str = str(getattr(response.prompt_feedback, 'safety_ratings', "N/A")) 
                 generate_log.warning("Gemini response potentially blocked (no candidates)",
                                      finish_reason=finish_reason_str, safety_ratings=safety_ratings_str)
                 if response_pydantic_schema:
                     return self._create_error_json_response(
                         error_message=f"Respuesta bloqueada por Gemini (sin candidatos). Razón: {finish_reason_str}",
                         detailed_message=f"La generación de la respuesta fue bloqueada. Por favor, reformula tu pregunta o contacta a soporte si el problema persiste. Razón: {finish_reason_str}."
                     )
                 return f"[Respuesta bloqueada por Gemini (sin candidatos). Razón: {finish_reason_str}]"

            candidate = response.candidates[0]
            candidate_finish_reason = getattr(candidate, "finish_reason", "UNKNOWN")
            candidate_finish_reason_str = getattr(candidate_finish_reason, "name", str(candidate_finish_reason))
            generate_log.info(
                "Gemini candidate details",
                finish_reason=candidate_finish_reason_str,
                safety_ratings=str(candidate.safety_ratings),
            )


            if candidate_finish_reason_str == "MAX_TOKENS":
                generate_log.warning(
                    "Gemini response TRUNCATED due to max_output_tokens.",
                    configured_limit=settings.GEMINI_MAX_OUTPUT_TOKENS,
                )


            if not candidate.content or not candidate.content.parts:
                generate_log.warning("Gemini response candidate empty or missing parts",
                                     candidate_details=str(candidate))
                if response_pydantic_schema:
                     return self._create_error_json_response(
                         error_message=f"Respuesta vacía de Gemini (candidato sin contenido). Razón: {candidate_finish_reason_str}",
                         detailed_message=f"El asistente no pudo generar una respuesta completa. Razón: {candidate_finish_reason_str}."
                     )
                return f"[Respuesta vacía de Gemini (candidato sin contenido). Razón: {candidate_finish_reason_str}]"
            
            if candidate.content.parts[0].text:
                generated_text = candidate.content.parts[0].text
            else:
                generate_log.error("Gemini response part exists but has no text content.")
                if response_pydantic_schema:
                    return self._create_error_json_response(
                        error_message="Respuesta del LLM incompleta o en formato inesperado (sin texto).",
                        detailed_message="Error: El asistente devolvió una respuesta sin contenido textual."
                    )
                return "[Respuesta del LLM incompleta o sin contenido textual]"

            if response_pydantic_schema:
                generate_log.debug("Received potential JSON text from Gemini API.", response_length=len(generated_text))
            else: 
                generate_log.debug("Received plain text response from Gemini API", response_length=len(generated_text))
                
            return generated_text.strip()

        except (genai_types.generation_types.BlockedPromptException, genai_types.generation_types.StopCandidateException) as security_err: 
            finish_reason_err_str = getattr(security_err, 'finish_reason', 'N/A') if hasattr(security_err, 'finish_reason') else 'Unknown security block'
            generate_log.warning("Gemini request blocked or stopped due to safety/policy.",
                                 error_type=type(security_err).__name__,
                                 error_details=str(security_err),
                                 finish_reason=finish_reason_err_str)
            if response_pydantic_schema:
                return self._create_error_json_response(
                    error_message=f"Contenido bloqueado o detenido por Gemini: {type(security_err).__name__}",
                    detailed_message=f"La generación de la respuesta fue bloqueada o detenida por políticas de contenido. Por favor, ajusta tu consulta. (Razón: {finish_reason_err_str})"
                )
            return f"[Contenido bloqueado o detenido por Gemini: {type(security_err).__name__}. Razón: {finish_reason_err_str}]"
        
        except google_api_exceptions.GoogleAPICallError as api_call_err:
            # Captura errores más genéricos de la API de Google, que pueden incluir errores HTTP no capturados por las más específicas.
            generate_log.error("Gemini API call failed with GoogleAPICallError", error_details=str(api_call_err), exc_info=True)
            raise ConnectionError(f"Gemini API call error: {api_call_err}") from api_call_err
        except google_api_exceptions.InvalidArgument as invalid_arg_err:
            prompt_preview_for_error = truncate_text(prompt, 200)
            generate_log.error("Gemini API call failed due to invalid argument. This could be due to the prompt or JSON schema if provided.",
                               error_details=str(invalid_arg_err), 
                               prompt_preview=prompt_preview_for_error,
                               json_schema_expected=response_pydantic_schema.__name__ if response_pydantic_schema else "None",
                               exc_info=True)
            raise ValueError(f"Gemini API InvalidArgument: {invalid_arg_err}") from invalid_arg_err
        except Exception as e: 
            generate_log.exception("Unhandled error during Gemini API call")
            if response_pydantic_schema: 
                return self._create_error_json_response(
                    error_message=f"Error inesperado en la API de Gemini: {type(e).__name__}",
                    detailed_message=f"Error interno al comunicarse con el asistente: {type(e).__name__} - {truncate_text(str(e),100)}."
                )
            raise ConnectionError(f"Gemini API call failed unexpectedly: {e}") from e

    async def generate_stream(self, prompt: str) -> AsyncGenerator[str, None]:
        if not self._model:
            log.error("Gemini client (GenerativeModel) not initialized for streaming.")
            raise ConnectionError("Gemini client is not properly configured for streaming.")

        stream_log = log.bind(
            adapter="GeminiAdapter",
            action="generate_stream",
            model_name=self._model_name,
            prompt_length=len(prompt)
        )
        
        generation_config_parts: Dict[str, Any] = {
            "temperature": 0.6,
            "top_p": 0.9,
        }
        if self._max_output_tokens:
            generation_config_parts["max_output_tokens"] = self._max_output_tokens
            
        final_generation_config = genai_types.GenerationConfig(**generation_config_parts)

        try:
            stream_log.debug("Sending stream request to Gemini API...")
            
            response_stream = await self._model.generate_content_async(prompt, stream=True, generation_config=final_generation_config)
            
            async for chunk in response_stream:
                if chunk.text:
                    yield chunk.text
                if chunk.prompt_feedback and chunk.prompt_feedback.block_reason:
                    stream_log.warning("Stream chunk indicated prompt blocking", reason=chunk.prompt_feedback.block_reason)
            
            stream_log.debug("Streaming finished.")

        except (genai_types.generation_types.BlockedPromptException, genai_types.generation_types.StopCandidateException) as security_err:
            finish_reason_err_str = getattr(security_err, 'finish_reason', 'N/A') if hasattr(security_err, 'finish_reason') else 'Unknown security block'
            stream_log.warning("Gemini stream blocked or stopped.", error_type=type(security_err).__name__, reason=finish_reason_err_str)
            yield f"[STREAM ERROR: Contenido bloqueado por Gemini. Razón: {finish_reason_err_str}]"
        except google_api_exceptions.GoogleAPICallError as e: 
            stream_log.error("Gemini API stream call failed with GoogleAPICallError", error_details=str(e), exc_info=True)
            yield f"[STREAM ERROR: Error de API de Gemini - {type(e).__name__}]"
        except Exception as e:
            stream_log.exception("Error during Gemini API stream")
            yield f"[STREAM ERROR: {type(e).__name__} - {str(e)}]"
```

## File: `app/infrastructure/llms/llama_cpp_adapter.py`
```py
# query-service/app/infrastructure/llms/llama_cpp_adapter.py
import json
import logging
from typing import Optional, Type, Any, Dict

import httpx
import structlog
from pydantic import BaseModel, ValidationError
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log,
)

from app.application.ports.llm_port import LLMPort
from app.core.config import settings
from app.utils.helpers import truncate_text

log = structlog.get_logger(__name__)


class LlamaCppAdapter(LLMPort):
    """Adapter for llama.cpp HTTP server exposing an OpenAI-compatible API."""

    def __init__(
        self,
        base_url: str,
        model_name: str,
        timeout: int = settings.HTTP_CLIENT_TIMEOUT,
        max_output_tokens: Optional[int] = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model_name = model_name
        self._max_output_tokens = max_output_tokens
        self._client = httpx.AsyncClient(timeout=timeout)

        self.chat_completions_endpoint = f"{self.base_url}/v1/chat/completions"

        log.info(
            "LlamaCppAdapter initialized",
            base_url=self.base_url,
            chat_endpoint=self.chat_completions_endpoint,
            model_name=self.model_name,
            max_output_tokens=self._max_output_tokens,
        )

    async def close(self) -> None:
        await self._client.aclose()
        log.info("LlamaCppAdapter HTTP client closed")

    async def health_check(self) -> bool:
        health_endpoint = f"{self.base_url}/health"
        for url in (health_endpoint, self.base_url):
            try:
                response = await self._client.get(url, timeout=5.0)
                if response.status_code >= 500:
                    log.warning(
                        "llama.cpp health check returned server error",
                        status_code=response.status_code,
                        url=url,
                    )
                    continue
                if response.status_code == 404 and url == health_endpoint:
                    # Algunos builds no exponen /health, considera 404 como éxito cuando el host es alcanzable.
                    log.debug("llama.cpp health endpoint missing, falling back to base URL check", url=url)
                return True
            except httpx.RequestError as exc:
                log.warning("llama.cpp health check failed", error=str(exc), url=url)
        return False

    @retry(
        stop=stop_after_attempt(max(1, (settings.HTTP_CLIENT_MAX_RETRIES or 0) + 1)),
        wait=wait_exponential(
            multiplier=max(settings.HTTP_CLIENT_BACKOFF_FACTOR, 0.1),
            min=1,
            max=10,
        ),
        retry=retry_if_exception_type(
            (
                httpx.TimeoutException,
                httpx.NetworkError,
                httpx.ConnectError,
            )
        ),
        reraise=True,
        before_sleep=before_sleep_log(logging.getLogger(__name__), logging.WARNING),
    )
    async def generate(
        self,
        prompt: str,
        response_pydantic_schema: Optional[Type[BaseModel]] = None,
    ) -> str:
        if not prompt or not prompt.strip():
            raise ValueError("Prompt cannot be empty.")

        expecting_json = bool(response_pydantic_schema)
        gen_log = log.bind(
            adapter="LlamaCppAdapter",
            model_name=self.model_name,
            prompt_length=len(prompt),
            expecting_json=expecting_json,
        )

        schema_json: Optional[str] = None
        if response_pydantic_schema:
            try:
                schema_json = json.dumps(
                    response_pydantic_schema.model_json_schema(),
                    ensure_ascii=False,
                )
            except TypeError as schema_err:
                gen_log.warning(
                    "Failed to serialize response schema for llama.cpp guidance",
                    error=str(schema_err),
                )

        messages = []
        if expecting_json:
            if schema_json:
                system_prompt = (
                    "Eres Atenex, un asistente empresarial. Devuelve exclusivamente JSON válido que cumpla "
                    "exactamente con el siguiente esquema Pydantic. No incluyas código, comentarios ni texto adicional.\n"
                    f"Esquema: {schema_json}"
                )
            else:
                system_prompt = (
                    "Eres Atenex, un asistente empresarial. Responde única y exclusivamente con JSON válido. "
                    "No incluyas texto fuera del JSON ni bloques de código."
                )
            messages.append({"role": "system", "content": system_prompt})

        messages.append({"role": "user", "content": prompt})

        payload: Dict[str, Any] = {
            "model": self.model_name,
            "messages": messages,
            "temperature": 0.6,
            "top_p": 0.9,
        }

        if expecting_json:
            payload["response_format"] = {"type": "json_object"}

        if self._max_output_tokens:
            payload["max_tokens"] = self._max_output_tokens

        prompt_preview = truncate_text(prompt, 500)
        gen_log.debug(
            "Sending request to llama.cpp",
            endpoint=self.chat_completions_endpoint,
            prompt_preview=prompt_preview,
        )

        response: Optional[httpx.Response] = None
        attempt_payload = dict(payload)

        for attempt in range(2 if expecting_json else 1):
            try:
                response = await self._client.post(
                    self.chat_completions_endpoint,
                    json=attempt_payload,
                )
                response.raise_for_status()
                break
            except httpx.HTTPStatusError as exc:
                if (
                    expecting_json
                    and attempt == 0
                    and exc.response.status_code == 400
                    and "response_format" in attempt_payload
                ):
                    gen_log.warning(
                        "llama.cpp rejected response_format hint; retrying without it",
                        response_text=truncate_text(exc.response.text, 200),
                    )
                    attempt_payload.pop("response_format", None)
                    continue

                gen_log.error(
                    "HTTP error from llama.cpp",
                    status_code=exc.response.status_code,
                    response_text=truncate_text(exc.response.text, 300),
                )
                raise ConnectionError(
                    f"LLM service (llama.cpp) returned HTTP {exc.response.status_code}: {exc.response.text}"
                ) from exc
            except httpx.RequestError as exc:
                gen_log.error("Request error contacting llama.cpp", error=str(exc))
                raise ConnectionError(
                    f"Could not connect to LLM service (llama.cpp server): {exc}"
                ) from exc

        if response is None:
            raise ConnectionError("LLM service (llama.cpp) did not return a response.")

        try:
            data = response.json()
        except json.JSONDecodeError as exc:
            gen_log.error(
                "Invalid JSON returned by llama.cpp",
                raw_response=truncate_text(response.text, 300),
            )
            raise ValueError("Invalid JSON response from LLM service.") from exc

        if not isinstance(data, dict) or not data.get("choices"):
            gen_log.error("Unexpected response format from llama.cpp", data=data)
            raise ValueError("Unexpected response format from LLM service.")

        first_choice = data["choices"][0]
        content: Optional[str] = None

        if isinstance(first_choice, dict):
            message = first_choice.get("message")
            if isinstance(message, dict):
                content = message.get("content")
            if content is None:
                content = first_choice.get("text")

        if not content:
            gen_log.error("No content found in llama.cpp response", data=data)
            raise ValueError("LLM service response does not contain content.")

        if expecting_json:
            normalized = self._normalize_json_output(content)
            if normalized:
                content = normalized
            else:
                gen_log.warning(
                    "Unable to normalize llama.cpp JSON response; returning raw content",
                    content_preview=truncate_text(content, 200),
                )

            if response_pydantic_schema:
                try:
                    response_pydantic_schema.model_validate_json(content)
                except (ValidationError, json.JSONDecodeError) as validation_err:
                    gen_log.warning(
                        "llama.cpp response failed schema validation",
                        error=str(validation_err),
                    )

        gen_log.info(
            "LLM response received from llama.cpp",
            content_preview=truncate_text(content, 200),
        )
        return content

    @staticmethod
    def _normalize_json_output(raw_text: str) -> Optional[str]:
        if not raw_text:
            return None

        text = raw_text.strip()

        if text.startswith("```"):
            text = text[3:]
            if text.startswith("json"):
                text = text[4:]
            if text.startswith("\n"):
                text = text[1:]
            closing_fence = text.rfind("```")
            if closing_fence != -1:
                text = text[:closing_fence]
            text = text.strip()

        try:
            json.loads(text)
            return text
        except json.JSONDecodeError:
            pass

        for open_char, close_char in (("{", "}"), ("[", "]")):
            start = text.find(open_char)
            end = text.rfind(close_char)
            if start != -1 and end != -1 and end > start:
                candidate = text[start : end + 1].strip()
                try:
                    json.loads(candidate)
                    return candidate
                except json.JSONDecodeError:
                    continue

        return None
```

## File: `app/infrastructure/persistence/__init__.py`
```py
# query-service/app/infrastructure/persistence/__init__.py
```

## File: `app/infrastructure/persistence/postgres_connector.py`
```py
# query-service/app/infrastructure/persistence/postgres_connector.py
import asyncpg
import structlog
import json
from typing import Optional

# LLM_REFACTOR_STEP_1: Update import path
from app.core.config import settings

log = structlog.get_logger(__name__)

_pool: Optional[asyncpg.Pool] = None

async def get_db_pool() -> asyncpg.Pool:
    """Gets the existing asyncpg pool or creates a new one."""
    global _pool
    if _pool is None or _pool._closed:
        log.info("Creating PostgreSQL connection pool...",
                 host=settings.POSTGRES_SERVER, port=settings.POSTGRES_PORT,
                 user=settings.POSTGRES_USER, db=settings.POSTGRES_DB)
        try:
            def _json_encoder(value): return json.dumps(value)
            def _json_decoder(value): return json.loads(value)
            async def init_connection(conn):
                await conn.set_type_codec('jsonb', encoder=_json_encoder, decoder=_json_decoder, schema='pg_catalog', format='text')
                await conn.set_type_codec('json', encoder=_json_encoder, decoder=_json_decoder, schema='pg_catalog', format='text')

            _pool = await asyncpg.create_pool(
                user=settings.POSTGRES_USER,
                password=settings.POSTGRES_PASSWORD.get_secret_value(),
                database=settings.POSTGRES_DB,
                host=settings.POSTGRES_SERVER,
                port=settings.POSTGRES_PORT,
                min_size=2, max_size=10, timeout=30.0, command_timeout=60.0,
                init=init_connection,
                statement_cache_size=0
            )
            log.info("PostgreSQL connection pool created successfully.")
        except (asyncpg.exceptions.InvalidPasswordError, OSError, ConnectionRefusedError) as conn_err:
            log.critical("CRITICAL: Failed to connect to PostgreSQL", error=str(conn_err), exc_info=True)
            _pool = None
            raise ConnectionError(f"Failed to connect to PostgreSQL: {conn_err}") from conn_err
        except Exception as e:
            log.critical("CRITICAL: Failed to create PostgreSQL connection pool", error=str(e), exc_info=True)
            _pool = None
            raise RuntimeError(f"Failed to create PostgreSQL pool: {e}") from e
    return _pool

async def close_db_pool():
    """Closes the asyncpg connection pool."""
    global _pool
    if _pool and not _pool._closed:
        log.info("Closing PostgreSQL connection pool...")
        await _pool.close()
        _pool = None
        log.info("PostgreSQL connection pool closed.")
    elif _pool and _pool._closed:
        log.warning("Attempted to close an already closed PostgreSQL pool.")
        _pool = None
    else:
        log.info("No active PostgreSQL connection pool to close.")

async def check_db_connection() -> bool:
    """Checks if a connection to the database can be established."""
    pool = None
    conn = None
    try:
        pool = await get_db_pool()
        conn = await pool.acquire()
        result = await conn.fetchval("SELECT 1")
        return result == 1
    except Exception as e:
        log.error("Database connection check failed", error=str(e))
        return False
    finally:
        if conn:
             await pool.release(conn) # Use await here
```

## File: `app/infrastructure/persistence/postgres_repositories.py`
```py
# query-service/app/infrastructure/persistence/postgres_repositories.py
import uuid
from typing import Any, Optional, Dict, List, Tuple
import asyncpg
import structlog
import json
from datetime import datetime, timezone

from app.core.config import settings
from app.api.v1 import schemas 
from app.domain.models import Chat, ChatMessage, ChatSummary, QueryLog 
from app.application.ports.repository_ports import ChatRepositoryPort, LogRepositoryPort, ChunkContentRepositoryPort 
from .postgres_connector import get_db_pool

log = structlog.get_logger(__name__)


# --- Chat Repository Implementation ---
class PostgresChatRepository(ChatRepositoryPort):
    """Implementación concreta del repositorio de chats usando PostgreSQL."""

    async def create_chat(self, user_id: uuid.UUID, company_id: uuid.UUID, title: Optional[str] = None) -> uuid.UUID:
        pool = await get_db_pool()
        chat_id = uuid.uuid4()
        query = """
        INSERT INTO chats (id, user_id, company_id, title, created_at, updated_at)
        VALUES ($1, $2, $3, $4, NOW() AT TIME ZONE 'UTC', NOW() AT TIME ZONE 'UTC') RETURNING id;
        """
        repo_log = log.bind(repo="PostgresChatRepository", action="create_chat", user_id=str(user_id), company_id=str(company_id))
        try:
            async with pool.acquire() as conn:
                result = await conn.fetchval(query, chat_id, user_id, company_id, title or f"Chat {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M')}")
            if result and result == chat_id:
                repo_log.info("Chat created successfully", chat_id=str(chat_id))
                return chat_id
            else:
                repo_log.error("Failed to create chat, no ID returned", returned_id=result)
                raise RuntimeError("Failed to create chat, no ID returned")
        except Exception as e:
            repo_log.exception("Failed to create chat")
            raise 

    async def get_user_chats(self, user_id: uuid.UUID, company_id: uuid.UUID, limit: int = 50, offset: int = 0) -> List[ChatSummary]:
        pool = await get_db_pool()
        query = """
        SELECT id, title, updated_at FROM chats
        WHERE user_id = $1 AND company_id = $2
        ORDER BY updated_at DESC LIMIT $3 OFFSET $4;
        """
        repo_log = log.bind(repo="PostgresChatRepository", action="get_user_chats", user_id=str(user_id), company_id=str(company_id))
        try:
            async with pool.acquire() as conn:
                rows = await conn.fetch(query, user_id, company_id, limit, offset)
            
            chats = [ChatSummary(**dict(row)) for row in rows]
            repo_log.info(f"Retrieved {len(chats)} chat summaries")
            return chats
        except Exception as e:
            repo_log.exception("Failed to get user chats")
            raise

    async def check_chat_ownership(self, chat_id: uuid.UUID, user_id: uuid.UUID, company_id: uuid.UUID) -> bool:
        pool = await get_db_pool()
        query = "SELECT EXISTS (SELECT 1 FROM chats WHERE id = $1 AND user_id = $2 AND company_id = $3);"
        repo_log = log.bind(repo="PostgresChatRepository", action="check_chat_ownership", chat_id=str(chat_id), user_id=str(user_id))
        try:
            async with pool.acquire() as conn:
                exists = await conn.fetchval(query, chat_id, user_id, company_id)
            repo_log.debug("Ownership check result", exists=exists)
            return exists is True
        except Exception as e:
            repo_log.exception("Failed to check chat ownership")
            return False 

    async def get_chat_messages(self, chat_id: uuid.UUID, user_id: uuid.UUID, company_id: uuid.UUID, limit: int = 100, offset: int = 0) -> List[ChatMessage]:
        pool = await get_db_pool()
        repo_log = log.bind(repo="PostgresChatRepository", action="get_chat_messages", chat_id=str(chat_id))

        owner = await self.check_chat_ownership(chat_id, user_id, company_id)
        if not owner:
            repo_log.warning("Attempt to get messages for chat not owned or non-existent")
            return []

        messages_query = """
        SELECT id, chat_id, role, content, sources, created_at FROM messages
        WHERE chat_id = $1 ORDER BY created_at ASC LIMIT $2 OFFSET $3;
        """
        try:
            async with pool.acquire() as conn:
                message_rows = await conn.fetch(messages_query, chat_id, limit, offset)

            messages = []
            for row in message_rows:
                msg_dict = dict(row)
                
                if msg_dict.get('sources') is None:
                     msg_dict['sources'] = None
                elif not isinstance(msg_dict.get('sources'), (list, dict, type(None))):
                    
                    log.warning("Unexpected type for 'sources' from DB", type=type(msg_dict['sources']).__name__, message_id=str(msg_dict.get('id')))
                    try:
                        
                        if isinstance(msg_dict['sources'], str):
                            msg_dict['sources'] = json.loads(msg_dict['sources'])
                        else:
                             msg_dict['sources'] = None 
                    except (json.JSONDecodeError, TypeError):
                         log.error("Failed to manually decode 'sources'", message_id=str(msg_dict.get('id')))
                         msg_dict['sources'] = None

                
                if not isinstance(msg_dict.get('sources'), (list, type(None))):
                    msg_dict['sources'] = None

                messages.append(ChatMessage(**msg_dict)) 

            repo_log.info(f"Retrieved {len(messages)} messages")
            return messages
        except Exception as e:
            repo_log.exception("Failed to get chat messages")
            raise

    async def save_message(self, chat_id: uuid.UUID, role: str, content: str, sources: Optional[List[Dict[str, Any]]] = None) -> uuid.UUID:
        pool = await get_db_pool()
        message_id = uuid.uuid4()
        repo_log = log.bind(repo="PostgresChatRepository", action="save_message", chat_id=str(chat_id), role=role)
        conn = None
        try:
            conn = await pool.acquire()
            async with conn.transaction():
                
                update_chat_query = "UPDATE chats SET updated_at = NOW() AT TIME ZONE 'UTC' WHERE id = $1 RETURNING id;"
                chat_updated = await conn.fetchval(update_chat_query, chat_id)
                if not chat_updated:
                    repo_log.error("Failed to update chat timestamp, chat might not exist")
                    raise ValueError(f"Chat with ID {chat_id} not found for saving message.")

                
                insert_message_query = """
                INSERT INTO messages (id, chat_id, role, content, sources, created_at)
                VALUES ($1, $2, $3, $4, $5, NOW() AT TIME ZONE 'UTC') RETURNING id;
                """
                
                result = await conn.fetchval(insert_message_query, message_id, chat_id, role, content, sources)

                if result and result == message_id:
                    repo_log.info("Message saved successfully", message_id=str(message_id))
                    return message_id
                else:
                    repo_log.error("Failed to save message, unexpected result", returned_id=result)
                    raise RuntimeError("Failed to save message, ID mismatch or not returned.")
        except Exception as e:
            repo_log.exception("Failed to save message")
            raise
        finally:
            if conn:
                await pool.release(conn)

    async def delete_chat(self, chat_id: uuid.UUID, user_id: uuid.UUID, company_id: uuid.UUID) -> bool:
        pool = await get_db_pool()
        repo_log = log.bind(repo="PostgresChatRepository", action="delete_chat", chat_id=str(chat_id), user_id=str(user_id))

        owner = await self.check_chat_ownership(chat_id, user_id, company_id)
        if not owner:
            repo_log.warning("Chat not found or does not belong to user, deletion skipped")
            return False

        conn = None
        try:
            conn = await pool.acquire()
            async with conn.transaction():
                repo_log.debug("Deleting associated messages...")
                await conn.execute("DELETE FROM messages WHERE chat_id = $1;", chat_id)
                repo_log.debug("Deleting associated query logs...")
                await conn.execute("DELETE FROM query_logs WHERE chat_id = $1;", chat_id)
                repo_log.debug("Deleting chat entry...")
                deleted_id = await conn.fetchval("DELETE FROM chats WHERE id = $1 RETURNING id;", chat_id)
                success = deleted_id is not None
                if success:
                    repo_log.info("Chat deleted successfully")
                else:
                    repo_log.error("Chat deletion failed after deleting dependencies")
                return success
        except Exception as e:
            repo_log.exception("Failed to delete chat")
            raise
        finally:
            if conn:
                await pool.release(conn)


# --- Log Repository Implementation ---
class PostgresLogRepository(LogRepositoryPort):
    """Implementación concreta del repositorio de logs usando PostgreSQL."""

    async def log_query_interaction(
        self,
        user_id: Optional[uuid.UUID],
        company_id: uuid.UUID,
        query: str,
        answer: str,
        retrieved_documents_data: List[Dict[str, Any]], 
        metadata: Optional[Dict[str, Any]] = None,
        chat_id: Optional[uuid.UUID] = None,
    ) -> uuid.UUID:
        pool = await get_db_pool()
        log_id = uuid.uuid4()
        repo_log = log.bind(repo="PostgresLogRepository", action="log_query_interaction", log_id=str(log_id))

        query_sql = """
        INSERT INTO query_logs (
            id, user_id, company_id, query, response,
            metadata, chat_id, created_at
        ) VALUES (
            $1, $2, $3, $4, $5, $6, $7, NOW() AT TIME ZONE 'UTC'
        ) RETURNING id;
        """
        
        final_metadata = metadata or {}
        final_metadata["retrieved_summary"] = [
            {"id": d.get("id"), "score": d.get("score"), "file_name": d.get("file_name")}
            for d in retrieved_documents_data
        ]

        try:
            async with pool.acquire() as connection:
                
                result = await connection.fetchval(
                    query_sql,
                    log_id, user_id, company_id, query, answer,
                    final_metadata, 
                    chat_id
                )
            if not result or result != log_id:
                repo_log.error("Failed to create query log entry", returned_id=result)
                raise RuntimeError("Failed to create query log entry")
            repo_log.info("Query interaction logged successfully")
            return log_id
        except Exception as e:
            repo_log.exception("Failed to log query interaction")
            raise RuntimeError(f"Failed to log query interaction: {e}") from e


# --- Chunk Content Repository Implementation ---
class PostgresChunkContentRepository(ChunkContentRepositoryPort):
    """Implementación concreta para obtener contenido de chunks desde PostgreSQL."""

    async def get_chunk_contents_by_company(self, company_id: uuid.UUID) -> Dict[str, Dict[str, Any]]:
        pool = await get_db_pool()
        query = """
        SELECT 
            dc.embedding_id, 
            dc.content,
            dc.metadata,
            dc.chunk_index,
            d.id as document_id,
            d.file_name
        FROM document_chunks dc
        JOIN documents d ON dc.document_id = d.id
        WHERE d.company_id = $1 AND dc.embedding_id IS NOT NULL;
        """
        repo_log = log.bind(repo="PostgresChunkContentRepository", action="get_chunk_contents_by_company", company_id=str(company_id))
        repo_log.warning("Fetching all chunk contents (keyed by embedding_id) for company, this might be memory intensive!")
        try:
            async with pool.acquire() as conn:
                rows = await conn.fetch(query, company_id)
            
            contents_with_meta = {}
            for row in rows:
                row_dict = dict(row)
                embedding_id = row_dict.get('embedding_id')
                content = row_dict.get('content')
                if not embedding_id or not content:
                    continue

                row_metadata = row_dict.get('metadata') or {}
                if not isinstance(row_metadata, dict):
                    row_metadata = {"metadata_raw": row_metadata}

                row_metadata.setdefault("chunk_index", row_dict.get('chunk_index'))
                document_id_str = str(row_dict['document_id']) if row_dict.get('document_id') else None
                row_metadata.setdefault("document_id", document_id_str)
                row_metadata.setdefault("file_name", row_dict.get('file_name'))

                contents_with_meta[embedding_id] = {
                    "content": content,
                    "document_id": document_id_str,
                    "file_name": row_dict.get('file_name'),
                    "metadata": row_metadata,
                }
            repo_log.info(f"Retrieved content and metadata for {len(contents_with_meta)} chunks (keyed by embedding_id)")
            return contents_with_meta
        except Exception as e:
            repo_log.exception("Failed to get chunk contents by company (keyed by embedding_id)")
            raise

    async def get_chunk_contents_by_ids(self, chunk_ids: List[str]) -> Dict[str, Dict[str, Any]]:
        if not chunk_ids:
            return {}
        pool = await get_db_pool()
        
        query = """
        SELECT 
            dc.embedding_id, 
            dc.content,
            dc.metadata,
            dc.chunk_index,
            d.id as document_id,
            d.file_name
        FROM document_chunks dc
        JOIN documents d ON dc.document_id = d.id
        WHERE dc.embedding_id = ANY($1::text[]);
        """
        repo_log = log.bind(repo="PostgresChunkContentRepository", action="get_chunk_contents_by_ids", count=len(chunk_ids))
        try:
            async with pool.acquire() as conn:
                rows = await conn.fetch(query, chunk_ids) 
            
            contents_with_meta = {}
            for row in rows:
                row_dict = dict(row)
                embedding_id = row_dict.get('embedding_id')
                content = row_dict.get('content')
                if not embedding_id or not content:
                    continue

                row_metadata = row_dict.get('metadata') or {}
                if not isinstance(row_metadata, dict):
                    row_metadata = {"metadata_raw": row_metadata}

                row_metadata.setdefault("chunk_index", row_dict.get('chunk_index'))
                document_id_str = str(row_dict['document_id']) if row_dict.get('document_id') else None
                row_metadata.setdefault("document_id", document_id_str)
                row_metadata.setdefault("file_name", row_dict.get('file_name'))

                contents_with_meta[embedding_id] = {
                    "content": content,
                    "document_id": document_id_str,
                    "file_name": row_dict.get('file_name'),
                    "metadata": row_metadata,
                }
            repo_log.info(f"Retrieved content and metadata for {len(contents_with_meta)} chunks (keyed by embedding_id) out of {len(chunk_ids)} requested")
            
            if len(contents_with_meta) != len(set(chunk_ids)): # Use set for accurate missing check
                found_ids = set(contents_with_meta.keys())
                missing_ids = [cid for cid in chunk_ids if cid not in found_ids]
                if missing_ids:
                    repo_log.warning("Could not find content/metadata for some requested chunk IDs (embedding_ids)", missing_ids=missing_ids)
            return contents_with_meta
        except Exception as e:
            repo_log.exception("Failed to get chunk contents and metadata by IDs (embedding_ids)")
            raise
```

## File: `app/infrastructure/retrievers/__init__.py`
```py

```

## File: `app/infrastructure/retrievers/remote_sparse_retriever_adapter.py`
```py
# query-service/app/infrastructure/retrievers/remote_sparse_retriever_adapter.py
import structlog
import uuid
from typing import List, Tuple

from app.application.ports.retrieval_ports import SparseRetrieverPort
from app.infrastructure.clients.sparse_search_service_client import SparseSearchServiceClient
from app.domain.models import SparseSearchResultItem # Para el tipo de resultado del cliente

log = structlog.get_logger(__name__)

class RemoteSparseRetrieverAdapter(SparseRetrieverPort):
    """
    Adaptador que utiliza SparseSearchServiceClient para realizar búsquedas dispersas
    llamando al servicio externo sparse-search-service.
    """
    def __init__(self, client: SparseSearchServiceClient):
        self.client = client
        log.info("RemoteSparseRetrieverAdapter initialized")

    async def search(self, query: str, company_id: uuid.UUID, top_k: int) -> List[Tuple[str, float]]:
        """
        Realiza una búsqueda dispersa llamando al servicio remoto.
        Devuelve una lista de tuplas (chunk_id, score).
        """
        adapter_log = log.bind(adapter="RemoteSparseRetrieverAdapter", action="search",
                               company_id=str(company_id), top_k=top_k)
        try:
            # El cliente devuelve una lista de SparseSearchResultItem
            search_results_domain: List[SparseSearchResultItem] = await self.client.search(
                query_text=query,
                company_id=company_id,
                top_k=top_k
            )

            # Mapear los resultados del dominio a List[Tuple[str, float]]
            # SparseSearchResultItem tiene 'chunk_id' y 'score'
            mapped_results: List[Tuple[str, float]] = [
                (item.chunk_id, item.score) for item in search_results_domain
            ]

            adapter_log.info(f"Sparse search successful via remote service. Returned {len(mapped_results)} results.")
            return mapped_results
        except ConnectionError as e:
            adapter_log.error("Connection error during remote sparse search.", error=str(e), exc_info=False)
            # Devolver una lista vacía para que el pipeline RAG pueda continuar
            # si la búsqueda dispersa no es estrictamente crítica.
            return []
        except ValueError as e: # Por ej. si la respuesta del servicio es inválida
            adapter_log.error("Value error during remote sparse search (invalid response from service?).", error=str(e), exc_info=True)
            return []
        except Exception as e:
            adapter_log.exception("Unexpected error during remote sparse search.")
            return [] # Devolver vacío en caso de error inesperado.

    async def health_check(self) -> bool:
        """
        Delega la verificación de salud al cliente del servicio de búsqueda dispersa.
        """
        return await self.client.check_health()
```

## File: `app/infrastructure/vectorstores/__init__.py`
```py

```

## File: `app/infrastructure/vectorstores/milvus_adapter.py`
```py
# query-service/app/infrastructure/vectorstores/milvus_adapter.py
import structlog
import asyncio
from typing import List, Optional, Dict, Any
import json 

from pymilvus import Collection, connections, utility, MilvusException, DataType
from haystack import Document 

from app.core.config import settings
from app.application.ports.vector_store_port import VectorStorePort
from app.domain.models import RetrievedChunk 

try:
    MILVUS_PK_FIELD = "pk_id"
    MILVUS_VECTOR_FIELD = "embedding" 
    MILVUS_CONTENT_FIELD = "content" 
    MILVUS_COMPANY_ID_FIELD = "company_id"
    MILVUS_DOCUMENT_ID_FIELD = "document_id"
    MILVUS_FILENAME_FIELD = "file_name"
    MILVUS_PAGE_FIELD = "page"
    MILVUS_TITLE_FIELD = "title"

    INGEST_SCHEMA_FIELDS = {
        "pk": MILVUS_PK_FIELD,
        "vector": MILVUS_VECTOR_FIELD,
        "content": MILVUS_CONTENT_FIELD,
        "company": MILVUS_COMPANY_ID_FIELD,
        "document": MILVUS_DOCUMENT_ID_FIELD,
        "filename": MILVUS_FILENAME_FIELD,
        "page": MILVUS_PAGE_FIELD,
        "title": MILVUS_TITLE_FIELD,
    }
except ImportError:
     structlog.getLogger(__name__).warning("Could not import ingest schema constants, using settings and fallbacks for field names.")
     INGEST_SCHEMA_FIELDS = {
        "pk": "pk_id",
        "vector": settings.MILVUS_EMBEDDING_FIELD,
        "content": settings.MILVUS_CONTENT_FIELD,
        "company": settings.MILVUS_COMPANY_ID_FIELD,
        "document": settings.MILVUS_DOCUMENT_ID_FIELD,
        "filename": settings.MILVUS_FILENAME_FIELD,
        "page": "page",
        "title": "title",
    }


log = structlog.get_logger(__name__)

class MilvusAdapter(VectorStorePort):
    _collection: Optional[Collection] = None
    _connected = False
    _alias = "query_service_milvus_adapter"
    _pk_field_name: str
    _vector_field_name: str
    _content_field_name: str # REFACTOR_5_2: Added
    _doc_id_field_name: str  # REFACTOR_5_2: Added
    _filename_field_name: str# REFACTOR_5_2: Added

    def __init__(self):
        self._pk_field_name = INGEST_SCHEMA_FIELDS["pk"]
        self._vector_field_name = INGEST_SCHEMA_FIELDS["vector"]
        self._content_field_name = INGEST_SCHEMA_FIELDS["content"] # REFACTOR_5_2
        self._doc_id_field_name = INGEST_SCHEMA_FIELDS["document"] # REFACTOR_5_2
        self._filename_field_name = INGEST_SCHEMA_FIELDS["filename"] # REFACTOR_5_2


    async def _ensure_connection(self):
        if not self._connected or self._alias not in connections.list_connections():
            uri = str(settings.MILVUS_URI)
            connect_log = log.bind(adapter="MilvusAdapter", action="connect", uri=uri, alias=self._alias)
            connect_log.debug("Attempting to connect to Milvus (Zilliz)...")
            try:
                connections.connect(
                    alias=self._alias,
                    uri=uri,
                    token=settings.ZILLIZ_API_KEY.get_secret_value(), 
                    timeout=settings.MILVUS_GRPC_TIMEOUT
                )
                self._connected = True
                connect_log.info("Connected to Milvus (Zilliz) successfully.")
            except MilvusException as e:
                connect_log.error("Failed to connect to Milvus (Zilliz).", error_code=e.code, error_message=e.message)
                self._connected = False
                raise ConnectionError(f"Milvus (Zilliz) connection failed (Code: {e.code}): {e.message}") from e
            except Exception as e:
                connect_log.error("Unexpected error connecting to Milvus (Zilliz).", error=str(e), exc_info=True)
                self._connected = False
                raise ConnectionError(f"Unexpected Milvus (Zilliz) connection error: {e}") from e

    async def _get_collection(self) -> Collection:
        await self._ensure_connection()

        if self._collection is None:
            collection_name = settings.MILVUS_COLLECTION_NAME
            collection_log = log.bind(adapter="MilvusAdapter", action="get_collection", collection=collection_name, alias=self._alias)
            collection_log.info(f"Attempting to access Milvus collection: '{collection_name}'")
            try:
                if not utility.has_collection(collection_name, using=self._alias):
                    collection_log.error("Milvus collection does not exist.", target_collection=collection_name)
                    raise RuntimeError(f"Milvus collection '{collection_name}' not found. Ensure ingest-service has created it.")

                collection = Collection(name=collection_name, using=self._alias)
                collection_log.debug("Loading Milvus collection into memory...")
                collection.load()
                collection_log.info("Milvus collection loaded successfully.")
                self._collection = collection

            except MilvusException as e:
                collection_log.error("Failed to get or load Milvus collection", error_code=e.code, error_message=e.message)
                if "multiple indexes" in e.message.lower(): 
                    collection_log.critical("Potential 'Ambiguous Index' error encountered. Please check Milvus indices for this collection.")
                raise RuntimeError(f"Milvus collection access error (Code: {e.code}): {e.message}") from e
            except Exception as e:
                 collection_log.exception("Unexpected error accessing Milvus collection")
                 raise RuntimeError(f"Unexpected error accessing Milvus collection: {e}") from e

        if not isinstance(self._collection, Collection): 
            log.critical("Milvus collection object is unexpectedly None or invalid type after initialization attempt.")
            raise RuntimeError("Failed to obtain a valid Milvus collection object.")

        return self._collection

    async def search(self, embedding: List[float], company_id: str, top_k: int) -> List[RetrievedChunk]:
        search_log = log.bind(adapter="MilvusAdapter", action="search", company_id=company_id, top_k=top_k)
        try:
            collection = await self._get_collection()

            search_params = settings.MILVUS_SEARCH_PARAMS.copy() 
            filter_expr = f'{INGEST_SCHEMA_FIELDS["company"]} == "{company_id}"'
            search_log.debug("Using filter expression", expr=filter_expr)

            # REFACTOR_5_2: Ensure all necessary fields for RetrievedChunk are fetched
            # including the ones previously only in MILVUS_METADATA_FIELDS
            # self._content_field_name, self._doc_id_field_name, self._filename_field_name
            # should be explicitly requested if they are not already part of INGEST_SCHEMA_FIELDS for general metadata
            output_fields_list = list(
                set(
                    [
                        self._pk_field_name,
                        self._vector_field_name, # Needed for MMR or other post-processing
                        self._content_field_name,
                        INGEST_SCHEMA_FIELDS["company"], # company_id
                        self._doc_id_field_name,   # document_id
                        self._filename_field_name, # file_name
                    ]
                    + settings.MILVUS_METADATA_FIELDS # Adds page, title etc.
                )
            )
            
            search_log.debug("Performing Milvus vector search...",
                             vector_field=self._vector_field_name,
                             output_fields=output_fields_list)

            loop = asyncio.get_running_loop()
            search_results = await loop.run_in_executor(
                None,
                lambda: collection.search(
                    data=[embedding],
                    anns_field=self._vector_field_name,
                    param=search_params,
                    limit=top_k,
                    expr=filter_expr,
                    output_fields=output_fields_list,
                    consistency_level="Strong" 
                )
            )

            search_log.debug(f"Milvus search completed. Hits: {len(search_results[0]) if search_results and search_results[0] else 0}")

            domain_chunks: List[RetrievedChunk] = []
            if search_results and search_results[0]:
                for hit in search_results[0]:
                    entity_data = hit.entity.to_dict() if hasattr(hit, 'entity') and hasattr(hit.entity, 'to_dict') else {}
                    
                    if not entity_data: # Fallback
                        entity_data = {field: hit.get(field) for field in output_fields_list if hit.get(field) is not None}

                    pk_id = str(hit.id) 
                    content = entity_data.get(self._content_field_name, "") # REFACTOR_5_2 Use specific field
                    embedding_vector = entity_data.get(self._vector_field_name) 
                    
                    metadata_dict = {k: v for k, v in entity_data.items() if k not in [self._vector_field_name, self._pk_field_name, self._content_field_name]}
                    
                    doc_id_val = entity_data.get(self._doc_id_field_name) # REFACTOR_5_2 Use specific field
                    comp_id_val = entity_data.get(INGEST_SCHEMA_FIELDS["company"])
                    fname_val = entity_data.get(self._filename_field_name) # REFACTOR_5_2 Use specific field

                    # REFACTOR_5_2: Populate content field directly in RetrievedChunk
                    chunk = RetrievedChunk(
                        id=pk_id,
                        content=content, # Populate content
                        score=hit.score,
                        metadata=metadata_dict,
                        embedding=embedding_vector,
                        document_id=str(doc_id_val) if doc_id_val else None,
                        file_name=str(fname_val) if fname_val else None,
                        company_id=str(comp_id_val) if comp_id_val else None
                    )
                    domain_chunks.append(chunk)

            search_log.info(f"Converted {len(domain_chunks)} Milvus hits to domain objects.")
            return domain_chunks

        except MilvusException as me:
             search_log.error("Milvus search failed", error_code=me.code, error_message=me.message)
             raise ConnectionError(f"Vector DB search error (Code: {me.code}): {me.message}") from me
        except Exception as e:
            search_log.exception("Unexpected error during Milvus search")
            raise ConnectionError(f"Vector DB search service error: {e}") from e

    async def fetch_vectors_by_ids(
        self,
        ids: List[str],
        *,
        collection_name: str | None = None,
    ) -> Dict[str, List[float]]:
        fetch_log = log.bind(adapter="MilvusAdapter", action="fetch_vectors_by_ids", num_ids=len(ids))
        if not ids:
            fetch_log.debug("No IDs provided, returning empty dict.")
            return {}

        try:
            _collection_obj = await self._get_collection()
            ids_json_array_str = json.dumps(ids)
            expr = f'{self._pk_field_name} in {ids_json_array_str}'
            
            fetch_log.debug("Querying Milvus for vectors by PKs", expr=expr, pk_field=self._pk_field_name, vector_field=self._vector_field_name)

            output_fields_to_fetch = [self._pk_field_name, self._vector_field_name]

            loop = asyncio.get_running_loop()
            res = await loop.run_in_executor(
                None,
                lambda: _collection_obj.query(
                    expr=expr,
                    output_fields=output_fields_to_fetch,
                    consistency_level="Strong"
                )
            )
            
            fetched_vectors = {str(row[self._pk_field_name]): row[self._vector_field_name] for row in res if self._pk_field_name in row and self._vector_field_name in row}
            fetch_log.info(f"Fetched {len(fetched_vectors)} vectors from Milvus out of {len(ids)} requested.")
            return fetched_vectors
        except MilvusException as me:
            fetch_log.error("Milvus query for vectors by IDs failed", error_code=me.code, error_message=me.message)
            raise ConnectionError(f"Vector DB query error (Code: {me.code}): {me.message}") from me
        except Exception as e:
            fetch_log.exception("Unexpected error during Milvus vector fetch by IDs")
            raise ConnectionError(f"Vector DB query service error: {e}") from e


    async def connect(self):
        await self._ensure_connection()

    async def disconnect(self):
        if self._connected and self._alias in connections.list_connections():
            log.info("Disconnecting from Milvus...", adapter="MilvusAdapter", alias=self._alias)
            try:
                connections.disconnect(self._alias)
                self._connected = False
                self._collection = None
                log.info("Disconnected from Milvus.", adapter="MilvusAdapter")
            except Exception as e:
                log.error("Error during Milvus disconnect", error=str(e), exc_info=True)
```

## File: `app/main.py`
```py
# query-service/app/main.py
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

# Configurar logging primero
from app.core.config import settings
from app.core.logging_config import setup_logging
setup_logging()
log = structlog.get_logger("query_service.main")

# Import Routers
from app.api.v1.endpoints import query as query_router_module
from app.api.v1.endpoints import chat as chat_router_module

# Import Ports and Adapters/Repositories for Dependency Injection
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

# Global state
SERVICE_READY = False
# Global instances for simplified DI
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


# --- Lifespan Manager ---
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

    # 0. Initialize Global HTTP Client
    try:
        http_client_instance = httpx.AsyncClient(
            timeout=settings.HTTP_CLIENT_TIMEOUT,
            limits=httpx.Limits(max_connections=100, max_keepalive_connections=20) 
        )
        log.info("Global HTTP client initialized.")
    except Exception as e_http_client:
        critical_failure_message = "Failed to initialize global HTTP client."
        log.critical(f"CRITICAL: {critical_failure_message}", error=str(e_http_client), exc_info=True)
        dependencies_ok = False


    # 1. Initialize DB Pool
    if dependencies_ok:
        try:
            await postgres_connector.get_db_pool()
            db_ready = await postgres_connector.check_db_connection()
            if db_ready:
                log.info("PostgreSQL connection pool initialized and verified.")
                chat_repo_instance = PostgresChatRepository()
                log_repo_instance = PostgresLogRepository()
                chunk_content_repo_instance = PostgresChunkContentRepository() 
            else:
                critical_failure_message = "Failed PostgreSQL connection verification during startup."
                log.critical(f"CRITICAL: {critical_failure_message}")
                dependencies_ok = False
        except Exception as e_pg:
            critical_failure_message = "Failed PostgreSQL pool initialization."
            log.critical(f"CRITICAL: {critical_failure_message}", error=str(e_pg), exc_info=True)
            dependencies_ok = False

    # 2. Initialize Embedding Service Client & Adapter
    if dependencies_ok:
        try:
            embedding_service_client_instance = EmbeddingServiceClient(
                base_url=str(settings.EMBEDDING_SERVICE_URL),
                timeout=settings.EMBEDDING_CLIENT_TIMEOUT
            )
            embedding_adapter_instance = RemoteEmbeddingAdapter(client=embedding_service_client_instance)
            await embedding_adapter_instance.initialize()
            
            emb_service_healthy = await embedding_adapter_instance.health_check()
            if emb_service_healthy:
                log.info("Embedding Service client and adapter initialized, health check passed.")
            else:
                # REFACTOR_5_4: Log as critical, but service *can* start if other core components are fine.
                # User queries needing new embeddings will fail later.
                critical_failure_message += " Embedding Service health check failed during startup."
                log.critical(f"CRITICAL (but non-blocking for startup): {critical_failure_message} URL: {settings.EMBEDDING_SERVICE_URL}")
                # dependencies_ok = False # Non-blocking, allow startup if other critical parts OK
        except Exception as e_embed:
            critical_failure_message += " Failed to initialize Embedding Service client/adapter."
            log.critical(f"CRITICAL (but non-blocking for startup): {critical_failure_message}", error=str(e_embed), exc_info=True, url=settings.EMBEDDING_SERVICE_URL)
            # dependencies_ok = False

    # 2.B. Initialize Sparse Search Service Client & Adapter
    if dependencies_ok and settings.BM25_ENABLED: 
        try:
            sparse_search_service_client_instance = SparseSearchServiceClient(
                base_url=str(settings.SPARSE_SEARCH_SERVICE_URL),
                timeout=settings.SPARSE_SEARCH_CLIENT_TIMEOUT
            )
            sparse_retriever_instance = RemoteSparseRetrieverAdapter(client=sparse_search_service_client_instance)
            
            sparse_service_healthy = await sparse_search_service_client_instance.check_health() 
            if sparse_service_healthy:
                log.info("Sparse Search Service client and adapter initialized, health check passed.")
            else:
                log.warning(f"Sparse Search Service health check failed during startup. URL: {settings.SPARSE_SEARCH_SERVICE_URL}. Sparse search may be unavailable but service will continue.")
                # Do not set dependencies_ok = False, as sparse search is optional enhancement
        except Exception as e_sparse:
            log.error(f"Failed to initialize Sparse Search Service client/adapter. Sparse search will be unavailable.", error=str(e_sparse), exc_info=True, url=str(settings.SPARSE_SEARCH_SERVICE_URL))
            sparse_retriever_instance = None 

    # 3. Initialize Milvus Adapter
    if dependencies_ok:
        try:
            vector_store_instance = MilvusAdapter()
            await vector_store_instance.connect() 
            log.info("Milvus Adapter initialized and collection checked/loaded.")
        except Exception as e_milvus:
            critical_failure_message += " Failed to initialize Milvus Adapter or load collection."
            log.critical(
                f"CRITICAL: {critical_failure_message} Ensure collection '{settings.MILVUS_COLLECTION_NAME}' exists and is accessible.",
                error=str(e_milvus), exc_info=True, adapter_error=getattr(e_milvus, 'message', 'N/A')
            )
            dependencies_ok = False

    # 4. Initialize LLM Adapter
    if dependencies_ok:
        try:
            llm_instance = LlamaCppAdapter(
                base_url=str(settings.LLM_API_BASE_URL),
                model_name=settings.LLM_MODEL_NAME,
                timeout=settings.HTTP_CLIENT_TIMEOUT,
                max_output_tokens=settings.LLM_MAX_OUTPUT_TOKENS,
            )
            llm_healthy = await llm_instance.health_check()
            if not llm_healthy:
                critical_failure_message += " LLM Adapter health check failed (llama.cpp unreachable)."
                log.critical(
                    f"CRITICAL: {critical_failure_message}",
                    llm_base_url=str(settings.LLM_API_BASE_URL),
                )
                dependencies_ok = False
            else:
                log.info(
                    "LlamaCppAdapter initialized successfully.",
                    llm_base_url=str(settings.LLM_API_BASE_URL),
                    llm_model=settings.LLM_MODEL_NAME,
                )
        except Exception as e_llm:
            critical_failure_message += " Failed to initialize LlamaCppAdapter."
            log.critical(
                f"CRITICAL: {critical_failure_message}",
                error=str(e_llm),
                exc_info=True,
                llm_base_url=str(settings.LLM_API_BASE_URL),
            )
            dependencies_ok = False
    
    # Initialize optional components (Diversity Filter)
    if dependencies_ok: # Check dependencies_ok before initializing optional that might depend on critical ones
        if settings.DIVERSITY_FILTER_ENABLED:
            try:
                if embedding_adapter_instance and embedding_adapter_instance.get_embedding_dimension() > 0 :
                    diversity_filter_instance = MMRDiversityFilter(lambda_mult=settings.QUERY_DIVERSITY_LAMBDA)
                    log.info("MMR Diversity Filter initialized.")
                else: # REFACTOR_5_4: Log warning if embedding adapter not ready for MMR
                    log.warning("MMR Diversity Filter enabled but embedding adapter is not available or has no dimension. Falling back to StubDiversityFilter.")
                    diversity_filter_instance = StubDiversityFilter()
            except Exception as e_diversity:
                log.error("Failed to initialize MMR Diversity Filter. Falling back to StubDiversityFilter.", error=str(e_diversity), exc_info=True)
                diversity_filter_instance = StubDiversityFilter()
        else: 
            log.info("Diversity filter disabled in settings, using StubDiversityFilter as placeholder.")
            diversity_filter_instance = StubDiversityFilter()

    # 5. Instantiate Use Case
    if dependencies_ok:
         try:
             if not http_client_instance: # This check should pass due to earlier initialization
                 raise RuntimeError("HTTP client instance is not available for AskQueryUseCase.")
             if not chat_repo_instance or not log_repo_instance or not vector_store_instance or \
                not llm_instance or not embedding_adapter_instance or not chunk_content_repo_instance: # REFACTOR_5_4: Add chunk_content_repo
                 raise RuntimeError("One or more critical repository/adapter instances are missing for AskQueryUseCase.")


             ask_query_use_case_instance = AskQueryUseCase(
                 chat_repo=chat_repo_instance,
                 log_repo=log_repo_instance,
                 vector_store=vector_store_instance,
                 llm=llm_instance,
                 embedding_adapter=embedding_adapter_instance,
                 http_client=http_client_instance,
                 sparse_retriever=sparse_retriever_instance, # Can be None if BM25_ENABLED=false or init failed
                 chunk_content_repo=chunk_content_repo_instance, 
                 diversity_filter=diversity_filter_instance # Can be StubDiversityFilter
             )
             log.info("AskQueryUseCase instantiated successfully.")
             SERVICE_READY = True 
             set_ask_query_use_case_instance(ask_query_use_case_instance, SERVICE_READY)
             log.info(f"{settings.PROJECT_NAME} service components initialized. SERVICE READY.")

         except Exception as e_usecase:
              critical_failure_message += " Failed to instantiate AskQueryUseCase." # REFACTOR_5_4: Append to message
              log.critical(f"CRITICAL: {critical_failure_message}", error=str(e_usecase), exc_info=True)
              SERVICE_READY = False
              set_ask_query_use_case_instance(None, False)
    else:
        # Log critical failure if not already caught by a more specific message
        if not critical_failure_message: critical_failure_message = "Unknown critical dependency failure during startup."
        log.critical(f"{settings.PROJECT_NAME} startup sequence aborted due to critical failure: {critical_failure_message}")
        log.critical("SERVICE NOT READY.")
        set_ask_query_use_case_instance(None, False)


    if not SERVICE_READY:
        if not critical_failure_message: critical_failure_message = "Unknown critical dependency failure during startup."
        log.critical(f"Startup finished. Critical failure detected: {critical_failure_message}. SERVICE NOT READY.")


    yield 

    # --- Shutdown Logic ---
    log.info(f"Shutting down {settings.PROJECT_NAME}...")
    if http_client_instance:
        await http_client_instance.aclose()
        log.info("Global HTTP client closed.")
    await postgres_connector.close_db_pool()
    if vector_store_instance and hasattr(vector_store_instance, 'disconnect'):
        try: await vector_store_instance.disconnect()
        except Exception as e_milvus_close: log.error("Error during Milvus disconnect", error=str(e_milvus_close), exc_info=True)
    
    if embedding_service_client_instance:
        try: await embedding_service_client_instance.close()
        except Exception as e_emb_client_close: log.error("Error closing EmbeddingServiceClient", error=str(e_emb_client_close), exc_info=True)
    
    if sparse_search_service_client_instance: 
        try: await sparse_search_service_client_instance.close()
        except Exception as e_sparse_client_close: log.error("Error closing SparseSearchServiceClient", error=str(e_sparse_client_close), exc_info=True)
    if llm_instance and hasattr(llm_instance, "close"):
        try: await llm_instance.close()
        except Exception as e_llm_close: log.error("Error closing LLM adapter", error=str(e_llm_close), exc_info=True)
        
    log.info("Shutdown complete.")

app = FastAPI(
    title=settings.PROJECT_NAME,
    openapi_url=f"{settings.API_V1_STR}/openapi.json",
    version="0.3.3", 
    description="Microservice to handle user queries using RAG pipeline, chat history, remote embedding, remote reranking, and remote sparse search.",
    lifespan=lifespan
)

@app.middleware("http")
async def add_request_id_timing_logging(request: Request, call_next):
    start_time = asyncio.get_event_loop().time()
    request_id = request.headers.get("x-request-id", str(uuid.uuid4()))
    structlog.contextvars.bind_contextvars(request_id=request_id)
    req_log = log.bind(method=request.method, path=str(request.url.path), client=request.client.host if request.client else "unknown")
    req_log.info("Request received")
    response = None
    try:
        response = await call_next(request)
        process_time = (asyncio.get_event_loop().time() - start_time) * 1000
        resp_log = req_log.bind(status_code=response.status_code, duration_ms=round(process_time, 2))
        log_level_method = "warning" if 400 <= response.status_code < 500 else "error" if response.status_code >= 500 else "info"
        getattr(resp_log, log_level_method)("Request finished")
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Process-Time-Ms"] = f"{process_time:.2f}"
    except Exception as e_middleware:
        process_time = (asyncio.get_event_loop().time() - start_time) * 1000
        exc_log = req_log.bind(status_code=500, duration_ms=round(process_time, 2))
        exc_log.exception("Unhandled exception during request processing middleware") 
        response = JSONResponse(status_code=fastapi_status.HTTP_500_INTERNAL_SERVER_ERROR, content={"detail": "Internal Server Error"})
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Process-Time-Ms"] = f"{process_time:.2f}"
    finally: structlog.contextvars.clear_contextvars()
    return response

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    log_level_method = log.warning if exc.status_code < 500 else log.error
    log_level_method("HTTP Exception caught by handler", status_code=exc.status_code, detail=exc.detail)
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    error_details_val = []
    try: error_details_val = exc.errors()
    except Exception: error_details_val = [{"loc": [], "msg": "Failed to parse validation errors.", "type": "internal_parsing_error"}]
    log.warning("Request Validation Error caught by handler", errors=error_details_val)
    return JSONResponse(status_code=fastapi_status.HTTP_422_UNPROCESSABLE_ENTITY, content={"detail": error_details_val})

@app.exception_handler(ResponseValidationError)
async def response_validation_exception_handler(request: Request, exc: ResponseValidationError):
    log.error("Response Validation Error caught by handler", errors=exc.errors(), exc_info=True)
    return JSONResponse(status_code=fastapi_status.HTTP_500_INTERNAL_SERVER_ERROR, content={"detail": "Internal Server Error: Failed to serialize response."})

@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    log.exception("Unhandled Exception caught by generic handler") 
    return JSONResponse(status_code=fastapi_status.HTTP_500_INTERNAL_SERVER_ERROR, content={"detail": "Internal Server Error"})


def get_chat_repository() -> ChatRepositoryPort:
    if not chat_repo_instance:
        log.error("Dependency Injection Failed: ChatRepository requested but not initialized.")
        raise HTTPException(status_code=503, detail="Chat service component not available.")
    return chat_repo_instance

app.include_router(query_router_module.router, prefix=settings.API_V1_STR, tags=["Query Interaction"])
app.include_router(chat_router_module.router, prefix=settings.API_V1_STR, tags=["Chat Management"])
log.info("Routers included", prefix=settings.API_V1_STR)

@app.get("/", tags=["Health Check"], summary="Service Liveness/Readiness Check")
async def read_root():
    health_log = log.bind(check="liveness_readiness_root")
    if not SERVICE_READY:
        health_log.warning("Health check (root) failed: Service not ready.", service_ready_flag=SERVICE_READY)
        raise HTTPException(status_code=fastapi_status.HTTP_503_SERVICE_UNAVAILABLE, detail="Service Not Ready. Check startup logs for critical failures.")

    # Check Embedding Service Health (Considered critical for RAG)
    if not embedding_adapter_instance :
        health_log.error("Health check (root) CRITICAL: Embedding Adapter instance not available, inconsistent with SERVICE_READY state.")
        raise HTTPException(status_code=fastapi_status.HTTP_503_SERVICE_UNAVAILABLE, detail="Critical dependency (Embedding Adapter) missing.")
    
    emb_adapter_healthy = await embedding_adapter_instance.health_check()
    if not emb_adapter_healthy:
        health_log.error("Health check (root) CRITICAL: Embedding Adapter reports unhealthy dependency (Embedding Service).")
        # REFACTOR_5_4: If embedding service is critical for any response, this should fail the health check.
        raise HTTPException(status_code=fastapi_status.HTTP_503_SERVICE_UNAVAILABLE, detail="Critical dependency (Embedding Service) is unhealthy.")

    if not llm_instance:
        health_log.error("Health check (root) CRITICAL: LLM adapter instance missing.")
        raise HTTPException(status_code=fastapi_status.HTTP_503_SERVICE_UNAVAILABLE, detail="Critical dependency (LLM adapter) missing.")
    if hasattr(llm_instance, "health_check"):
        llm_healthy = await llm_instance.health_check()
        if not llm_healthy:
            health_log.error("Health check (root) CRITICAL: LLM adapter reports unhealthy dependency (llama.cpp).")
            raise HTTPException(status_code=fastapi_status.HTTP_503_SERVICE_UNAVAILABLE, detail="Critical dependency (LLM service) is unhealthy.")
    
    # Check Sparse Search Service Health (if enabled, but non-blocking for overall service health)
    if settings.BM25_ENABLED: 
        if sparse_search_service_client_instance: 
            sparse_service_healthy = await sparse_search_service_client_instance.check_health()
            if not sparse_service_healthy:
                health_log.warning("Health check (root) warning: Sparse Search Service reports unhealthy. Sparse search functionality may be impaired but service can continue if RAG is primary.")
            else:
                health_log.debug("Sparse Search Service health check successful via root.")
        else: 
             # This indicates BM25_ENABLED is true, but the client didn't initialize, which is a configuration/startup issue.
             # While non-blocking for a basic RAG response, it's a degradation of expected functionality.
            health_log.warning("Health check (root) warning: BM25_ENABLED is true, but Sparse Search Service client is not available. Sparse search functionality will be missing.")

    health_log.debug("Health check (root) passed (core dependencies OK).")
    return PlainTextResponse("OK", status_code=fastapi_status.HTTP_200_OK)

if __name__ == "__main__":
    port = int(os.getenv("PORT", "8001")) 
    log_level_str = settings.LOG_LEVEL.lower()
    print(f"----- Starting {settings.PROJECT_NAME} locally on port {port} -----")
    uvicorn.run("app.main:app", host="0.0.0.0", port=port, reload=True, log_level=log_level_str)


#jfu 3
```

## File: `app/models/__init__.py`
```py

```

## File: `app/pipelines/rag_pipeline.py`
```py

```

## File: `app/prompts/general_template_gemini_v2.txt`
```txt
════════════════════════════════════════════════════════════════════
A T E N E X · RESPUESTA SIN CONTEXTO
════════════════════════════════════════════════════════════════════

IDENTIDAD Y TONO
- Te llamas **Atenex**. Ofreces apoyo profesional en español latino claro.
- Tu objetivo es ser útil aunque no tengas fragmentos de documentos para citar.
- Mantén un tono cordial, directo y honesto; evita inventar datos específicos que no se deriven de la pregunta u otras entradas.

ENTRADAS DISPONIBLES
- Pregunta actual del usuario: {{ query }}
{% if chat_history %}- Historial reciente (de más antiguo a más nuevo):
{{ chat_history }}
{% else %}- No hay historial de conversación disponible.
{% endif %}

LINEAMIENTOS PRINCIPALES
1. **Transparencia:** Aclara que no se encontraron fragmentos de documentos relevantes para responder directamente.
2. **Utilidad:** Ofrece una orientación general basada en conocimientos comunes, pasos sugeridos o preguntas de seguimiento que puedan guiar al usuario.
3. **Precisión:** Si la pregunta requiere datos concretos que no posees, dilo explícitamente y sugiere cómo obtenerlos.
4. **Formato Markdown:** Usa Markdown sencillo (`**negritas**`, listas con `-` o `1.`) para mejorar la legibilidad de `respuesta_detallada`.
5. **JSON obligatorio:** Devuelve únicamente un objeto JSON válido con la estructura indicada. No añadas texto fuera del JSON.
6. **Fuentes citadas:** Como no hay fragmentos, la lista `fuentes_citadas` debe ser un arreglo vacío `[]`.
7. **Pregunta siguiente:** Propón una `siguiente_pregunta_sugerida` solo si aporta valor; de lo contrario, usa `null`.

FORMATO DE RESPUESTA (OBLIGATORIO)
{
	"resumen_ejecutivo": string o null,
	"respuesta_detallada": string,
	"fuentes_citadas": [],
	"siguiente_pregunta_sugerida": string o null
}

RECUERDA
- Debes mencionar explícitamente que no se encontraron documentos relevantes.
- No incluyas referencias `[Doc N]` ni enlaces a documentos inexistentes.
- Mantén la respuesta concisa pero empática.
- Devuelve únicamente el JSON; cualquier texto adicional invalida la salida.


```

## File: `app/prompts/map_prompt_template.txt`
```txt
Eres un analista experto. Tu única tarea es extraer información de los siguientes fragmentos que sea relevante para la pregunta del usuario.

PREGUNTA: "{{ original_query }}"

FRAGMENTOS A ANALIZAR:
{% for doc in documents %}
---
Fragmento ID: {{ doc.id }} (Archivo: {{ doc.meta.file_name }})
Contenido:
{{ doc.content | trim }}
---
{% endfor %}

INSTRUCCIONES:
1. Si encuentras información que ayuda a responder la pregunta, extrae los puntos clave y resúmelos. Incluye referencias al archivo de origen.
2. Si los fragmentos NO tienen nada que ver con la pregunta, responde EXACTAMENTE: "NO_RELEVANT_INFO".
3. No inventes información.

TU ANÁLISIS:
```

## File: `app/prompts/rag_template_gemini_v2.txt`
```txt
Eres Atenex, un asistente empresarial útil y veraz.
Tu tarea es responder a la pregunta del usuario basándote EXCLUSIVAMENTE en los fragmentos de documentos proporcionados abajo.

INSTRUCCIONES DE RESPUESTA:
1. Analiza los fragmentos proporcionados.
2. Genera una respuesta detallada en formato Markdown (listas, negritas).
3. CITA SIEMPRE tus fuentes usando la etiqueta [Doc N] correspondiente al fragmento usado.
4. Si la información no está en los fragmentos, responde "No encontré información específica en los documentos." y deja la lista de fuentes vacía.
5. Devuelve SOLAMENTE un JSON válido con esta estructura:
{
  "resumen_ejecutivo": "string breve o null",
  "respuesta_detallada": "string con la respuesta completa y citas [Doc N]",
  "fuentes_citadas": [
    { "id_documento": "ID_EXACTO_DEL_FRAGMENTO", "nombre_archivo": "nombre del archivo", "pagina": "número o null", "score": 0.0, "cita_tag": "[Doc N]" }
  ],
  "siguiente_pregunta_sugerida": "string o null"
}

PREGUNTA DEL USUARIO:
{{ query }}

HISTORIAL RECIENTE:
{% if chat_history %}{{ chat_history }}{% else %}N/A{% endif %}

FRAGMENTOS DE DOCUMENTOS (CONTEXTO):
{% if documents %}
{% for doc in documents %}
---
[Doc {{ loop.index }}]
ID_Fragmento: {{ doc.id }}
Archivo: {{ doc.meta.file_name | default("Desconocido") }}
ID_Doc_Origen: {{ doc.meta.document_id | default("N/A") }}
Página: {{ doc.meta.page | default("?") }}
Score: {{ "%.3f"|format(doc.score) if doc.score is not none else "0.0" }}

CONTENIDO:
{{ doc.content | trim }}
---
{% endfor %}
{% else %}
(No hay documentos relevantes disponibles)
{% endif %}

RESPUESTA JSON:
```

## File: `app/prompts/reduce_prompt_template_v2.txt`
```txt
Eres Atenex. Sintetiza la información extraída para responder al usuario en formato JSON.

PREGUNTA: {{ original_query }}

INFORMACIÓN EXTRAÍDA (De fase previa):
{{ mapped_responses }}

DATOS DE FUENTES ORIGINALES (Para citas):
{% for doc in original_documents_for_citation %}
[Doc {{ loop.index }}] ID: {{ doc.id }}, Archivo: {{ doc.meta.file_name }}, Score: {{ "%.2f"|format(doc.score) if doc.score else 0 }}
{% endfor %}

INSTRUCCIONES:
1. Genera una respuesta final unificando la información extraída.
2. Usa Markdown en "respuesta_detallada".
3. Cita usando [Doc N] basándote en la lista de "DATOS DE FUENTES ORIGINALES".
4. Devuelve SOLAMENTE JSON válido con esta estructura:
{
  "resumen_ejecutivo": "string o null",
  "respuesta_detallada": "respuesta completa con citas",
  "fuentes_citadas": [ { "id_documento": "ID", "nombre_archivo": "nombre", "pagina": "pag", "score": 0.0, "cita_tag": "[Doc N]" } ],
  "siguiente_pregunta_sugerida": "string o null"
}

RESPUESTA JSON:
```

## File: `app/utils/__init__.py`
```py

```

## File: `app/utils/helpers.py`
```py
# ./app/utils/helpers.py
# (Actualmente vacío, añadir funciones de utilidad si son necesarias)
import structlog

log = structlog.get_logger(__name__)

# Ejemplo de función de utilidad potencial:
def truncate_text(text: str, max_length: int) -> str:
    """Trunca un texto a una longitud máxima, añadiendo puntos suspensivos."""
    if not text:
        return ""
    if len(text) <= max_length:
        return text
    return text[:max_length - 3] + "..."
```

## File: `pyproject.toml`
```toml
[tool.poetry]
name = "query-service"
version = "1.4.0"
description = "Query service for SaaS B2B using Clean Architecture, PyMilvus, Haystack & Advanced RAG with remote embeddings, reranking, and sparse search."
authors = ["Nyro <dev@atenex.com>"]
readme = "README.md"

[tool.poetry.dependencies]
python = "^3.10"

fastapi = "^0.110.0"
uvicorn = {extras = ["standard"], version = "^0.28.0"}
gunicorn = "^21.2.0"
pydantic = {extras = ["email"], version = "^2.6.4"} 
pydantic-settings = "^2.2.1"
httpx = ">=0.27.0,<1.0.0" 
asyncpg = "^0.29.0"
python-jose = {extras = ["cryptography"], version = "^3.3.0"}
tenacity = "^8.2.3"
structlog = "^24.1.0"
haystack-ai = "^2.0.1" 
pymilvus = "==2.5.3" 
numpy = "1.26.4" 
tiktoken = "^0.9.0"
google-generativeai = "^0.4.1"

[tool.poetry.dev-dependencies]
pytest = "^7.4.4"
pytest-asyncio = "^0.21.1"

[build-system]
requires = ["poetry-core>=1.0.0"]
build-backend = "poetry.core.masonry.api"
```
