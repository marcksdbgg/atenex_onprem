# query-service/app/application/use_cases/ask_query_use_case.py
import structlog
import asyncio
import uuid
import re
import time
from typing import Dict, Any, List, Tuple, Optional
from datetime import datetime, timezone, timedelta
import httpx
import json
from pydantic import ValidationError

from app.application.ports import (
    ChatRepositoryPort, LogRepositoryPort, VectorStorePort, LLMPort,
    SparseRetrieverPort, DiversityFilterPort, ChunkContentRepositoryPort,
    EmbeddingPort, RerankerPort 
)
from app.domain.models import RetrievedChunk, ChatMessage, RespuestaEstructurada, FuenteCitada 
from app.application.use_cases.ask_query.prompt_service import PromptService, PromptType
from app.application.use_cases.ask_query.token_accountant import TokenAccountant
from app.application.use_cases.ask_query.types import TokenAnalysis
from app.application.use_cases.ask_query.config_types import PromptBudgetConfig, MapReduceConfig, RetrievalConfig
from app.application.use_cases.ask_query.pipeline import RAGPipeline
from app.application.use_cases.ask_query.steps import (
    EmbeddingStep, RetrievalStep, FusionStep, ContentFetchStep, RerankStep, FilterStep,
    DirectGenerationStep, MapReduceGenerationStep, AdaptiveGenerationStep
)
from app.core.config import settings
from app.utils.helpers import truncate_text
from fastapi import HTTPException, status

log = structlog.get_logger(__name__)

GREETING_REGEX = re.compile(r"^\s*(hola|hello|hi|buenos días|buenas tardes|buenas noches|hey|qué tal|hi there)\s*[\.,!?]*\s*$", re.IGNORECASE)
RRF_K = 60 

MAP_REDUCE_NO_RELEVANT_INFO = "No hay información relevante en el fragmento"


def format_time_delta(dt: datetime) -> str:
    now = datetime.now(timezone.utc)
    delta = now - dt
    if delta < timedelta(minutes=1):
        return "justo ahora"
    elif delta < timedelta(hours=1):
        minutes = int(delta.total_seconds() / 60)
        return f"hace {minutes} min" if minutes > 1 else "hace 1 min"
    elif delta < timedelta(days=1):
        hours = int(delta.total_seconds() / 3600)
        return f"hace {hours} h" if hours > 1 else "hace 1 h"
    else:
        days = delta.days
        return f"hace {days} días" if days > 1 else "hace 1 día"


class AskQueryUseCase:
    def __init__(self,
                 chat_repo: ChatRepositoryPort,
                 log_repo: LogRepositoryPort,
        return self._token_accountant.count_tokens_for_text(text)

    async def _build_prompt(
        self,
        prompt_type: PromptType,
        *,
        query: str = "",
        chunks: Optional[List[RetrievedChunk]] = None,
        chat_history: Optional[str] = None,
        prompt_data_override: Optional[Dict[str, Any]] = None,
    ) -> str:
        prompt_payload: Dict[str, Any] = dict(prompt_data_override) if prompt_data_override else {}
        prompt_payload.setdefault("query", query)

        if chunks:
            prompt_payload.setdefault("documents", self._prompt_service.create_documents(chunks))

        self._prompt_service.include_chat_history(prompt_payload, chat_history)
        return await self._prompt_service.build(prompt_type, **prompt_payload)

    async def _embed_query(self, query: str) -> List[float]:
        embed_log = log.bind(action="_embed_query_use_case_call_remote")
        try:
            embedding = await self.embedding_adapter.embed_query(query)
            if not embedding or not isinstance(embedding, list) or not all(isinstance(f, float) for f in embedding):
                embed_log.error("Invalid embedding received from adapter", received_embedding_type=type(embedding).__name__)
                raise ValueError("Embedding adapter returned invalid or empty vector.")
            embed_log.debug("Query embedded successfully via remote adapter", vector_dim=len(embedding))
            return embedding
        except ConnectionError as e: 
            embed_log.error("Embedding failed: Connection to embedding service failed.", error=str(e), exc_info=False)
            raise ConnectionError(f"Embedding service error: {e}") from e
        except ValueError as e: 
            embed_log.error("Embedding failed: Invalid data from embedding service.", error=str(e), exc_info=False)
            raise ValueError(f"Embedding service data error: {e}") from e
        except Exception as e: 
            embed_log.error("Unexpected error during query embedding via adapter", error=str(e), exc_info=True)
            raise ConnectionError(f"Unexpected error contacting embedding service: {e}") from e

    def _format_chat_history(self, messages: List[ChatMessage]) -> str:
        if not messages:
            return ""
        history_str = []
        for msg in reversed(messages): 
            role = "Usuario" if msg.role == 'user' else "Atenex"
            time_mark = format_time_delta(msg.created_at)
            history_str.append(f"{role} ({time_mark}): {msg.content}")
        return "\n".join(reversed(history_str))

    def _reciprocal_rank_fusion(self,
                                dense_results: List[RetrievedChunk],
                                sparse_results: List[Tuple[str, float]], 
                                k: int = RRF_K) -> Dict[str, float]:
        fused_scores: Dict[str, float] = {}
        for rank, chunk in enumerate(dense_results):
            if chunk.id: 
                fused_scores[chunk.id] = fused_scores.get(chunk.id, 0.0) + 1.0 / (k + rank + 1)
        
        for rank, (chunk_id, _) in enumerate(sparse_results):
            if chunk_id: 
                fused_scores[chunk_id] = fused_scores.get(chunk_id, 0.0) + 1.0 / (k + rank + 1)
        return fused_scores

    async def _fetch_content_for_fused_results(
        self,
        fused_scores: Dict[str, float], 
        dense_map: Dict[str, RetrievedChunk], 
        top_n: int
        ) -> List[RetrievedChunk]:
        fetch_log = log.bind(action="fetch_content_for_fused", top_n=top_n, fused_count=len(fused_scores))
        if not fused_scores: return []

        sorted_chunk_ids_with_scores = sorted(fused_scores.items(), key=lambda item: item[1], reverse=True)
        top_ids_with_scores_tuples: List[Tuple[str, float]] = sorted_chunk_ids_with_scores[:top_n]
        
        fetch_log.debug("Top IDs after fusion", top_ids_count=len(top_ids_with_scores_tuples))

        chunks_with_content: List[RetrievedChunk] = []
        ids_needing_data: List[str] = [] 
        placeholder_map: Dict[str, RetrievedChunk] = {}

        for cid, fused_score_val in top_ids_with_scores_tuples:
            if not cid:
                 fetch_log.warning("Skipping invalid chunk ID found during fusion processing.")
                 continue
            
            original_chunk_from_dense = dense_map.get(cid)

            if original_chunk_from_dense and original_chunk_from_dense.content:
                original_chunk_from_dense.score = fused_score_val 
                chunks_with_content.append(original_chunk_from_dense)
            else: 
                
                chunk_placeholder = RetrievedChunk(
                    id=cid,
                    score=fused_score_val, 
                    content=None, 
                    metadata=original_chunk_from_dense.metadata if original_chunk_from_dense and original_chunk_from_dense.metadata else {"retrieval_source": "sparse_or_fused_no_initial_meta"},
                    embedding=original_chunk_from_dense.embedding if original_chunk_from_dense and original_chunk_from_dense.embedding else None,
                    document_id=original_chunk_from_dense.document_id if original_chunk_from_dense else None,
                    file_name=original_chunk_from_dense.file_name if original_chunk_from_dense else None,
                    company_id=original_chunk_from_dense.company_id if original_chunk_from_dense else None 
                )
                chunks_with_content.append(chunk_placeholder) 
                placeholder_map[cid] = chunk_placeholder 
                ids_needing_data.append(cid) 

        if ids_needing_data and self.chunk_content_repo:
             fetch_log.info("Fetching content and metadata for chunks missing data", count=len(ids_needing_data))
             try:
                 
                 chunk_data_map: Dict[str, Dict[str, Any]] = await self.chunk_content_repo.get_chunk_contents_by_ids(ids_needing_data)
                 
                 for cid_item, data in chunk_data_map.items():
                     if cid_item in placeholder_map: 
                          placeholder_map[cid_item].content = data.get("content")
                          
                          if not placeholder_map[cid_item].document_id:
                            placeholder_map[cid_item].document_id = data.get("document_id")
                          if not placeholder_map[cid_item].file_name:
                            placeholder_map[cid_item].file_name = data.get("file_name")
                          
                          
                          if placeholder_map[cid_item].metadata:
                            placeholder_map[cid_item].metadata.update({
                                "content_fetched_for_sparse": True,
                                "fetched_document_id": data.get("document_id"),
                                "fetched_file_name": data.get("file_name")
                            })
                          else: 
                            placeholder_map[cid_item].metadata = {
                                "content_fetched_for_sparse": True,
                                "fetched_document_id": data.get("document_id"),
                                "fetched_file_name": data.get("file_name")
                            }

                 missing_after_fetch = [cid_item_check for cid_item_check in ids_needing_data if cid_item_check not in chunk_data_map or not chunk_data_map[cid_item_check].get("content")]
                 if missing_after_fetch:
                      fetch_log.warning("Content/metadata not found or empty for some chunks after fetch", missing_ids=missing_after_fetch)
             except Exception as e_content_fetch:
                 fetch_log.exception("Failed to fetch content/metadata for fused results", error=str(e_content_fetch))
        elif ids_needing_data:
            fetch_log.warning("Cannot fetch content/metadata for sparse/fused results, ChunkContentRepository not available.")

        final_chunks_with_content = [c for c in chunks_with_content if c.content and c.content.strip()]
        fetch_log.debug("Chunks remaining after content check and fetch", count=len(final_chunks_with_content))
        
        final_chunks_with_content.sort(key=lambda c: c.score or 0.0, reverse=True)
        return final_chunks_with_content
        
    async def _handle_llm_response(
        self,
        json_answer_str: str,
        query: str,
        company_id: uuid.UUID,
        user_id: uuid.UUID,
        final_chat_id: uuid.UUID,
        original_chunks_for_citation: List[RetrievedChunk], 
        pipeline_stages_used: List[str],
        map_reduce_used: bool = False,
        retriever_k_effective: int = 0,
        fusion_fetch_k_effective: int = 0,
        num_chunks_after_rerank_or_fusion_fetch_effective: int = 0,
        num_final_chunks_sent_to_llm_effective: int = 0,
        num_history_messages_effective: int = 0
    ) -> Tuple[str, List[RetrievedChunk], Optional[uuid.UUID]]:
        
        llm_handler_log = log.bind(action="_handle_llm_response", chat_id=str(final_chat_id))
        answer_for_user: str
        retrieved_chunks_for_api_response: List[RetrievedChunk] = []
        assistant_sources_for_db: List[Dict[str, Any]] = []
        log_id: Optional[uuid.UUID] = None
        
        validation_json_err = None
        json_decode_err = None

        try:
            structured_answer_obj = RespuestaEstructurada.model_validate_json(json_answer_str)
            answer_for_user = structured_answer_obj.respuesta_detallada
            
            llm_handler_log.info("LLM response successfully parsed as RespuestaEstructurada.",
                                 has_summary=bool(structured_answer_obj.resumen_ejecutivo),
                                 num_fuentes_citadas_by_llm=len(structured_answer_obj.fuentes_citadas),
                                 siguiente_pregunta_sugerida=structured_answer_obj.siguiente_pregunta_sugerida)

            # Use fuentues_citadas from LLM response as the primary source for building `retrieved_chunks_for_api_response`
            map_id_to_original_chunk = {chunk.id: chunk for chunk in original_chunks_for_citation if chunk.id and chunk.content}

            processed_chunk_ids_for_response = set()

            if not structured_answer_obj.fuentes_citadas:
                 llm_handler_log.info("LLM response did not include any 'fuentes_citadas'.")

            for cited_source_by_llm in structured_answer_obj.fuentes_citadas:
                if cited_source_by_llm.id_documento and cited_source_by_llm.id_documento in map_id_to_original_chunk:
                    original_chunk = map_id_to_original_chunk[cited_source_by_llm.id_documento]
                    if original_chunk.id not in processed_chunk_ids_for_response:
                        retrieved_chunks_for_api_response.append(original_chunk)
                        processed_chunk_ids_for_response.add(original_chunk.id)

            if not retrieved_chunks_for_api_response and structured_answer_obj.fuentes_citadas:
                llm_handler_log.warning("LLM cited sources, but no direct match found by id_documento. Using filename as fallback or top N.")
                for cited_source_by_llm in structured_answer_obj.fuentes_citadas:
                    if len(retrieved_chunks_for_api_response) >= self.settings.NUM_SOURCES_TO_SHOW: break
                    found_by_name = False
                    for orig_chunk in original_chunks_for_citation:
                        if orig_chunk.id not in processed_chunk_ids_for_response and \
                           orig_chunk.file_name == cited_source_by_llm.nombre_archivo:
                            retrieved_chunks_for_api_response.append(orig_chunk)
                            processed_chunk_ids_for_response.add(orig_chunk.id)
                            found_by_name = True
                            break
                    if not found_by_name:
                         llm_handler_log.info("LLM cited source not found by filename either", cited_source_name=cited_source_by_llm.nombre_archivo)
            
            if len(retrieved_chunks_for_api_response) < self.settings.NUM_SOURCES_TO_SHOW and original_chunks_for_citation:
                llm_handler_log.debug("Filling remaining source slots with top original chunks provided to LLM/MapReduce.")
                for chunk in original_chunks_for_citation:
                    if len(retrieved_chunks_for_api_response) >= self.settings.NUM_SOURCES_TO_SHOW:
                        break
                    if chunk.id not in processed_chunk_ids_for_response:
                        retrieved_chunks_for_api_response.append(chunk)
                        processed_chunk_ids_for_response.add(chunk.id)

            # Build assistant sources payload for chat history persistence.
            if retrieved_chunks_for_api_response:
                retrieved_chunks_map = {chunk.id: chunk for chunk in retrieved_chunks_for_api_response if chunk.id}
                assistant_sources_for_db = []

                sources_limit = self.settings.NUM_SOURCES_TO_SHOW
                for cited_source_by_llm in structured_answer_obj.fuentes_citadas:
                    if len(assistant_sources_for_db) >= sources_limit:
                        break
                    citation_payload: Dict[str, Any] = {
                        "document_id": cited_source_by_llm.id_documento,
                        "file_name": cited_source_by_llm.nombre_archivo,
                        "pagina": cited_source_by_llm.pagina,
                        "score": cited_source_by_llm.score,
                        "cita_tag": cited_source_by_llm.cita_tag,
                    }

                    chunk_for_citation = None
                    if cited_source_by_llm.id_documento and cited_source_by_llm.id_documento in retrieved_chunks_map:
                        chunk_for_citation = retrieved_chunks_map[cited_source_by_llm.id_documento]
                    else:
                        chunk_for_citation = next(
                            (chunk for chunk in retrieved_chunks_for_api_response if chunk.file_name == cited_source_by_llm.nombre_archivo),
                            None,
                        )

                    if chunk_for_citation:
                        chunk_for_citation.cita_tag = cited_source_by_llm.cita_tag
                        citation_payload.update(
                            {
                                "id": chunk_for_citation.id,
                                "content_preview": truncate_text(chunk_for_citation.content, 400)
                                if chunk_for_citation.content
                                else None,
                                "metadata": chunk_for_citation.metadata,
                            }
                        )

                    assistant_sources_for_db.append(citation_payload)

                if not assistant_sources_for_db:
                    for chunk in retrieved_chunks_for_api_response:
                        if len(assistant_sources_for_db) >= sources_limit:
                            break
                        assistant_sources_for_db.append(
                            {
                                "id": chunk.id,
                                "document_id": chunk.document_id,
                                "file_name": chunk.file_name,
                                "content_preview": truncate_text(chunk.content, 400) if chunk.content else None,
                                "metadata": chunk.metadata,
                                "cita_tag": chunk.cita_tag,
                            }
                        )


        except ValidationError as pydantic_err:
            validation_json_err = pydantic_err.errors()
            llm_handler_log.error("LLM JSON response failed Pydantic validation", raw_response=truncate_text(json_answer_str, 500), errors=validation_json_err)
            answer_for_user = "La respuesta del asistente no tuvo el formato esperado. Por favor, intenta de nuevo."
            assistant_sources_for_db = [{"error": "Pydantic validation failed", "details": validation_json_err}]
            retrieved_chunks_for_api_response = original_chunks_for_citation[:self.settings.NUM_SOURCES_TO_SHOW]
        except json.JSONDecodeError as json_err_detail:
            json_decode_err = str(json_err_detail)
            llm_handler_log.error("Failed to parse JSON response from LLM", raw_response=truncate_text(json_answer_str, 500), error=json_decode_err)
            answer_for_user = f"Hubo un error al procesar la respuesta del asistente (JSON malformado): {truncate_text(json_answer_str,100)}. Por favor, intenta de nuevo."
            assistant_sources_for_db = [{"error": "JSON decode error", "details": json_decode_err}]
            retrieved_chunks_for_api_response = original_chunks_for_citation[:self.settings.NUM_SOURCES_TO_SHOW]

        await self.chat_repo.save_message(
            chat_id=final_chat_id, role='assistant',
            content=answer_for_user, 
            sources=assistant_sources_for_db # Usar las fuentes procesadas
        )
        llm_handler_log.info(f"Assistant message saved with up to {self.settings.NUM_SOURCES_TO_SHOW} sources.", num_sources_saved_to_db=len(assistant_sources_for_db))

        # Loguear la interacción
        try:
            # Preparar los retrieved_documents_data para el log
            # Usa retrieved_chunks_for_api_response que ahora sí está alineado con lo que el LLM citó
            docs_for_log_summary = []
            if retrieved_chunks_for_api_response: # Solo si hay fuentes validadas
                docs_for_log_summary = [
                     # Usar model_dump para serializar el RetrievedChunk a dict para el log.
                     # El schema RetrievedDocumentSchema no es necesario aquí, solo un dict.
                    chunk.model_dump(exclude={'embedding'}, exclude_none=True)
                    for chunk in retrieved_chunks_for_api_response
                ]

            log_metadata_details = {
                "pipeline_stages": pipeline_stages_used,
                "map_reduce_used": map_reduce_used,
                "retriever_k_initial": retriever_k_effective,
                "fusion_fetch_k": fusion_fetch_k_effective,
                "max_context_chunks_direct_rag_limit": self.settings.MAX_CONTEXT_CHUNKS, 
                "num_chunks_after_rerank_or_fusion_content_fetch": num_chunks_after_rerank_or_fusion_fetch_effective,
                "num_final_chunks_sent_to_llm": num_final_chunks_sent_to_llm_effective,
                "num_sources_processed_from_llm_response": len(assistant_sources_for_db),
                "num_retrieved_docs_in_api_response": len(retrieved_chunks_for_api_response),
                "chat_history_messages_included_in_prompt": num_history_messages_effective,
                "diversity_filter_enabled_in_settings": self.settings.DIVERSITY_FILTER_ENABLED,
                "reranker_enabled_in_settings": self.settings.RERANKER_ENABLED,
                "bm25_enabled_in_settings": self.settings.BM25_ENABLED,
                "json_validation_error": validation_json_err,
                "json_decode_error": json_decode_err,
            }
            log_id = await self.log_repo.log_query_interaction(
                company_id=company_id, user_id=user_id, query=query, answer=answer_for_user,
                retrieved_documents_data=docs_for_log_summary, 
                chat_id=final_chat_id, metadata={k: v for k, v in log_metadata_details.items() if v is not None}
            )
            llm_handler_log.info("Query interaction logged successfully.", log_id=str(log_id) if log_id else "None")
        except Exception as e_log:
            llm_handler_log.error("Failed to log query interaction", error=str(e_log), exc_info=True)
            # log_id remains as initialized (None)

        return answer_for_user, retrieved_chunks_for_api_response, log_id

    async def _manage_chat_state(
        self, query: str, company_id: uuid.UUID, user_id: uuid.UUID,
        chat_id_param: Optional[uuid.UUID], exec_log: structlog.BoundLogger
    ) -> Tuple[uuid.UUID, Optional[str], List[ChatMessage]]:
        final_chat_id: uuid.UUID
        chat_history_str: Optional[str] = None
        history_messages: List[ChatMessage] = []

        if chat_id_param:
            if not await self.chat_repo.check_chat_ownership(chat_id_param, user_id, company_id):
                exec_log.warning("Chat ownership check failed.", provided_chat_id=str(chat_id_param))
                raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Chat not found or access denied.")
            final_chat_id = chat_id_param
            if self.settings.MAX_CHAT_HISTORY_MESSAGES > 0:
                history_messages = await self.chat_repo.get_chat_messages(
                    chat_id=final_chat_id, user_id=user_id, company_id=company_id,
                    limit=self.settings.MAX_CHAT_HISTORY_MESSAGES, offset=0
                )
                chat_history_str = self._format_chat_history(history_messages)
                exec_log.info("Existing chat, history retrieved", num_messages=len(history_messages))
        else:
            initial_title = f"Chat: {truncate_text(query, 40)}"
            final_chat_id = await self.chat_repo.create_chat(user_id=user_id, company_id=company_id, title=initial_title)
            exec_log.info("New chat created", new_chat_id=str(final_chat_id))
        
        # Actualizar exec_log para tener el chat_id correcto
        exec_log = exec_log.bind(chat_id=str(final_chat_id))
        await self.chat_repo.save_message(chat_id=final_chat_id, role='user', content=query)
        exec_log.info("User message saved", is_new_chat=(not chat_id_param)) 
        
        return final_chat_id, chat_history_str, history_messages

    async def _handle_greeting(
        self, query: str, company_id: uuid.UUID, user_id: uuid.UUID,
        final_chat_id: uuid.UUID, exec_log: structlog.BoundLogger
    ) -> Tuple[str, List[RetrievedChunk], Optional[uuid.UUID], uuid.UUID]:
        answer = "¡Hola! ¿En qué puedo ayudarte hoy con la información de tus documentos?"
        await self.chat_repo.save_message(chat_id=final_chat_id, role='assistant', content=answer, sources=None)
        exec_log.info("Greeting detected, responded directly.")
        simple_log_id = await self.log_repo.log_query_interaction(
            user_id=user_id, company_id=company_id, query=query, answer=answer,
            retrieved_documents_data=[], metadata={"type": "greeting"}, chat_id=final_chat_id
        )
        return answer, [], simple_log_id, final_chat_id


    async def execute(
        self, query: str, company_id: uuid.UUID, user_id: uuid.UUID,
        chat_id: Optional[uuid.UUID] = None, top_k: Optional[int] = None
    ) -> Tuple[str, List[RetrievedChunk], Optional[uuid.UUID], uuid.UUID]:
        
        exec_log = log.bind(use_case="AskQueryUseCase", company_id=str(company_id), user_id=str(user_id), query_preview=truncate_text(query, 50))
        
        # 1. Chat Management (Legacy Logic kept for now, could be a step later)
        final_chat_id: uuid.UUID
        chat_history_str: Optional[str] = None
        history_messages: List[ChatMessage] = []

        try:
            if chat_id:
                if not await self.chat_repo.check_chat_ownership(chat_id, user_id, company_id):
                    exec_log.warning("Chat ownership check failed for existing chat.", provided_chat_id=str(chat_id))
                    raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Chat no encontrado o acceso denegado.")
                final_chat_id = chat_id
                history_messages = await self.chat_repo.get_chat_messages(
                    chat_id=final_chat_id, user_id=user_id, company_id=company_id,
                    limit=self.settings.MAX_CHAT_HISTORY_MESSAGES, offset=0
                )
                chat_history_str = self._format_chat_history(history_messages)
                exec_log.info("Usando chat existente. Historial recuperado.", num_messages=len(history_messages))
            else:
                initial_title = f"Chat: {truncate_text(query, 40)}"
                final_chat_id = await self.chat_repo.create_chat(user_id=user_id, company_id=company_id, title=initial_title)
                exec_log.info("Nuevo chat creado para esta consulta.", new_chat_id=str(final_chat_id))

            exec_log = exec_log.bind(chat_id=str(final_chat_id), is_new_chat=(not chat_id))

            await self.chat_repo.save_message(chat_id=final_chat_id, role='user', content=query)
            exec_log.info("Mensaje del usuario guardado.")

            if GREETING_REGEX.match(query):
                return await self._handle_greeting(query, company_id, user_id, final_chat_id, exec_log)

            # 2. Build Pipeline
            # We create the pipeline here. In a more advanced setup, we could use a factory.
            pipeline_steps = [
                self.embedding_step,
                self.retrieval_step,
                self.fusion_step,
                self.content_fetch_step,
                self.rerank_step,
                self.filter_step,
                self.adaptive_gen_step # Decides between Direct and MapReduce
            ]
            
            pipeline = RAGPipeline(pipeline_steps)
            
            # 3. Prepare Context
            context = {
                "query": query,
                "company_id": company_id,
                "user_id": user_id,
                "chat_id": final_chat_id,
                "chat_history": chat_history_str, # String format for prompt
                "history_messages": history_messages, # List format for token counting
                "top_k": top_k,
                "pipeline_stages_used": [],
                "request_id": str(uuid.uuid4())
            }
            
            # 4. Run Pipeline
            exec_log.info("Iniciando ejecución del pipeline RAG")
            final_context = await pipeline.run(context)
            exec_log.info("Ejecución del pipeline RAG completada")
            
            # 5. Extract Results
            answer = final_context.get("answer", "")
            final_chunks = final_context.get("final_chunks", [])
            pipeline_stages_used = final_context.get("pipeline_stages_used", [])
            generation_mode = final_context.get("generation_mode", "unknown")
            
            # 6. Handle Response & Logging (Legacy logic adapted)
            # The pipeline returns the raw LLM response (string). We need to parse it if it's structured.
            # Wait, `DirectGenerationStep` calls `llm.generate`. `LLMPort.generate` usually returns a string.
            # The original code had `_handle_llm_response` which parsed JSON.
            # Does `llm.generate` return JSON string? Yes, usually.
            
            # We need to call `_handle_llm_response` to parse and save to DB.
            # Or we should have moved `_handle_llm_response` into the GenerationStep?
            # Ideally yes, but for now let's keep it here to minimize changes to response handling logic.
            
            # However, `_handle_llm_response` needs `original_chunks_for_citation`.
            # `final_chunks` from pipeline should be it.
            
            # Also `_handle_llm_response` saves the assistant message to DB.
            
            answer_text, cited_chunks, log_id = await self._handle_llm_response(
                json_answer_str=answer,
                query=query,
                company_id=company_id,
                user_id=user_id,
                final_chat_id=final_chat_id,
                original_chunks_for_citation=final_chunks,
                pipeline_stages_used=pipeline_stages_used,
                map_reduce_used=(generation_mode == "map_reduce")
            )
            
            return answer_text, cited_chunks, log_id, final_chat_id

        except Exception as e:
            exec_log.exception("Error crítico en AskQueryUseCase", error=str(e))
            # Fallback or re-raise
            raise e
_history_str,
                )
                json_answer_str = await self.llm.generate(
                    rag_fallback_prompt,
                    response_pydantic_schema=RespuestaEstructurada,
                )
                return await self._handle_llm_response(
                    json_answer_str=json_answer_str,
                    query=query,
                    company_id=company_id,
                    user_id=user_id,
                    final_chat_id=final_chat_id,
                    original_chunks_for_citation=[],
                    pipeline_stages_used=pipeline_stages_used,
                    map_reduce_used=False,
                    retriever_k_effective=retriever_k_effective,
                    fusion_fetch_k_effective=fusion_fetch_k_effective,
                    num_chunks_after_rerank_or_fusion_fetch_effective=0,
                    num_final_chunks_sent_to_llm_effective=0,
                    num_history_messages_effective=len(history_messages),
                )

            
            chunks_after_postprocessing = combined_chunks_with_content 
            if self.settings.RERANKER_ENABLED and chunks_after_postprocessing:
                pipeline_stages_used.append("reranking (remote_reranker_service)")
                rerank_log = exec_log.bind(action="rerank_remote", num_chunks_to_rerank=len(chunks_after_postprocessing))
                
                documents_for_reranker = []
                map_id_to_original_chunk_before_rerank = {c.id: c for c in chunks_after_postprocessing}

                for chk_id, original_chunk_obj in map_id_to_original_chunk_before_rerank.items():
                    if original_chunk_obj.content and original_chunk_obj.id: 
                        documents_for_reranker.append({
                            "id": original_chunk_obj.id,
                            "text": original_chunk_obj.content, 
                            "metadata": original_chunk_obj.metadata or {} 
                        })
                
                if documents_for_reranker:
                    reranker_payload = {
                        "query": query,
                        "documents": documents_for_reranker,
                        "top_n": self.settings.MAX_CONTEXT_CHUNKS 
                    }
                    try:
                        rerank_log.debug("Sending request to reranker service...")
                        base_reranker_url = str(settings.RERANKER_SERVICE_URL).rstrip('/')
                        
                        if "/api/v1/rerank" not in base_reranker_url:
                            if base_reranker_url.endswith("/api/v1"):
                                reranker_url = f"{base_reranker_url}/rerank"
                            elif base_reranker_url.endswith("/api"):
                                reranker_url = f"{base_reranker_url}/v1/rerank"
                            else:
                                reranker_url = f"{base_reranker_url}/api/v1/rerank"
                        else: 
                            reranker_url = base_reranker_url

                        rerank_log.debug(f"Final Reranker URL: {reranker_url}")
                        reranker_specific_timeout = httpx.Timeout(self.settings.RERANKER_CLIENT_TIMEOUT, connect=10.0)

                        reranker_response = await self.http_client.post(
                            reranker_url,
                            json=reranker_payload,
                            timeout=reranker_specific_timeout
                        )
                        reranker_response.raise_for_status()
                        reranked_data = reranker_response.json()

                        if "data" in reranked_data and "reranked_documents" in reranked_data["data"]:
                            reranked_docs_from_service = reranked_data["data"]["reranked_documents"]
                                                        
                            updated_reranked_chunks = []
                            for reranked_item_data in reranked_docs_from_service: 
                                chunk_id_val = reranked_item_data.get("id")
                                new_score = reranked_item_data.get("score")
                                
                                if chunk_id_val in map_id_to_original_chunk_before_rerank:
                                    original_retrieved_chunk = map_id_to_original_chunk_before_rerank[chunk_id_val]
                                    
                                    updated_chunk = RetrievedChunk(
                                        id=original_retrieved_chunk.id,
                                        content=original_retrieved_chunk.content, 
                                        score=new_score, 
                                        metadata={
                                            **(original_retrieved_chunk.metadata or {}),
                                            **(reranked_item_data.get("metadata", {})), 
                                            "reranked_score": new_score 
                                        },
                                        embedding=original_retrieved_chunk.embedding, 
                                        document_id=original_retrieved_chunk.document_id,
                                        file_name=original_retrieved_chunk.file_name,
                                        company_id=original_retrieved_chunk.company_id
                                    )
                                    updated_reranked_chunks.append(updated_chunk)
                                else:
                                    rerank_log.warning("Reranked chunk ID not found in original map.", reranked_id=chunk_id_val)

                            if updated_reranked_chunks: 
                                chunks_after_postprocessing = updated_reranked_chunks
                                rerank_log.info(f"Reranking successful. {len(chunks_after_postprocessing)} chunks after reranking.")
                            else:
                                rerank_log.warning("Reranking seemed successful but no chunks could be mapped back. Using pre-reranked chunks.")
                        else:
                            rerank_log.warning("Reranker service response format invalid.", response_data=reranked_data)
                    except httpx.HTTPStatusError as http_err:
                        rerank_log.error(
                            "HTTP error from Reranker service",
                            status_code=http_err.response.status_code,
                            response_text=truncate_text(http_err.response.text, 200),
                            error_details=repr(http_err),
                            exc_info=True 
                        )
                    except httpx.RequestError as req_err: 
                        rerank_log.error(
                            "Request error contacting Reranker service",
                            error_details=repr(req_err),
                            exc_info=True 
                        )
                    except Exception as e_rerank:
                        rerank_log.error(
                            "Unexpected error during reranking call.",
                            error_details=repr(e_rerank),
                            exc_info=True
                        )
                else:
                    rerank_log.warning("No valid documents with content/id to send for reranking.")
            else: 
                pipeline_stages_used.append(f"reranking ({'disabled' if not self.settings.RERANKER_ENABLED else 'skipped_no_chunks_or_content'})")
            
            num_chunks_after_rerank = len(chunks_after_postprocessing)

            if chunks_after_postprocessing and self.diversity_filter : 
                pipeline_stages_used.append("embedding_population_for_mmr")
                mmr_prep_log = exec_log.bind(action="mmr_embedding_population")
                
                chunk_ids_for_mmr_filter = [doc.id for doc in chunks_after_postprocessing if doc.id]
                mmr_prep_log.debug("Preparing embeddings for MMR", num_chunks_input=len(chunks_after_postprocessing), num_ids_to_fetch=len(chunk_ids_for_mmr_filter))

                vectors_from_milvus_by_id: Dict[str, List[float]] = {}
                if chunk_ids_for_mmr_filter:
                    try:
                        # Assuming VectorStorePort has fetch_vectors_by_ids
                        if hasattr(self.vector_store, 'fetch_vectors_by_ids'):
                            vectors_from_milvus_by_id = await self.vector_store.fetch_vectors_by_ids(chunk_ids_for_mmr_filter)
                            mmr_prep_log.info(f"Fetched {len(vectors_from_milvus_by_id)} vectors from Milvus for MMR.")
                        else:
                            mmr_prep_log.warning("VectorStorePort does not have 'fetch_vectors_by_ids'. Embeddings for MMR might be incomplete.")
                    except Exception as e_milvus_fetch:
                         mmr_prep_log.error("Failed to fetch vectors from Milvus for MMR, fallback may occur.", error=str(e_milvus_fetch))

                texts_needing_embedding_generation: List[str] = []
                map_text_index_to_chunk_obj: Dict[int, RetrievedChunk] = {} 
                chunks_ready_for_mmr: List[RetrievedChunk] = []

                for original_chunk in chunks_after_postprocessing:
                    retrieved_embedding = vectors_from_milvus_by_id.get(original_chunk.id)
                    
                    if retrieved_embedding is None and original_chunk.content: 
                        if original_chunk.embedding is None: 
                            map_text_index_to_chunk_obj[len(texts_needing_embedding_generation)] = original_chunk
                            texts_needing_embedding_generation.append(original_chunk.content)
                            
                    chunks_ready_for_mmr.append(
                        RetrievedChunk(
                            id=original_chunk.id,
                            content=original_chunk.content,
                            score=original_chunk.score,
                            metadata=original_chunk.metadata,
                            embedding=retrieved_embedding if retrieved_embedding else original_chunk.embedding, 
                            document_id=original_chunk.document_id,
                            file_name=original_chunk.file_name,
                            company_id=original_chunk.company_id
                        )
                    )
                
                if texts_needing_embedding_generation:
                    mmr_prep_log.info(f"Requesting {len(texts_needing_embedding_generation)} missing embeddings from embedding_adapter for MMR.")
                    try:
                        generated_embeddings_list = await self.embedding_adapter.embed_texts(texts_needing_embedding_generation)
                        
                        if len(generated_embeddings_list) == len(texts_needing_embedding_generation):
                            for idx, generated_emb_vector in enumerate(generated_embeddings_list):
                                chunk_to_update = map_text_index_to_chunk_obj[idx] 
                                for ch_ready in chunks_ready_for_mmr:
                                    if ch_ready.id == chunk_to_update.id:
                                        ch_ready.embedding = generated_emb_vector
                                        break
                            mmr_prep_log.info("Successfully updated chunks with newly generated embeddings for MMR.")
                        else:
                            mmr_prep_log.error("Mismatch in count of generated embeddings and requested texts for MMR fallback.",
                                               requested_count=len(texts_needing_embedding_generation),
                                               generated_count=len(generated_embeddings_list))
                    except Exception as e_embed_fallback:
                        mmr_prep_log.error("Failed to generate embeddings via adapter for MMR fallback.", error=str(e_embed_fallback))
                chunks_after_postprocessing = chunks_ready_for_mmr
            
            num_with_embeddings_before_mmr = sum(1 for c_chk in chunks_after_postprocessing if c_chk.embedding is not None)
            exec_log.debug(
                "Chunks before diversity filter",
                total_chunks=len(chunks_after_postprocessing),
                chunks_with_embeddings=num_with_embeddings_before_mmr,
                first_few_ids_and_embedding_status=[(c.id, c.embedding is not None) for c in chunks_after_postprocessing[:min(5, len(chunks_after_postprocessing))]]
            )

            if self.diversity_filter and chunks_after_postprocessing:
                 k_final_diversity = self.settings.MAX_CONTEXT_CHUNKS 
                 filter_type = type(self.diversity_filter).__name__
                 pipeline_stages_used.append(f"diversity_filter ({filter_type})")
                 exec_log.debug(f"Applying {filter_type} k={k_final_diversity}...", count=len(chunks_after_postprocessing))
                 chunks_after_postprocessing = await self.diversity_filter.filter(chunks_after_postprocessing, k_final_diversity)
                 exec_log.info(f"{filter_type} applied.", final_count=len(chunks_after_postprocessing))
            else: 
                 pipeline_stages_used.append(f"diversity_filter ({'disabled' if not self.settings.DIVERSITY_FILTER_ENABLED else 'skipped_no_chunks'})")
                 chunks_after_postprocessing = chunks_after_postprocessing[:self.settings.MAX_CONTEXT_CHUNKS]
                 exec_log.info(f"Diversity filter not applied or no chunks. Truncating to MAX_CONTEXT_CHUNKS.", 
                               count=len(chunks_after_postprocessing), limit=self.settings.MAX_CONTEXT_CHUNKS)

            final_chunks_for_processing = [c for c in chunks_after_postprocessing if c.content and c.content.strip()]
            final_chunks_for_processing, truncated_chunk_count = self._enforce_chunk_size_limits(final_chunks_for_processing)
            if truncated_chunk_count:
                exec_log.info(
                    "Applied chunk size limits before prompt assembly",
                    truncated_chunks=truncated_chunk_count,
                    max_tokens_per_chunk=self.settings.MAX_TOKENS_PER_CHUNK,
                    max_chars_per_chunk=self.settings.MAX_CHARS_PER_CHUNK,
                )

            num_final_chunks_for_llm_or_mapreduce = len(final_chunks_for_processing)

            if not final_chunks_for_processing: 
                exec_log.warning("No chunks with content after all postprocessing. Using structured RAG fallback without context.")
                pipeline_stages_used.append("rag_fallback_no_context")
                rag_fallback_prompt = await self._build_prompt(
                    PromptType.RAG,
                    query=query,
                    chat_history=chat_history_str,
                )
                json_answer_str = await self.llm.generate(
                    rag_fallback_prompt,
                    response_pydantic_schema=RespuestaEstructurada,
                )
                return await self._handle_llm_response(
                    json_answer_str=json_answer_str,
                    query=query,
                    company_id=company_id,
                    user_id=user_id,
                    final_chat_id=final_chat_id,
                    original_chunks_for_citation=[],
                    pipeline_stages_used=pipeline_stages_used,
                    map_reduce_used=False,
                    retriever_k_effective=retriever_k_effective,
                    fusion_fetch_k_effective=fusion_fetch_k_effective,
                    num_chunks_after_rerank_or_fusion_fetch_effective=0,
                    num_final_chunks_sent_to_llm_effective=0,
                    num_history_messages_effective=len(history_messages),
                )


            map_reduce_active = False
            json_answer_str: str
            chunks_to_send_to_llm: List[RetrievedChunk]

            token_count_start = time.perf_counter()
            token_analysis = self._analyze_tokens(final_chunks_for_processing)
            token_count_duration = time.perf_counter() - token_count_start

            total_tokens_for_llm = token_analysis.total_tokens
            chunk_token_counts = token_analysis.per_chunk_tokens

            query_tokens = self._count_tokens_for_text(query)
            history_tokens = self._count_tokens_for_text(chat_history_str) if chat_history_str else 0

            cache_stats = token_analysis.cache_info
            
            exec_log.info(
                "Token count for final list of processed chunks",
                num_chunks=num_final_chunks_for_llm_or_mapreduce,
                total_tokens=total_tokens_for_llm,
                query_tokens=query_tokens,
                history_tokens=history_tokens,
                token_count_duration_ms=round(token_count_duration * 1000, 2),
                cache_size=cache_stats.get("cache_size"),
                cache_max_size=cache_stats.get("cache_max_size"),
                direct_rag_token_limit=self.prompt_budget_config.direct_rag_token_limit,
            )

            should_activate_mapreduce = (
                self.map_reduce_config.enabled
                and num_final_chunks_for_llm_or_mapreduce > 1
                and total_tokens_for_llm > self.prompt_budget_config.direct_rag_token_limit
            )

            if should_activate_mapreduce:
                exec_log.info(
                    f"Activating MapReduce. Total tokens ({total_tokens_for_llm}) exceed limit ({self.prompt_budget_config.direct_rag_token_limit}).",
                    flow_type="MapReduce",
                )
                
                pipeline_stages_used.append("map_reduce_flow")
                map_reduce_active = True
                chunks_to_send_to_llm = final_chunks_for_processing

                map_prompt_budget = max(1, int(self.prompt_budget_config.llm_context_window * self.prompt_budget_config.map_prompt_ratio))
                per_chunk_overhead = 50 # Simplified overhead
                map_fixed_overhead = 200 # Simplified overhead

                pipeline_stages_used.append("map_phase")
                mapped_responses_parts: List[str] = []
                documents_for_prompt = self._prompt_service.create_documents(chunks_to_send_to_llm)
                if len(documents_for_prompt) != len(chunk_token_counts):
                    exec_log.warning(
                        "Mismatch between generated documents and token counts; truncating to shortest length.",
                        documents=len(documents_for_prompt),
                        token_counts=len(chunk_token_counts),
                    )

                documents_with_tokens: List[Tuple[Any, int]] = list(
                    zip(documents_for_prompt, chunk_token_counts)
                )

                map_batches: List[List[Any]] = []
                current_batch: List[Any] = []
                current_tokens = map_fixed_overhead
                max_tokens_per_batch = max(1, map_prompt_budget)

                for doc, chunk_tokens in documents_with_tokens:
                    projected_tokens = current_tokens + chunk_tokens + per_chunk_overhead
                    reached_size_limit = len(current_batch) >= self.map_reduce_config.chunk_batch_size
                    if current_batch and (projected_tokens > max_tokens_per_batch or reached_size_limit):
                        map_batches.append(current_batch)
                        current_batch = []
                        current_tokens = map_fixed_overhead
                        projected_tokens = current_tokens + chunk_tokens + per_chunk_overhead
                    if not current_batch and projected_tokens > max_tokens_per_batch:
                        exec_log.warning(
                            "Single chunk exceeds map prompt budget; sending alone",
                            chunk_id=doc.id,
                            chunk_tokens=chunk_tokens,
                            map_prompt_budget=max_tokens_per_batch,
                        )
                    current_batch.append(doc)
                    current_tokens = min(max_tokens_per_batch, projected_tokens)
                if current_batch:
                    map_batches.append(current_batch)

                exec_log.info(
                    "Prepared map batches",
                    map_batch_count=len(map_batches),
                    total_documents=len(documents_with_tokens),
                    map_prompt_budget=max_tokens_per_batch,
                    map_fixed_overhead=map_fixed_overhead,
                )

                total_documents = len(documents_with_tokens)
                map_tasks = []
                document_index_counter = 0
                for batch_docs in map_batches:
                    map_prompt_data = {
                        "original_query": query,
                        "documents": batch_docs,
                        "document_index": document_index_counter,
                        "total_documents": total_documents,
                    }
                    map_prompt_str_task = self._build_prompt(
                        PromptType.MAP,
                        query="",
                        prompt_data_override=map_prompt_data,
                    )
                    map_tasks.append(map_prompt_str_task)
                    document_index_counter += len(batch_docs)

                map_prompts = await asyncio.gather(*map_tasks)

                map_phase_results = []
                for idx, map_prompt in enumerate(map_prompts):
                    map_log_batch = exec_log.bind(map_batch_index=idx)
                    map_log_batch.info(
                        "Dispatching map batch to LLM",
                        documents_in_batch=len(map_batches[idx]) if idx < len(map_batches) else 0,
                    )
                    try:
                        result = await self.llm.generate(map_prompt, response_pydantic_schema=None)
                        map_phase_results.append(result)
                    except Exception as batch_error:
                        map_phase_results.append(batch_error)

                reduce_prompt_budget = max(1, int(self.prompt_budget_config.llm_context_window * self.prompt_budget_config.reduce_prompt_ratio))
                reduce_tokens_used = (
                    500 # Simplified fixed overhead
                    + query_tokens
                    + history_tokens
                )

                for idx, result in enumerate(map_phase_results):
                    map_log_batch = exec_log.bind(map_batch_index=idx) 
                    if isinstance(result, Exception):
                        map_log_batch.error(
                            "LLM call failed for map batch",
                            error=str(result),
                            exc_info=result,
                        )
                    elif result and MAP_REDUCE_NO_RELEVANT_INFO not in result:
                        result_tokens = self._count_tokens_for_text(result)
                        projected_reduce_tokens = reduce_tokens_used + result_tokens + 50 # Simplified overhead
                        if projected_reduce_tokens > reduce_prompt_budget:
                            map_log_batch.warning(
                                "Skipping map batch result due to reduce prompt budget",
                                reduce_tokens_used=reduce_tokens_used,
                                result_tokens=result_tokens,
                                reduce_prompt_budget=reduce_prompt_budget,
                            )
                            continue
                        mapped_responses_parts.append(f"--- Extracto del Lote de Documentos {idx + 1} ---\n{result}\n")
                        reduce_tokens_used = projected_reduce_tokens
                        map_log_batch.info(
                            "Map request processed for batch.",
                            response_length=len(result),
                            is_relevant=True,
                            result_tokens=result_tokens,
                        )
                    else:
                         map_log_batch.info("Map request processed for batch, no relevant info found by LLM.", response_length=len(result or ""))

                if not mapped_responses_parts:
                    exec_log.warning("MapReduce: All map steps reported no relevant information or failed.")
                    concatenated_mapped_responses = "Todos los fragmentos procesados indicaron no tener información relevante para la consulta o hubo errores en su procesamiento."
                else:
                    concatenated_mapped_responses = "\n".join(mapped_responses_parts)

                pipeline_stages_used.append("reduce_phase")
                haystack_docs_for_reduce_citation = [doc for doc, _ in documents_with_tokens] 
                reduce_prompt_data = {
                    "original_query": query,
                    "chat_history": chat_history_str,
                    "mapped_responses": concatenated_mapped_responses,
                    "original_documents_for_citation": haystack_docs_for_reduce_citation 
                }
                reduce_prompt_str = await self._build_prompt(
                    PromptType.REDUCE,
                    query="",
                    prompt_data_override=reduce_prompt_data,
                )
                
                exec_log.info(
                    "Sending reduce request to LLM for final JSON response.",
                    reduce_prompt_budget=reduce_prompt_budget,
                    reduce_tokens_used=reduce_tokens_used,
                )
                json_answer_str = await self.llm.generate(reduce_prompt_str, response_pydantic_schema=RespuestaEstructurada)

            else: 
                exec_log.info(f"Using Direct RAG strategy. Chunks available: {len(final_chunks_for_processing)}", flow_type="DirectRAG")
                pipeline_stages_used.append("direct_rag_flow")
                
                candidate_chunks = final_chunks_for_processing[:self.retrieval_config.max_context_chunks]
                candidate_token_counts = chunk_token_counts[:len(candidate_chunks)]
                
                # Recalculate direct_prompt_budget locally since we removed the earlier calculation
                direct_prompt_budget = max(1, int(self.prompt_budget_config.llm_context_window * self.prompt_budget_config.margin_ratio))

                tokens_consumed = 200 + query_tokens + history_tokens # Simplified overhead
                selected_chunks: List[RetrievedChunk] = []
                for chunk, chunk_tokens in zip(candidate_chunks, candidate_token_counts):
                    projected_tokens = tokens_consumed + chunk_tokens + 50 # Simplified overhead
                    if selected_chunks and projected_tokens > direct_prompt_budget:
                        break
                    if not selected_chunks and projected_tokens > direct_prompt_budget:
                        exec_log.warning(
                            "First chunk exceeds direct RAG budget; forcing inclusion",
                            chunk_id=chunk.id,
                            chunk_tokens=chunk_tokens,
                            direct_prompt_budget=direct_prompt_budget,
                        )
                        selected_chunks.append(chunk)
                        tokens_consumed = direct_prompt_budget
                        break
                    selected_chunks.append(chunk)
                    tokens_consumed = projected_tokens

                if not selected_chunks and candidate_chunks:
                    selected_chunks.append(candidate_chunks[0])

                chunks_to_send_to_llm = selected_chunks

                exec_log.info(
                    "Chunks selected for Direct RAG",
                    chunk_count=len(chunks_to_send_to_llm),
                    direct_prompt_budget=direct_prompt_budget,
                    estimated_tokens_consumed=tokens_consumed,
                )

                direct_rag_prompt = await self._build_prompt(
                    PromptType.RAG,
                    query=query,
                    chunks=chunks_to_send_to_llm,
                    chat_history=chat_history_str,
                )
                
                exec_log.info("Sending direct RAG request to LLM for JSON response.")
                json_answer_str = await self.llm.generate(direct_rag_prompt, response_pydantic_schema=RespuestaEstructurada)

            answer_text, relevant_chunks_for_api, final_log_id = await self._handle_llm_response(
                json_answer_str=json_answer_str,
                query=query,
                company_id=company_id,
                user_id=user_id,
                final_chat_id=final_chat_id,
                original_chunks_for_citation=chunks_to_send_to_llm, 
                pipeline_stages_used=pipeline_stages_used,
                map_reduce_used=map_reduce_active,
                retriever_k_effective=retriever_k_effective,
                fusion_fetch_k_effective=fusion_fetch_k_effective,
                num_chunks_after_rerank_or_fusion_fetch_effective=num_chunks_after_rerank, 
                num_final_chunks_sent_to_llm_effective=len(chunks_to_send_to_llm), 
                num_history_messages_effective=len(history_messages)
            )
            
            exec_log.info("Use case execution finished successfully.")
            return answer_text, relevant_chunks_for_api, final_log_id, final_chat_id

        except ConnectionError as ce: 
            exec_log.error("Connection error during use case execution", error=str(ce), exc_info=False)
            detail_message = "A required external service is unavailable. Please try again later."
            if "Embedding service" in str(ce): detail_message = "The embedding service is currently unavailable."
            elif "Reranker service" in str(ce): detail_message = "The reranking service is currently unavailable."
            elif "Sparse search service" in str(ce): detail_message = "The sparse search service is currently unavailable."
            elif "LLM service" in str(ce) or "llama.cpp" in str(ce): detail_message = "The language model service is currently unavailable."
            elif "Vector DB" in str(ce): detail_message = "The vector database service is currently unavailable."
            raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=detail_message) from ce
        except ValueError as ve: 
            exec_log.error("Value error during use case execution", error=str(ve), exc_info=True) 
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Data processing error: {ve}") from ve
        except HTTPException as http_exc: 
            exec_log.warning("HTTPException caught in use case", status_code=http_exc.status_code, detail=http_exc.detail)
            raise http_exc
        except Exception as e: 
            exec_log.exception("Unexpected error during use case execution") 
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"An internal server error occurred: {type(e).__name__}. Please contact support if this persists.") from e
