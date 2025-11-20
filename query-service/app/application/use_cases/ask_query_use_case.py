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