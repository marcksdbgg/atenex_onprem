import asyncio
import uuid
import structlog
from typing import Any, Dict, List, Optional, Tuple

from app.application.use_cases.ask_query.pipeline import PipelineStep
from app.application.use_cases.ask_query.config_types import RetrievalConfig, MapReduceConfig, PromptBudgetConfig
from app.application.use_cases.ask_query.token_accountant import TokenAccountant
from app.application.use_cases.ask_query.prompt_service import PromptService
from app.application.services.fusion_service import FusionService
from app.domain.models import RetrievedChunk
from app.application.ports import (
    VectorStorePort, SparseRetrieverPort, 
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
        # Retrieve slightly more than final context to allow RRF to filter effectively
        top_k = (context.get("top_k") or self.config.top_k) 
        
        context["pipeline_stages_used"].append("dense_retrieval")
        # We fetch directly RetrievedChunk objects
        dense_task = self.vector.search(embedding, str(company_id), top_k)
        
        sparse_task = asyncio.create_task(self._noop_sparse())

        if self.config.bm25_enabled and self.sparse:
            context["pipeline_stages_used"].append("sparse_retrieval")
            sparse_task = self.sparse.search(query, company_id, top_k)
            
        results = await asyncio.gather(dense_task, sparse_task, return_exceptions=True)
        
        dense_res = results[0]
        sparse_res = results[1]

        if isinstance(dense_res, Exception):
            log.error("Dense retrieval failed", error=str(dense_res))
            dense_res = []
        
        if isinstance(sparse_res, Exception):
            log.error("Sparse retrieval failed", error=str(sparse_res))
            sparse_res = []
            
        context["dense_chunks"] = dense_res
        context["sparse_results"] = sparse_res # List[Tuple[str, float]]
        return context
    
    async def _noop_sparse(self):
        return []

class FusionStep(PipelineStep):
    """
    Reemplaza la fusión simple con Weighted RRF (Weighted Reciprocal Rank Fusion).
    Esta etapa actúa como el reranker del sistema.
    """
    def __init__(self, fusion_service: FusionService, config: RetrievalConfig):
        super().__init__("FusionStep")
        self.fusion_service = fusion_service
        self.config = config

    async def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        dense_chunks: List[RetrievedChunk] = context["dense_chunks"]
        sparse_tuples: List[Tuple[str, float]] = context["sparse_results"]
        company_id = str(context["company_id"])

        # Convertir tuplas sparse a RetrievedChunk placeholders para que el servicio de fusión pueda trabajar
        # Esto homogeniza las listas de entrada
        sparse_chunks = []
        for chunk_id, score in sparse_tuples:
            # Solo tenemos ID y Score. El contenido se buscará después si este chunk gana en el RRF.
            sparse_chunks.append(
                RetrievedChunk(
                    id=chunk_id,
                    score=score,
                    content=None, # Explicitly None, fetching happens later
                    metadata={"retrieval_source": "sparse_only"},
                    company_id=company_id
                )
            )

        log.info("Executing Weighted RRF Fusion", 
                 dense_count=len(dense_chunks), 
                 sparse_count=len(sparse_chunks),
                 k=self.config.rrf_k)

        fused_chunks = self.fusion_service.weighted_rrf(
            dense_results=dense_chunks,
            sparse_results=sparse_chunks,
            dense_weight=self.config.rrf_weight_dense,
            sparse_weight=self.config.rrf_weight_sparse,
            top_k=self.config.top_k, # Maintain Top K for downstream
            id_field="id"
        )
        
        context["fused_chunks"] = fused_chunks
        context["pipeline_stages_used"].append("rrf_fusion")
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
                log.debug(f"Fetching content for {len(missing_ids)} chunks prioritized by RRF")
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
                             # Fusionar metadatos si ya existían (e.g. de sparse)
                             current_meta = c.metadata or {}
                             current_meta.update(data["metadata"] or {})
                             c.metadata = current_meta
                        valid_chunks.append(c)
                    else:
                        log.warning(f"Content not found for chunk {c.id}, dropping from results.")
                context["fused_chunks"] = valid_chunks
            except Exception as e:
                log.error("Content fetch failed", error=str(e))
                context["fused_chunks"] = [c for c in chunks if c.content]
        
        return context

class FilterStep(PipelineStep):
    def __init__(self, diver_filter: Optional[DiversityFilterPort], config: RetrievalConfig):
        super().__init__("FilterStep")
        self.diver_filter = diver_filter
        self.config = config

    async def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        # Toma input directo de ContentFetch (que tomó de Fusion)
        chunks = context["fused_chunks"]
        limit = self.config.max_context_chunks
        
        # Aplicar filtro de diversidad si es necesario y si hay embeddings
        # Nota: ContentFetch no recupera embeddings para items que vinieron solo de sparse.
        # Si diversidad es crítica, necesitaríamos un paso adicional "EmbeddingFetchStep",
        # pero para el caso SLLM actual, RRF es suficientemente bueno como filtro primario.
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
        
        valid_chunks = []
        current_tokens = 300 + self.accountant.count_tokens_for_text(query + history)
        
        for chunk in chunks:
            ct = self.accountant.count_tokens_for_text(chunk.content)
            if current_tokens + ct > self.config.direct_rag_token_limit:
                break
            valid_chunks.append(chunk)
            current_tokens += ct
            
        context["final_used_chunks"] = valid_chunks
        prompt = await self.prompts.build_rag_prompt(query, valid_chunks, history)
        
        context["llm_response_raw"] = await self.llm.generate(prompt, response_pydantic_schema=None) 
        
        context["generation_mode"] = "direct_rag"
        return context

class MapReduceGenerationStep(PipelineStep):
    def __init__(self, llm: LLMPort, prompt_service: PromptService, map_config: MapReduceConfig):
        super().__init__("MapReduceGenerationStep")
        self.llm = llm
        self.prompts = prompt_service
        self.config = map_config
        self._sem = asyncio.Semaphore(self.config.concurrency_limit)

    async def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        chunks = context["final_chunks"]
        query = context["query"]
        
        context["pipeline_stages_used"].append("map_reduce")
        
        batch_size = self.config.chunk_batch_size
        batches = [chunks[i:i + batch_size] for i in range(0, len(chunks), batch_size)]
        
        total_docs = len(chunks)
        
        async def _process_batch_safely(b_chunks, current_idx):
            async with self._sem:
                try:
                    p = await self.prompts.build_map_prompt(query, b_chunks, current_idx, total_docs)
                    response = await self.llm.generate(p)
                    
                    if "IRRELEVANTE" in response.upper() and len(response.strip()) < 50:
                        return None
                    return response
                except Exception as e:
                    log.error(f"Error in Map batch processing: {e}")
                    return None

        tasks = []
        doc_idx = 0
        for b in batches:
            tasks.append(_process_batch_safely(b, doc_idx))
            doc_idx += len(b)
            
        raw_map_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        valid_maps = []
        for res in raw_map_results:
            if isinstance(res, str) and res:
                valid_maps.append(res)
            elif isinstance(res, Exception):
                log.error("Map task failed with exception", error=str(res))
        
        if not valid_maps:
             log.warning("Generative Filter found no relevant info in chunks. Falling back to Direct Generation (fallback mode).")
             context["generation_mode"] = "map_reduce_fallback"
             context["final_chunks"] = chunks[:2]
             
             combined_map = "No se encontró información específica en los documentos para responder a la pregunta. Intenta responder usando el conocimiento general si aplica, o indica que no hay datos."
        else:
             combined_map = "\n".join(valid_maps)
             log.info(f"Generative Filter reduced context from {len(chunks)} chunks to {len(valid_maps)} relevant extracts.")

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
        analysis = self.accountant.calculate_token_usage(chunks)
        total_tokens = analysis.total_tokens
        
        log.info(f"Token analysis: {total_tokens} tokens in {len(chunks)} chunks. Limit: {self.budget.direct_rag_token_limit}")

        if self.map_config.enabled and total_tokens > self.budget.direct_rag_token_limit:
            return await self.mapred.execute(context)
        else:
            return await self.direct.execute(context)