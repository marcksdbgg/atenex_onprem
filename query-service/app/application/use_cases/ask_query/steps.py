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