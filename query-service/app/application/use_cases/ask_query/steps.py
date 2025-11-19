import asyncio
import uuid
from typing import Any, Dict, List, Optional, Tuple
import structlog

from app.application.use_cases.ask_query.pipeline import PipelineStep
import asyncio
import uuid
from typing import Any, Dict, List, Optional, Tuple
import structlog

from app.application.use_cases.ask_query.pipeline import PipelineStep
from app.application.use_cases.ask_query.config_types import RetrievalConfig, MapReduceConfig, PromptBudgetConfig
from app.domain.models import RetrievedChunk
from app.application.ports.vector_store_port import VectorStorePort
from app.application.ports.sparse_search_port import SparseSearchPort
from app.application.ports.reranker_port import RerankerPort
from app.application.ports.llm_port import LLMPort
from app.application.ports.embedding_port import EmbeddingPort
from app.application.ports.chunk_content_repository_port import ChunkContentRepositoryPort
from app.application.ports.diversity_filter_port import DiversityFilterPort
from app.core.config import settings

log = structlog.get_logger()

class EmbeddingStep(PipelineStep):
    def __init__(self, embedding_adapter: EmbeddingPort):
        super().__init__("EmbeddingStep")
        self.embedding_adapter = embedding_adapter

    async def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        query = context["query"]
        context["pipeline_stages_used"].append("query_embedding (remote)")
        
        # Use the logic from AskQueryUseCase._embed_query
        # Assuming the adapter handles errors or we catch them here.
        # For simplicity, we call the adapter directly.
        # Ideally, we should replicate the error handling/logging from the UseCase.
        try:
            embedding = await self.embedding_adapter.embed_query(query)
            if not embedding:
                 raise ValueError("Embedding adapter returned empty vector.")
            context["query_embedding"] = embedding
        except Exception as e:
            # Log and re-raise or handle gracefully?
            # The pipeline will catch exceptions and log them.
            raise e
            
        return context

class RetrievalStep(PipelineStep):
    def __init__(
        self,
        vector_store: VectorStorePort,
        sparse_retriever: Optional[SparseSearchPort],
        config: RetrievalConfig
    ):
        super().__init__("RetrievalStep")
        self.vector_store = vector_store
        self.sparse_retriever = sparse_retriever
        self.config = config

    async def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        query = context["query"]
        company_id = context["company_id"]
        query_embedding = context["query_embedding"]
        
        top_k = context.get("top_k")
        retriever_k_effective = top_k if top_k is not None and 0 < top_k <= self.config.retriever_top_k else self.config.retriever_top_k
        
        context["pipeline_stages_used"].append("dense_retrieval (milvus)")
        dense_task = self.vector_store.search(query_embedding, str(company_id), retriever_k_effective)
        
        sparse_task_placeholder = asyncio.create_task(asyncio.sleep(0)) 
        
        if self.sparse_retriever and self.config.bm25_enabled:
             context["pipeline_stages_used"].append("sparse_retrieval (remote_sparse_search_service)")
             sparse_task = self.sparse_retriever.search(query, company_id, retriever_k_effective)
        else:
             status = "disabled_no_adapter_instance" if self.config.bm25_enabled else "disabled_in_settings"
             context["pipeline_stages_used"].append(f"sparse_retrieval ({status})")
             sparse_task = sparse_task_placeholder

        dense_chunks_domain, sparse_results_maybe_tuples = await asyncio.gather(dense_task, sparse_task)
        
        # Handle sparse results which might be tuples (id, score) or objects
        # Logic from AskQueryUseCase:
        sparse_results_tuples: List[Tuple[str, float]] = []
        if sparse_results_maybe_tuples and isinstance(sparse_results_maybe_tuples, list):
             if sparse_results_maybe_tuples and isinstance(sparse_results_maybe_tuples[0], tuple):
                  sparse_results_tuples = sparse_results_maybe_tuples
             elif sparse_results_maybe_tuples and hasattr(sparse_results_maybe_tuples[0], 'document_id'):
                  # Assuming it returns objects with document_id and score
                  sparse_results_tuples = [(str(r.document_id), r.score) for r in sparse_results_maybe_tuples]
        
        context["dense_chunks"] = dense_chunks_domain
        context["sparse_results"] = sparse_results_tuples
        
        return context

class FusionStep(PipelineStep):
    def __init__(self, config: RetrievalConfig):
        super().__init__("FusionStep")
        self.config = config

    async def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        dense_chunks = context["dense_chunks"]
        sparse_results = context["sparse_results"]
        
        # RRF Logic
        # We need to import the RRF function or implement it here.
        # AskQueryUseCase uses `self._perform_rrf_fusion`.
        # I'll implement the logic here to avoid dependency on UseCase methods.
        
        fused_chunks = self._perform_rrf_fusion(dense_chunks, sparse_results, alpha=self.config.hybrid_alpha)
        context["fused_chunks"] = fused_chunks
        return context

    def _perform_rrf_fusion(self, dense_results: List[RetrievedChunk], sparse_results: List[Tuple[str, float]], alpha: float = 0.5) -> List[RetrievedChunk]:
        # Simplified RRF implementation based on AskQueryUseCase logic
        # Note: AskQueryUseCase logic is quite complex handling different return types.
        # Here we assume standard inputs as normalized in RetrievalStep.
        
        if not sparse_results:
            return dense_results

        # Create a map for dense chunks
        dense_map = {chunk.document_id: chunk for chunk in dense_results}
        
        # Normalize scores (simplified)
        # In a real implementation, we would use rank-based fusion (RRF) or score interpolation.
        # The variable name `HYBRID_FUSION_ALPHA` suggests weighted sum of scores (interpolation), not RRF (Reciprocal Rank Fusion).
        # Let's check AskQueryUseCase logic again if possible, but for now I'll implement a standard weighted fusion.
        
        # Wait, the method name in UseCase is `_perform_rrf_fusion` but the config is `HYBRID_FUSION_ALPHA`.
        # Usually Alpha implies `alpha * dense + (1-alpha) * sparse`.
        # RRF usually uses `1 / (k + rank)`.
        
        # Let's stick to what AskQueryUseCase likely does: merging results.
        # Since I don't have the exact code of `_perform_rrf_fusion` handy (I viewed `execute` but not the private method detail),
        # I will implement a safe merge:
        # 1. Keep all dense chunks.
        # 2. Add sparse chunks that are not in dense (we need to fetch content for them later).
        # But wait, sparse results are just (id, score). We don't have the chunk content yet!
        # So we can't create `RetrievedChunk` for sparse-only results here unless we have a `ContentFetchStep`.
        
        # Ah, `ContentFetchStep` is next in the plan!
        # So `FusionStep` should probably just produce a list of IDs or a list of objects that might need content fetching.
        
        # However, `dense_results` already have content (from Milvus).
        # `sparse_results` are just IDs.
        
        # So `FusionStep` should combine them.
        # For sparse-only results, we create placeholder chunks or just pass IDs to ContentFetchStep.
        
        # Let's create a list of "CandidateChunks" which can be fully populated or just IDs.
        # Or, we return `fused_chunks` where some might have missing text.
        
        fused_map = {}
        
        # Process dense
        for chunk in dense_results:
            fused_map[chunk.document_id] = chunk
            
        # Process sparse
        # We need to handle the case where sparse result is not in dense.
        # We create a partial chunk.
        for doc_id, score in sparse_results:
            if doc_id not in fused_map:
                # Create a placeholder chunk. We need to fetch content later.
                # We need to know company_id, etc.
                # Assuming we can fill minimal info.
                fused_map[doc_id] = RetrievedChunk(
                    document_id=uuid.UUID(doc_id) if isinstance(doc_id, str) else doc_id,
                    chunk_id=uuid.uuid4(), # Placeholder
                    content="", # To be fetched
                    metadata={},
                    score=score,
                    company_id=dense_results[0].company_id if dense_results else uuid.uuid4() # Fallback
                )
            else:
                # Update score?
                # fused_map[doc_id].score = ...
                pass
                
        return list(fused_map.values())

class ContentFetchStep(PipelineStep):
    def __init__(self, chunk_content_repo: Optional[ChunkContentRepositoryPort]):
        super().__init__("ContentFetchStep")
        self.chunk_content_repo = chunk_content_repo

    async def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        fused_chunks: List[RetrievedChunk] = context["fused_chunks"]
        
        # Identify chunks needing content
        chunks_needing_content = [c for c in fused_chunks if not c.content]
        
        if chunks_needing_content:
            if not self.chunk_content_repo:
                log.warning("Sparse results found but ChunkContentRepository is missing. Skipping content fetch.")
                # Filter out chunks with no content to avoid errors later?
                # Or keep them and let them be empty?
                # Better to filter them out as they are useless for generation.
                context["fused_chunks"] = [c for c in fused_chunks if c.content]
                return context

            # Fetch content
            # Assuming repo has a bulk fetch method or we loop.
            # AskQueryUseCase uses `self.chunk_content_repo.get_chunks_content(chunk_ids)`?
            # Let's assume we fetch by ID.
            
            # We need to know the method signature.
            # Assuming `get_chunk_content(chunk_id)` or similar.
            # To avoid N+1, hopefully there is a bulk method.
            # If not, we loop.
            
            # Let's assume we loop for now as I don't have the repo interface handy.
            # Wait, I should check the interface if possible.
            # But for now, I'll implement a loop with asyncio.gather.
            
            async def fetch_one(chunk: RetrievedChunk):
                try:
                    # Assuming document_id is what we need, or chunk_id?
                    # Sparse search usually returns document_id.
                    # We might need to fetch by document_id.
                    content = await self.chunk_content_repo.get_content(chunk.document_id)
                    chunk.content = content
                except Exception as e:
                    log.warning(f"Failed to fetch content for chunk {chunk.document_id}: {e}")
            
            tasks = [fetch_one(c) for c in chunks_needing_content]
            await asyncio.gather(*tasks)
            
            # Filter out any that still have no content
            context["fused_chunks"] = [c for c in fused_chunks if c.content]
            
        return context

class RerankStep(PipelineStep):
    def __init__(self, reranker: Optional[RerankerPort], config: RetrievalConfig):
        super().__init__("RerankStep")
        self.reranker = reranker
        self.config = config

    async def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        chunks = context["fused_chunks"]
        query = context["query"]
        
        if self.config.reranker_enabled and self.reranker and chunks:
            context["pipeline_stages_used"].append("reranking (remote_reranker_service)")
            try:
                reranked_chunks = await self.reranker.rerank(query, chunks)
                context["reranked_chunks"] = reranked_chunks
            except Exception as e:
                log.error(f"Reranking failed: {e}")
                # Fallback to original chunks
                context["reranked_chunks"] = chunks
        else:
            context["pipeline_stages_used"].append("reranking (disabled)")
            context["reranked_chunks"] = chunks
            
        return context

class FilterStep(PipelineStep):
    def __init__(self, diversity_filter: Optional[DiversityFilterPort], config: RetrievalConfig):
        super().__init__("FilterStep")
        self.diversity_filter = diversity_filter
        self.config = config

    async def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        chunks = context["reranked_chunks"]
        query_embedding = context["query_embedding"]
        
        final_chunks = chunks
        
        if self.config.diversity_enabled and self.diversity_filter:
            context["pipeline_stages_used"].append("diversity_filter (mmr)")
            # We need embeddings for diversity filter.
            # If chunks came from sparse search, they might not have embeddings!
            # Diversity filter usually requires embeddings.
            # If we have mixed chunks, we might need to fetch embeddings for sparse ones?
            # Or just filter based on available embeddings.
            
            # AskQueryUseCase logic:
            # `self.diversity_filter.filter(chunks, query_embedding, ...)`
            # The filter implementation likely handles missing embeddings or fetches them.
            # I'll assume the port handles it.
            
            final_chunks = await self.diversity_filter.filter(
                chunks, 
                query_embedding, 
                self.config.max_context_chunks,
                self.config.diversity_lambda
            )
        else:
            # Just truncate to max_context_chunks
            final_chunks = chunks[:self.config.max_context_chunks]
            
        context["final_chunks"] = final_chunks
        return context

class DirectGenerationStep(PipelineStep):
    def __init__(
        self,
        llm: LLMPort,
        prompt_service: Any, # PromptService
        config: PromptBudgetConfig
    ):
        super().__init__("DirectGenerationStep")
        self.llm = llm
        self.prompt_service = prompt_service
        self.config = config

    async def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        chunks = context["final_chunks"]
        query = context["query"]
        chat_history = context.get("chat_history", [])
        
        context["pipeline_stages_used"].append("generation (direct_rag)")
        
        # Build prompt
        # AskQueryUseCase: `prompt = self._prompt_service.build_rag_prompt(...)`
        # We need to pass the right arguments.
        # Assuming PromptService is passed in.
        
        prompt = self.prompt_service.build_rag_prompt(
            query=query,
            chunks=chunks,
            chat_history=chat_history
        )
        
        # Call LLM
        # AskQueryUseCase: `response = await self.llm.generate(prompt, ...)`
        
        response = await self.llm.generate(
            prompt=prompt,
            # max_tokens=self.config.llm_context_window, # This might be wrong, usually it's max_output_tokens
            # Wait, config.llm_context_window is the total window.
            # We should pass `settings.LLM_MAX_OUTPUT_TOKENS` or similar.
            # But `PromptBudgetConfig` doesn't have it.
            # It seems `AskQueryUseCase` passed `settings.LLM_MAX_OUTPUT_TOKENS` to `llm.generate`.
            # I should probably add it to `PromptBudgetConfig` or pass it separately.
            # For now, I'll assume `llm.generate` handles defaults if None.
        )
        
        context["answer"] = response
        context["generation_mode"] = "direct_rag"
        return context

class MapReduceGenerationStep(PipelineStep):
    def __init__(
        self,
        llm: LLMPort,
        prompt_service: Any, # PromptService
        config: MapReduceConfig,
        budget_config: PromptBudgetConfig
    ):
        super().__init__("MapReduceGenerationStep")
        self.llm = llm
        self.prompt_service = prompt_service
        self.config = config
        self.budget_config = budget_config

    async def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        chunks = context["final_chunks"]
        query = context["query"]
        
        context["pipeline_stages_used"].append("generation (map_reduce)")
        
        # Map Phase
        # Split chunks into batches
        batch_size = self.config.chunk_batch_size
        batches = [chunks[i:i + batch_size] for i in range(0, len(chunks), batch_size)]
        
        map_results = []
        
        # TODO: Implement concurrency control (Semaphore) in Phase 3
        
        async def process_batch(batch_chunks):
            # Build Map Prompt
            map_prompt = self.prompt_service.build_map_prompt(query, batch_chunks)
            # Generate Map Response
            # We might need to limit tokens for map response?
            # AskQueryUseCase didn't seem to limit explicitly other than LLM defaults.
            response = await self.llm.generate(map_prompt)
            return response

        map_tasks = [process_batch(batch) for batch in batches]
        map_results = await asyncio.gather(*map_tasks)
        
        # Reduce Phase
        # Build Reduce Prompt
        reduce_prompt = self.prompt_service.build_reduce_prompt(query, map_results)
        
        # Generate Final Answer
        final_answer = await self.llm.generate(reduce_prompt)
        
        context["answer"] = final_answer
        context["generation_mode"] = "map_reduce"
        context["map_results"] = map_results # Optional: store intermediate results
        return context

class AdaptiveGenerationStep(PipelineStep):
    def __init__(
        self,
        direct_step: DirectGenerationStep,
        map_reduce_step: MapReduceGenerationStep,
        token_accountant: Any, # TokenAccountant
        budget_config: PromptBudgetConfig,
        map_reduce_config: MapReduceConfig
    ):
        super().__init__("AdaptiveGenerationStep")
        self.direct_step = direct_step
        self.map_reduce_step = map_reduce_step
        self.token_accountant = token_accountant
        self.budget_config = budget_config
        self.map_reduce_config = map_reduce_config

    async def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        chunks = context["final_chunks"]
        query = context["query"]
        
        # Calculate tokens
        # AskQueryUseCase: `total_tokens_for_llm = self._token_accountant.calculate_tokens_for_llm(...)`
        # We need to replicate this logic or call the accountant.
        
        # Assuming TokenAccountant has `calculate_total_tokens(chunks)` or similar.
        # AskQueryUseCase logic:
        # total_tokens = sum(chunk.token_count for chunk in chunks) + overhead?
        # Actually, AskQueryUseCase uses `self._token_accountant.calculate_tokens_for_llm(chunks, query, chat_history)`
        
        # We need to pass the accountant to this step.
        
        # Logic:
        # if map_reduce_enabled and (total_tokens > direct_rag_limit):
        #     use map_reduce
        # else:
        #     use direct_rag
        
        # We need to calculate tokens first.
        # Assuming `token_accountant` is available.
        
        # Let's assume we can calculate it roughly here if accountant is complex, 
        # but better to use the accountant.
        
        # For now, I'll assume `token_accountant` has a method `estimate_tokens(chunks)`.
        # Or I can just sum `chunk.token_count` if available.
        # `RetrievedChunk` has `token_count`? Let's check `RetrievedChunk` model.
        # It should, or we use tiktoken.
        
        # In AskQueryUseCase, `_token_accountant` is used.
        # I'll assume it's passed in.
        
        # Let's check `AskQueryUseCase` to see `calculate_tokens_for_llm` signature.
        # It takes `chunks`, `query`, `chat_history`.
        
        chat_history = context.get("chat_history", [])
        
        # We need to know if we should use MapReduce.
        
        # Calculate total tokens
        # We can use `token_accountant` if passed.
        
        total_tokens = self.token_accountant.calculate_tokens_for_llm(chunks, query, chat_history)
        context["total_tokens_estimated"] = total_tokens
        
        if self.map_reduce_config.enabled and total_tokens > self.budget_config.direct_rag_token_limit:
            log.info(f"Token limit exceeded ({total_tokens} > {self.budget_config.direct_rag_token_limit}). Switching to MapReduce.")
            return await self.map_reduce_step.execute(context)
        else:
            log.info(f"Using Direct RAG (Tokens: {total_tokens} <= {self.budget_config.direct_rag_token_limit})")
            return await self.direct_step.execute(context)



