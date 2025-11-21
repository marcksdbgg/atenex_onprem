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
    concurrency_limit: int

@dataclass
class RetrievalConfig:
    top_k: int
    bm25_enabled: bool
    diversity_enabled: bool
    diversity_lambda: float
    max_context_chunks: int
    # RRF Config
    rrf_k: int
    rrf_weight_dense: float
    rrf_weight_sparse: float