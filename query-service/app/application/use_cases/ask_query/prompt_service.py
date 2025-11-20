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