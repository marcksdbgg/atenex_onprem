from __future__ import annotations

import asyncio
import os
from enum import Enum
from typing import Dict, List, Optional

import structlog
from haystack import Document
from haystack.components.builders.prompt_builder import PromptBuilder

from app.core.config import Settings
from app.domain.models import RetrievedChunk

log = structlog.get_logger(__name__)


class PromptType(Enum):
    RAG = "rag"
    GENERAL = "general"
    MAP = "map"
    REDUCE = "reduce"


class PromptService:
    """Gestiona los PromptBuilder de Haystack y provee utilidades para preparar datos."""

    def __init__(self, app_settings: Settings) -> None:
        self._settings = app_settings
        self._builders: Dict[PromptType, PromptBuilder] = {
            PromptType.RAG: self._load_builder(app_settings.RAG_PROMPT_TEMPLATE_PATH),
            PromptType.GENERAL: self._load_builder(app_settings.GENERAL_PROMPT_TEMPLATE_PATH),
            PromptType.MAP: self._load_builder(app_settings.MAP_PROMPT_TEMPLATE_PATH),
            PromptType.REDUCE: self._load_builder(app_settings.REDUCE_PROMPT_TEMPLATE_PATH),
        }

    @staticmethod
    def _load_builder(template_path: str) -> PromptBuilder:
        init_log = log.bind(action="load_prompt_builder", path=template_path)
        init_log.debug("Loading PromptBuilder from path...")
        try:
            if not os.path.exists(template_path):
                raise FileNotFoundError(f"Prompt template file not found at {template_path}")

            with open(template_path, "r", encoding="utf-8") as template_file:
                template_content = template_file.read()

            if not template_content.strip():
                raise ValueError(f"Prompt template file is empty: {template_path}")

            init_log.info("PromptBuilder initialized successfully from file.")
            return PromptBuilder(template=template_content)
        except FileNotFoundError as file_error:
            init_log.error("Prompt template file not found.")
            fallback_template = (
                "Query: {{ query }}\n"
                "{% if documents %}Context: {{ documents }}{% endif %}\n"
                "Answer:"
            )
            init_log.warning(
                "Falling back to basic template due to missing file.",
                fallback_path=template_path,
            )
            return PromptBuilder(template=fallback_template)
        except Exception as exc:
            init_log.exception("Failed to load or initialize PromptBuilder from path.")
            raise RuntimeError(
                f"Critical error loading prompt template from {template_path}: {exc}"
            ) from exc

    async def build(self, prompt_type: PromptType, **prompt_payload) -> str:
        builder = self._builders[prompt_type]
        result = await asyncio.to_thread(builder.run, **prompt_payload)
        prompt = result.get("prompt")
        if not prompt:
            raise ValueError("Prompt generation returned empty result.")
        log.debug(
            "Prompt built successfully.",
            builder_type=type(builder).__name__,
            payload_keys=list(prompt_payload.keys()),
            length=len(prompt),
        )
        return prompt

    def create_documents(self, chunks: List[RetrievedChunk]) -> List[Document]:
        documents: List[Document] = []
        for chunk in chunks:
            content = chunk.content or ""

            meta = dict(chunk.metadata or {})
            if chunk.file_name is not None:
                meta.setdefault("file_name", chunk.file_name)
                meta.setdefault("file_name_normalized", chunk.file_name.lower())
            if chunk.document_id is not None:
                meta.setdefault("document_id", chunk.document_id)
            if chunk.company_id is not None:
                meta.setdefault("company_id", chunk.company_id)
            if "page" not in meta and chunk.metadata:
                meta["page"] = chunk.metadata.get("page")
            if "title" not in meta and chunk.metadata:
                meta["title"] = chunk.metadata.get("title")

            documents.append(
                Document(
                    id=chunk.id,
                    content=content,
                    meta=meta,
                    score=chunk.score,
                )
            )
        return documents

    def include_chat_history(
        self, prompt_payload: Dict[str, object], chat_history: Optional[str]
    ) -> Dict[str, object]:
        if chat_history:
            prompt_payload["chat_history"] = chat_history
        return prompt_payload
