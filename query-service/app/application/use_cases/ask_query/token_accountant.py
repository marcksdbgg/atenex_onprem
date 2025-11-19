from __future__ import annotations

import hashlib
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple, Union

import structlog
import tiktoken

from app.core.config import Settings
from app.domain.models import RetrievedChunk

from .types import TokenAnalysis

log = structlog.get_logger(__name__)


class _BoundedCache(OrderedDict):
    def __init__(self, max_size: int) -> None:
        super().__init__()
        self._max_size = max_size

    def __setitem__(self, key, value):  # type: ignore[override]
        if key not in self and len(self) >= self._max_size:
            self.popitem(last=False)
        super().__setitem__(key, value)
        self.move_to_end(key)


class TokenAccountant:
    """Centraliza el conteo y control de tokens para los prompts enviados al LLM."""

    def __init__(self, app_settings: Settings, cache_max_size: int = 1000) -> None:
        self._encoding: Optional[tiktoken.Encoding] = None
        self._token_cache: _BoundedCache = _BoundedCache(cache_max_size)
        self._settings = app_settings
        self._cache_max_size = cache_max_size

    # ------------------------------------------------------------------
    # Encoding helpers
    # ------------------------------------------------------------------
    def _get_encoding(self) -> tiktoken.Encoding:
        if self._encoding is None:
            try:
                self._encoding = tiktoken.get_encoding(self._settings.TIKTOKEN_ENCODING_NAME)
            except Exception as exc:  # pragma: no cover - defensive fallback
                log.error(
                    "Failed to load tiktoken encoding; falling back to gpt2",
                    encoding_name=self._settings.TIKTOKEN_ENCODING_NAME,
                    error=str(exc),
                )
                self._encoding = tiktoken.get_encoding("gpt2")
        return self._encoding

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def clear_cache(self) -> None:
        self._token_cache.clear()

    def cache_stats(self) -> Dict[str, Union[int, float]]:
        memory_usage_bytes = len(self._token_cache) * 100
        return {
            "cache_size": len(self._token_cache),
            "cache_max_size": self._cache_max_size,
            "memory_usage_estimate_kb": round(memory_usage_bytes / 1024, 2),
        }

    def count_tokens_for_chunks(self, chunks: List[RetrievedChunk]) -> Tuple[int, List[int]]:
        if not chunks:
            return 0, []

        encoding = self._get_encoding()
        per_chunk_counts: List[int] = []

        for chunk in chunks:
            if not chunk.content or not chunk.content.strip():
                per_chunk_counts.append(0)
                continue

            content_hash = hashlib.md5(chunk.content.encode("utf-8")).hexdigest()
            cached_token_count = self._token_cache.get(content_hash)
            if cached_token_count is None:
                cached_token_count = len(encoding.encode(chunk.content))
                self._token_cache[content_hash] = cached_token_count
            per_chunk_counts.append(cached_token_count)

        return sum(per_chunk_counts), per_chunk_counts

    def count_tokens_for_text(self, text: Optional[str]) -> int:
        if not text:
            return 0

        try:
            return len(self._get_encoding().encode(text))
        except Exception:  # pragma: no cover - fallback al cálculo por caracteres
            return max(1, int(len(text) / 4))

    def enforce_chunk_size_limits(
        self, chunks: List[RetrievedChunk]
    ) -> Tuple[List[RetrievedChunk], int]:
        if not chunks:
            return [], 0

        max_tokens = max(self._settings.MAX_TOKENS_PER_CHUNK or 0, 0)
        max_chars = max(self._settings.MAX_CHARS_PER_CHUNK or 0, 0)
        if max_tokens <= 0 and max_chars <= 0:
            return chunks, 0

        encoding = self._get_encoding()
        truncated_chunks: List[RetrievedChunk] = []
        truncated_count = 0

        for chunk in chunks:
            content = chunk.content
            if not content:
                truncated_chunks.append(chunk)
                continue

            truncated_content = content
            try:
                if max_tokens > 0:
                    token_ids = encoding.encode(truncated_content)
                    if len(token_ids) > max_tokens:
                        truncated_content = encoding.decode(token_ids[:max_tokens])
                if max_chars > 0 and len(truncated_content) > max_chars:
                    truncated_content = truncated_content[:max_chars]
            except Exception as trunc_err:  # pragma: no cover - fallback seguro
                log.warning(
                    "Failed to apply token-based truncation; falling back to char-based limit",
                    error=str(trunc_err),
                    chunk_id=chunk.id,
                )
                if max_chars > 0 and len(truncated_content) > max_chars:
                    truncated_content = truncated_content[:max_chars]

            if truncated_content != content:
                truncated_chunks.append(chunk.model_copy(update={"content": truncated_content}))
                truncated_count += 1
            else:
                truncated_chunks.append(chunk)

        return truncated_chunks, truncated_count

    def calculate_token_usage(self, chunks: List[RetrievedChunk]) -> TokenAnalysis:
        total_tokens, per_chunk_tokens = self.count_tokens_for_chunks(chunks)
        return TokenAnalysis(
            total_tokens=total_tokens,
            per_chunk_tokens=per_chunk_tokens,
            cache_info=self.cache_stats(),
        )
