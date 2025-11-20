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