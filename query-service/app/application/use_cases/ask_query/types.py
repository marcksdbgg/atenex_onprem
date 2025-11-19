from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Dict, List, Optional, Union

from app.domain.models import ChatMessage, RetrievedChunk


@dataclass
class ChatSession:
    chat_id: uuid.UUID
    chat_history: Optional[str]
    history_messages: List[ChatMessage]
    is_new_chat: bool


@dataclass
class RetrievalOutcome:
    chunks: List[RetrievedChunk]
    fusion_fetch_k: int
    num_chunks_after_rerank: int
    stages: List[str]


@dataclass
class TokenAnalysis:
    total_tokens: int
    per_chunk_tokens: List[int]
    cache_info: Dict[str, Union[int, float]]


@dataclass
class ResponseGenerationResult:
    json_answer: str
    chunks_used_for_answer: List[RetrievedChunk]
    map_reduce_used: bool
    stages: List[str]
