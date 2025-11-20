from __future__ import annotations
import uuid
from dataclasses import dataclass
from typing import Dict, List, Optional, Union, Any
from app.domain.models import ChatMessage, RetrievedChunk

@dataclass
class TokenAnalysis:
    total_tokens: int
    per_chunk_tokens: List[int]
    cache_info: Dict[str, Union[int, float]]

@dataclass
class RetrievalOutcome:
    chunks: List[RetrievedChunk]
    stages: List[str]