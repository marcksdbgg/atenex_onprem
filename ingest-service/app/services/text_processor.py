# ingest-service/app/services/text_processor.py
import structlog
from typing import List, Dict, Any, Optional
import tiktoken

from app.core.config import settings

log = structlog.get_logger(__name__)

class TextProcessor:
    """
    Handles high-density chunking and context injection for SLLM (Granite-2B) alignment.
    Ensures that chunks passed to the embedding service are:
    1. Within strict token limits.
    2. Context-aware (Header Injection).
    3. Overlapped correctly.
    """

    def __init__(self):
        try:
            self.encoder = tiktoken.get_encoding(settings.TIKTOKEN_ENCODING_NAME)
            log.info(f"TextProcessor initialized with encoding: {settings.TIKTOKEN_ENCODING_NAME}")
        except Exception as e:
            log.error(f"Failed to initialize tiktoken encoding {settings.TIKTOKEN_ENCODING_NAME}", error=str(e))
            # Fallback to cl100k_base if configured fails, or raise depending on strictness.
            # Raising to prevent inconsistent data ingestion.
            raise RuntimeError(f"Could not load tiktoken encoding: {e}")

    def refine_chunks(self, raw_chunks: List[Dict[str, Any]], filename: str) -> List[Dict[str, Any]]:
        """
        Takes raw chunks (likely large) from DocProc and refines them into 
        high-density, context-rich micro-chunks.

        Args:
            raw_chunks: List of dicts with 'text' and 'source_metadata'.
            filename: The original filename for context injection.

        Returns:
            A flattened list of refined chunk dictionaries.
        """
        refined_chunks = []
        total_raw_input = len(raw_chunks)
        
        # Prepare logging context
        refine_log = log.bind(filename=filename, input_chunks=total_raw_input)

        for i, raw_chunk in enumerate(raw_chunks):
            text = raw_chunk.get("text", "").strip()
            if not text:
                continue

            metadata = raw_chunk.get("source_metadata", {})
            page_num = metadata.get("page_number", "N/A")

            # 1. Prepare Header
            # "Filename: manual.pdf | Page: 10 >>> "
            header = settings.INGEST_CONTEXT_HEADER_TEMPLATE.format(
                filename=filename,
                page=page_num
            )
            
            # 2. Calculate Token Budgets
            try:
                header_tokens = self.encoder.encode(header)
                len_header = len(header_tokens)
                
                # Safety check: if header is huge (unlikely), we must truncate it or warn
                if len_header > (settings.INGEST_CHUNK_TOKEN_LIMIT // 2):
                    refine_log.warning("Header is suspiciously long, truncating context.", header_len=len_header)
                    header = header[:100] + " >>> "
                    header_tokens = self.encoder.encode(header)
                    len_header = len(header_tokens)

                available_tokens = settings.INGEST_CHUNK_TOKEN_LIMIT - len_header
                if available_tokens <= 0:
                     refine_log.error("Header consumes entire token budget. Skipping chunk.")
                     continue
                
                # 3. Tokenize Content
                content_tokens = self.encoder.encode(text)
                
                # 4. Sliding Window Logic (Token-based)
                step = available_tokens - settings.INGEST_CHUNK_OVERLAP
                if step <= 0: 
                    # Configuration error safeguard
                    step = available_tokens 

                chunks_generated_from_single_raw = 0
                
                # If content fits entirely, just create one chunk
                if len(content_tokens) <= available_tokens:
                    final_text = header + text
                    self._add_refined_chunk(refined_chunks, final_text, metadata, page_num)
                    chunks_generated_from_single_raw = 1
                else:
                    # Slice and dice
                    for start_idx in range(0, len(content_tokens), step):
                        end_idx = min(start_idx + available_tokens, len(content_tokens))
                        
                        chunk_token_slice = content_tokens[start_idx:end_idx]
                        chunk_text_decoded = self.encoder.decode(chunk_token_slice)
                        
                        final_text = header + chunk_text_decoded
                        self._add_refined_chunk(refined_chunks, final_text, metadata, page_num)
                        chunks_generated_from_single_raw += 1
                        
                        if end_idx >= len(content_tokens):
                            break
                
            except Exception as e:
                refine_log.error("Error processing individual chunk refinement", chunk_index=i, error=str(e))
                continue

        refine_log.info("Refinement complete.", output_chunks=len(refined_chunks), expansion_factor=round(len(refined_chunks)/total_raw_input, 2) if total_raw_input else 0)
        return refined_chunks

    def _add_refined_chunk(self, list_ref: List, text: str, source_metadata: Dict, page_num: Any):
        """Helper to append standard dict structure."""
        list_ref.append({
            "text": text,
            # We define a new key to differentiate explicitly if needed, 
            # but 'text' is what the embedding service expects.
            "content_with_header": text, 
            "source_metadata": {
                **source_metadata,
                "is_refined": True,
                "original_page": page_num
            }
        })