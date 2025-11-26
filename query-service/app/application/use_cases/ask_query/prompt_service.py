import asyncio
import os
from enum import Enum
from typing import Dict, List, Optional, Any
import structlog
from jinja2 import Environment, FileSystemLoader, StrictUndefined, TemplateNotFound, meta, TemplateSyntaxError
from app.core.config import settings
from app.domain.models import RetrievedChunk

log = structlog.get_logger(__name__)

class PromptRenderError(Exception):
    """Excepción personalizada para errores de renderizado de prompts."""
    pass

class PromptType(Enum):
    RAG = "rag"
    GENERAL = "general"
    MAP = "map"
    REDUCE = "reduce"

class PromptManager:
    def __init__(self, template_dir: str, auto_reload: bool = False):
        """
        Inicializa el gestor de prompts con configuración optimizada para LLMs.
        """
        self.template_dir = template_dir
        
        # Configuración del Entorno Jinja2
        self.env = Environment(
            loader=FileSystemLoader(template_dir),
            auto_reload=auto_reload,  # Hot-reloading
            autoescape=False,         # Texto plano para LLM
            trim_blocks=True,         # Elimina newlines después de bloques
            lstrip_blocks=True,       # Elimina espacios antes de bloques
            keep_trailing_newline=True,
            undefined=StrictUndefined # Lanza error si falta una variable
        )
        log.info("PromptManager initialized", template_dir=template_dir, auto_reload=auto_reload)

    def render(self, template_name: str, **kwargs: Any) -> str:
        """
        Renderiza un prompt de manera segura. Acepta dicts o modelos Pydantic.
        """
        try:
            template = self.env.get_template(template_name)
            return template.render(**kwargs)
        except TemplateNotFound:
            log.error(f"Template not found: {template_name} in {self.template_dir}")
            raise PromptRenderError(f"El template '{template_name}' no existe en {self.template_dir}")
        except Exception as e:
            log.error(f"Error rendering template {template_name}", error=str(e))
            raise PromptRenderError(f"Fallo al renderizar prompt '{template_name}': {str(e)}")

class PromptService:
    def __init__(self) -> None:
        # Inferimos el directorio de prompts desde la configuración
        # Asumimos que todos los templates están en el mismo directorio
        rag_path = settings.RAG_PROMPT_TEMPLATE_PATH
        self.prompt_dir = os.path.dirname(rag_path)
        
        # Inicializamos el Manager
        # auto_reload=True permite editar los .txt sin reiniciar el servicio
        self.manager = PromptManager(template_dir=self.prompt_dir, auto_reload=True)
        
        # Mapeamos tipos a nombres de archivo (basename)
        self.templates = {
            PromptType.RAG: os.path.basename(settings.RAG_PROMPT_TEMPLATE_PATH),
            PromptType.GENERAL: os.path.basename(settings.GENERAL_PROMPT_TEMPLATE_PATH),
            PromptType.MAP: os.path.basename(settings.MAP_PROMPT_TEMPLATE_PATH),
            PromptType.REDUCE: os.path.basename(settings.REDUCE_PROMPT_TEMPLATE_PATH),
        }

    def _chunk_to_dict(self, chunk: RetrievedChunk, index: int) -> Dict[str, Any]:
        """Convierte chunk a dict para template rendering."""
        return {
            "index": index,
            "id": chunk.id,
            "content": chunk.content,
            "file_name": chunk.file_name or "N/A",
            "page": chunk.metadata.get("page") if chunk.metadata else "?",
            "score": round(chunk.score, 2) if chunk.score else 0.0
        }

    async def build_rag_prompt(self, query: str, chunks: List[RetrievedChunk], chat_history: str) -> str:
        formatted_docs = [self._chunk_to_dict(c, i+1) for i, c in enumerate(chunks)]
        return self.manager.render(
            self.templates[PromptType.RAG],
            query=query,
            documents=formatted_docs,
            chat_history=chat_history
        )

    async def build_map_prompt(self, query: str, chunks: List[RetrievedChunk], index_offset: int, total: int) -> str:
        formatted_docs = [self._chunk_to_dict(c, 0) for c in chunks] # Index handled in template loop
        return self.manager.render(
            self.templates[PromptType.MAP],
            original_query=query,
            documents=formatted_docs,
            document_index=index_offset,
            total_documents=total
        )

    async def build_reduce_prompt(self, query: str, map_results: str, original_chunks: List[RetrievedChunk], chat_history: str) -> str:
        formatted_docs = [self._chunk_to_dict(c, i+1) for i, c in enumerate(original_chunks)]
        return self.manager.render(
            self.templates[PromptType.REDUCE],
            original_query=query,
            mapped_responses=map_results,
            original_documents_for_citation=formatted_docs,
            chat_history=chat_history
        )