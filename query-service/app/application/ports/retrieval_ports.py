import abc
import uuid 
from typing import List, Tuple
from app.domain.models import RetrievedChunk

# Puerto para Retrievers dispersos (como BM25)
class SparseRetrieverPort(abc.ABC):
    """Puerto abstracto para recuperar chunks usando métodos dispersos (keyword-based)."""

    @abc.abstractmethod
    async def search(self, query: str, company_id: uuid.UUID, top_k: int) -> List[Tuple[str, float]]: 
        """
        Busca chunks relevantes basados en la consulta textual y filtra por compañía.

        Args:
            query: La consulta del usuario.
            company_id: El ID de la compañía para filtrar (UUID).
            top_k: El número máximo de IDs de chunks a devolver.

        Returns:
            Una lista de tuplas (chunk_id: str, score: float).
            Nota: Este puerto solo devuelve IDs y scores, el contenido se recupera por separado.
        """
        raise NotImplementedError

# Puerto para Filtros de Diversidad
class DiversityFilterPort(abc.ABC):
    """Puerto abstracto para aplicar filtros de diversidad a los chunks."""

    @abc.abstractmethod
    async def filter(self, chunks: List[RetrievedChunk], k_final: int) -> List[RetrievedChunk]:
        """
        Filtra una lista de chunks para maximizar diversidad y relevancia.

        Args:
            chunks: Lista de RetrievedChunk.
            k_final: El número deseado de chunks finales.

        Returns:
            Una sublista filtrada de RetrievedChunk.
        """
        raise NotImplementedError