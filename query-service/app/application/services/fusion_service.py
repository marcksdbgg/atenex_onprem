from typing import List, Dict, Any, TypeVar, Optional
from collections import defaultdict
import structlog

log = structlog.get_logger(__name__)

T = TypeVar("T")

class FusionService:
    """
    Servicio de Fusión de Rankings optimizado para arquitecturas RAG Híbridas
    sin Reranker Neuronal. Implementa Weighted RRF (Reciprocal Rank Fusion).
    """

    def __init__(self, default_k: int = 30):
        """
        Args:
            default_k (int): Constante de suavizado. 
                             Un k=60 es estándar. 
                             Un k=20-30 es 'agresivo' para favorecer documentos top.
        """
        self.default_k = default_k

    def weighted_rrf(
        self,
        dense_results: List[Any],
        sparse_results: List[Any],
        dense_weight: float = 1.0,
        sparse_weight: float = 1.5,
        top_k: int = 10,
        id_field: str = "id" 
    ) -> List[Any]:
        """
        Ejecuta la fusión RRF ponderada.
        """
        fusion_log = log.bind(action="weighted_rrf", dense_count=len(dense_results), sparse_count=len(sparse_results))
        
        # 1. Mapa acumulador de scores: {id: score}
        rrf_score_map: Dict[str, float] = defaultdict(float)
        
        # 2. Mapa para retener el objeto completo
        content_map: Dict[str, Any] = {}

        def process_list(results: List[Any], weight: float):
            for rank, item in enumerate(results):
                # Obtener ID
                if isinstance(item, dict):
                    item_id = item.get(id_field)
                else:
                    item_id = getattr(item, id_field, None)

                if not item_id:
                    continue

                # Fórmula RRF Ponderada
                score = weight * (1.0 / (self.default_k + rank + 1))
                
                rrf_score_map[item_id] += score
                
                # Estrategia de preservación de objetos
                # Priorizamos objetos que ya tengan contenido poblado (dense usualmente lo tiene)
                if item_id not in content_map:
                    content_map[item_id] = item
                else:
                    existing = content_map[item_id]
                    # Chequeo simple: si el existente no tiene 'content' y el nuevo sí, actualizamos.
                    # Esto maneja el caso donde Sparse llega primero pero Dense tiene el texto completo.
                    existing_content = getattr(existing, 'content', None) or (existing.get('content') if isinstance(existing, dict) else None)
                    new_content = getattr(item, 'content', None) or (item.get('content') if isinstance(item, dict) else None)
                    
                    if not existing_content and new_content:
                        content_map[item_id] = item

        # 3. Procesar ambas listas
        process_list(dense_results, dense_weight)
        process_list(sparse_results, sparse_weight)

        # 4. Ordenar por Score RRF descendente
        sorted_items = sorted(rrf_score_map.items(), key=lambda x: x[1], reverse=True)

        # 5. Reconstruir lista final y asignar Score RRF normalizado para trazabilidad
        final_results = []
        for item_id, score in sorted_items[:top_k]:
            original_obj = content_map[item_id]
            
            # Inyectamos el score RRF en el objeto (para debugging/metrics)
            if isinstance(original_obj, dict):
                original_obj['_rrf_score'] = round(score, 5)
                # Overwrite score for UI consistency if needed, implies retrieval logic relies on score sort
                original_obj['score'] = score 
            elif hasattr(original_obj, 'score'):
                original_obj.score = score 
                # También podemos guardar el original si el modelo lo permite
                if hasattr(original_obj, 'metadata') and isinstance(original_obj.metadata, dict):
                    original_obj.metadata['rrf_score'] = score

            final_results.append(original_obj)

        fusion_log.info(f"RRF completed. Top-K fused results: {len(final_results)}")
        return final_results