# Plan Maestro de Optimización: Atenex RAG para SLLM en Hardware Restringido

**Versión del Plan:** 2.0 (Post-Incidente)
**Objetivo:** Estabilizar la latencia, maximizar la precisión factual y eliminar cuellos de botella en la arquitectura de microservicios, adaptando el pipeline para operar eficientemente con modelos cuantizados (<4GB VRAM/RAM) sin un servicio de Reranking dedicado.

---

## 1. Diagnóstico Científico del Estado Actual
Basado en la telemetría del incidente (`request_id: 8db8954a...`) y la revisión del código, el sistema actual sufre de una **saturación de contexto y colapso de inferencia**.

### 1.1. Identificación del Fallo en Cascada
1.  **Ingesta Ineficiente:** Los *chunks* recuperados poseen una longitud promedio de **~1,500 tokens**. Esto contradice las mejores prácticas para SLLMs, cuya ventana de atención efectiva se degrada rápidamente.
2.  **Fallo en Diversidad (MMR):** El `MilvusAdapter` no está retornando los vectores (*embeddings*) a la aplicación. Como resultado, el filtro de diversidad (`MMRDiversityFilter`) falla silenciosamente y deja pasar redundancia masiva.
3.  **Sobrecarga de MapReduce:** Al recibir ~24,000 tokens de contexto total, el sistema intentó paralelizar llamadas a `llama.cpp`. Al ser un host basado en CPU, el *pre-fill time* (tiempo de lectura del prompt) superó exponencialmente el `HTTP_CLIENT_TIMEOUT` del `query-service`, causando un bucle de reintentos que actuó como un ataque de denegación de servicio interno.
4.  **Dependencia Crítica del Reranker:** La arquitectura original dependía de un modelo *Cross-Encoder* pesado para limpiar la búsqueda. Al eliminarlo sin optimizar la recuperación previa, se alimenta "ruido" al LLM.

---

## 2. Estrategia de Optimización: "Calidad sobre Cantidad"

Para compensar la eliminación del servicio Reranker y el tamaño reducido del modelo Granite, implementaremos una estrategia de **Densidad Semántica** y **Filtrado Generativo**.

### 2.1. Reingeniería de Ingesta y Chunking (Fundamentado en Tesis Atenex)
El modelo Granite-2B no tiene la capacidad de "aguja en el pajar" de GPT-4. Necesitamos que la unidad de información sea atómica y precisa.

*   **Acción:** Implementar **Chunking Semántico-Estructural**.
    *   **Límite Duro:** Reducir el tamaño del chunk de 1000+ tokens a **256-384 tokens**.
    *   **Justificación:** Fragmentos más pequeños aumentan la precisión del *embedding* denso (vector) y reducen la carga de lectura del LLM.
    *   **Enriquecimiento:** Cada chunk debe incluir metadatos inyectados en el texto (ej: `[Título Documento] >> [Sección] >> Contenido`) para que el chunk sea auto-contenido, mitigando la pérdida de contexto global.

### 2.2. Optimización de Recuperación Híbrida (Sustitución del Reranker)
Al eliminar el modelo *Cross-Encoder* (Reranker), debemos delegar la precisión a la fase de recuperación inicial mediante algoritmos matemáticos de bajo costo.

*   **Acción:** Implementar **Weighted Reciprocal Rank Fusion (RRF)** Estricto.
    *   En lugar de un reranker neuronal, utilizaremos RRF para fusionar los resultados de `Milvus` (Semántico) y `Sparse-Search` (BM25/Léxico).
    *   **Calibración:** Ajustar el parámetro `alpha` para dar un ligero peso superior a BM25, ya que los SLLMs se benefician de la coincidencia exacta de palabras clave para terminología específica (nombres, códigos), compensando la posible "alucinación semántica" de los embeddings pequeños.
    *   **Optimización de Carga:** Configurar el `VectorStorePort` para traer **siempre** el embedding. Esto es obligatorio para que funcione cualquier filtro de diversidad posterior.

### 2.3. Estrategia de Caching Adaptativo (Referencia: EdgeRAG)
Como sugiere el paper *EdgeRAG*, el costo de generar embeddings y recuperar contexto es alto en el *edge*.

*   **Acción:** Implementar **Cacheo de Resultados de Recuperación**.
    *   Antes de llamar al pipeline RAG, verificar si la consulta (hasheada) ya existe en una tabla de caché rápida (Redis o memoria LRU) junto con sus IDs de chunks recuperados.
    *   Esto evita invocar a `sparse-search-service` y `milvus` en preguntas repetitivas, reduciendo la latencia base en un 40-60% para preguntas frecuentes.

---

## 3. El Nuevo Pipeline MapReduce: "Filtrado Generativo"
Esta es la pieza clave para que el modelo de 2B parámetros funcione correctamente. Convertiremos la fase **Map** en nuestro nuevo "Reranker Lógico".

### 3.1. Protocolo de "Filtrado Negativo"
En lugar de pedirle al modelo que extraiga información de todo lo que lee, le enseñaremos a **descartar agresivamente**.

*   **Prompt de Fase MAP (Optimizado para SLLM):**
    *   Instrucción explícita: *"Eres un filtro de calidad. Si el fragmento NO contiene la respuesta a la pregunta '{{query}}', responde ÚNICAMENTE la palabra: 'IRRELEVANTE'. Si contiene información parcial, extrae solo las frases clave."*
    *   **Beneficio:** Esto limpia el contexto para la fase *Reduce*. Si 8 de 10 chunks son ruido, el prompt final de *Reduce* será muy corto y limpio, aumentando la precisión del modelo pequeño.

### 3.2. Control de Concurrencia y Timeouts
Para solucionar el cuello de botella observado en los logs (timeout por sobrecarga de CPU):

*   **Acción:** **Procesamiento Serializado o Semáforo Estricto**.
    *   No lanzar 10 peticiones al LLM en paralelo (`asyncio.gather` sin límites).
    *   Configurar un `Semaphore` de concurrencia limitado a **1 o 2** (dependiendo de los hilos físicos del CPU). Esto asegura que `llama.cpp` termine una generación antes de empezar otra, evitando el *context switching* excesivo y el timeout por espera en cola.
    *   **Alineación de Timeouts:** Aumentar el `HTTP_CLIENT_TIMEOUT` en `query-service` basándose en la fórmula: `(Max Tokens Input / Velocidad de Ingesta) + (Max Tokens Output / Velocidad de Generación) + Buffer`. Según tus logs, esto debería ser al menos **120 segundos** para cargas pesadas, no 30s.

---

## 4. Arquitectura Lógica Final (Flujo de Datos)

Este flujo reemplaza el componente de Reranking dedicado por lógica algorítmica y generativa:

1.  **User Query** -> `API Gateway` -> `Query Service`.
2.  **Check Salud**: Si es un saludo simple -> Responder directo (Regla Regex).
3.  **Retrieval Híbrido Optimizado**:
    *   Milvus (Top-20) + Sparse (Top-20).
    *   Fusión RRF -> Genera Top-10 ordenado.
    *   *Filtro Diversidad (Light)*: Aplicar filtro de umbral simple (eliminar duplicados exactos por ID o hash de contenido).
4.  **Pipeline MapReduce (El "Reranker" Generativo)**:
    *   **Entrada:** Top-10 chunks.
    *   **Proceso:** Iterar sobre los chunks (lotes de 1 o 2 máximo).
    *   **LLM (Map):** Evalúa pertinencia. Si es ruido -> Output: "IRRELEVANTE".
5.  **Fase Reduce (Síntesis)**:
    *   **Entrada:** Solo los outputs de la fase Map que NO son "IRRELEVANTE".
    *   **Contexto:** Historial de chat + Resúmenes validados.
    *   **LLM (Reduce):** Genera respuesta final en JSON estructurado.
6.  **Output**: Respuesta al usuario + Citas (Ids de los chunks sobrevivientes).

## 5. Justificación basada en los Papers Adjuntos

1.  **Tesis Atenex (Cap 3.3.2):** Valida el uso de "Filtrado Negativo" en la fase MAP para reducir alucinaciones en SLMs.
2.  **EdgeRAG:** Soporta la decisión de no usar un índice plano completo en memoria y la necesidad de gestionar la memoria caché agresivamente debido a las limitaciones del hardware local.
3.  **Rendimiento en SLLMs (Medical Paper):** Confirma que modelos pequeños (Llama-3 8B, comparable a Granite en optimización) pueden superar a modelos grandes si el contexto inyectado es de alta calidad (Logrado aquí por chunking pequeño + RRF + Filtrado Generativo).

## 6. Conclusión del Plan
Al ejecutar este plan, se espera:
1.  **Reducción drástica de latencia:** Al reducir el tamaño de chunk y serializar las peticiones al LLM.
2.  **Eliminación de Timeouts:** Al alinear la configuración del cliente HTTP con la realidad física del hardware (CPU inference).
3.  **Mantenimiento de Calidad:** El Reranking semántico (modelo pesado) es sustituido eficazmente por el RRF matemático (rápido) y el juicio crítico del LLM en la fase Map (preciso), cumpliendo la promesa de la arquitectura monolítica-modular eficiente.