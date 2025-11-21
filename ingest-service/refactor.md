# Plan de Refactorización: Ingesta de Alta Densidad para SLLM (Refactor-Ingest)

**Estado:** Aprobado para Implementación
**Objetivo:** Alinear el `ingest-service` con el nuevo pipeline de consulta (RRF + MapReduce) para maximizar la precisión de **Granite-2B** en hardware restringido.

---

## 1. Diagnóstico y Justificación Técnica

El análisis cruzado entre el `query-service` refactorizado y el `ingest-service` actual revela una desconexión crítica:

1.  **Desajuste de Granularidad:**
    *   **Query-Service:** Utiliza un pipeline sofisticado (RRF) y un filtro generativo (MapReduce) diseñados para trabajar con evidencias precisas.
    *   **Ingest-Service (Actual):** Ingiere lo que entrega `docproc` (chunks grandes, ~1000-1500 tokens).
    *   **Consecuencia:** Al incrustar (embed) 1500 tokens en un solo vector de 384 dimensiones, la información se "diluye". El vector representa el *promedio* del texto, perdiendo detalles específicos. Granite-2B, con su ventana de atención limitada, se confunde al recibir tanto ruido en la fase de generación.

2.  **Ausencia de Contexto Explícito:**
    *   Los chunks actuales son texto plano. Si un chunk dice "El límite es de 500", el LLM no sabe si se refiere a una tarjeta de crédito o a una velocidad de internet, a menos que el texto lo diga explícitamente.
    *   **Solución:** *Context Injection*. Cada chunk debe llevar incrustado su origen (`File: manual.pdf | Section: Limites`).

3.  **Latencia en Pre-fill (Cuello de Botella CPU):**
    *   Enviarle chunks grandes a `llama.cpp` (CPU) durante la fase *Map* provoca tiempos de lectura (pre-fill) excesivos, causando los timeouts HTTP observados en los logs del incidente anterior. Chunks más pequeños = Inferencia más rápida.

---

## 2. Estrategia: Ingesta Semántica de Alta Densidad

Transformaremos el pipeline de un flujo "Passthrough" (DocProc -> Milvus) a un flujo de **Refinamiento Activo**.

### Nuevo Flujo Lógico:
1.  **Extract (DocProc):** Obtener texto crudo y particiones estructurales básicas.
2.  **Refine (Ingest - Nuevo):**
    *   Inyección de Metadatos (Header Injection).
    *   Token-Split estricto (Hard limit: 384 tokens).
    *   Ventana deslizante (Overlap: 50 tokens).
3.  **Embed (Embedding Svc):** Vectorizar los chunks refinados.
4.  **Persist (Milvus/PG):** Guardar con trazabilidad de metadatos mejorada.

---

## 3. Especificaciones Técnicas por Archivo

A continuación, se detalla la implementación requerida para cada componente.

### 3.1. Configuración (`app/core/config.py`)

Se deben añadir las constantes que gobiernan la física del chunking.

```python
class Settings(BaseSettings):
    # ... configs existentes ...

    # --- SLLM Optimization: High Density Ingestion ---
    # Límite estricto para alinearse con all-MiniLM-L6-v2 / E5 y la ventana de Granite
    INGEST_CHUNK_TOKEN_LIMIT: int = 384 
    
    # Superposición para mantener coherencia entre cortes
    INGEST_CHUNK_OVERLAP: int = 50
    
    # Template para inyección de contexto en el vector
    # Esto hace que el chunk sea "Self-Contained"
    INGEST_CONTEXT_HEADER_TEMPLATE: str = "Filename: {filename} | Page: {page} >>> "
    
    # Encoding para conteo preciso (usar el mismo que el modelo embedding si es posible, sino cl100k)
    TIKTOKEN_ENCODING_NAME: str = "cl100k_base"
```

### 3.2. Nuevo Servicio de Procesamiento (`app/services/text_processor.py`)

Crear este servicio para encapsular la lógica de división y enriquecimiento. No debemos confiar ciegamente en `docproc` para la granularidad final.

**Responsabilidades:**
*   Recibir una lista de chunks "crudos".
*   Calcular el tamaño del header de metadatos en tokens.
*   Dividir el contenido restante usando `tiktoken` y sliding window.
*   Devolver una lista aplanada de chunks optimizados.

**Lógica Crítica:**
```python
# Pseudocódigo de implementación
def refine_chunks(raw_chunks: List[Dict], filename: str) -> List[Dict]:
    refined = []
    for chunk in raw_chunks:
        header = settings.INGEST_CONTEXT_HEADER_TEMPLATE.format(...)
        # Calcular tokens disponibles: 384 - len(header_tokens)
        # Dividir texto original respetando available_tokens + overlap
        # Crear nuevos objetos chunk con el texto = header + sub_slice
        # Preservar metadatos originales
    return refined
```

### 3.3. Modificación de la Tarea Celery (`app/tasks/process_document.py`)

El flujo de la tarea `process_document_standalone` debe interceptar la respuesta de `DocProc` antes de enviarla a `EmbeddingService`.

**Cambios en `process_document_standalone`:**

1.  **Llamada a DocProc:** (Sin cambios, devuelve chunks grandes).
2.  **Fase de Refinamiento (NUEVO):**
    *   Instanciar `TextProcessor`.
    *   Llamar `refined_chunks = text_processor.refine_chunks(processed_chunks_from_docproc, normalized_filename)`.
    *   Loggear la expansión: *"DocProc returned 5 chunks. Refined into 18 high-density chunks."*
3.  **Llamada a Embedding:**
    *   Usar `refined_chunks` para generar la lista de textos (`[c['content_with_header'] for c in refined_chunks]`).
    *   Enviar a `EmbeddingServiceClient`.
4.  **Indexación:**
    *   Pasar `refined_chunks` y los vectores resultantes a `index_chunks_in_milvus_and_prepare_for_pg`.

### 3.4. Ajuste del Pipeline de Ingesta (`app/services/ingest_pipeline.py`)

La función `index_chunks_in_milvus_and_prepare_for_pg` necesita adaptarse para manejar la estructura de datos refinada.

**Adaptaciones:**
*   El cálculo de hash (`content_hash`) debe hacerse sobre el contenido **con el header incluido** para asegurar unicidad semántica.
*   El campo `chunk_index` debe ser secuencial basado en la lista aplanada (0, 1, 2... N), ignorando la estructura jerárquica original para simplificar la recuperación en Milvus.
*   **Validación:** Asegurar que los metadatos originales (ej. número de página real) se propaguen a los sub-chunks divididos.

### 3.5. Base de Datos y Migraciones

*   **Tabla `document_chunks` (Postgres):** No requiere cambio de esquema (DDL), pero el volumen de filas aumentará (factor 3x-4x).
*   **Milvus:** No requiere cambio de esquema, ya que los campos (`embedding`, `content`, `page`) se mantienen. La "magia" está en que el contenido guardado ahora incluye el header explícito.

---

## 4. Alineación con Query-Service (El "Por Qué")

Esta refactorización no es aislada; es el prerrequisito para que el `query-service` funcione correctamente:

1.  **RRF (Reciprocal Rank Fusion):** RRF funciona comparando rankings. Si los chunks son demasiado largos y difusos, los vectores de Milvus tendrán scores de similitud bajos y ruidosos. Chunks pequeños y densos producen vectores "agudos" (sharp) que rankean mejor, haciendo que la fusión con BM25 sea efectiva.
2.  **MapReduce Generativo:**
    *   **Query-Service Input:** Recibe 5-10 chunks.
    *   **Antes:** 10 chunks * 1500 tokens = 15,000 tokens. `llama.cpp` colapsa por pre-fill time (Timeouts > 60s).
    *   **Ahora:** 10 chunks * 384 tokens = 3,840 tokens. Procesamiento rápido (<15s). El LLM puede leer todo y filtrar "IRRELEVANTE" eficientemente.
3.  **Alucinaciones de Granite:** Al inyectar `Filename: ... | Page: ...` en el texto, forzamos al LLM a asociar la información con una fuente. Esto permite que el query-service extraiga citas precisas `[Doc 1]` en la respuesta final.

---

## 5. Plan de Ejecución Paso a Paso

1.  **Fase 1: Core Logic (Sin Side Effects)**
    *   Implementar `app/services/text_processor.py`.
    *   Añadir configuraciones en `config.py`.

2.  **Fase 2: Integración en Worker**
    *   Modificar `app/tasks/process_document.py` para integrar el `TextProcessor`.
    *   Actualizar logs para visualizar la explosión de chunks (Chunks entrada vs Salida).

3.  **Fase 3: Validación E2E**
    *   Subir un documento complejo (ej. PDF legal de 20 pág).
    *   Verificar en Milvus que los chunks son pequeños (~384 dims) y contienen el header.
    *   Ejecutar una consulta en `query-service` y observar si mejora la latencia y la precisión de las citas.

4.  **Fase 4: Limpieza (Opcional)**
    *   Si se requiere re-indexar documentos antiguos, crear un script que los baje de MinIO y los re-procese a través del nuevo pipeline.

---

## 6. Archivos a Modificar/Crear

| Archivo | Acción | Descripción |
| :--- | :--- | :--- |
| `app/core/config.py` | Modificar | Agregar constantes de límites de tokens y template de header. |
| `app/services/text_processor.py` | **Crear** | Lógica de tiktoken, header injection y splitting. |
| `app/tasks/process_document.py` | Modificar | Integrar la fase de refinamiento antes del embedding. |
| `app/services/ingest_pipeline.py` | Modificar | Adaptar la preparación de datos para usar textos optimizados. |
| `app/api/v1/endpoints/ingest.py` | Revisar | Asegurar que los endpoints de estado reflejen métricas correctas. |

---