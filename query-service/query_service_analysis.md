# Análisis Técnico del Query Service

## 1. Resumen Ejecutivo

El `query-service` es un microservicio robusto construido sobre **FastAPI** que implementa un sistema de **RAG (Retrieval-Augmented Generation)** avanzado. Su arquitectura sigue estrictamente el patrón de **Arquitectura Hexagonal (Puertos y Adaptadores)**, lo que garantiza un desacoplamiento efectivo entre la lógica de negocio, el modelo de dominio y las dependencias de infraestructura (bases de datos, LLMs, servicios vectoriales).

El núcleo del servicio es el `AskQueryUseCase`, que orquesta un pipeline de procesamiento de consultas altamente configurable, capaz de realizar búsquedas híbridas (densa + dispersa), fusión de resultados (RRF), y generación de respuestas adaptativa (Directa vs Map-Reduce).

## 2. Arquitectura y Patrones

### 2.1. Arquitectura Hexagonal
El proyecto está estructurado canónicamente:
- **Domain (`app/domain`)**: Contiene los modelos de datos puros (`RetrievedChunk`, `ChatMessage`) sin dependencias externas.
- **Application (`app/application`)**: Define los puertos (interfaces) y los casos de uso. Aquí reside la lógica de orquestación del RAG.
- **Infrastructure (`app/infrastructure`)**: Implementa los puertos definidos en la capa de aplicación. Aquí encontramos los adaptadores para Postgres, Milvus, LlamaCpp, etc.
- **API (`app/api`)**: La capa de entrada (Drivers) que expone los endpoints REST.

### 2.2. Pipeline Pattern
El servicio implementa un patrón de **Pipeline** explícito para el procesamiento del RAG (`app/application/use_cases/ask_query/pipeline.py`).
- **`RAGPipeline`**: Ejecuta una secuencia de pasos.
- **`PipelineStep`**: Clase base abstracta para cada etapa del proceso (Embedding, Retrieval, Fusion, etc.).
- **Contexto**: Un diccionario (`context`) fluye a través de todos los pasos, acumulando estado (query, embeddings, chunks recuperados, logs).

Este diseño es excelente para la mantenibilidad y testabilidad, permitiendo añadir, quitar o reordenar pasos sin afectar al orquestador principal.

## 3. Análisis de Componentes Clave

### 3.1. Gestión de Dependencias y Ciclo de Vida (`main.py`)
- Uso de `lifespan` para la inicialización asíncrona de recursos pesados (conexiones a DB, Milvus, clientes HTTP).
- Inyección de dependencias manual en `AskQueryUseCase`, lo cual es explícito y claro, aunque escala verbosamente.
- **Punto de Atención**: Hay muchas variables globales en `main.py` (`SERVICE_READY`, instancias). Aunque funcional, un contenedor de inyección de dependencias más formal podría limpiar este archivo.

### 3.2. El Pipeline RAG (`AskQueryUseCase`)
El flujo de ejecución es sofisticado:
1.  **EmbeddingStep**: Convierte la query a vector.
2.  **RetrievalStep**: Ejecuta búsqueda paralela:
    - **Densa**: Milvus (vectores).
    - **Dispersa (Sparse)**: BM25 (vía servicio externo).
3.  **FusionStep**: Implementa **Weighted Reciprocal Rank Fusion (RRF)**. Este es un componente crítico y avanzado que normaliza y combina los rankings de búsqueda semántica y de palabras clave.
4.  **ContentFetchStep**: Optimización inteligente. El paso de recuperación (Retrieval) a veces solo trae IDs y Scores (especialmente de búsquedas dispersas o índices ligeros). Este paso "hidrata" los chunks con su contenido real desde Postgres (`ChunkContentRepository`) solo para los ganadores del RRF.
5.  **FilterStep**: Aplica filtros de diversidad (MMR) o límites simples.
6.  **AdaptiveGenerationStep**: Decide dinámicamente entre:
    - **DirectGeneration**: RAG estándar si el contexto cabe en la ventana.
    - **MapReduceGeneration**: Si el contexto es demasiado grande, divide y conquista.

### 3.3. Infraestructura y Adaptadores

#### **MilvusAdapter** (`app/infrastructure/vectorstores/milvus_adapter.py`)
- **Fortaleza**: Manejo robusto de la conexión y reconexión.
- **Complejidad**: La gestión de esquemas (`INGEST_SCHEMA_FIELDS`) y la construcción dinámica de queries es compleja.
- **Riesgo**: La dependencia de `pymilvus` y la lógica de "fallback" para nombres de campos sugiere que el esquema de la base de datos vectorial podría no estar 100% estandarizado entre entornos.
- **Detalle Técnico**: Realiza la búsqueda vectorial y mapea manualmente los resultados a `RetrievedChunk`. Es vital que el mapeo de campos (`pk_id`, `embedding`, `content`) se mantenga sincronizado con el servicio de ingesta.

#### **LlamaCppAdapter** (`app/infrastructure/llms/llama_cpp_adapter.py`)
- Implementación personalizada de un cliente HTTP para un servidor compatible con OpenAI (llama.cpp).
- **Resiliencia**: Usa `tenacity` para reintentos con backoff exponencial, lo cual es crucial para llamadas a LLMs que pueden fallar por timeout o sobrecarga.
- **Validación**: Intenta forzar salidas JSON válidas, con lógica de "reparación" (`_normalize_json_output`) para limpiar bloques de código markdown, algo muy común en modelos locales pequeños.

## 4. Puntos Fuertes (Pros)

1.  **Separación de Responsabilidades**: El código es muy limpio. Cambiar de Milvus a Qdrant o de LlamaCpp a OpenAI sería trivial gracias a los puertos.
2.  **RAG Avanzado**: No es un RAG ingenuo. Implementa RRF, búsqueda híbrida y Map-Reduce, lo que indica un sistema preparado para producción y casos de uso complejos.
3.  **Observabilidad**: Uso extensivo de `structlog` con contexto (request_id) y logging detallado de los pasos del pipeline.
4.  **Resiliencia**: Manejo de errores en pasos individuales (e.g., si falla la búsqueda dispersa, el pipeline continúa solo con la densa).

## 5. Áreas de Mejora y Riesgos (Cons)

1.  **Complejidad de Configuración**: `AskQueryUseCase` recibe demasiados objetos de configuración (`PromptBudgetConfig`, `MapReduceConfig`, `RetrievalConfig`). Podría simplificarse agrupándolos en un objeto `RAGConfig` único.
2.  **Manejo de Errores Silenciosos**: En `RetrievalStep`, si `dense_task` o `sparse_task` fallan, se loguea el error y se devuelve una lista vacía. Esto es bueno para la disponibilidad, pero peligroso si el fallo es sistémico (e.g., Milvus caído), ya que el usuario podría recibir respuestas alucinadas por falta de contexto sin saber que la búsqueda falló.
3.  **Hardcoding en Prompts**: Los prompts parecen estar construidos en `PromptService` (no inspeccionado a fondo, pero referenciado). Si están hardcodeados en código, sería mejor moverlos a archivos de texto o una base de datos para iterar sin redesplegar.
4.  **Concurrencia en MapReduce**: `MapReduceGenerationStep` usa un semáforo para limitar concurrencia. Esto es bueno, pero en cargas altas, muchas corutinas esperando el semáforo podrían saturar el event loop si no se maneja con cuidado.

## 6. Conclusión

El `query-service` es una pieza de ingeniería sólida. Muestra un nivel de madurez alto en cuanto a patrones de diseño y comprensión de los retos de los sistemas RAG (alucinaciones, ventana de contexto, recuperación híbrida). El código está listo para escalar, aunque se beneficiaría de una limpieza en la inyección de dependencias y una revisión de la estrategia de "fail-open" en la recuperación de documentos.
