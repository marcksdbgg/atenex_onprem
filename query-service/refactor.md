# Plan de Refactorización y Optimización del Query-Service (v1.3.5)

## 1. Diagnóstico General de la Codebase

Tras un análisis completo del microservicio, se identifican las siguientes fortalezas y áreas de mejora.

### Fortalezas y Buenas Prácticas Actuales:

*   **Clean Architecture Sólida:** La separación entre `api`, `application`, `domain`, e `infrastructure` es clara y sigue los principios de la Arquitectura Limpia. Las dependencias se invierten correctamente a través de los puertos (interfaces).
*   **Abstracción de Responsabilidades:** La creación de `TokenAccountant` y `PromptService` es un excelente ejemplo de aplicación del Principio de Responsabilidad Única (SRP). El código actual confirma que toda la lógica de conteo de tokens y construcción de prompts ya vive en estas clases auxiliares.
*   **Modularidad Mejorada:** La nueva estructura dentro de `application/use_cases/ask_query` agrupa lógicamente las utilidades que dan soporte al caso de uso, mejorando la legibilidad. No quedan referencias a las implementaciones anteriores (`_prompt_builder_*`, caché manual de tokens, etc.).
*   **Uso de Puertos y Adaptadores:** El sistema es extensible. Cambiar de Milvus a otro vector store o de Llama.cpp a otro LLM solo requiere implementar un nuevo adaptador, lo cual es ideal.
*   **Integración Verificada:** Tras la refactorización reciente no se aprecian regresiones funcionales en la codebase; el flujo del caso de uso compila con las nuevas dependencias internas y los tipos exportados en `types.py`.

### Áreas Clave para Refactorización y Optimización:

1.  **Complejidad de Configuración del MapReduce:** La lógica para decidir si se activa el MapReduce es innecesariamente compleja. Múltiples variables de entorno (`MAPREDUCE_ACTIVATION_THRESHOLD_CHUNKS`, `MAX_PROMPT_TOKENS`, y toda la familia de `_OVERHEAD_TOKENS`) crean un sistema de umbrales redundante y difícil de ajustar. El objetivo real es simple: "usar MapReduce si el contexto no cabe en una sola llamada". La configuración actual lo ofusca.

2.  **El "Método Orquestador" (`AskQueryUseCase.execute`):** A pesar de la refactorización, el método `execute` sigue siendo un script procedural muy largo. Contiene toda la lógica del pipeline RAG (recuperación, fusión, reranking, decisión de MapReduce, etc.) de forma secuencial y con múltiples condicionales (`if settings.RERANKER_ENABLED:`). Esto hace que el flujo sea rígido, difícil de modificar, probar y optimizar.

3.  **Lógica de Pipeline Implícita:** El orden y la activación de cada paso del RAG están codificados directamente en el método `execute`. No existe una abstracción del "pipeline" como tal, lo que dificulta añadir, quitar o reordenar pasos sin modificar una gran cantidad de código.

4.  **Optimización para LLMs Pequeños:** Aunque se han hecho mejoras, el flujo actual no está completamente optimizado para LLMs con recursos limitados. Las llamadas en la fase Map, aunque ahora son concurrentes, podrían gestionarse de forma más robusta y el proceso de decisión de truncamiento de contexto podría ser más dinámico.

5.  **Gestión de Dependencias y Estados Compartidos:** El repositorio aún no declara explícitamente dependencias vitales (`structlog`, `haystack`, `tiktoken`, `fastapi`, `pydantic`) en `pyproject.toml`; esto puede romper despliegues limpios. Además, `TokenAccountant` mantiene una caché en memoria sin política de expiración ni garantías de thread-safety cuando se despliega con varios workers.

6.  **Rutas de Prompt y Fallbacks:** `PromptType.GENERAL` no se utiliza en el flujo actual. Mantener plantillas no utilizadas añade ruido y oculta problemas si falta un archivo, ya que el servicio cae en un fallback silencioso en lugar de fallar rápido.

---

## 2. Plan de Refactorización Propuesto

El objetivo es transformar el `AskQueryUseCase` de un "orquestador procedural" a un "director de un pipeline configurable y dinámico", simplificando drásticamente la configuración y mejorando la modularidad.

### Fase 1: Simplificar Radicalmente la Configuración y Lógica de Decisión del MapReduce

El problema central es la sobrecarga de parámetros para una decisión simple.

*   **Propuesta:** Eliminar la dependencia de múltiples umbrales y cálculos de overhead complejos.

*   **Acciones a Realizar:**
    1.  **Eliminar Parámetros Redundantes:** Quitar de `config.py` las siguientes variables:
        *   `MAPREDUCE_ACTIVATION_THRESHOLD_CHUNKS`: El conteo de chunks es un mal proxy del tamaño del contexto. La decisión debe basarse únicamente en tokens.
        *   Toda la familia `_OVERHEAD_TOKENS` (`PROMPT_BASE_OVERHEAD_TOKENS`, `PROMPT_PER_CHUNK_OVERHEAD_TOKENS`, etc.): Estos valores son frágiles y difíciles de mantener.
    2.  **Introducir un Umbral Único y Claro:** Añadir un solo parámetro en `config.py`:
        *   `DIRECT_RAG_TOKEN_LIMIT`: Un número entero que define el máximo de tokens de *documentos* que se pueden enviar en una llamada directa. Por ejemplo, `16000`.
    3.  **Simplificar la Lógica de Decisión:** En `AskQueryUseCase`, la decisión se reducirá a:
        ```
        # Pseudocódigo
        token_analysis = token_accountant.calculate_token_usage(final_chunks)
        
        if settings.MAPREDUCE_ENABLED and token_analysis.total_tokens > settings.DIRECT_RAG_TOKEN_LIMIT:
            # Activar MapReduce
        else:
            # Usar Direct RAG (truncando los chunks si es necesario para que quepan)
        ```
    4.  **Limpieza:** El método `execute` ya no necesitará calcular `estimated_direct_prompt_tokens` ni manejar múltiples condiciones, haciendo el código mucho más limpio.
    5.  **Agrupar Configuración:** Crear dataclasses como `PromptBudgetConfig`, `MapReduceConfig` y `RetrievalConfig` que agrupen los parámetros que sobrevivan. El caso de uso recibirá estas estructuras en lugar de leer directamente de `settings`, facilitando validaciones y pruebas.

*   **Beneficios:**
    *   **Configuración Intuitiva:** El ajuste del sistema se reduce a un solo valor claro.
    *   **Robustez:** La decisión es directa y predecible.
    *   **Optimización para LLMs pequeños:** Es muy fácil ajustar `DIRECT_RAG_TOKEN_LIMIT` a la baja para modelos con ventanas de contexto pequeñas sin tener que recalcular múltiples variables de overhead.

### Fase 2: Abstraer el Pipeline RAG con un Patrón de Diseño "Strategy" o "Chain of Responsibility"

El método `execute` no debe *implementar* el pipeline, sino *construirlo y ejecutarlo*.

*   **Propuesta:** Crear una serie de clases que representen cada etapa del pipeline, donde cada una se encarga de una única tarea (p.ej. `RetrievalPipeline`, `ChunkPostProcessor`, `PromptAssemblyService`, `LLMOrchestrator`, `ResponseAssembler`).

*   **Acciones a Realizar:**
    1.  **Definir una Interfaz de Etapa (Step):** Crear una clase base abstracta `PipelineStep` con un método `execute(context)`, donde `context` es un objeto que contiene los datos que fluyen a través del pipeline (query, chunks, etc.).
    2.  **Crear Implementaciones Concretas:**
        *   `RetrievalStep`: Encapsula la búsqueda densa y dispersa y su `asyncio.gather`.
        *   `FusionStep`: Realiza la fusión RRF.
        *   `ContentFetchStep`: Obtiene el contenido de los chunks.
        *   `RerankStep`: Llama al servicio de reranking (solo si está habilitado).
        *   `FilterStep`: Aplica el filtro de diversidad.
        *   `GenerationStep`: Esta es una etapa clave. Tendrá dos implementaciones (estrategias):
            *   `DirectGenerationStep`: Construye el prompt RAG y llama al LLM una vez.
            *   `MapReduceGenerationStep`: Encapsula toda la lógica de MapReduce (batching, llamadas concurrentes con semáforo, y la llamada final de reduce).
    3.  **Crear una Clase `RAGPipeline`:**
        *   Esta clase se inicializa con una lista de `PipelineStep`.
        *   Tendrá un método `run(initial_context)` que itera sobre los pasos, pasando el contexto modificado de uno a otro, y encapsulará la recolección de métricas del pipeline.
    4.  **Refactorizar `AskQueryUseCase.execute`:**
        *   El método `execute` ahora se encargará de:
            a.  Gestionar el estado del chat (como ya hace).
            b.  **Construir dinámicamente el pipeline:** Añadir `RerankStep` solo si `settings.RERANKER_ENABLED` es true.
            c.  Decidir qué `GenerationStep` usar basado en el umbral de tokens simplificado de la Fase 1.
            d.  Ejecutar `pipeline.run()`.
            e.  Manejar la respuesta final y el logging a través de un `ResponseAssembler` dedicado.

*   **Beneficios:**
    *   **Desacoplamiento Total:** Cada etapa del pipeline es independiente y reutilizable.
    *   **Extensibilidad (Principio Abierto/Cerrado):** Añadir un nuevo paso (ej. un "Query Expansion Step") solo requiere crear una nueva clase e insertarla en la construcción del pipeline, sin modificar el código existente.
    *   **Claridad y Testeabilidad:** El método `execute` se volverá corto y legible. Cada `PipelineStep` puede ser probado de forma aislada.

### Fase 3: Optimización del Rendimiento para LLMs Pequeños

*   **Propuesta:** Refinar la gestión de concurrencia y el truncamiento de contexto.

*   **Acciones a Realizar:**
    1.  **Introducir un Semáforo de Concurrencia:** Actualmente no existe limitador; hay que añadir un `asyncio.Semaphore` (o configurarlo vía settings) dentro de `MapReduceGenerationStep` para proteger al LLM cuando se ejecutan muchos lotes en paralelo.
    2.  **Truncamiento Inteligente en Direct RAG:** La estrategia `DirectGenerationStep` debe ser responsable de truncar la lista de `final_chunks` para asegurar que el contexto total no exceda `DIRECT_RAG_TOKEN_LIMIT`, en lugar de que esta lógica esté mezclada en `execute`.
    3.  **Reducir el "Ruido" en los Prompts:** Revisar los prompts, especialmente `map_prompt_template.txt`, para eliminar cualquier texto redundante. Cada token cuenta para un LLM pequeño. La meta es que el prompt contenga casi exclusivamente la pregunta y el contexto de los fragmentos.
    4.  **Métricas Operativas:** Emitir eventos estructurados (`TokenBudgetExceeded`, `MapReduceActivated`, etc.) y exponer métricas para vigilar latencia y tamaño de prompts en producción.

*   **Beneficios:**
    *   **Protección del LLM:** Se evita sobrecargar el servidor `llama.cpp`, previniendo timeouts y mejorando la estabilidad.
    *   **Rendimiento Predecible:** El tiempo de respuesta será más consistente, ya que el tamaño del contexto enviado al LLM está estrictamente controlado.

---

## 3. Consideraciones Adicionales

*   **Dependencias Declaradas:** Añadir explícitamente `structlog`, `haystack`, `tiktoken`, `fastapi` y `pydantic` en `pyproject.toml` para evitar sorpresas en despliegues limpios o pipelines CI.
*   **Caché de Tokens:** Documentar que `TokenAccountant` mantiene estado en memoria. Evaluar estrategias de rotación (TTL, `functools.lru_cache`, métricas de impacto) y aclarar su seguridad en casos con múltiples workers.
*   **Plantillas Inexistentes:** Forzar un `FileNotFoundError` en `PromptService` si una plantilla declarada falta, evitando fallbacks silenciosos. Eliminar `PromptType.GENERAL` o cubrirlo con pruebas para justificar su permanencia.
*   **Cobertura de Pruebas:** Planificar pruebas unitarias por etapa del pipeline y un smoke-test end-to-end que cubra Direct RAG y MapReduce después de la refactorización.

---

## 4. Checklist de Refactorización

-   [ ] **Fase 1: Simplificación de Configuración de MapReduce**
    -   [ ] Eliminar `MAPREDUCE_ACTIVATION_THRESHOLD_CHUNKS` de `config.py` y de la lógica.
    -   [ ] Eliminar `PROMPT_BASE_OVERHEAD_TOKENS`, `PROMPT_PER_CHUNK_OVERHEAD_TOKENS` y otros parámetros de "overhead" de `config.py`.
    -   [ ] Añadir el nuevo parámetro `DIRECT_RAG_TOKEN_LIMIT` a `config.py`.
    -   [ ] Actualizar el método `execute` para usar este único umbral en su lógica de decisión.
    -   [ ] Extraer dataclasses de configuración (`PromptBudgetConfig`, `MapReduceConfig`, `RetrievalConfig`) y pasarlas al caso de uso.

-   [ ] **Fase 2: Abstracción del Pipeline RAG**
    -   [ ] Crear la clase base abstracta `PipelineStep`.
    -   [ ] Crear las clases concretas para cada etapa: `RetrievalStep`, `FusionStep`, `ContentFetchStep`, `RerankStep`, `FilterStep`.
    -   [ ] Crear la interfaz `GenerationStep` y sus dos implementaciones: `DirectGenerationStep` y `MapReduceGenerationStep`.
    -   [ ] Mover la lógica de MapReduce (batching, semáforo, etc.) a `MapReduceGenerationStep`.
    -   [ ] Crear la clase `RAGPipeline` que orquesta la ejecución de los pasos.
    -   [ ] Refactorizar `AskQueryUseCase.execute` para que construya y ejecute el `RAGPipeline`.
    -   [ ] Introducir `ResponseAssembler` y `PromptAssemblyService` para mantener el método `execute` mínimo.

-   [ ] **Fase 3: Optimización y Limpieza**
    -   [ ] Asegurarse de que `MapReduceGenerationStep` utiliza correctamente `asyncio.Semaphore` para limitar las llamadas concurrentes.
    -   [ ] Implementar la lógica de truncamiento de chunks dentro de `DirectGenerationStep` para respetar `DIRECT_RAG_TOKEN_LIMIT`.
    -   [ ] Revisar y minimizar el texto estático en `map_prompt_template.txt` para reducir la carga de tokens.
    -   [ ] Mover la lógica de `_prompt_service.create_documents` a una etapa inicial del pipeline si se considera un paso de transformación.
    -   [ ] Eliminar `PromptType.GENERAL` o cubrirlo con pruebas si se mantiene.
    -   [ ] Documentar y testear la política de caché de `TokenAccountant`.

-   [ ] **Fase 4: Verificación Final**
    -   [ ] Revisar que la cohesión de las nuevas clases sea alta (hacen una sola cosa).
    -   [ ] Comprobar que el acoplamiento entre `AskQueryUseCase` y los detalles de implementación del pipeline se haya eliminado.
    -   [ ] Validar que todo el flujo sigue funcionando como se espera a través de pruebas de integración.
    -   [ ] Actualizar `pyproject.toml` e incluir un smoke-test automatizado que cubra Direct RAG y MapReduce.