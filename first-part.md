### **Plan de Refactorización: De Microservicios a Monolito Local (Fase 1)**

**Objetivo:** Consolidar la lógica del backend de Atenex en una única aplicación Python, eliminando dependencias de la nube y la complejidad de la arquitectura distribuida, y adaptar el pipeline de ingesta para que funcione de manera completamente local y offline.

**Principios Guía:**
*   **Simplicidad Primero:** Eliminar cualquier capa de abstracción que no sea estrictamente necesaria para una aplicación de escritorio (ej. colas de mensajes, gateways de API complejos).
*   **Reutilización Inteligente:** Reciclar la lógica de negocio y los modelos de datos bien definidos, desechando el código de infraestructura específico de la nube (clientes de GCS, Zilliz, etc.).
*   **Multi-tenant local (mantener y adaptar):** Conservar la lógica multi-tenant (`company_id`, `user_id`) pero adaptada al entorno offline. La aplicación soportará múltiples tenants locales (perfiles/empresas) aislando datos por tenant mediante columnas en SQLite, prefijos/carpetas en el filesystem y metadatos en el almacén vectorial.

---

### **Parte 1: Creación del Proyecto y Configuración de Dependencias**

Esta fase establece las bases del nuevo proyecto monolítico `atenex-offline`.

1.  **Iniciar el Proyecto con Poetry:**
    *   Crear un nuevo directorio para el proyecto y ejecutar `poetry init atenex-offline`.
    *   Esto generará un archivo `pyproject.toml` que centralizará toda la gestión de dependencias.

2.  **Añadir Dependencias Clave (Análisis y Fusión):**
    *   Se analizarán los `pyproject.toml` de todos los microservicios para consolidar las dependencias necesarias y añadir las nuevas para el entorno local.

    | Dependencia | Origen/Propósito | Acción |
    | :--- | :--- | :--- |
    | `fastapi`, `uvicorn` | **Reciclar** (de todos los servicios) | El núcleo del servidor web local. |
    | `pydantic`, `pydantic-settings` | **Reciclar** (de todos los servicios) | Se mantendrá para la configuración y validación de datos. |
    | `structlog` | **Reciclar** (de todos los servicios) | Se mantendrá como el sistema de logging estándar. |
    | `sqlalchemy` | **Nuevo** | Reemplazo de `asyncpg`. Será el ORM para interactuar con la nueva base de datos **SQLite**. |
    | `alembic` | **Nuevo** | Se añadirá para gestionar las migraciones del esquema de la base de datos SQLite de forma robusta. |
    | `python-jose`, `passlib` | **Eliminar** | La lógica de JWT y hashing de contraseñas complejas ya no es necesaria. |
    | `httpx`, `celery`, `redis` | **Eliminar** | Las llamadas entre servicios y la cola de tareas asíncrona se reemplazan por llamadas a funciones directas y `BackgroundTasks`. |
    | `pymilvus`, `google-cloud-storage`| **Eliminar** | Los clientes para servicios en la nube son reemplazados por sus alternativas locales. |
    | `chromadb-client` | **Nuevo** | Reemplazo de Milvus/Zilliz. **ChromaDB** es una excelente opción por ser ligera, basada en archivos y fácil de integrar en Python. |
    | `sentence-transformers` | **Reciclar** (de `reranker-service`) | Se usará tanto para el **reranking** como para la generación de **embeddings**, reemplazando la dependencia de OpenAI. |
    | `ctransformers[cuda]` | **Nuevo** | Reemplazo de la API de Gemini. Esta librería permite ejecutar modelos LLM locales (en formato GGUF) de manera eficiente en CPU y/o GPU. |
    | `PyMuPDF`, `python-docx`... | **Reciclar** (de `docproc-service`) | Se mantendrán todas las librerías de parseo de documentos. |
    | `tiktoken` | **Reciclar** (de `ingest-service`) | Se mantendrá para el conteo de tokens en los chunks. |
    | `rank_bm25` | **Nuevo** | Reemplazo de `bm2s` de `sparse-search-service`. `rank_bm25` es una librería más estándar y directa para BM25 en memoria, ideal para este caso. |

---

### **Parte 2: Fusión de Servicios y Estructura del Código Monolítico**

Se definirá una nueva estructura de directorios lógica y se migrará el código de los microservicios a su nuevo hogar.

**Nueva Estructura de Directorios Propuesta:**

```
atenex-offline/
├── atenex_offline/
│   ├── api/
│   │   ├── ingest_router.py   # Endpoints de /ingest/*
│   │   └── query_router.py    # Endpoints de /query/*
│   ├── core/
│   │   ├── config.py          # Configuración Pydantic unificada
│   │   └── logging_config.py  # Configuración de Structlog
│   ├── data/                  # Directorio de datos del usuario (se creará en runtime)
│   │   ├── atenex.db          # Base de datos SQLite
│   │   ├── files/             # Archivos de documentos originales
│   │   ├── vector_store/      # Datos de ChromaDB
│   │   └── models/            # Modelos de IA descargados
│   ├── domain/
│   │   └── models.py          # Modelos de datos Pydantic (Document, Chunk, etc.)
│   ├── infrastructure/
│   │   ├── database.py        # Conexión y sesión de SQLAlchemy para SQLite
│   │   ├── vector_store.py    # Adaptador para ChromaDB
│   │   └── models/
│   │       ├── embedding.py   # Clase contenedora del modelo SentenceTransformer
│   │       └── reranker.py    # Clase contenedora del modelo CrossEncoder
│   ├── processing/
│   │   ├── chunking.py        # Lógica de división de texto
│   │   └── extraction.py      # Lógica de extracción de texto de archivos
│   ├── services/
│   │   ├── ingest_service.py  # Lógica del pipeline de ingesta
│   │   └── query_service.py   # Lógica del pipeline de RAG
│   └── main.py                # Punto de entrada de la aplicación FastAPI
├── migrations/                # Directorio de Alembic para migraciones de DB
├── pyproject.toml
└── README.md
```

**Plan de Fusión Detallado (Archivo por Archivo):**

1.  **`docproc-service` -> `atenex_offline/processing/`**
    *   **Reciclar:** La lógica interna es perfecta. Mover el contenido de `infrastructure/extractors/` a `processing/extraction.py` y `infrastructure/chunkers/` a `processing/chunking.py`. La arquitectura de puertos y adaptadores se simplifica a funciones directas o clases de servicio.
    *   **Eliminar:** Toda la capa de API (`main.py`, `api/`, `dependencies.py`). El `ProcessDocumentUseCase` se convierte en una función o clase dentro de `ingest_service.py` que llama directamente a las funciones de `extraction` y `chunking`.

2.  **`embedding-service` y `reranker-service` -> `atenex_offline/infrastructure/models/`**
    *   **Reciclar:** La lógica central de los adaptadores (`SentenceTransformerRerankerAdapter`, `SentenceTransformerAdapter` de `embedding-service`).
    *   **Adaptar:** Se crearán dos nuevas clases: `EmbeddingModel` en `embedding.py` y `RerankerModel` en `reranker.py`. Estas clases cargarán los modelos de `sentence-transformers` en memoria una sola vez durante el inicio de la aplicación (en el `lifespan` del `main.py`). Se eliminará la capa de API y la dependencia de OpenAI del `embedding-service`.
    *   **Eliminar:** Toda la capa web (FastAPI) de ambos servicios.

3.  **`api-gateway` -> `atenex_offline/main.py` y `core/`**
    *   **Reciclar:** La estructura del `main.py` (lifespan, middlewares, configuración de logging) es una buena base para el nuevo `main.py` del monolito.
    *   **Eliminar:** La lógica de proxy `httpx` es completamente innecesaria. La autenticación (`auth/`) se eliminará por completo en esta fase, ya que no habrá gestión de usuarios. La configuración (`core/config.py`) se fusionará en el nuevo `atenex_offline/core/config.py`.

4.  **`ingest-service` y `query-service` -> `api/`, `services/`, `domain/`, `infrastructure/`**
    *   **`endpoints/ingest.py` y `endpoints/query.py`**: Se mueven a `api/ingest_router.py` y `api/query_router.py`. Las dependencias se actualizarán para apuntar a servicios locales en lugar de clientes HTTP. Se eliminan las cabeceras `X-Company-ID` y `X-User-ID`.
    *   **`services/ingest_pipeline.py` y `use_cases/ask_query_use_case.py`**: La lógica de negocio principal se traslada a `services/ingest_service.py` y `services/query_service.py`. Aquí es donde ocurrirán los cambios más significativos.
    *   **`db/postgres_client.py` e `infrastructure/persistence/`**: La lógica de acceso a datos se reescribirá en `infrastructure/database.py` usando SQLAlchemy para SQLite, adaptando y preservando las columnas y lógica `company_id` para soportar multi-tenant local (posible simplificación, índices y constraints adecuados a SQLite).
    *   **`models/domain.py`**: Se fusionarán los modelos Pydantic de todos los servicios en un único `domain/models.py`, eliminando duplicados y simplificando.

---

### **Parte 3: Adaptación Detallada del Pipeline de Ingesta**

Esta es la aplicación práctica de la fusión de servicios en un flujo concreto.

1.  **Reescritura del Endpoint `POST /api/v1/ingest/upload` en `ingest_router.py`:**
    *   **Firma:** Se simplificará drásticamente. Eliminará las dependencias de cabeceras (`X-Company-ID`, `X-User-ID`).
    *   **Lógica:** En lugar de encolar una tarea en Celery, inyectará `fastapi.BackgroundTasks` y añadirá la función principal del pipeline de ingesta como una tarea en segundo plano. Esto proporciona una respuesta inmediata a la UI mientras el procesamiento ocurre localmente.

2.  **Refactorización de `process_document_standalone` (lógica del worker) en `services/ingest_service.py`:**
    *   **Entrada:** Recibirá los bytes del archivo, el nombre y el tipo de contenido directamente.
    *   **Paso 1: Almacenamiento Local (Reemplaza GCSClient):**
        *   Creará un directorio de datos `data/files/` si no existe.
        *   Generará un `document_id` (UUID).
        *   Guardará el archivo en `data/files/{document_id}/{filename}` usando `pathlib`.
    *   **Paso 2: Persistencia Inicial (Reemplaza `postgres_client` con SQLAlchemy):**
        *   Creará una sesión de SQLAlchemy para `atenex.db`.
    *   Insertará un nuevo registro en la tabla `documents` con estado `pending`, guardando la ruta del archivo local. Se conservará la columna `company_id` para permitir identificar el tenant asociado al documento en modo offline.
    *   **Paso 3: Extracción y Chunking (Reemplaza llamada a `docproc-service`):**
        *   Importará y llamará directamente a las funciones de `atenex_offline/processing/extraction.py` y `chunking.py`. `result = extract_text(file_bytes)` -> `chunks = chunk_text(result)`.
    *   **Paso 4: Generación de Embeddings (Reemplaza llamada a `embedding-service`):**
        *   Usará la instancia global del modelo `EmbeddingModel` (cargado en el `lifespan`) para generar los vectores: `vectors = embedding_model.embed(chunks)`.
    *   **Paso 5: Indexación Vectorial (Reemplaza `pymilvus`):**
        *   Usará el cliente de **ChromaDB**.
        *   Creará una colección si no existe.
        *   Añadirá los chunks, sus vectores y metadatos a la colección: `collection.add(ids=[...], embeddings=[...], documents=[...], metadatas=[...])`.
    *   **Paso 6: Persistencia Final (Reemplaza `postgres_client` y SQLAlchemy síncrono):**
        *   Usará la misma sesión de SQLAlchemy para insertar los detalles de los chunks en la tabla `document_chunks` y actualizar el estado del documento principal en la tabla `documents` a `processed`.

---

### **Checklist de Refactorización (Fase 1)**

- [ ] **Proyecto:** Crear un nuevo proyecto `atenex-offline` con Poetry.
- [ ] **Dependencias:** Instalar el nuevo stack (FastAPI, SQLAlchemy, ChromaDB, Sentence-Transformers, CTransformers, etc.).
- [ ] **Estructura:** Crear la nueva estructura de directorios (`api`, `core`, `services`, `infrastructure`, etc.).
- [ ] **Eliminación:** Borrar todo el código relacionado con JWT, `passlib`, `httpx` para proxying, `celery`, y `redis`.
- [ ] **Fusión (DocProc):** Mover la lógica de extracción/chunking de `docproc-service` a `atenex_offline/processing/`.
- [ ] **Fusión (Modelos):** Mover los adaptadores de `embedding-service` y `reranker-service` a clases de modelo locales en `atenex_offline/infrastructure/models/` y cargarlos en el `lifespan`.
- [ ] **Fusión (Endpoints):** Mover los routers de `ingest-service` y `query-service` a `atenex_offline/api/`.
- [ ] **Fusión (Lógica de Negocio):** Trasladar la lógica de los pipelines a `atenex_offline/services/`.
- [ ] **Base de Datos:**
    - [ ] Implementar la conexión a SQLite con SQLAlchemy en `infrastructure/database.py`.
    - [ ] Crear los modelos de SQLAlchemy para las tablas adaptadas (con soporte para `companies` y `company_id`); ajustar constraints e índices para SQLite.
    - [ ] Configurar Alembic para las migraciones de SQLite.
- [ ] **Pipeline de Ingesta:**
    - [ ] Reescribir el endpoint `upload` para usar `BackgroundTasks`.
    - [ ] Reemplazar cliente GCS con `pathlib` para guardado local.
    - [ ] Reemplazar llamadas HTTP a `docproc` y `embedding` con llamadas a funciones/clases directas.
    - [ ] Reemplazar el cliente de Milvus con el cliente de ChromaDB para la indexación de vectores.
    - [ ] Adaptar todas las operaciones de base de datos para usar SQLAlchemy con SQLite.
- [ ] **Limpieza de Código (conservar multi-tenant):** Eliminar referencias innecesarias a infra nube (GCS, Milvus remotos, Celery), pero conservar y adaptar las referencias a `company_id` y `user_id` en los modelos de datos, queries y lógica de negocio para soportar multi-tenant local.