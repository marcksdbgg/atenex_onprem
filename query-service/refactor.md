Te resumo primero qué hay que tocar y luego vamos archivo por archivo con trozos concretos de código:

1. **ConfigMap**: dejar de hablar de Gemini y meter la URL del `llama-server` + nombre del modelo Qwen.
2. **config.py**: hacer que **Gemini deje de ser obligatorio** y añadir settings nuevos para el LLM local (llama.cpp + Qwen).
3. **Nuevo adapter LLM**: `LlamaCppAdapter` que llama vía HTTP a `llama-server` y cumple la interfaz de `LLMPort`.
4. **main.py**: donde ahora construyas `GeminiAdapter`, cambiarlo por `LlamaCppAdapter`.
5. **Limpiar imports y mensajes** que mencionan Gemini (sobre todo en `query.py` y `ask_query_use_case.py`).

Voy en orden.

---

## 1. `query-service/configmap.yaml`

Sustituye la parte de Gemini por configuración del LLM local.

### ANTES

```yaml
  # Gemini LLM Settings
  QUERY_GEMINI_MODEL_NAME: "gemini-2.0-flash"
  # QUERY_GEMINI_API_KEY -> Proveniente de Secret
```

### DESPUÉS (ejemplo para un servicio `qwen-llama-server` en el namespace `atenex`)

```yaml
  # LLM (llama.cpp + Qwen) Settings
  QUERY_LLM_API_BASE_URL: "http://qwen-llama-server.atenex.svc.cluster.local:8080"
  QUERY_LLM_MODEL_NAME: "qwen2.5-1.5b-instruct-q4_k_m.gguf"
  QUERY_LLM_MAX_OUTPUT_TOKENS: "4096"

  # (Opcional) si quieres dejar Gemini muerto pero sin romper nada:
  # QUERY_GEMINI_MODEL_NAME: "gemini-2.0-flash"
```

> Para desarrollo local con Docker, podrías usar por ejemplo:
>
> ```yaml
> QUERY_LLM_API_BASE_URL: "http://host.docker.internal:8080"
> ```

---

## 2. `app/core/config.py`

### 2.1. Defaults nuevos para el LLM local

Cerca de donde tienes los defaults de Gemini, añade:

```py
# LLM local (llama.cpp + Qwen)
LLM_API_BASE_URL_DEFAULT = "http://qwen-llama-server.atenex.svc.cluster.local:8080"
LLM_MODEL_NAME_DEFAULT = "qwen2.5-1.5b-instruct-q4_k_m.gguf"
LLM_MAX_OUTPUT_TOKENS_DEFAULT: Optional[int] = 4096
```

### 2.2. Hacer que GEMINI_API_KEY deje de ser obligatorio

En la clase `Settings`, cambia la definición de `GEMINI_API_KEY`:

#### ANTES

```py
    # --- LLM (Google Gemini) ---
    GEMINI_API_KEY: SecretStr
    GEMINI_MODEL_NAME: str = Field(default=DEFAULT_GEMINI_MODEL)
    GEMINI_MAX_OUTPUT_TOKENS: Optional[int] = Field(default=DEFAULT_GEMINI_MAX_OUTPUT_TOKENS, description="Optional: Maximum number of tokens to generate in the LLM response.")
```

#### DESPUÉS

```py
    # --- LLM (Google Gemini - opcional / legacy) ---
    GEMINI_API_KEY: Optional[SecretStr] = Field(
        default=None,
        description="Gemini API key (optional if using local LLM)."
    )
    GEMINI_MODEL_NAME: str = Field(default=DEFAULT_GEMINI_MODEL)
    GEMINI_MAX_OUTPUT_TOKENS: Optional[int] = Field(
        default=DEFAULT_GEMINI_MAX_OUTPUT_TOKENS,
        description="Optional: Maximum number of tokens to generate in the Gemini response."
    )
```

Y en el validador donde se exige que secretos no estén vacíos, **saca `GEMINI_API_KEY`**:

#### ANTES

```py
    @field_validator('POSTGRES_PASSWORD', 'GEMINI_API_KEY', mode='before')
    @classmethod
    def check_secret_value_present(cls, v: Any, info: ValidationInfo) -> Any:
        ...
```

#### DESPUÉS

```py
    @field_validator('POSTGRES_PASSWORD', mode='before')
    @classmethod
    def check_secret_value_present(cls, v: Any, info: ValidationInfo) -> Any:
        ...
```

De esta forma, el servicio ya **no te exige** una API key de Gemini para arrancar.

### 2.3. Añadir campos nuevos para llama.cpp + Qwen

En `Settings`, después del bloque de Gemini (o en lugar de él si quieres matarlo del todo), añade:

```py
    # --- LLM local (llama.cpp + Qwen) ---
    LLM_API_BASE_URL: AnyHttpUrl = Field(
        default=LLM_API_BASE_URL_DEFAULT,
        description="Base URL for local llama.cpp server (e.g. http://qwen-llama-server.atenex.svc.cluster.local:8080)."
    )
    LLM_MODEL_NAME: str = Field(
        default=LLM_MODEL_NAME_DEFAULT,
        description="Identifier of the model as seen by llama.cpp (usually the GGUF filename)."
    )
    LLM_MAX_OUTPUT_TOKENS: Optional[int] = Field(
        default=LLM_MAX_OUTPUT_TOKENS_DEFAULT,
        description="Maximum number of tokens to generate in the LLM response."
    )
```

Y un validador simple para `LLM_MAX_OUTPUT_TOKENS`:

```py
    @field_validator('LLM_MAX_OUTPUT_TOKENS')
    @classmethod
    def check_llm_max_output_tokens(cls, v: Optional[int]) -> Optional[int]:
        if v is not None and v <= 0:
            raise ValueError("LLM_MAX_OUTPUT_TOKENS, if set, must be a positive integer.")
        return v
```

Con esto, ya tienes settings limpios para el LLM local.

---

## 3. Nuevo adapter: `app/infrastructure/llms/llama_cpp_adapter.py`

Crea este archivo (nuevo) para hablar con `llama-server`:

```py
# query-service/app/infrastructure/llms/llama_cpp_adapter.py
import json
from typing import Optional, Type, Any, Dict

import httpx
import structlog
from pydantic import BaseModel
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type, before_sleep_log

from app.application.ports.llm_port import LLMPort
from app.core.config import settings
from app.utils.helpers import truncate_text

log = structlog.get_logger(__name__)


class LlamaCppAdapter(LLMPort):
    """
    Adaptador LLM que llama al servidor llama.cpp vía API OpenAI-compatible
    (POST /v1/chat/completions).
    """

    def __init__(
        self,
        base_url: str,
        model_name: str,
        timeout: int = settings.HTTP_CLIENT_TIMEOUT,
        max_output_tokens: Optional[int] = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.model_name = model_name
        self._max_output_tokens = max_output_tokens
        self._client = httpx.AsyncClient(timeout=timeout)

        self.chat_completions_endpoint = f"{self.base_url}/v1/chat/completions"

        log.info(
            "LlamaCppAdapter initialized",
            base_url=self.base_url,
            chat_endpoint=self.chat_completions_endpoint,
            model_name=self.model_name,
            max_output_tokens=self._max_output_tokens,
        )

    async def close(self) -> None:
        await self._client.aclose()
        log.info("LlamaCppAdapter HTTP client closed")

    @retry(
        stop=stop_after_attempt(settings.HTTP_CLIENT_MAX_RETRIES + 1),
        wait=wait_exponential(
            multiplier=settings.HTTP_CLIENT_BACKOFF_FACTOR, min=1, max=10
        ),
        retry=retry_if_exception_type(
            (httpx.TimeoutException, httpx.NetworkError, httpx.ConnectError)
        ),
        reraise=True,
        before_sleep=before_sleep_log(log, None),
    )
    async def generate(
        self,
        prompt: str,
        response_pydantic_schema: Optional[Type[BaseModel]] = None,
    ) -> str:
        """
        Implementa la interfaz LLMPort.generate usando /v1/chat/completions.
        El parámetro response_pydantic_schema se ignora (nos fiamos del prompt
        para que el modelo devuelva JSON bien formado).
        """
        if not prompt or not prompt.strip():
            raise ValueError("Prompt cannot be empty.")

        gen_log = log.bind(
            adapter="LlamaCppAdapter",
            model_name=self.model_name,
            prompt_length=len(prompt),
            expecting_json=bool(response_pydantic_schema),
        )

        # En llama.cpp, el campo "model" suele ignorarse, pero lo enviamos por compatibilidad
        payload: Dict[str, Any] = {
            "model": self.model_name,
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            "temperature": 0.6,
            "top_p": 0.9,
        }

        if self._max_output_tokens:
            # nombre de parámetro OpenAI-style; llama.cpp lo soporta en modo compat
            payload["max_tokens"] = self._max_output_tokens

        prompt_preview = (
            truncate_text(prompt, 500) if len(prompt) > 500 else prompt
        )
        gen_log.debug(
            "Sending request to llama.cpp",
            endpoint=self.chat_completions_endpoint,
            prompt_preview=prompt_preview,
        )

        try:
            resp = await self._client.post(
                self.chat_completions_endpoint, json=payload
            )
            resp.raise_for_status()
        except httpx.HTTPStatusError as e:
            gen_log.error(
                "HTTP error from llama.cpp",
                status_code=e.response.status_code,
                response_text=truncate_text(e.response.text, 300),
            )
            raise ConnectionError(
                f"LLM service (llama.cpp) returned HTTP {e.response.status_code}: {e.response.text}"
            ) from e
        except httpx.RequestError as e:
            gen_log.error("Request error contacting llama.cpp", error=str(e))
            raise ConnectionError(
                f"Could not connect to LLM service (llama.cpp server): {e}"
            ) from e

        try:
            data = resp.json()
        except json.JSONDecodeError as e:
            gen_log.error(
                "Invalid JSON returned by llama.cpp",
                raw_response=truncate_text(resp.text, 300),
            )
            raise ValueError(
                f"Invalid JSON response from LLM service: {e}"
            ) from e

        # OpenAI-compatible shape:
        # {"choices": [{"message": {"role": "assistant", "content": "..."}, ...}], ...}
        if not isinstance(data, dict) or "choices" not in data or not data["choices"]:
            gen_log.error("Unexpected response format from llama.cpp", data=data)
            raise ValueError("Unexpected response format from LLM service.")

        choice0 = data["choices"][0]

        # Algunos builds devuelven "message", otros "text"
        content: Optional[str] = None
        if isinstance(choice0, dict):
            if "message" in choice0 and isinstance(choice0["message"], dict):
                content = choice0["message"].get("content")
            elif "text" in choice0:
                content = choice0.get("text")

        if not content:
            gen_log.error("No content found in llama.cpp response", data=data)
            raise ValueError("LLM service response does not contain content.")

        gen_log.info(
            "LLM response received from llama.cpp",
            content_preview=truncate_text(content, 200),
        )
        return content
```

Y actualiza `app/infrastructure/llms/__init__.py` para exportarlo:

```py
# query-service/app/infrastructure/llms/__init__.py
from .llama_cpp_adapter import LlamaCppAdapter

__all__ = ["LlamaCppAdapter"]
```

---

## 4. `app/api/v1/endpoints/query.py` – quitar referencia directa a Gemini

Tu endpoint de `/ask` ya usa el `AskQueryUseCase` vía dependencia, así que aquí **no debe importarse el adapter de Gemini**.

### ANTES (parte de los imports)

```py
from app.infrastructure.vectorstores.milvus_adapter import MilvusAdapter
from app.infrastructure.llms.gemini_adapter import GeminiAdapter
```

### DESPUÉS

Si no usas esos símbolos en el archivo (y por el código que muestras, no los usas), simplemente elimínalos:

```py
# from app.infrastructure.vectorstores.milvus_adapter import MilvusAdapter
# from app.infrastructure.llms.gemini_adapter import GeminiAdapter
```

O bórralos del todo. Lo importante: **no se debe importar `gemini_adapter`** para que no intente cargar `google.generativeai`.

---

## 5. `app/application/use_cases/ask_query_use_case.py`

Aquí hay dos cosas:

1. Logging que menciona `gemini_model_name`.
2. Mensajes de error que hablan explícitamente de “Gemini API”.

### 5.1. Log de inicialización

En el `__init__`, en `log_params` tienes esto:

```py
            "gemini_model_name": settings.GEMINI_MODEL_NAME,
```

Cámbialo a algo genérico usando tus nuevos settings:

```py
            "llm_model_name": settings.LLM_MODEL_NAME,
```

### 5.2. Mapeo de errores de conexión

En el `except ConnectionError as ce:` del método `execute`, ahora mismo:

```py
        except ConnectionError as ce: 
            exec_log.error("Connection error during use case execution", error=str(ce), exc_info=False)
            detail_message = "A required external service is unavailable. Please try again later."
            if "Embedding service" in str(ce): detail_message = "The embedding service is currently unavailable."
            elif "Reranker service" in str(ce): detail_message = "The reranking service is currently unavailable."
            elif "Sparse search service" in str(ce): detail_message = "The sparse search service is currently unavailable."
            elif "Gemini API" in str(ce): detail_message = "The language model service (Gemini) is currently unavailable."
            elif "Vector DB" in str(ce): detail_message = "The vector database service is currently unavailable."
```

Cámbialo por algo compatible con el nuevo adapter, que lanza errores con texto “LLM service (llama.cpp)”:

```py
        except ConnectionError as ce: 
            exec_log.error("Connection error during use case execution", error=str(ce), exc_info=False)
            detail_message = "A required external service is unavailable. Please try again later."
            if "Embedding service" in str(ce):
                detail_message = "The embedding service is currently unavailable."
            elif "Reranker service" in str(ce):
                detail_message = "The reranking service is currently unavailable."
            elif "Sparse search service" in str(ce):
                detail_message = "The sparse search service is currently unavailable."
            elif "LLM service" in str(ce) or "llama.cpp" in str(ce):
                detail_message = "The language model service is currently unavailable."
            elif "Vector DB" in str(ce):
                detail_message = "The vector database service is currently unavailable."
```

Con eso, los errores del adapter nuevo se traducen a mensajes coherentes para el cliente.

---

## 6. `app/main.py` – cambiar de Gemini a LlamaCppAdapter

No has pegado `main.py`, pero casi seguro tienes algo así:

```py
from app.infrastructure.llms.gemini_adapter import GeminiAdapter
from app.dependencies import set_ask_query_use_case_instance
from app.core.config import settings
...
llm_adapter = GeminiAdapter()
use_case = AskQueryUseCase(
    chat_repo=chat_repo,
    log_repo=log_repo,
    vector_store=vector_store,
    llm=llm_adapter,
    embedding_adapter=embedding_adapter,
    http_client=http_client,
    sparse_retriever=sparse_retriever,
    chunk_content_repo=chunk_content_repo,
    diversity_filter=diversity_filter,
)
set_ask_query_use_case_instance(use_case, ready_flag=True)
```

Cámbialo por:

```py
from app.infrastructure.llms.llama_cpp_adapter import LlamaCppAdapter
from app.dependencies import set_ask_query_use_case_instance
from app.core.config import settings
...

llm_adapter = LlamaCppAdapter(
    base_url=str(settings.LLM_API_BASE_URL),
    model_name=settings.LLM_MODEL_NAME,
    timeout=settings.HTTP_CLIENT_TIMEOUT,
    max_output_tokens=settings.LLM_MAX_OUTPUT_TOKENS,
)

ask_query_use_case = AskQueryUseCase(
    chat_repo=chat_repo,
    log_repo=log_repo,
    vector_store=vector_store,
    llm=llm_adapter,
    embedding_adapter=embedding_adapter,
    http_client=http_client,
    sparse_retriever=sparse_retriever,
    chunk_content_repo=chunk_content_repo,
    diversity_filter=diversity_filter,
)

set_ask_query_use_case_instance(ask_query_use_case, ready_flag=True)
```

Y, si tienes eventos de shutdown (FastAPI `@app.on_event("shutdown")`), añade el cierre del adapter:

```py
@app.on_event("shutdown")
async def shutdown_event():
    await embedding_client.close()
    await sparse_client.close()
    await llm_adapter.close()
```

(Ajusta nombres de variables a como lo tengas definido realmente.)

---

## 7. Limpiezas menores / cosas a revisar

* **`logging_config.py`**: tienes una línea que ajusta el logger de `google.generativeai`. Puedes dejarla (no rompe nada aunque ya no uses Gemini), o si quieres limpiar, simplemente quítala:

  ```py
  # logging.getLogger("google.generativeai").setLevel(logging.INFO)
  ```

* **Secrets de Gemini en Kubernetes**: si ya no vas a usar Gemini, puedes:

  * Eliminar el Secret correspondiente, o
  * Dejarlo un tiempo por si quieres volver, ya no es obligatorio gracias a los cambios en `config.py`.

* **Prompts**: aunque los nombres de los archivos llevan “gemini” (`rag_template_gemini_v2.txt`, etc.), no pasa nada. Qwen va a leer el prompt igual. Si quieres afinar el modelo local, más adelante puedes retocar esos prompts para Qwen, pero no es imprescindible para que funcione.

---

## Resumen rápido de qué tienes que tocar

1. **ConfigMap**

   * Añadir `QUERY_LLM_API_BASE_URL`, `QUERY_LLM_MODEL_NAME`, `QUERY_LLM_MAX_OUTPUT_TOKENS`.
   * Dejar/ignorar `QUERY_GEMINI_*`.

2. **`core/config.py`**

   * Hacer `GEMINI_API_KEY` opcional y quitarlo del validador de secretos.
   * Añadir `LLM_API_BASE_URL`, `LLM_MODEL_NAME`, `LLM_MAX_OUTPUT_TOKENS` + validador.

3. **Nuevo archivo** `app/infrastructure/llms/llama_cpp_adapter.py`

   * Adapter HTTP contra `llama-server` en `/v1/chat/completions`.

4. **`app/infrastructure/llms/__init__.py`**

   * Exportar `LlamaCppAdapter`.

5. **`app/api/v1/endpoints/query.py`**

   * Eliminar import de `GeminiAdapter` (y MilvusAdapter si no se usa).

6. **`ask_query_use_case.py`**

   * Log inicial: `gemini_model_name` → `llm_model_name`.
   * Mapeo de errores: “Gemini API” → algo genérico que encaje con los errores del nuevo adapter.

7. **`main.py`**

   * Importar y usar `LlamaCppAdapter` en lugar de `GeminiAdapter`.
   * (Opcional) cerrar el cliente HTTP del adapter en shutdown.

Con esos cambios, el `query-service` dejará de depender de la API de Gemini y empezará a consumir tu modelo Qwen cargado en `llama-server`.
