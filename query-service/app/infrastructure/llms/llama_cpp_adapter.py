# query-service/app/infrastructure/llms/llama_cpp_adapter.py
import json
import logging
from typing import Optional, Type, Any, Dict

import httpx
import structlog
from pydantic import BaseModel, ValidationError
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log,
)

from app.application.ports.llm_port import LLMPort
from app.core.config import settings
from app.utils.helpers import truncate_text

log = structlog.get_logger(__name__)


class LlamaCppAdapter(LLMPort):
    """Adapter for llama.cpp HTTP server exposing an OpenAI-compatible API."""

    def __init__(
        self,
        base_url: str,
        model_name: str,
        timeout: int = settings.HTTP_CLIENT_TIMEOUT,
        max_output_tokens: Optional[int] = None,
    ) -> None:
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

    async def health_check(self) -> bool:
        health_endpoint = f"{self.base_url}/health"
        for url in (health_endpoint, self.base_url):
            try:
                response = await self._client.get(url, timeout=5.0)
                if response.status_code >= 500:
                    log.warning(
                        "llama.cpp health check returned server error",
                        status_code=response.status_code,
                        url=url,
                    )
                    continue
                if response.status_code == 404 and url == health_endpoint:
                    # Algunos builds no exponen /health, considera 404 como éxito cuando el host es alcanzable.
                    log.debug("llama.cpp health endpoint missing, falling back to base URL check", url=url)
                return True
            except httpx.RequestError as exc:
                log.warning("llama.cpp health check failed", error=str(exc), url=url)
        return False

    @retry(
        stop=stop_after_attempt(max(1, (settings.HTTP_CLIENT_MAX_RETRIES or 0) + 1)),
        wait=wait_exponential(
            multiplier=max(settings.HTTP_CLIENT_BACKOFF_FACTOR, 0.1),
            min=1,
            max=10,
        ),
        retry=retry_if_exception_type(
            (
                httpx.TimeoutException,
                httpx.NetworkError,
                httpx.ConnectError,
            )
        ),
        reraise=True,
        before_sleep=before_sleep_log(logging.getLogger(__name__), logging.WARNING),
    )
    async def generate(
        self,
        prompt: str,
        response_pydantic_schema: Optional[Type[BaseModel]] = None,
    ) -> str:
        if not prompt or not prompt.strip():
            raise ValueError("Prompt cannot be empty.")

        expecting_json = bool(response_pydantic_schema)
        gen_log = log.bind(
            adapter="LlamaCppAdapter",
            model_name=self.model_name,
            prompt_length=len(prompt),
            expecting_json=expecting_json,
        )

        schema_json: Optional[str] = None
        if response_pydantic_schema:
            try:
                schema_json = json.dumps(
                    response_pydantic_schema.model_json_schema(),
                    ensure_ascii=False,
                )
            except TypeError as schema_err:
                gen_log.warning(
                    "Failed to serialize response schema for llama.cpp guidance",
                    error=str(schema_err),
                )

        messages = []
        if expecting_json:
            if schema_json:
                system_prompt = (
                    "Eres Atenex, un asistente empresarial. Devuelve exclusivamente JSON válido que cumpla "
                    "exactamente con el siguiente esquema Pydantic. No incluyas código, comentarios ni texto adicional.\n"
                    f"Esquema: {schema_json}"
                )
            else:
                system_prompt = (
                    "Eres Atenex, un asistente empresarial. Responde única y exclusivamente con JSON válido. "
                    "No incluyas texto fuera del JSON ni bloques de código."
                )
            messages.append({"role": "system", "content": system_prompt})

        messages.append({"role": "user", "content": prompt})

        payload: Dict[str, Any] = {
            "model": self.model_name,
            "messages": messages,
            "temperature": 0.6,
            "top_p": 0.9,
        }

        if expecting_json:
            payload["response_format"] = {"type": "json_object"}

        if self._max_output_tokens:
            payload["max_tokens"] = self._max_output_tokens

        prompt_preview = truncate_text(prompt, 500)
        gen_log.debug(
            "Sending request to llama.cpp",
            endpoint=self.chat_completions_endpoint,
            prompt_preview=prompt_preview,
        )

        response: Optional[httpx.Response] = None
        attempt_payload = dict(payload)

        for attempt in range(2 if expecting_json else 1):
            try:
                response = await self._client.post(
                    self.chat_completions_endpoint,
                    json=attempt_payload,
                )
                response.raise_for_status()
                break
            except httpx.HTTPStatusError as exc:
                if (
                    expecting_json
                    and attempt == 0
                    and exc.response.status_code == 400
                    and "response_format" in attempt_payload
                ):
                    gen_log.warning(
                        "llama.cpp rejected response_format hint; retrying without it",
                        response_text=truncate_text(exc.response.text, 200),
                    )
                    attempt_payload.pop("response_format", None)
                    continue

                gen_log.error(
                    "HTTP error from llama.cpp",
                    status_code=exc.response.status_code,
                    response_text=truncate_text(exc.response.text, 300),
                )
                raise ConnectionError(
                    f"LLM service (llama.cpp) returned HTTP {exc.response.status_code}: {exc.response.text}"
                ) from exc
            except httpx.RequestError as exc:
                gen_log.error("Request error contacting llama.cpp", error=str(exc))
                raise ConnectionError(
                    f"Could not connect to LLM service (llama.cpp server): {exc}"
                ) from exc

        if response is None:
            raise ConnectionError("LLM service (llama.cpp) did not return a response.")

        try:
            data = response.json()
        except json.JSONDecodeError as exc:
            gen_log.error(
                "Invalid JSON returned by llama.cpp",
                raw_response=truncate_text(response.text, 300),
            )
            raise ValueError("Invalid JSON response from LLM service.") from exc

        if not isinstance(data, dict) or not data.get("choices"):
            gen_log.error("Unexpected response format from llama.cpp", data=data)
            raise ValueError("Unexpected response format from LLM service.")

        first_choice = data["choices"][0]
        content: Optional[str] = None

        if isinstance(first_choice, dict):
            message = first_choice.get("message")
            if isinstance(message, dict):
                content = message.get("content")
            if content is None:
                content = first_choice.get("text")

        if not content:
            gen_log.error("No content found in llama.cpp response", data=data)
            raise ValueError("LLM service response does not contain content.")

        if expecting_json:
            normalized = self._normalize_json_output(content)
            if normalized:
                content = normalized
            else:
                gen_log.warning(
                    "Unable to normalize llama.cpp JSON response; returning raw content",
                    content_preview=truncate_text(content, 200),
                )

            if response_pydantic_schema:
                try:
                    response_pydantic_schema.model_validate_json(content)
                except (ValidationError, json.JSONDecodeError) as validation_err:
                    gen_log.warning(
                        "llama.cpp response failed schema validation",
                        error=str(validation_err),
                    )

        gen_log.info(
            "LLM response received from llama.cpp",
            content_preview=truncate_text(content, 200),
        )
        return content

    @staticmethod
    def _normalize_json_output(raw_text: str) -> Optional[str]:
        if not raw_text:
            return None

        text = raw_text.strip()

        if text.startswith("```"):
            text = text[3:]
            if text.startswith("json"):
                text = text[4:]
            if text.startswith("\n"):
                text = text[1:]
            closing_fence = text.rfind("```")
            if closing_fence != -1:
                text = text[:closing_fence]
            text = text.strip()

        try:
            json.loads(text)
            return text
        except json.JSONDecodeError:
            pass

        for open_char, close_char in (("{", "}"), ("[", "]")):
            start = text.find(open_char)
            end = text.rfind(close_char)
            if start != -1 and end != -1 and end > start:
                candidate = text[start : end + 1].strip()
                try:
                    json.loads(candidate)
                    return candidate
                except json.JSONDecodeError:
                    continue

        return None