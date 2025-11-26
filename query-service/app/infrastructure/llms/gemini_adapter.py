# query-service/app/infrastructure/llms/gemini_adapter.py
import asyncio
import json
import structlog
from typing import Optional, Type, Any, Dict
from pydantic import BaseModel
from google import genai
from google.genai import types
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from app.application.ports.llm_port import LLMPort
from app.core.config import settings
from app.utils.helpers import truncate_text

log = structlog.get_logger(__name__)

class GeminiAdapter(LLMPort):
    """
    Adaptador para Google Gemini utilizando el SDK `google-genai`.
    Soporta generación de texto y respuestas estructuradas (JSON).
    """

    def __init__(
        self,
        api_key: str,
        model_name: str = "gemini-2.5-flash",
        max_output_tokens: int = 8192,
        temperature: float = 0.5,
    ) -> None:
        self.model_name = model_name
        self._max_output_tokens = max_output_tokens
        self._temperature = temperature
        
        try:
            self._client = genai.Client(api_key=api_key)
            log.info("GeminiAdapter initialized", model_name=self.model_name)
        except Exception as e:
            log.critical("Failed to initialize Gemini Client", error=str(e))
            raise e

    async def close(self) -> None:
        log.info("GeminiAdapter closed (stateless client).")

    async def health_check(self) -> bool:
        try:
            # Simple prompt to check connectivity
            loop = asyncio.get_running_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self._client.models.generate_content(
                    model=self.model_name,
                    contents="Ping"
                )
            )
            return response is not None
        except Exception as e:
            log.error("Gemini health check failed", error=str(e))
            return False

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1.5, min=1, max=10),
        retry=retry_if_exception_type(Exception), 
        reraise=True
    )
    async def generate(
        self,
        prompt: str,
        response_pydantic_schema: Optional[Type[BaseModel]] = None,
    ) -> str:
        if not prompt or not prompt.strip():
            raise ValueError("Prompt cannot be empty.")

        gen_log = log.bind(
            adapter="GeminiAdapter",
            model_name=self.model_name,
            prompt_length=len(prompt),
            expecting_json=bool(response_pydantic_schema)
        )

        config_params = {
            "max_output_tokens": self._max_output_tokens,
            "temperature": self._temperature,
        }

        # Configuración para respuesta JSON estructurada
        if response_pydantic_schema:
            config_params["response_mime_type"] = "application/json"
            config_params["response_schema"] = response_pydantic_schema
        
        generate_config = types.GenerateContentConfig(**config_params)

        try:
            gen_log.debug("Sending request to Gemini", prompt_preview=truncate_text(prompt, 200))
            
            # Ejecutar de forma asíncrona en un executor para no bloquear el loop de eventos
            loop = asyncio.get_running_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self._client.models.generate_content(
                    model=self.model_name,
                    contents=prompt,
                    config=generate_config
                )
            )

            if not response.text:
                gen_log.error("Gemini returned empty response text", response_dict=response.model_dump())
                raise ValueError("Gemini returned empty response.")

            content = response.text
            
            # Si se pidió JSON pero el modelo devolvió algo con bloques de código, limpiarlo
            if response_pydantic_schema:
                content = self._clean_json_markdown(content)

            gen_log.info("Response received from Gemini", response_preview=truncate_text(content, 200))
            return content

        except Exception as e:
            gen_log.error("Error generating content with Gemini", error=str(e))
            raise ConnectionError(f"Gemini generation failed: {str(e)}") from e

    @staticmethod
    def _clean_json_markdown(text: str) -> str:
        """Elimina bloques de código markdown si existen para obtener solo el JSON raw."""
        text = text.strip()
        if text.startswith("```json"):
            text = text[7:]
        elif text.startswith("```"):
            text = text[3:]
        
        if text.endswith("```"):
            text = text[:-3]
        
        return text.strip()