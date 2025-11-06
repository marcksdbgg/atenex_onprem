# query-service/app/application/ports/llm_port.py
import abc
from typing import Optional, Type

from pydantic import BaseModel

class LLMPort(abc.ABC):
    """Puerto abstracto para interactuar con un Large Language Model."""

    @abc.abstractmethod
    async def generate(
        self,
        prompt: str,
        response_pydantic_schema: Optional[Type[BaseModel]] = None,
    ) -> str:
        """
        Genera texto basado en el prompt proporcionado.

        Args:
            prompt: El prompt a enviar al LLM.
            response_pydantic_schema: Esquema opcional para indicar que la respuesta debería estar en JSON.

        Returns:
            La respuesta generada por el LLM.

        Raises:
            ConnectionError: Si falla la comunicación con el servicio LLM.
            Exception: Para otros errores inesperados.
        """
        raise NotImplementedError