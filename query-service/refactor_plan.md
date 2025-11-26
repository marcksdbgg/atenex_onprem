# Plan de Refactorización: Optimización de Prompts y RAG

## 1. Análisis de Situación Actual

### Estado de los Prompts
**Hallazgo**: Los prompts **NO** están hardcodeados en el código Python (strings dentro de funciones).
- Se cargan desde archivos de texto en `app/prompts/` (ej: `rag_template_granite.txt`).
- Se gestionan a través de `PromptService` que usa `PromptBuilder` de Haystack.

**Problema**: Aunque están en archivos, requieren un despliegue (deploy) para cambiarse. No se pueden editar en caliente ("hot-swap") ni gestionar versiones dinámicamente sin tocar el sistema de archivos del contenedor/servidor.

### Uso de Haystack
**Hallazgo**: La librería `haystack-ai` se está utilizando **exclusivamente** para:
1.  **Templating**: Usar `PromptBuilder` para renderizar plantillas Jinja2.
2.  **Estructura de Datos**: Usar la clase `Document` para pasar datos al template.

**Veredicto**: Es una dependencia **innecesaria y pesada** (bloatware) para este caso de uso. Jinja2 puro hace lo mismo con menos overhead y mayor control.

---

## 2. Objetivos del Refactor

1.  **Eliminar Dependencia de Haystack**: Reemplazar `haystack` con `jinja2` nativo. Reducirá el tamaño de la imagen Docker y la complejidad.
2.  **Sistema de Prompts Robusto**: Implementar un `PromptManager` que permita:
    - Cargar prompts desde archivos (por defecto).
    - (Futuro) Cargar prompts desde Base de Datos o Variables de Entorno para cambios en caliente.
    - Versionado simple.
3.  **Optimización RAG**: Estandarizar la inyección de contexto y mejorar el manejo de fallos.

---

## 3. Plan de Implementación

### Paso 1: Eliminar Haystack y Migrar a Jinja2
**Archivos afectados**: `app/application/use_cases/ask_query/prompt_service.py`, `pyproject.toml`

1.  **Desinstalar Haystack**: Remover `haystack-ai` de `pyproject.toml`.
2.  **Implementar Jinja2 Directo**:
    - Modificar `PromptService` para usar `jinja2.Template` o `jinja2.Environment`.
    - Eliminar la conversión `_to_haystack_docs`. Usar diccionarios o modelos Pydantic directamente en el template.
    - Esto simplifica el código y elimina la necesidad de transformar `RetrievedChunk` -> `Haystack Document`.

### Paso 2: Mejorar la Gestión de Prompts (`PromptService`)
**Archivos afectados**: `app/application/use_cases/ask_query/prompt_service.py`

1.  **Carga Resiliente**:
    - Al iniciar, cargar templates en memoria.
    - Añadir método `reload_prompts()` para recargar desde disco sin reiniciar el servicio (útil para desarrollo/montajes de volumen).
2.  **Validación**: Asegurar que los templates tengan las variables requeridas (`{{ query }}`, `{{ documents }}`) al cargarlos.

### Paso 3: Refinar Templates
**Archivos afectados**: `app/prompts/*.txt`

1.  **Limpieza**: Asegurar que la sintaxis Jinja2 sea estándar (Haystack usa Jinja2 por debajo, así que los cambios serán mínimos, principalmente en cómo se iteran los objetos).
2.  **Estructura**: Los objetos `documents` ahora serán dicts o Pydantic models, no objetos `Document` de Haystack. Verificar acceso a atributos (ej: `doc.content` vs `doc['content']`).

---

## 4. Beneficios Esperados

- **Menor Deuda Técnica**: Menos dependencias externas grandes.
- **Mayor Rendimiento**: Renderizado de templates más rápido y directo.
- **Flexibilidad**: Base sólida para mover prompts a una DB en el futuro si se requiere un "Prompt Registry" avanzado.
