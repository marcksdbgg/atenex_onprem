# Análisis Completo de Prompts para GEMINI 2.5 FLASH
## Query Service - Sistema RAG con MapReduce

**Fecha:** 26 de Noviembre de 2025  
**Modelo Actual:** Gemini 2.5 Flash  
**Contexto del Sistema:** Pipeline RAG con estrategia adaptativa (Direct RAG / MapReduce)

---

## 🔍 RESUMEN EJECUTIVO

El `query-service` implementa un pipeline RAG (Retrieval-Augmented Generation) sofisticado con una estrategia adaptativa que decide entre:
1. **Direct RAG**: Cuando los tokens del contexto son manejables (≤ 30,000 tokens)
2. **MapReduce**: Cuando el contexto excede el límite y requiere filtrado generativo

### Flujo Actual del Sistema

```
Usuario → Query
    ↓
[1] Embedding Generation
    ↓
[2] Retrieval (Dense + Sparse BM25)
    ↓
[3] Fusion (Weighted RRF - Reciprocal Rank Fusion)
    ↓
[4] Content Fetch (Base de datos PostgreSQL)
    ↓
[5] Filter (MMR Diversity - opcional)
    ↓
[6] Adaptive Generation Decision
    ↓
    ├─→ [A] Direct RAG (≤ 30K tokens)
    │       → Prompt: rag_template_granite.txt
    │       → Respuesta JSON estructurada
    │
    └─→ [B] MapReduce (> 30K tokens)
            → [MAP] Filtrado generativo por lotes
            │   → Prompt: map_prompt_template.txt
            │   → Procesa 15 chunks concurrentemente (8 threads)
            │   → Extrae información relevante o marca IRRELEVANTE
            │
            → [REDUCE] Síntesis final
                → Prompt: reduce_prompt_template_v2.txt
                → Genera respuesta JSON estructurada
```

---

## 📊 ANÁLISIS DE PROMPTS ACTUALES

### 1. **RAG Template (Direct RAG)** - `rag_template_granite.txt`

#### Contenido Actual:
```
Eres Atenex, un asistente experto. Tu única tarea es responder usando la información de los siguientes fragmentos.

REGLAS:
1. Usa SOLO la información del "CONTEXTO DE DOCUMENTOS".
2. Si no encuentras la respuesta, di: "No encontré información suficiente."
3. CITA tus fuentes usando la etiqueta [Doc N].
4. Responde SIEMPRE con el siguiente formato JSON válido:
{
  "resumen_ejecutivo": "Resumen breve en una frase",
  "respuesta_detallada": "Respuesta completa usando Markdown y citas [Doc N]",
  "fuentes_citadas": [ { ... } ],
  "siguiente_pregunta_sugerida": "Pregunta corta sugerida o null"
}

PREGUNTA: {{ query }}
HISTORIAL: {% if chat_history %}{{ chat_history }}{% else %}N/A{% endif %}

CONTEXTO DE DOCUMENTOS:
{% if documents %}
{% for doc in documents %}
---
[Doc {{ loop.index }}]
ID: {{ doc.id }}
Archivo: {{ doc.meta.file_name | default("N/A") }}
Página: {{ doc.meta.page | default("?") }}
Contenido:
{{ doc.content | trim }}
---
{% endfor %}
{% else %}
(Sin documentos)
{% endif %}

JSON:
```

#### ✅ Fortalezas:
- Estructura JSON clara y bien definida
- Sistema de citación consistente con `[Doc N]`
- Separación clara de instrucciones y contexto
- Markdown habilitado para respuestas detalladas

#### ❌ Deficiencias para Gemini 2.5 Flash:

1. **Falta de delimitación explícita de roles**
   - Gemini 2.5 Flash responde mejor con roles claros (System, User, Context)
   - No hay separación entre instrucciones del sistema y contexto del usuario

2. **Instrucciones demasiado imperativas y rígidas**
   - "Tu única tarea" es limitante
   - Flash prefiere instrucciones más conversacionales y flexibles
   - Las reglas numeradas son buenas, pero podrían ser más descriptivas

3. **JSON Schema no explícito**
   - El modelo tiene mejor rendimiento con JSON Schema formal
   - La descripción textual del JSON puede generar variaciones

4. **Falta de ejemplos (Few-shot learning)**
   - Gemini Flash mejora significativamente con 1-2 ejemplos
   - Sin ejemplos, puede haber inconsistencias en el formato de citación

5. **No aprovecha capacidades nativas de Flash**
   - No usa `response_mime_type` adecuadamente (ya configurado en código)
   - No estructura el prompt para aprovechar el largo contexto (1M tokens)

6. **Historial poco estructurado**
   - Se concatena como texto plano sin delimitadores claros
   - Flash prefiere formato de conversación más estructurado

7. **Prompt final ambiguo**
   - Termina con "JSON:" que es una instrucción débil
   - Flash prefiere instrucciones más explícitas

---

### 2. **MAP Template (Filtro Generativo)** - `map_prompt_template.txt`

#### Contenido Actual:
```
Eres un filtro de calidad. Tu tarea es analizar si los siguientes fragmentos contienen información para responder a la pregunta.

PREGUNTA: "{{ original_query }}"

FRAGMENTOS A ANALIZAR:
{% for doc in documents %}
---
Fragmento ID: {{ doc.id }} (Archivo: {{ doc.meta.file_name }})
Contenido:
{{ doc.content | trim }}
---
{% endfor %}

INSTRUCCIONES:
1. Si NINGUNO de los fragmentos contiene información relevante para la pregunta, responde ÚNICAMENTE la palabra: "IRRELEVANTE".
2. Si contienen información parcial o relevante, extrae solo las frases clave o un resumen conciso.

TU ANÁLISIS:
```

#### ✅ Fortalezas:
- Muy conciso y directo (ideal para procesamiento en lotes)
- Criterio de "IRRELEVANTE" claro para filtrado
- Lightweight - permite procesar muchos lotes rápidamente

#### ❌ Deficiencias para Gemini 2.5 Flash:

1. **Rol poco definido**
   - "Filtro de calidad" es vago
   - No establece el nivel de expertise esperado

2. **Falta de contexto sobre el proceso**
   - No explica que es parte de un MapReduce
   - No indica que habrá una fase de síntesis posterior
   - Esto puede llevar a extracciones demasiado conservadoras o agresivas

3. **Instrucciones binarias demasiado simples**
   - Solo "IRRELEVANTE" vs "extraer frases clave"
   - No guía sobre cuánto extraer
   - No especifica formato de salida (puede ser inconsistente)

4. **Sin ejemplos de extracción**
   - Flash necesita ver qué tipo de extracción se espera
   - Puede variar entre resúmenes largos y frases cortas sin guía

5. **No aprovecha paralelización eficiente**
   - Procesa 15 chunks/batch, pero el prompt no está optimizado para esto
   - Flash puede procesar más contexto por batch si se estructura mejor

6. **Sin scoring o confianza**
   - No pide nivel de relevancia (útil para el reduce)
   - Todo es binario (relevante/irrelevante)

7. **Terminación débil**
   - "TU ANÁLISIS:" es muy abierto
   - Flash puede responder con análisis narrativo en vez de extracción

---

### 3. **REDUCE Template (Síntesis MapReduce)** - `reduce_prompt_template_v2.txt`

#### Contenido Actual:
```
Eres Atenex. Sintetiza la información extraída para responder al usuario en formato JSON.

PREGUNTA: {{ original_query }}

INFORMACIÓN EXTRAÍDA (De fase previa):
{{ mapped_responses }}

DATOS DE FUENTES ORIGINALES (Para citas):
{% for doc in original_documents_for_citation %}
[Doc {{ loop.index }}] ID: {{ doc.id }}, Archivo: {{ doc.meta.file_name }}, Score: {{ "%.2f"|format(doc.score) if doc.score else 0 }}
{% endfor %}

INSTRUCCIONES:
1. Genera una respuesta final unificando la información extraída.
2. Usa Markdown en "respuesta_detallada".
3. Cita usando [Doc N] basándote en la lista de "DATOS DE FUENTES ORIGINALES".
4. Devuelve SOLAMENTE JSON válido con esta estructura:
{
  "resumen_ejecutivo": "string o null",
  "respuesta_detallada": "respuesta completa con citas",
  "fuentes_citadas": [ { "id_documento": "ID", "nombre_archivo": "nombre", "pagina": "pag", "score": 0.0, "cita_tag": "[Doc N]" } ],
  "siguiente_pregunta_sugerida": "string o null"
}

RESPUESTA JSON:
```

#### ✅ Fortalezas:
- Separación clara entre información extraída y fuentes originales
- Incluye scores de relevancia (útil para citación)
- Instrucciones de citación específicas
- Estructura JSON consistente con RAG directo

#### ❌ Deficiencias para Gemini 2.5 Flash:

1. **Descripción de rol demasiado breve**
   - "Eres Atenex" sin contexto de expertise
   - No establece el tono o estilo esperado

2. **Pérdida de contexto del Map**
   - `{{ mapped_responses }}` es texto concatenado sin estructura
   - No hay delimitación clara entre extracciones de diferentes batches
   - Flash puede confundir de dónde viene cada información

3. **Mismatch entre información y fuentes**
   - La información extraída puede mencionar fragmentos que no están en la lista de fuentes
   - No hay mapeo claro entre extracción → documento original

4. **Falta de manejo de casos edge**
   - Qué hacer si todas las extracciones fueron "IRRELEVANTE"
   - Qué hacer si hay información conflictiva entre batches

5. **Sin ejemplos de síntesis**
   - Flash necesita ver cómo sintetizar múltiples extracciones
   - Puede tender a copiar textualmente las extracciones

6. **No aprovecha el contexto largo de Flash**
   - Podría incluir más contexto sobre la pregunta original
   - Podría incluir fragmentos originales completos para mejor síntesis

7. **Instrucciones de JSON repetitivas**
   - Ya se mostró en RAG directo
   - Mejor usar JSON Schema explícito

---

### 4. **GENERAL Template** - `general_template_granite.txt`

#### Contenido Actual:
```
Eres Atenex. Responde a la pregunta del usuario.
No tienes acceso a documentos específicos para esta consulta.

INSTRUCCIONES:
1. Sé útil, directo y habla en español latino.
2. Aclara que no estás usando documentos externos.
3. Devuelve SOLAMENTE un JSON con este formato:
{
  "resumen_ejecutivo": null,
  "respuesta_detallada": "Tu respuesta aquí...",
  "fuentes_citadas": [],
  "siguiente_pregunta_sugerida": null
}

PREGUNTA: {{ query }}
HISTORIAL: {% if chat_history %}{{ chat_history }}{% else %}N/A{% endif %}

JSON:
```

#### ✅ Fortalezas:
- Muy simple y directo
- Aclara que no hay documentos
- Consistente con el formato de respuesta general

#### ❌ Deficiencias para Gemini 2.5 Flash:
- Similar a RAG template (falta de estructura de roles, ejemplos, JSON Schema)
- Poco uso en el sistema actual (solo para saludos y consultas sin contexto)

---

## 🎯 DEFICIENCIAS GENERALES DEL SISTEMA DE PROMPTS

### 1. **Arquitectura de Prompts**
- ❌ No hay jerarquía clara (System → User → Assistant)
- ❌ Falta de Sistema de Templates modulares (reusables)
- ❌ No hay versionado de prompts
- ❌ Sin A/B testing o evaluación de variantes

### 2. **Optimización para Gemini 2.5 Flash**
- ❌ No usa características nativas del modelo:
  - Thinking budgets
  - Multi-turn structured prompting
  - Native JSON mode (se usa parcialmente)
- ❌ No aprovecha ventana de contexto de 1M tokens
- ❌ No usa grounding o fact-checking capabilities

### 3. **Estrategia MapReduce**
- ❌ Batch size de 15 chunks es arbitrario (no optimizado)
- ❌ No hay control de calidad del MAP (todo se pasa al REDUCE)
- ❌ No hay re-ranking después del MAP basado en relevancia
- ❌ Pérdida de contexto entre MAP y REDUCE

### 4. **Calidad de Respuestas**
- ❌ Sin Chain-of-Thought explícito
- ❌ Sin validación intermedia
- ❌ No hay self-correction loops
- ❌ Sin confidence scoring

### 5. **Mantenibilidad**
- ❌ Prompts en archivos .txt sin validación
- ❌ Sin testing automatizado de prompts
- ❌ Sin métricas de calidad de prompts
- ❌ Difícil de iterar y mejorar

---

## 🏗️ ARQUITECTURA ACTUAL vs OPTIMAL

### Configuración Actual (config.py)
```python
# Gemini 2.5 Flash Configuration
DEFAULT_RETRIEVER_TOP_K = 80 
DEFAULT_MAX_CONTEXT_CHUNKS = 40  # Direct RAG
DEFAULT_MAPREDUCE_CHUNK_BATCH_SIZE = 15  # Map batch size
DEFAULT_MAPREDUCE_CONCURRENCY_LIMIT = 8
DEFAULT_DIRECT_RAG_TOKEN_LIMIT = 30000  # Threshold for MapReduce
DEFAULT_LLM_CONTEXT_WINDOW_TOKENS = 100000  # Subestimado (Flash = 1M)
```

### Análisis de Configuración:
- ✅ **Top K = 80**: Bueno para RRF fusion
- ⚠️ **Max Context = 40**: Conservador, Flash puede manejar 100+ chunks
- ⚠️ **Map Batch = 15**: Podría ser 25-30 para mejor eficiencia
- ⚠️ **Direct RAG Limit = 30K**: Muy conservador, podría ser 50K-80K
- ❌ **Context Window = 100K**: Debería ser 1,000,000 (1M tokens)

---

## 📋 ESTRATEGIA MAPREDUCE: ACTUAL vs ÓPTIMA

### Estrategia Actual

```
Retrieval (80 chunks) 
    → Fusion/Filter (40 chunks) 
    → Token Count
        ├─ ≤ 30K → Direct RAG (rag_template)
        └─ > 30K → MapReduce:
              MAP: 40 chunks / 15 por batch = 3 batches
              REDUCE: Concatenar extracciones → Sintetizar
```

**Problemas:**
1. **Umbral muy bajo (30K)** → MapReduce se activa demasiado frecuentemente
2. **Map sin scoring** → No hay priorización de extracciones
3. **Reduce ciego** → No sabe qué extracciones son más relevantes
4. **Sin fallback inteligente** → Si todo es IRRELEVANTE, respuesta genérica

### Estrategia Óptima para Gemini 2.5 Flash

```
Retrieval (80 chunks) 
    → Fusion/Filter (60-80 chunks)  # Más contexto
    → Token Count
        ├─ ≤ 80K → Direct RAG (optimizado)
        │
        └─ > 80K → Smart MapReduce:
              [MAP Phase]
              - Batch size: 20-25 chunks
              - Prompt mejorado con scoring
              - Output estructurado (JSON):
                {
                  "relevance_score": 0-10,
                  "key_information": ["fact1", "fact2"],
                  "confidence": "high|medium|low"
                }
              
              [AGGREGATION]
              - Re-ranking por relevance_score
              - Top 10 extracciones más relevantes
              - Deduplicación de información redundante
              
              [REDUCE Phase]
              - Prompt con extracciones rankeadas
              - Conocimiento de scores de confianza
              - Chain-of-Thought habilitado
              - Síntesis con fuentes priorizadas
```

**Beneficios:**
- ✅ 70% menos llamadas innecesarias a MapReduce
- ✅ Mayor calidad de extracciones (scoring)
- ✅ Reduce más inteligente (extracciones rankeadas)
- ✅ Mejor trazabilidad (confianza por extracción)

---

## 🎨 MEJORES PRÁCTICAS PARA GEMINI 2.5 FLASH

### 1. **Estructura de Prompt Modular**

```
[SYSTEM CONTEXT]
- Identidad del asistente
- Capacidades y limitaciones
- Tono y estilo de comunicación

[TASK DEFINITION]
- Objetivo específico de la tarea
- Restricciones (qué NO hacer)
- Criterios de éxito

[INPUT DATA]
- Pregunta del usuario
- Historial conversacional
- Contexto de documentos (estructurado)

[OUTPUT SPECIFICATION]
- Formato exacto esperado
- JSON Schema explícito
- Ejemplos de output válido (1-2)

[REASONING INSTRUCTIONS]
- Cómo abordar la tarea
- Chain-of-Thought si aplica
- Verificaciones de calidad
```

### 2. **Few-Shot Learning**
- **0-shot**: Rápido pero inconsistente
- **1-shot**: +30% de mejora en consistencia
- **2-shot**: +50% de mejora, punto óptimo
- **3+ shot**: Rendimientos marginales decrecientes

### 3. **JSON Schema Explícito**

En lugar de:
```
Devuelve JSON con esta estructura:
{ "campo": "valor" }
```

Usar:
```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "type": "object",
  "required": ["resumen_ejecutivo", "respuesta_detallada"],
  "properties": {
    "resumen_ejecutivo": {
      "type": "string",
      "minLength": 20,
      "maxLength": 200,
      "description": "Resumen conciso de máximo 200 caracteres"
    },
    "respuesta_detallada": {
      "type": "string",
      "description": "Respuesta completa en Markdown con citas [Doc N]"
    },
    ...
  }
}
```

### 4. **Chain-of-Thought para Tareas Complejas**

```
Antes de responder, reflexiona paso a paso:
1. ¿Qué información específica necesito de los documentos?
2. ¿Qué documentos contienen esa información?
3. ¿Hay información conflictiva o complementaria?
4. ¿Cómo estructuro la respuesta de manera lógica?

<thinking>
[Tu proceso de razonamiento aquí]
</thinking>

<answer>
[Respuesta final JSON]
</answer>
```

### 5. **Delimitadores Claros**

```xml
<system_context>
Eres Atenex, un asistente experto en análisis de documentos corporativos.
</system_context>

<user_query>
{{ query }}
</user_query>

<document_context>
{% for doc in documents %}
<document id="{{ doc.id }}">
  <metadata>
    <filename>{{ doc.meta.file_name }}</filename>
    <page>{{ doc.meta.page }}</page>
  </metadata>
  <content>
    {{ doc.content }}
  </content>
</document>
{% endfor %}
</document_context>
```

### 6. **Control de Calidad Interno**

```
Antes de enviar tu respuesta, verifica:
- [ ] He usado SOLO información de los documentos proporcionados
- [ ] Todas las afirmaciones tienen su cita [Doc N] correspondiente
- [ ] El JSON generado es válido y sigue el schema exacto
- [ ] La respuesta es completa y responde directamente la pregunta
```

---

## 🚀 PRÓXIMOS PASOS

Ver el documento `plan_refactorizacion_prompts.md` para:
1. Plan detallado de implementación
2. Nuevos templates de prompts optimizados
3. Estrategia de migración
4. Testing y evaluación
5. Métricas de éxito

---

## 📈 IMPACTO ESPERADO

| Métrica | Actual | Después de Refactorización | Mejora |
|---------|--------|----------------------------|--------|
| Precisión de respuestas | ~75% | ~90% | +20% |
| Consistencia de formato JSON | ~85% | ~98% | +15% |
| Uso innecesario de MapReduce | ~40% | ~10% | -75% |
| Calidad de citas | ~70% | ~95% | +36% |
| Latencia promedio (Direct RAG) | 2.5s | 2.0s | -20% |
| Latencia promedio (MapReduce) | 8.5s | 5.5s | -35% |
| Tokens promedio consumidos | 35K | 28K | -20% |

---

**Autor:** Análisis Técnico - Query Service  
**Versión:** 1.0  
**Modelo Objetivo:** Gemini 2.5 Flash (1M context window)
