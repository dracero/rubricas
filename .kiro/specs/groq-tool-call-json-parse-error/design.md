# groq-tool-call-json-parse-error Bugfix Design

## Overview

El agente `ontologo` falla al llamar a `guardar_ontologia_en_qdrant` cuando la ontología extraída contiene muchas entidades y relaciones. Groq no puede parsear los argumentos JSON de la tool call porque el payload supera su límite de tamaño/complejidad, devolviendo `litellm.BadRequestError: GroqException - {"error":{"message":"Failed to parse tool call arguments as JSON"}}` con el JSON truncado.

La estrategia de fix es eliminar el payload estructurado grande de la tool call. En lugar de pasar listas completas de entidades y relaciones como argumentos JSON, el agente recibirá el texto de la ontología como string serializado (JSON-as-string) o se dividirá en lotes pequeños, manteniéndose dentro de los límites de Groq.

## Glossary

- **Bug_Condition (C)**: La condición que dispara el bug — cuando `guardar_ontologia_en_qdrant` es llamada con listas de entidades/relaciones cuyo JSON serializado supera el límite de Groq para argumentos de tool call (~4KB aproximadamente).
- **Property (P)**: El comportamiento correcto — la herramienta debe guardar todas las entidades en Qdrant sin que Groq falle al parsear los argumentos.
- **Preservation**: El comportamiento de búsqueda (`buscar_contexto_qdrant`, `buscar_contexto_para_evaluacion`) y el guardado correcto de los campos de cada entidad que deben permanecer sin cambios.
- **guardar_ontologia_en_qdrant**: Función tool en `agents/generator/app/adk_agents.py` que recibe entidades y relaciones y las persiste en Qdrant vía `QdrantService.save_ontology`.
- **payload_size**: El tamaño en bytes del JSON serializado de los argumentos de la tool call enviados a Groq.
- **GROQ_TOOL_CALL_LIMIT**: Límite implícito de Groq para el tamaño de argumentos JSON en una tool call (evidenciado por el JSON truncado en el error).

## Bug Details

### Bug Condition

El bug se manifiesta cuando el agente `ontologo` extrae una ontología con suficientes entidades y relaciones como para que el JSON de los argumentos de `guardar_ontologia_en_qdrant` supere el límite de Groq. El modelo genera el payload completo pero Groq lo trunca al serializarlo, produciendo JSON inválido.

**Formal Specification:**
```
FUNCTION isBugCondition(tool_call_args)
  INPUT: tool_call_args = {"entidades": [...], "relaciones": [...]}
  OUTPUT: boolean

  serialized := JSON.serialize(tool_call_args)
  RETURN len(serialized) > GROQ_TOOL_CALL_LIMIT
         AND tool_name == "guardar_ontologia_en_qdrant"
END FUNCTION
```

### Examples

- **Caso buggy**: Documento normativo de 8000 chars → ontólogo extrae 15 entidades con propiedades anidadas + 45 relaciones → JSON de args ~6KB → Groq trunca → `Failed to parse tool call arguments as JSON`.
- **Caso buggy**: Documento con 10 entidades pero cada una con `propiedades` dict grande → JSON supera límite → mismo error.
- **Caso no-buggy**: Documento pequeño → 3 entidades simples → JSON de args ~500 bytes → Groq parsea correctamente → guardado exitoso.
- **Edge case**: Exactamente en el límite → comportamiento no determinista dependiendo del modelo.

## Expected Behavior

### Preservation Requirements

**Unchanged Behaviors:**
- `buscar_contexto_qdrant` debe continuar retornando resultados relevantes de Qdrant con scores de similitud.
- `buscar_contexto_para_evaluacion` del agente evaluador no debe verse afectada en absoluto.
- Cada entidad guardada en Qdrant debe continuar persistiendo correctamente: `nombre`, `tipo`, `contexto`, `propiedades` y `relaciones_salientes`.
- El pipeline completo (ontólogo → rubricador) debe continuar funcionando para documentos pequeños que ya funcionaban antes del fix.

**Scope:**
Todos los inputs que NO involucren un payload de tool call grande para `guardar_ontologia_en_qdrant` deben quedar completamente sin cambios. Esto incluye:
- Llamadas a `buscar_contexto_qdrant` (argumento es solo un string corto).
- Llamadas a `buscar_contexto_para_evaluacion` (argumento es solo un string corto).
- El flujo del agente evaluador completo.
- Documentos normativos pequeños que generan pocas entidades.

## Hypothesized Root Cause

Basado en el análisis del bug, las causas más probables son:

1. **Payload demasiado grande en un solo tool call**: La firma actual de `guardar_ontologia_en_qdrant` acepta `entidades: List[Dict]` y `relaciones: List[Dict]` como argumentos directos. Cuando el LLM genera el JSON para estos argumentos, el tamaño total supera el límite de Groq para tool call arguments.
   - El JSON de una lista de 15 entidades con propiedades anidadas puede fácilmente superar 4-6KB.
   - Groq trunca el JSON al límite, produciendo JSON inválido (termina con `\"` incompleto).

2. **Ausencia de batching**: No existe ningún mecanismo que divida el payload en lotes más pequeños antes de enviarlo a Groq.

3. **Schema de la tool demasiado complejo**: El schema JSON inferido por ADK para `List[Dict[str, Any]]` puede generar un schema muy verboso que contribuye al tamaño del payload.

4. **Sin validación de tamaño pre-llamada**: No hay ningún guardrail que detecte cuándo el payload sería demasiado grande y tome una acción alternativa.

## Correctness Properties

Property 1: Bug Condition - Tool Call Arguments Fit Within Groq Limits

_For any_ invocation of `guardar_ontologia_en_qdrant` where the serialized JSON of the arguments would exceed Groq's tool call limit (isBugCondition returns true), the fixed implementation SHALL successfully save all entities to Qdrant without raising `litellm.BadRequestError`, by ensuring no single tool call argument payload exceeds the safe size threshold.

**Validates: Requirements 2.1, 2.2**

Property 2: Preservation - Small Payload Behavior Unchanged

_For any_ invocation of `guardar_ontologia_en_qdrant` where the serialized JSON of the arguments does NOT exceed Groq's tool call limit (isBugCondition returns false), the fixed implementation SHALL produce the same result as the original implementation, preserving correct entity persistence (nombre, tipo, contexto, propiedades, relaciones_salientes) in Qdrant.

**Validates: Requirements 3.1, 3.2, 3.3, 3.4**

## Fix Implementation

### Changes Required

Asumiendo que la causa raíz es el payload demasiado grande en un solo tool call:

**File**: `agents/generator/app/adk_agents.py`

**Function**: `guardar_ontologia_en_qdrant`

**Specific Changes**:

1. **Cambiar la firma para aceptar JSON-as-string**: En lugar de `entidades: List[Dict]`, aceptar `ontologia_json: str` — el LLM serializa la ontología como string, lo que evita que Groq intente parsear una estructura anidada compleja como argumentos tipados.
   - Alternativa: mantener la firma pero agregar batching interno.

2. **Implementar batching de entidades**: Si se mantiene la firma actual, dividir las entidades en lotes de máximo 5 entidades por llamada, haciendo múltiples upserts a Qdrant en lugar de uno solo.

3. **Simplificar el schema de la tool**: Reducir la complejidad del schema JSON que ADK genera para la herramienta, usando tipos más simples o documentación más concisa.

4. **Agregar truncado de propiedades**: Limitar el tamaño de `propiedades` y `contexto` de cada entidad antes de serializar, para reducir el tamaño del payload.

5. **Actualizar la instrucción del agente `ontologo`**: Indicar explícitamente que debe limitar el número de propiedades por entidad y el número de entidades por llamada si se usa batching.

**Approach recomendado** (menor cambio, mayor impacto):
Cambiar `guardar_ontologia_en_qdrant` para aceptar `ontologia_json: str` (JSON serializado como string). El LLM pasa un string, no una estructura anidada, lo que mantiene el argumento de la tool call pequeño. La función deserializa internamente.

## Testing Strategy

### Validation Approach

La estrategia sigue dos fases: primero reproducir el bug en el código sin fix para confirmar la causa raíz, luego verificar que el fix funciona y no introduce regresiones.

### Exploratory Bug Condition Checking

**Goal**: Reproducir el error `Failed to parse tool call arguments as JSON` en el código actual para confirmar que el payload grande es la causa raíz. Si no se reproduce, re-hipotizar.

**Test Plan**: Crear tests unitarios que simulen la llamada a `guardar_ontologia_en_qdrant` con payloads de distintos tamaños y verificar cuándo Groq falla. Ejecutar en el código SIN fix.

**Test Cases**:
1. **Large Payload Test**: Llamar `guardar_ontologia_en_qdrant` con 15 entidades y 45 relaciones (simulando el caso real) — fallará en código sin fix.
2. **Medium Payload Test**: Llamar con 8 entidades y 20 relaciones — puede fallar dependiendo del tamaño de propiedades.
3. **Small Payload Test**: Llamar con 3 entidades y 5 relaciones — debería pasar incluso sin fix.
4. **Boundary Test**: Encontrar el número exacto de entidades donde Groq empieza a fallar.

**Expected Counterexamples**:
- `litellm.BadRequestError` con mensaje "Failed to parse tool call arguments as JSON" al pasar listas grandes.
- JSON truncado terminando con `\"` incompleto en el payload enviado a Groq.

### Fix Checking

**Goal**: Verificar que para todos los inputs donde la condición de bug se cumple, la implementación fija guarda exitosamente en Qdrant.

**Pseudocode:**
```
FOR ALL tool_call_args WHERE isBugCondition(tool_call_args) DO
  result := guardar_ontologia_en_qdrant_fixed(tool_call_args)
  ASSERT result starts with "✅"
  ASSERT qdrant.count() > 0
END FOR
```

### Preservation Checking

**Goal**: Verificar que para todos los inputs donde la condición de bug NO se cumple, la implementación fija produce el mismo resultado que la original.

**Pseudocode:**
```
FOR ALL tool_call_args WHERE NOT isBugCondition(tool_call_args) DO
  result_original := guardar_ontologia_en_qdrant_original(tool_call_args)
  result_fixed    := guardar_ontologia_en_qdrant_fixed(tool_call_args)
  ASSERT result_original == result_fixed
  ASSERT entities_in_qdrant(original) == entities_in_qdrant(fixed)
END FOR
```

**Testing Approach**: Property-based testing es recomendado para preservation checking porque:
- Genera automáticamente muchos tamaños y estructuras de payload.
- Detecta edge cases donde el fix podría alterar el comportamiento para payloads pequeños.
- Provee garantías fuertes de que el comportamiento es idéntico para todos los inputs no-buggy.

**Test Plan**: Observar el comportamiento en código sin fix para payloads pequeños, luego escribir property-based tests que capturen ese comportamiento.

**Test Cases**:
1. **Small Payload Preservation**: Verificar que 3 entidades se guardan con los mismos campos (nombre, tipo, contexto, propiedades, relaciones_salientes) antes y después del fix.
2. **Search Preservation**: Verificar que `buscar_contexto_qdrant` retorna los mismos resultados después del fix para las mismas entidades guardadas.
3. **Evaluator Tool Preservation**: Verificar que `buscar_contexto_para_evaluacion` no se ve afectada por el fix.
4. **Pipeline Preservation**: Verificar que el pipeline completo (ontólogo → rubricador) funciona para documentos pequeños.

### Unit Tests

- Test de `guardar_ontologia_en_qdrant` con payload pequeño (< límite) — debe retornar `✅`.
- Test de `guardar_ontologia_en_qdrant` con payload grande (> límite) — debe retornar `✅` con el fix aplicado.
- Test de que los campos de cada entidad se persisten correctamente en Qdrant.
- Test de edge case: lista vacía de entidades.
- Test de edge case: entidades sin relaciones.

### Property-Based Tests

- Generar listas de entidades de tamaño aleatorio (1-50) y verificar que el fix siempre guarda exitosamente sin `BadRequestError`.
- Generar entidades con propiedades de tamaño aleatorio y verificar que los campos se preservan correctamente en Qdrant.
- Verificar que para cualquier payload pequeño (< límite), el resultado del fix es idéntico al original.

### Integration Tests

- Test del pipeline completo con un documento normativo real de tamaño mediano (5000 chars).
- Test del pipeline completo con un documento normativo grande (15000 chars) que antes fallaba.
- Test de que después del fix, el rubricador puede buscar contexto en Qdrant y generar una rúbrica.
