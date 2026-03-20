# Bugfix Requirements Document

## Introduction

El agente `ontologo` (en `agents/generator/app/adk_agents.py`) falla al intentar llamar a la herramienta `guardar_ontologia_en_qdrant` a través de Groq. El modelo genera un payload JSON para la tool call que es demasiado grande o complejo (listas de entidades y relaciones con múltiples campos anidados), lo que provoca que Groq no pueda parsearlo y devuelva `litellm.BadRequestError: GroqException - {"error":{"message":"Failed to parse tool call arguments as JSON"}}`. El JSON se corta al final (termina con `\"` incompleto), confirmando que el payload excede los límites de Groq para argumentos de tool calls.

## Bug Analysis

### Current Behavior (Defect)

1.1 WHEN el agente `ontologo` extrae una ontología con múltiples entidades y relaciones de un documento normativo y llama a `guardar_ontologia_en_qdrant` con esas listas completas como argumentos de tool call THEN el sistema lanza `litellm.BadRequestError` con el mensaje "Failed to parse tool call arguments as JSON" y el pipeline se interrumpe sin guardar nada en Qdrant.

1.2 WHEN el JSON generado por Groq para los argumentos de la tool call supera el límite de tamaño/complejidad del parser de Groq THEN el sistema produce un JSON truncado (terminando con `\"` incompleto) que no puede ser deserializado.

1.3 WHEN el pipeline falla en el paso del ontólogo THEN el sistema no transfiere al agente `rubricador` y no se genera ninguna rúbrica.

### Expected Behavior (Correct)

2.1 WHEN el agente `ontologo` extrae una ontología con múltiples entidades y relaciones THEN el sistema SHALL guardar las entidades en Qdrant sin que Groq falle al parsear los argumentos de la tool call, independientemente del número de entidades extraídas.

2.2 WHEN el payload de la tool call sería demasiado grande para Groq THEN el sistema SHALL dividir o simplificar los argumentos de forma que cada llamada individual permanezca dentro de los límites de Groq (por ejemplo, guardando entidades en lotes pequeños o aceptando el JSON como string serializado en lugar de objeto estructurado).

2.3 WHEN el guardado en Qdrant se completa exitosamente THEN el sistema SHALL continuar el pipeline transfiriendo al agente `rubricador` para generar la rúbrica.

### Unchanged Behavior (Regression Prevention)

3.1 WHEN el agente `ontologo` guarda entidades en Qdrant THEN el sistema SHALL CONTINUE TO persistir correctamente el nombre, tipo, contexto, propiedades y relaciones salientes de cada entidad.

3.2 WHEN el agente `rubricador` busca contexto con `buscar_contexto_qdrant` THEN el sistema SHALL CONTINUE TO retornar resultados relevantes de Qdrant con scores de similitud.

3.3 WHEN el agente evaluador llama a `buscar_contexto_para_evaluacion` THEN el sistema SHALL CONTINUE TO funcionar sin cambios, ya que esa herramienta no recibe payloads grandes.

3.4 WHEN se procesa un documento normativo pequeño que genera pocas entidades THEN el sistema SHALL CONTINUE TO completar el pipeline completo (ontólogo → rubricador) y devolver una rúbrica generada.
