# LangSmith Tracing - Documentación

## 📊 Información Capturada

Este sistema está configurado para capturar información detallada de ejecución en LangSmith, incluyendo:

### 1. Trayectoria de Agentes
- **Agente Orquestador (BeeAI Router)**: Decisiones de ruteo y selección de agentes
- **Agente Generador**: 
  - Sub-agente Ontólogo: Extracción de entidades y relaciones
  - Sub-agente Rubricador: Generación de rúbricas
- **Agente Evaluador**: Evaluación de trabajos contra rúbricas
- **Agente Greeter**: Interacciones de bienvenida

### 2. Consumo de Tokens LLM
- **Estimación de tokens por llamada**: Aproximadamente 1 token = 4 caracteres
- **Total de tokens por sesión**: Suma acumulada de todas las llamadas
- **Número de llamadas al LLM**: Contador de interacciones con Gemini
- **Tokens por paso del agente**: Desglose detallado por cada paso

### 3. Interacciones con Qdrant
- **Operaciones de escritura (upsert)**:
  - Nombre de la colección
  - Número de entidades guardadas
  - Número de relaciones guardadas
  - Número de puntos vectoriales
  - Dimensión de vectores (384)
  
- **Operaciones de búsqueda (search)**:
  - Query de búsqueda
  - Límite de resultados
  - Umbral de score
  - Número de resultados obtenidos
  - Score promedio
  - Top 5 scores

### 4. Metadata de Ejecución
- **Tiempo de ejecución**: Duración de cada función traceable
- **Tipo de operación**: chain, retriever, tool, etc.
- **Errores y excepciones**: Stack traces completos
- **Argumentos de función**: Conteo y claves (sanitizado)
- **Session IDs**: Identificadores de sesión para seguimiento

### 5. Información del Pipeline ADK
- **Tipo de agente**: ADK Multi-Agent
- **Sub-agentes involucrados**: Lista de agentes delegados
- **Pasos del agente**: Secuencia completa de ejecución
- **Roles**: user, model, tool
- **Timestamps**: Marcas de tiempo de cada evento

## 🔗 Acceso al Dashboard

Puedes ver toda esta información en tiempo real en:

```
https://smith.langchain.com/o/projects/p/rubricas_qdrant_system
```

## 📝 Variables de Entorno Configuradas

```bash
LANGSMITH_API_KEY=<tu_api_key>
LANGSMITH_PROJECT=rubricas_qdrant_system
LANGSMITH_TRACING=true
LANGSMITH_TRACING_V2=true
LANGCHAIN_CALLBACKS_BACKGROUND=false  # Sincrónico para capturar todo
LANGCHAIN_VERBOSE=true                # Modo verbose para máximo detalle
```

## 🎯 Funciones Traceables

### Agente Generador
- `RubricGeneratorAgent.invoke` (chain)
- `QdrantService.save_ontology` (chain)
- `QdrantService.search` (retriever)

### Agente Evaluador
- `RubricEvaluatorAgent.invoke` (chain)
- `QdrantService.search` (retriever)

### Orquestador
- `BeeRouter.route` (chain)

## 📈 Métricas Disponibles

En el dashboard de LangSmith podrás ver:

1. **Latencia**: Tiempo de respuesta de cada componente
2. **Throughput**: Número de requests procesados
3. **Error Rate**: Tasa de errores por componente
4. **Token Usage**: Consumo estimado de tokens
5. **Trace Tree**: Árbol completo de ejecución con todos los pasos
6. **Cost Estimation**: Estimación de costos basada en tokens (si está configurado)

## 🔍 Ejemplo de Trace

Un trace típico de generación de rúbrica mostrará:

```
BeeRouter.route
└── RubricGeneratorAgent.invoke
    ├── QdrantService.clear_collection (si aplica)
    ├── ADK Pipeline Execution
    │   ├── Ontólogo Agent
    │   │   └── QdrantService.save_ontology
    │   │       └── [Metadata: 15 entidades, 45 relaciones]
    │   └── Rubricador Agent
    │       └── QdrantService.search
    │           └── [Metadata: 10 resultados, avg_score: 0.85]
    └── [Metadata: 5 llamadas LLM, ~2500 tokens estimados]
```

## 🛠️ Troubleshooting

Si no ves traces en LangSmith:

1. Verifica que `LANGSMITH_API_KEY` esté configurada correctamente
2. Confirma que `LANGSMITH_TRACING=true` en tu `.env`
3. Revisa los logs de inicio para ver el mensaje de confirmación:
   ```
   ✅ LangSmith configurado con trazabilidad completa
   📊 Capturando: Tokens, Latencia, Interacciones Qdrant, Trayectoria de Agentes
   ```
4. Verifica la conectividad con `https://api.smith.langchain.com`

## 📚 Recursos Adicionales

- [LangSmith Documentation](https://docs.smith.langchain.com/)
- [OpenTelemetry Integration](https://docs.smith.langchain.com/observability/opentelemetry)
- [Google ADK Tracing](https://github.com/google/generative-ai-python)
