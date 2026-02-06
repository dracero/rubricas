# Arquitectura de Agentes y Protocolos

Este diagrama ilustra la arquitectura A2A (Agent-to-Agent) implementada en el sistema RubricAI.

```mermaid
graph TD
    %% Actors
    User(Usuario)
    
    %% Frontend
    subgraph Frontend ["Interfaz de Chat (React)"]
        UI[Chat UI]
        ActionHandler[Manejador de Acciones]
    end

    %% Backend / Orchestration Layer
    subgraph Backend ["Servidor FastAPI"]
        subgraph API_Layer ["Capa de API"]
            API_Chat["/api/chat"]
            API_Gen["/api/invocaciones_especificas (upload/generate/evaluate)"]
        end
        
        subgraph Orchestrator [AgentOrchestrator]
            OrchLogic["Lógica de Enrutamiento (Gemini 2.5)"]
            Registry[Registro de Agentes]
        end
        
        %% Agents
        subgraph SocialAgents ["Agentes Sociales"]
            Greeter["GreetingAgent (LangChain)"]
        end
        
        subgraph GeneratorSystem ["Sistema Generador"]
            Ontologo[Agente Ontólogo]
            Rubricador[Agente Rubricador]
            QdrantGen[("Qdrant DB")]
        end
        
        subgraph EvaluatorSystem ["Sistema Evaluador"]
            Contexto[Agente Contexto]
            Auditor[Agente Auditor]
            Documento[Agente Documento]
        end
    end

    %% Protocol Definitions
    classDef protocol fill:#f9f,stroke:#333,stroke-width:2px;
    classDef user fill:#fff,stroke:#333,stroke-width:2px;
    
    %% Flows
    User -->|Envía Mensaje| UI
    UI -->|"POST /api/chat (ChatRequest)"| API_Chat
    API_Chat --> OrchLogic
    
    OrchLogic -->|Consulta| Registry
    
    %% Decisions
    OrchLogic -->|"Intent: Saludo"| Greeter
    Greeter -->|"Respuesta Texto"| OrchLogic
    
    OrchLogic -- "Intent: Generar" --> ActionGen["Action: show_component (RubricGenerator)"]
    OrchLogic -- "Intent: Evaluar" --> ActionEval["Action: show_component (RubricEvaluator)"]
    
    %% Responses back to UI
    ActionGen -->|"JSON (Protocolo A2UI)"| UI
    ActionEval -->|"JSON (Protocolo A2UI)"| UI
    OrchLogic -->|"JSON (Texto/Error)"| UI
    
    %% Action Handling in UI
    UI -->|Renderiza| ActionHandler
    ActionHandler -.->|"Muestra Componente"| GeneratorComp["Rubric Generator UI"]
    ActionHandler -.->|"Muestra Componente"| EvaluatorComp["Rubric Evaluator UI"]

    %% Specialized Flows (Triggered by UI Components)
    GeneratorComp -->|Requests| API_Gen
    EvaluatorComp -->|Requests| API_Gen
    
    API_Gen -->|"Llamada Directa a Agente (A2A Backend)"| GeneratorSystem
    API_Gen -->|"Llamada Directa a Agente (A2A Backend)"| EvaluatorSystem
    
    Ontologo <--> QdrantGen
    Rubricador <--> QdrantGen
    
    Contexto --> QdrantGen
    Auditor --> Contexto
    Auditor --> Documento

    %% Styling
    class User user;
```

## Protocolo de Comunicación (A2A)

El sistema utiliza un protocolo definido en `a2a_protocol.py` para estandarizar la comunicación.

### Estructura del Mensaje (`AgentMessage`)

| Campo | Tipo | Descripción |
|-------|------|-------------|
| `source` | `str` | ID del agente emisor (ej. "orchestrator") |
| `target` | `str` | ID del agente receptor (ej. "user") |
| `type` | `Enum` | Tipo de mensaje (`text`, `action_request`, `error`) |
| `content` | `str` | Contenido legible para el usuario |
| `metadata` | `dict` | Datos estructurado para la UI (ej. `{"component": "RubricGenerator"}`) |

### Tipos de Mensaje Principales

1.  **`text`**: Mensaje de respuesta estándar (chat).
2.  **`action_request`**: Instrucción para que el Frontend realice una acción (ej. mostrar un componente UI específico).
3.  **`error`**: Notificación de fallo en el procesamiento.

### Flujo de Orquestación

1.  **Entrada**: El Orquestador recibe el mensaje del usuario.
2.  **Análisis**: Usa un LLM para determinar la intención y seleccionar el agente del registro.
3.  **Enrutamiento**:
    *   Si es un agente "Social" (ej. Greeter), lo invoca directamente y devuelve su respuesta.
    *   Si es una "Herramienta Compleja" (Generador/Evaluador), devuelve un `action_request` para que el Frontend active la interfaz correspondiente.
