# RubricAI - Sistema Multi-Agente de Rúbricas (A2A)

Este sistema implementa una arquitectura **Multi-Agente (A2A)** orquestada por **BeeAI Router**, diseñada para la generación y evaluación de rúbricas académicas utilizando Inteligencia Artificial Generativa y RAG (Retrieval-Augmented Generation).

## 🧠 Arquitectura del Sistema

El sistema se compone de un Orquestador central y varios Agentes especializados que se comunican a través del protocolo **A2A (Agent-to-Agent)** sobre HTTP.

### Diagrama de Arquitectura

```mermaid
graph TD
    %% Estilos
    classDef user fill:#f9f,stroke:#333,stroke-width:2px;
    classDef frontend fill:#e1f5fe,stroke:#01579b,stroke-width:2px;
    classDef orchestrator fill:#fff9c4,stroke:#fbc02d,stroke-width:2px;
    classDef agent fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px;
    classDef subagent fill:#f3e5f5,stroke:#7b1fa2,stroke-width:1px,stroke-dasharray: 5 5;
    classDef db fill:#e0e0e0,stroke:#616161,stroke-width:2px;

    User((👤 Usuario)):::user
    Frontend["💻 Frontend A2UI (React + Vite)"]:::frontend
    
    subgraph Host ["🌐 Host (Orquestador)"]
        Router["🔀 SimpleRouter (Orquestador Central)"]:::orchestrator
        A2A_Client["📡 Remote Agent Connection (Cliente A2A)"]:::orchestrator
    end

    subgraph Agentes ["🤖 Red de Agentes (A2A Servers)"]
        direction TB
        
        subgraph GreeterPod ["👋 Greeter Agent"]
            Greeter["Greeter (LangGraph)"]:::agent
        end

        subgraph GeneratorPod ["📝 Generator Agent (Google ADK)"]
            GenRoot[Orquestador Generador]:::agent
            Ontologo[Ontólogo]:::subagent
            Rubricador[Rubricador]:::subagent
            GenRoot --> Ontologo
            GenRoot --> Rubricador
        end

        subgraph EvaluatorPod ["⚖️ Evaluator Agent (Google ADK)"]
            EvalRoot[Evaluador]:::agent
        end
    end

    subgraph Storage ["💾 Persistencia"]
        Qdrant["Qdrant Vector DB"]:::db
    end

    %% Conexiones
    User <-->|Chat / Acciones| Frontend
    Frontend <-->|API / JSON| Router
    Router -->|Ruteo Inteligente| A2A_Client
    
    %% Conexiones A2A (HTTP / JSON-RPC)
    A2A_Client <-->|A2A Protocol : message/send| Greeter
    A2A_Client <-->|A2A Protocol : message/send| GenRoot
    A2A_Client <-->|A2A Protocol : message/send| EvalRoot

    %% Conexiones a Datos
    Ontologo -->|Guarda Ontología| Qdrant
    Rubricador -->|Lee Contexto| Qdrant
    EvalRoot -->|Lee Contexto| Qdrant
```

## 🤖 Descripción de los Agentes

### 1. 🔀 SimpleRouter (Orquestador)
*   **Tecnología**: Groq LLM (llama-3.3-70b-versatile) para clasificación de intenciones.
*   **Rol**: Es el cerebro central del sistema. No realiza tareas por sí mismo, sino que analiza la intención del usuario y "enruta" la solicitud al agente especializado correspondiente.
*   **Funcionamiento**: Utiliza un modelo LLM con temperatura baja para clasificar la consulta del usuario en una de tres categorías (generator/evaluator/greeter) y delega al agente apropiado.

### 2. 👋 Greeter Agent (Bienvenida)
*   **Tecnología**: [LangGraph](https://langchain-ai.github.io/langgraph/).
*   **Puerto**: `10003`
*   **Rol**: Agente conversacional ligero encargado de dar la bienvenida, explicar el propósito del sistema y guiar al usuario en sus primeros pasos.
*   **Personalidad**: Amigable, entusiasta y servicial.

### 3. 📝 Generator Agent (Generador de Rúbricas)
*   **Tecnología**: [Google ADK (Agent Development Kit)](https://github.com/google/generative-ai-python).
*   **Puerto**: `10001`
*   **Rol**: Genera instrumentos de evaluación complejos basándose en normativas.
*   **Sub-Agentes**:
    *   **Ontólogo**: Analiza documentos normativos (PDFs), extrae entidades y relaciones semánticas, y las guarda en Qdrant.
    *   **Rubricador**: Consulta la base de conocimiento (Qdrant) para recuperar el contexto normativo y redacta la rúbrica detallada en Markdown.

### 4. ⚖️ Evaluator Agent (Evaluador)
*   **Tecnología**: Google ADK.
*   **Puerto**: `10002`
*   **Rol**: Realiza auditorías académicas. Compara un trabajo entregado por un estudiante contra una rúbrica específica y el contexto institucional.
*   **Capacidades**:
    *   Lectura de documentos (PDF).
    *   Búsqueda de contexto normativo en Qdrant (`buscar_contexto_para_evaluacion`).
    *   Generación de informes de retroalimentación constructiva.

## 📡 Protocolo A2A (Agent-to-Agent)

El sistema utiliza un protocolo de comunicación estandarizado basado en **JSON-RPC 2.0** sobre HTTP.

*   **Discovery**: El orquestador descubre las capacidades de los agentes consultando el endpoint `/.well-known/agent.json` de cada servicio.
*   **Mensajería**: Las interacciones se envían mediante el método `message/send`.
    ```json
    {
      "jsonrpc": "2.0",
      "method": "message/send",
      "params": {
        "message": {
          "role": "user",
          "parts": [{"type": "text", "text": "Hola"}],
          "contextId": "..."
        }
      },
      "id": "..."
    }
    ```

## 🛠️ Tecnologías Clave

*   **Backend**: Python, FastAPI/Starlette, `uv`.
*   **Frontend**: React, Vite, TailwindCSS.
*   **IA / LLM**: Google Gemini 2.5 Flash.
*   **Base de Datos Vectorial**: Qdrant (para almacenamiento de ontologías y RAG).
*   **Frameworks de Agentes**: BeeAI (IBM), LangGraph, Google ADK.
