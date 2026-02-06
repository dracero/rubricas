# RubricAI - Sistema de Generación y Evaluación de Rúbricas (A2A/A2UI)

Este sistema implementa una arquitectura basada en agentes (A2A) con una interfaz de usuario generativa (A2UI). Permite generar rúbricas académicas a partir de normativas PDF y evaluar trabajos estudiantiles utilizando estas rúbricas y tecnología RAG (Retrieval-Augmented Generation).

## 🧠 Arquitectura del Sistema

El sistema consta de tres componentes principales:

1.  **Frontend (A2UI)**: Una aplicación React/Vite que actúa como cliente del protocolo A2A. No tiene lógica de negocio dura; renderiza la interfaz basándose en las solicitudes de acción (`ACTION_REQUEST`) del orquestador.
2.  **Backend (A2A)**: Un servidor FastAPI que aloja varios agentes inteligentes:
    *   **Orquestador (`server.py`)**: Recibe mensajes del usuario, decide qué agente debe atenderlos y envía instrucciones al frontend.
    *   **Generador (`rubricas_qdrant_local.py`)**: Crea rúbricas académicas procesando documentos normativos.
    *   **Evaluador (`rubricador_qdrant_local.py`)**: Audita apuntes o trabajos contra una rúbrica.
    *   **Base de Datos Vectorial**: Qdrant (para RAG y contexto).
    *   **LLM**: Google Gemini 2.5 Flash.

## 📋 Prerrequisitos

*   **Python 3.12+** (Gestionado con `uv` preferiblemente)
*   **Node.js 18+** y `npm`
*   **Clave de API de Google Gemini**
*   **Instancia de Qdrant** (URL y API Key)

## 🚀 Instalación Paso a Paso

### 1. Clonar y Preparar el Entorno

```bash
# Clonar repositorio (si aplica)
# cd rubricas-app
```

### 2. Configurar Variables de Entorno

Crea un archivo `.env` en la raíz del proyecto (puedes copiar `.env.example`):

```env
GOOGLE_API_KEY="tu_clave_de_gemini"
QDRANT_URL="https://tu-cluster.qdrant.tech"
QDRANT_API_KEY="tu_clave_de_qdrant"

# Opcional: LangSmith para observabilidad
LANGSMITH_API_KEY="tu_clave_langsmith"
```

### 3. Instalar Dependencias

Desde la raíz del proyecto, ejecuta el comando unificado:

```bash
npm run install:all
```

> **Nota**: Esto instalará las dependencias de Python (via `uv`) y las dependencias de Node.js en la carpeta `frontend/`.

## ▶️ Ejecución

Para iniciar todo el sistema (Backend + Frontend) con un solo comando:

```bash
npm run dev
```

*   **Frontend**: http://localhost:5173
*   **Backend**: http://localhost:8000
*   **Documentación API**: http://localhost:8000/docs

## 📖 Uso del Sistema

1.  **Chat Orquestador**: Al abrir la aplicación, verás una interfaz de chat.
    *   Escribe: *"Quiero crear una rúbrica"* o *"Generar evaluación"*.
    *   El orquestador analizará tu intención y desplegará el componente correspondiente.

2.  **Generación de Rúbricas**:
    *   Sube un archivo PDF con la normativa (ej: "Reglamento de Tesis").
    *   Selecciona el nivel educativo (Primer año, Avanzado, Posgrado).
    *   El sistema extraerá la ontología, la guardará en Qdrant y generará una rúbrica Markdown descargable.

3.  **Evaluación de Apuntes**:
    *   Sube la rúbrica generada anteriormente (archivo `.txt` o `.md`).
    *   Sube el documento del estudiante (PDF).
    *   El agente "Auditor" leerá ambos, buscará contexto relevante en Qdrant y generará un informe de evaluación detallado.

## 🛠️ Desarrollo

*   **Backend**: El código está en `server.py` y los módulos `rubricas_*.py`. Usa `uv run uvicorn server:app --reload` para correr solo el backend.
*   **Frontend**: El código React está en `frontend/src`. Usa `cd frontend && npm run dev` para correr solo el frontend.
*   **Protocolo**: Las definiciones de comunicación están en `a2a_protocol.py`.

## 📦 Estructura de Archivos Clave

*   `server.py`: Punto de entrada del API y lógica del Orquestador.
*   `a2a_protocol.py`: Definiciones de tipos de mensajes (Text, ActionRequest).
*   `rubricas_qdrant_local.py`: Lógica del agente Generador.
*   `rubricador_qdrant_local.py`: Lógica del agente Evaluador.
*   `frontend/src/components/ChatInterface.jsx`: Cliente del protocolo A2A.
