# RubricAI — Sistema Genérico de Agentes basado en Skills

Sistema multi-agente configurable que permite cargar skills dinámicamente para procesar documentos normativos, generar rúbricas de cumplimiento y evaluar documentos. Construido con Google ADK, FastAPI, Qdrant y React.

## Arquitectura

```
┌─────────────────────────────────────────────────────────┐
│                    Frontend (React + Vite)               │
│  ┌──────────────┐  ┌──────────────┐  ┌───────────────┐  │
│  │ ChatInterface│  │ SkillManager │  │RubricEvaluator│  │
│  │              │  │              │  │RubricGenerator│  │
│  └──────┬───────┘  └──────┬───────┘  └───────┬───────┘  │
└─────────┼─────────────────┼──────────────────┼──────────┘
          │                 │                  │
          ▼                 ▼                  ▼
┌─────────────────────────────────────────────────────────┐
│                  FastAPI Server (Python)                  │
│                                                          │
│  /api/chat ──► ADK Runner ──► Root Agent                 │
│                                  │                       │
│                    ┌─────────────┼─────────────┐         │
│                    ▼             ▼             ▼         │
│              [Skill 1]    [Skill 2]    [Skill N]        │
│              (cargados dinámicamente desde skills/)       │
│                                                          │
│  /api/skills/*     Gestión de skills (.md)               │
│  /api/upload       Subida de PDFs                        │
│  /api/generate     Generación de rúbricas                │
│  /api/evaluate/*   Evaluación de documentos              │
│  /api/download/*   Descarga de archivos generados        │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
              ┌─────────────────┐
              │   Qdrant Cloud  │
              │  (Vector Store) │
              │                 │
              │  Ontologías     │
              │  Embeddings     │
              │  3072-dim       │
              └─────────────────┘
```

## Stack Tecnológico

| Componente | Tecnología |
|---|---|
| Backend | Python 3.11+, FastAPI, Uvicorn |
| Agentes | Google ADK (Agent Development Kit) |
| LLM | Google Gemini 2.5 Flash |
| Embeddings | Gemini `gemini-embedding-001` (3072 dim) |
| Vector Store | Qdrant Cloud |
| Frontend | React, Vite, TailwindCSS, Framer Motion |
| Observabilidad | LangSmith (OpenTelemetry) |

## Variables de Entorno

Crear un archivo `.env` en la raíz del proyecto:

```env
GOOGLE_API_KEY=tu_api_key_de_google
QDRANT_URL=https://tu-cluster.qdrant.io:6333
QDRANT_API_KEY=tu_api_key_de_qdrant
LANGSMITH_API_KEY=tu_api_key_de_langsmith    # opcional
LANGSMITH_PROJECT=rubricas_qdrant_system     # opcional
```

## Instalación y Ejecución

```bash
# Backend
cd rubricas
uv sync
uv run python -m app.server

# Frontend (en otra terminal)
cd rubricas/frontend
npm install
npm run dev
```

---

## Backend

### Servidor (`app/server.py`)

Servidor FastAPI que expone la API REST y gestiona el ciclo de vida del agente.

**Endpoints principales:**

| Método | Ruta | Descripción |
|---|---|---|
| `POST` | `/api/chat` | Chat principal — enruta mensajes al Root Agent |
| `POST` | `/api/upload` | Sube un PDF para procesamiento |
| `POST` | `/api/generate` | Genera rúbrica desde un PDF subido (extrae texto, envía al agente, genera .docx) |
| `GET` | `/api/download/{filename}` | Descarga archivos generados (.docx, .pdf, .txt) |
| `POST` | `/api/evaluate/upload_rubric` | Sube rúbrica para evaluación (.docx, .pdf, .txt, .md) |
| `POST` | `/api/evaluate/upload_doc` | Sube documento PDF para evaluación |
| `POST` | `/api/evaluate/run` | Ejecuta evaluación: documento vs rúbrica |
| `GET` | `/api/skills` | Lista skills cargados con metadatos |
| `POST` | `/api/skills/upload` | Sube un skill (.md), reconstruye el agente |
| `DELETE` | `/api/skills/{name}` | Elimina un skill, reconstruye el agente |
| `GET` | `/api/skills/{name}/download` | Descarga el archivo .md de un skill |
| `GET` | `/api/skills/tools` | Lista herramientas disponibles con documentación |

**Flujo de reconstrucción del agente:**
Cuando se sube o elimina un skill, el servidor reconstruye el Root Agent con los skills actualizados. Esto invalida las sesiones activas.

### Root Agent (`app/main_agent.py`)

Agente orquestador que enruta solicitudes a los skills cargados.

- Sin skills: responde conversacionalmente, explica el sistema, guía la carga de skills
- Con skills: analiza el mensaje del usuario y transfiere al skill apropiado
- La instrucción del agente se regenera dinámicamente con la descripción de cada skill cargado

### Skill Loader (`app/skill_loader.py`)

Parsea archivos `.md` con frontmatter YAML y crea agentes ADK dinámicamente.

**Formato de un skill:**

```markdown
---
name: nombre-del-skill
description: Qué hace este skill
model: gemini-2.5-flash
tools:
  - nombre_herramienta
sub_agents:
  - nombre_sub_agente
---

# Instrucciones principales del skill

Texto con las instrucciones para el agente...

## sub_agent: nombre_sub_agente

### Instrucciones
Instrucciones específicas del sub-agente...

### Tools
- nombre_herramienta
```

### Qdrant Service (`app/qdrant_service.py`)

Servicio unificado de base de datos vectorial. Gestiona embeddings, almacenamiento y búsqueda semántica.

- Usa Gemini `gemini-embedding-001` para generar embeddings de 3072 dimensiones
- Colección: `rubricas_entidades`
- Auto-detecta y recrea la colección si las dimensiones no coinciden

### Domain (`app/domain.py`)

Estructuras de datos compartidas:

- `Entidad`: nombre, tipo, contexto, propiedades
- `Relacion`: origen, destino, tipo, propiedades
- `Ontologia`: lista de entidades + relaciones + metadata
- Utilidades: `parsear_json_con_fallback`, rate limiter, LLM cache

---

## Herramientas Externas (Tool Registry)

Las herramientas son funciones Python registradas en `TOOL_REGISTRY` que los skills pueden usar declarándolas en su frontmatter YAML.

### `guardar_ontologia_en_qdrant`

Parsea un JSON de ontología y guarda entidades/relaciones en Qdrant.

- **Parámetro:** `ontologia_json` (str) — JSON con estructura `{entidades: [...], relaciones: [...]}`
- **Retorna:** Mensaje de confirmación con cantidad de entidades y relaciones guardadas
- **Comportamiento:** Limpia la colección antes de guardar (reemplaza ontología anterior)

### `buscar_contexto_qdrant`

Busca contexto normativo relevante en Qdrant por similitud semántica.

- **Parámetro:** `consulta` (str) — Texto de búsqueda
- **Retorna:** Texto formateado con entidades encontradas, scores y relaciones
- **Configuración:** limit=10, score_threshold=0.4

### `leer_rubrica_subida`

Lee el contenido de una rúbrica previamente subida por el usuario.

- **Parámetro:** `rubric_id` (str) — ID único de la rúbrica (retornado por el upload)
- **Retorna:** Texto extraído del archivo
- **Formatos soportados:** `.docx`, `.pdf`, `.txt`, `.md`

### `leer_documento_subido`

Lee y extrae texto de un documento PDF subido por el usuario.

- **Parámetro:** `document_id` (str) — ID único del documento
- **Retorna:** Texto extraído de todas las páginas del PDF

---

## Skills

Los skills se almacenan en `skills/` como directorios con un archivo `SKILL.md` dentro. Se cargan dinámicamente al iniciar el servidor o cuando se suben/eliminan via la API.

### normativa-a-rubrica

Genera rúbricas de cumplimiento a partir de documentos normativos PDF.

- **Herramientas:** `guardar_ontologia_en_qdrant`, `buscar_contexto_qdrant`
- **Sub-agentes:** `ontologo`, `rubricador`
- **Flujo:**
  1. Pide al usuario que suba un PDF normativo (muestra botón de carga con `[UI:RubricGenerator]`)
  2. El `ontologo` extrae entidades y relaciones del documento y las guarda en Qdrant
  3. El `rubricador` busca contexto en Qdrant y genera la rúbrica en formato Markdown con tablas
  4. El sistema genera un archivo `.docx` descargable

### evaluador-de-cumplimiento

Evalúa documentos contra una rúbrica de cumplimiento.

- **Herramientas:** `leer_rubrica_subida`, `leer_documento_subido`, `buscar_contexto_qdrant`
- **Flujo:**
  1. Pide al usuario que suba la rúbrica (.docx/.pdf/.txt/.md) y el documento a evaluar (.pdf) con `[UI:RubricEvaluator]`
  2. Lee ambos archivos con las herramientas
  3. Busca contexto normativo en Qdrant para enriquecer la evaluación
  4. Genera informe con: resumen ejecutivo, tabla de evaluación por criterio, contexto normativo aplicado, conclusiones
  5. Genera archivo `.docx` descargable

### asistente-de-redaccion

Orienta al usuario en la redacción de documentos normativos.

- **Herramientas:** `buscar_contexto_qdrant`, `leer_documento_subido`
- **Flujo conversacional:**
  - **Planificación:** El usuario dice qué documento quiere escribir → busca requisitos en Qdrant → sugiere estructura y puntos críticos
  - **Revisión:** El usuario pega texto o sube PDF → analiza contra normativa en Qdrant → da feedback constructivo
  - **Consultas:** Responde dudas puntuales sobre redacción técnica con ejemplos de Qdrant

---

## Frontend

Aplicación React con Vite y TailwindCSS.

### ChatInterface

Interfaz de chat principal. Envía mensajes a `/api/chat` y renderiza respuestas del agente. Detecta etiquetas `[UI:RubricGenerator]` y `[UI:RubricEvaluator]` en las respuestas para mostrar componentes interactivos inline.

### SkillManager

Panel desplegable en el header para gestionar skills:
- Lista skills cargados con nombre, descripción, herramientas y sub-agentes
- Subir nuevos skills (.md)
- Eliminar skills existentes
- Descargar skills para edición
- Sección expandible de "Herramientas Disponibles" con documentación de cada tool

### RubricGenerator

Componente de carga de archivos para generación de rúbricas:
- Upload de PDF normativo
- Selector de nivel de exigencia (Básico / Intermedio / Alta Criticidad)
- Muestra resultado en el chat y ofrece descarga .docx

### RubricEvaluator

Componente de carga de archivos para evaluación:
- Upload de rúbrica (.docx/.pdf/.txt/.md)
- Upload de documento a evaluar (.pdf)
- Muestra informe de evaluación y ofrece descarga .docx

---

## Cómo crear un nuevo Skill

1. Crear un archivo `.md` con frontmatter YAML:

```markdown
---
name: mi-nuevo-skill
description: Descripción de lo que hace
model: gemini-2.5-flash
tools:
  - buscar_contexto_qdrant
---

# Instrucciones para el agente

Describe aquí el comportamiento del agente...
```

2. Subirlo desde el panel de Skills en la interfaz o via API:

```bash
curl -X POST http://localhost:8000/api/skills/upload \
  -F "file=@mi-skill.md"
```

3. El sistema reconstruye el agente automáticamente con el nuevo skill disponible.

**Herramientas disponibles para usar en skills:**
- `guardar_ontologia_en_qdrant` — Guardar ontología extraída
- `buscar_contexto_qdrant` — Buscar contexto normativo
- `leer_rubrica_subida` — Leer rúbrica subida por el usuario
- `leer_documento_subido` — Leer documento PDF subido
