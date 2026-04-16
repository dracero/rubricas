# Arquitectura de RubricAI

## 1. Visión General

RubricAI es un sistema de generación y evaluación de rúbricas de cumplimiento normativo basado en Inteligencia Artificial. Combina un orquestador multi-agente (Google ADK), RAG sobre una base vectorial (Qdrant) y una interfaz web reactiva (React + Vite) para guiar al usuario desde la ingesta de documentos normativos hasta la obtención de rúbricas exportables en DOCX.

---

## 2. Mapa de Componentes de Alto Nivel

```
┌────────────────────────────────────────────────┐
│              Navegador (React SPA)             │
│  ChatInterface · RubricGenerator · Evaluator   │
│  SkillManager · MultiUpload · SuggestionPanel  │
└───────────────────┬────────────────────────────┘
                    │ HTTP (proxy Vite → :8000)
┌───────────────────▼────────────────────────────┐
│          FastAPI Server  (:8000)               │
│  REST API · ADK Runner · Session Service       │
│                                                │
│  ┌─────────────────────────────────────────┐   │
│  │      Orquestador ADK (Root Agent)       │   │
│  │  ┌──────────┐ ┌──────────┐ ┌─────────┐  │   │
│  │  │normativa │ │evaluador │ │asistente│  │   │
│  │  │-a-rubrica│ │-cumpli.  │ │-redacc. │  │   │
│  │  └──────────┘ └──────────┘ └─────────┘  │   │
│  └──────────────────┬──────────────────────┘   │
│                     │ Tools                    │
│  ┌──────────────────▼──────────────────────┐   │
│  │           Tool Registry                 │   │
│  │  leer_rubrica · leer_documento          │   │
│  │  buscar_contexto_qdrant                 │   │
│  │  guardar_ontologia_en_qdrant            │   │
│  └──────────────────┬──────────────────────┘   │
└─────────────────────┼──────────────────────────┘
                      │
        ┌─────────────▼──────────────┐
        │    Qdrant Vector DB        │
        │  rubricas_entidades        │
        │  rubricas_repositorio      │
        └────────────────────────────┘
```

---

## 3. Capas de la Solución

### 3.1 Frontend (React + Vite)

**Ubicación:** `frontend/`  
**Puerto dev:** `http://localhost:5173`  
**Proxy:** todas las rutas `/api/*` se redirigen a `http://localhost:8000` via `vite.config.js`.

> **IMPORTANTE — Internacionalización (i18n):**
> Un aspecto esencial de la arquitectura de la aplicación es que **debe soportar 4 idiomas en su interfaz**:
> - Español
> - Gallego
> - Portugués (BR)
> - Inglés
>
> 1. **Control de idioma universal:** En todas las páginas debe haber un control que permita seleccionar el idioma.
> 2. **Ubicación (Autenticado):** Como una opción junto a la opción de "cerrar sesión" en el ícono del usuario.
> 3. **Ubicación (No Autenticado / Login):** Se colocará bajo el formulario de inicio de sesión.
> 4. **Traducción integral:** Todos los strings que se desplieguen en el frontend deberán estar localizados en los 4 idiomas listados.
> 5. **Diccionarios editables:** Todos los strings que utiliza la aplicación deberán estar agrupados e indexados en un archivo de texto o módulo centralizado que permita ser editado fácilmente para corregir problemas de redacción, gramática y ortografía, separando el contenido del código de UI.
> 6. **Idioma Predefinido Institucional:** Si en las variables de entorno se establece una determinada institución, el idioma predeterminado de la landing debe acomodarse automáticamente según el origen (ej. UDC → gl, UFRJ → pt).

> **IMPORTANTE — Tematización Dinámica Institucional:**
> El frontend recibe en tiempo de construcción la variable `INSTITUCION` desde `.env` usando `vite.config.js`.
> 1. **Fondo Global (GlobalBackground):** Con base en el nombre de la institución (ej: `UCHILE`), el sistema resolverá y cargará un fondo fotográfico dinámico (`{institucion}.(jpg|png)`) aplicando un gradiente vertical tipo "overlay" de opacidad 100 hasta transparencia 50 (usando clases de Tailwind como `bg-gradient-to-r from-slate-900 to-slate-900/50`) para asegurar el contraste de la interfaz frontal.
> 2. **Logo de la Institución:** El logo respectivo (`logo_{institucion}.(png|jpg)`) se ubica dinámicamente como un elemento **flotante en la esquina superior izquierda** tanto en la vista inicial de autenticación como en el marco principal detrás de un difuminado (backdrop-blur).
> 3. **Logo de la Aplicación Fijo:** La tarjeta central de Login mostrará un asset *inmutable* correspondiente a la aplicación misma (`logo_app.png`), estáticamente empaquetado, sin depender de las variaciones institucionales dinámicas.

| Componente | Responsabilidad |
|---|---|
| `ChatInterface.jsx` | Conversación con el orquestador ADK vía `POST /api/chat` |
| `RubricGenerator.jsx` | Subida de PDF normativo y solicitud de generación `POST /api/generate` |
| `RubricEvaluator.jsx` | Subida de rúbrica + documento para evaluación |
| `SkillManager.jsx` | Listar, subir y eliminar skill `.md` en caliente |
| `MultiUploadPanel.jsx` | Carga masiva de PDFs normativos con seguimiento de lote |
| `ExtractionProgress.jsx` | Polling de estado de extracción por `batch_id` |
| `SuggestionPanel.jsx` | Visualización de rúbricas similares desde repositorio |
| `MarkdownTable.jsx` | Renderizado enriquecido de tablas Markdown en el chat |
| `ReferencePrompt.jsx` | Prompt de referencias bibliográficas del lote |

**UI Tags**: los agentes incluyen etiquetas especiales en sus respuestas que activan componentes específicos de la UI:
- `[UI:RubricGenerator]` → abre panel de subida de PDF normativo
- `[UI:RubricEvaluator]` → abre panel de evaluación de documentos

### 3.2 Backend — Servidor (FastAPI)

**Ubicación:** `app/server.py`  
**Puerto:** `http://localhost:8000`  
**Arranque:** `python -m app.server` (o via `uv run`)

El servidor inicializa en su `lifespan`:
1. Configura LangSmith/OpenTelemetry.
2. Crea el agente raíz con las skills disponibles.
3. Instancia `InMemorySessionService` y `ADK Runner`.

#### API REST

| Método | Ruta | Descripción |
|---|---|---|
| `GET` | `/` | Health check |
| `POST` | `/api/chat` | Mensaje al orquestador → respuesta del agente |
| `POST` | `/api/upload` | Subida de un único PDF para evaluación |
| `POST` | `/api/upload/batch` | Carga masiva + extracción de ontología en background |
| `GET` | `/api/upload/status/{batch_id}` | Estado del lote de extracción |
| `POST` | `/api/generate` | Generación de rúbrica con RAG |
| `GET` | `/api/download/{filename}` | Descarga DOCX generado |
| `GET` | `/api/rubrics` | Listar repositorio de rúbricas (paginado + búsqueda semántica) |
| `GET` | `/api/rubrics/{rubric_id}` | Detalle de una rúbrica |
| `DELETE` | `/api/rubrics/{rubric_id}` | Eliminar rúbrica del repositorio |
| `POST` | `/api/evaluate/upload_rubric` | Subir rúbrica para evaluación |
| `POST` | `/api/evaluate/upload_doc` | Subir documento a evaluar |
| `POST` | `/api/evaluate/run` | Ejecutar evaluación de cumplimiento vía agente |
| `GET` | `/api/skills/tools` | Listar herramientas disponibles en el Tool Registry |
| `GET` | `/api/skills` | Listar skills cargadas |
| `POST` | `/api/skills/upload` | Subir nueva skill `.md` en caliente |
| `DELETE` | `/api/skills/{skill_name}` | Eliminar skill |
| `GET` | `/api/skills/{skill_name}/download` | Descargar el `.md` de una skill |

### 3.3 Capa de Agentes (Google ADK)

**Ubicación:** `app/main_agent.py`, `app/skill_loader.py`

El sistema usa el **Agent Development Kit (ADK)** de Google para organizar una jerarquía de agentes:

```
rubricai_orchestrator  (Root Agent — LiteLlm/gpt-4o-mini)
├── normativa_a_rubrica        (L1 agent)
│   ├── ontologo               (L2 sub-agent)
│   └── rubricador             (L2 sub-agent)
├── evaluador_de_cumplimiento  (L1 agent)
└── asistente_de_redaccion     (L1 agent)
```

#### Orquestador (Root Agent)
- Analiza el mensaje e intención del usuario.
- Transfiere la conversación al skill más adecuado según la solicitud.
- No ejecuta tareas por sí mismo; delega siempre.

#### Carga dinámica de skills (`skill_loader.py`)
Cada skill reside en `skills/<nombre>/SKILL.md` con **YAML frontmatter + cuerpo Markdown**:

```yaml
---
name: nombre-del-skill
description: Descripción del agente
model: openai/gpt-4o-mini
tools:
  - nombre_de_herramienta
sub_agents:
  - sub_agente_opcional
---
# Instrucciones del agente en Markdown
```

El `skill_loader` usa `google.adk.skills.list_skills_in_dir` y crea instancias `Agent` en tiempo de carga. Las skills se pueden agregar o eliminar **sin reiniciar el servidor** a través de la API `/api/skills/upload`.

#### Skills disponibles

| Skill | Modelo | Tools | Sub-agentes |
|---|---|---|---|
| `normativa-a-rubrica` | gpt-4o-mini | `guardar_ontologia_en_qdrant`, `buscar_contexto_qdrant` | `ontologo`, `rubricador` |
| `evaluador-de-cumplimiento` | gpt-4o-mini | `leer_rubrica_subida`, `leer_documento_subido`, `buscar_contexto_qdrant` | — |
| `asistente-de-redaccion` | gpt-4o-mini | `buscar_contexto_qdrant`, `leer_documento_subido` | — |

### 3.4 Tool Registry (`app/qdrant_service.py`)

Punto central de herramientas que los agentes pueden invocar. Cada herramienta es una función Python registrada en `TOOL_REGISTRY`:

| Herramienta | Descripción |
|---|---|
| `leer_rubrica_subida(rubric_id)` | Lee el texto de una rúbrica cargada temporalmente |
| `leer_documento_subido(document_id)` | Extrae texto PDF/DOCX de un documento subido |
| `buscar_contexto_qdrant(consulta)` | Búsqueda vectorial RAG en la colección `rubricas_entidades` |
| `guardar_ontologia_en_qdrant(ontologia_json)` | Guarda entidades y relaciones en Qdrant con embeddings |

### 3.5 Capa de Persistencia Vectorial (Qdrant)

**Proveedor:** Qdrant (cloud o self-hosted)  
**Modelo de embeddings:** `text-embedding-3-small` (OpenAI via LiteLLM) — 1536 dimensiones  
**Distancia:** Coseno

| Colección | Propósito |
|---|---|
| `rubricas_entidades` | Ontología extraída de documentos normativos (entidades + relaciones) |
| `rubricas_repositorio` | Rúbricas generadas persistidas con metadata y vectores de búsqueda |

### 3.6 Módulos de Soporte

| Módulo | Descripción |
|---|---|
| `common/config.py` | Carga de `.env`, configuración de LangSmith/OpenTelemetry, rate limiter |
| `app/ontology_extractor.py` | Extracción de ontología directa vía LiteLLM (sin ADK) para carga masiva |
| `app/batch_manager.py` | Gestión en memoria del estado de lotes de extracción multi-documento |
| `app/rubric_repository.py` | CRUD de rúbricas sobre Qdrant (`rubricas_repositorio`) |
| `app/docx_converter.py` | Conversión de Markdown a DOCX y extracción de texto |
| `app/domain.py` | Estructuras de datos (`Entidad`, `Relacion`, `Ontologia`), rate limiter, LLM cache |
| `app/models.py` | Modelos Pydantic de respuesta para la API REST |

---

## 4. Flujos Principales

### 4.1 Generación de Rúbrica desde PDF Normativo

```
Usuario sube PDF
      │
      ▼
POST /api/upload/batch
      │
      ├─► OntologyExtractor.extract(text)
      │       └─ LiteLLM (gpt-4o-mini): extrae entidades + relaciones
      │
      ├─► QdrantService.save_ontology_additive()
      │       └─ Embeddings → Qdrant (rubricas_entidades)
      │
      ▼
POST /api/generate  (o vía chat → normativa_a_rubrica skill)
      │
      ├─► buscar_contexto_qdrant(prompt)
      │       └─ Qdrant semantic search
      │
      ├─► LLM genera rúbrica Markdown
      │
      ├─► docx_converter.md_to_docx()
      │
      ├─► RubricRepositoryService.store_rubric()
      │       └─ Embedding + upsert en rubricas_repositorio
      │
      └─► Responde con URL de descarga DOCX + rúbricas similares
```

### 4.2 Evaluación de Documento vs Rúbrica

```
Usuario sube rúbrica + documento
      │
      ├─► POST /api/evaluate/upload_rubric  →  rubric_id
      ├─► POST /api/evaluate/upload_doc     →  doc_id
      │
      ▼
POST /api/evaluate/run  (o vía chat → evaluador_de_cumplimiento skill)
      │
      ├─► leer_rubrica_subida(rubric_id)
      ├─► leer_documento_subido(doc_id)
      ├─► (opcional) buscar_contexto_qdrant(normativas_referenciadas)
      │
      └─► LLM genera informe de cumplimiento en Markdown con tabla
```

### 4.3 Chat con Orquestador

```
Usuario escribe mensaje
      │
      ▼
POST /api/chat  {message, session_id}
      │
      ▼
ADK Runner.run_async()
      │
      ├─► rubricai_orchestrator analiza intención
      │
      ├─[generar rúbrica]──► normativa_a_rubrica
      │       ├─► ontologo  (L2)
      │       └─► rubricador (L2)
      │
      ├─[evaluar documento]──► evaluador_de_cumplimiento
      │
      ├─[ayuda redacción]──► asistente_de_redaccion
      │
      └─[conversación general]──► responde directamente
```

---

## 5. Modelo de Datos

### Ontología (Qdrant: `rubricas_entidades`)

```json
{
  "nombre": "requisito_calidad",
  "tipo": "requisito",
  "contexto": "Descripción normalizada del concepto",
  "propiedades": {},
  "relaciones_salientes": [
    {"origen": "requisito_calidad", "destino": "criterio_x", "tipo": "REQUIERE"}
  ],
  "source_document_id": "uuid",
  "source_filename": "norma.pdf"
}
```

### Repositorio de Rúbricas (Qdrant: `rubricas_repositorio`)

```json
{
  "rubric_id": "uuid",
  "rubric_text": "# Rúbrica...",
  "summary": "Primeros 300 caracteres",
  "level": "avanzado",
  "source_filenames": ["norma.pdf"],
  "source_document_ids": ["uuid"],
  "created_at": "2026-04-14T..."
}
```

---

## 6. Stack Tecnológico

| Capa | Tecnología | Versión mínima |
|---|---|---|
| LLM principal | OpenAI GPT-4o-mini (via LiteLLM) | — |
| Agent SDK | Google ADK | 0.1.0 |
| LLM alternativo | Google Gemini 2.5 Flash | — |
| Embeddings | `text-embedding-3-small` (OpenAI) | 1536d |
| Backend | FastAPI + Uvicorn | 0.123 / 0.40 |
| Base vectorial | Qdrant | 1.7.0 |
| Frontend | React 19 + Vite 7 | — |
| Estilos | TailwindCSS 3 + Framer Motion | — |
| Observabilidad | LangSmith + OpenTelemetry | 0.4.26 |
| Python | CPython | 3.11+ |

---

## 7. Configuración (Variables de Entorno)

El archivo `.env` debe residir en la raíz del proyecto. Las variables con (*) son obligatorias:

| Variable | Descripción | Default |
|---|---|---|
| `GOOGLE_API_KEY` * | API key de Google Gemini | — |
| `QDRANT_URL` * | URL del cluster Qdrant | — |
| `QDRANT_API_KEY` * | API key de Qdrant | — |
| `OPENAI_API_KEY` | API key de OpenAI (LiteLLM/embeddings) | — |

---

## 10. Autenticación y Autorización

### 10.1 Descripción General

RubricAI implementa un sistema de autenticación multi-proveedor controlado por la variable `AUTH_MODE`. Todos los endpoints `/api/*` requieren un JWT válido en el header `Authorization: Bearer <token>`.

### 10.2 Flujo de Autenticación

```
                    ┌─────────────┐
                    │  LoginPage  │  (React, fetch /auth/mode)
                    └──────┬──────┘
           ┌───────────────┼────────────────┐
           │               │                │
    OAuth click      Local form       OAuth click
    (Google/MS/       POST              (UChile)
     UChile)      /auth/login/local
           │               │                │
    GET /auth/login/{prov} │        GET /auth/login/uchile
           │               │                │
    Redirect → Provider    │        Redirect → UChile IdP
           │               │                │
    GET /auth/callback/{prov}  ←────────────┘
           │
    JWT emitido → RedirectResponse(FRONTEND_URL/?token=JWT)
           │
    AuthContext lee ?token=, guarda en localStorage
           │
    Todas las llamadas /api/* → Authorization: Bearer JWT (patchFetch)
```

### 10.3 Proveedores

| Proveedor | `AUTH_MODE` | Protocolo | Variables requeridas |
|---|---|---|---|
| Google | `GOOGLE` | OIDC / OAuth2 | `GOOGLE_CLIENT_ID`, `GOOGLE_CLIENT_SECRET` |
| Microsoft | `MICROSOFT` | OIDC (tenant) | `MICROSOFT_CLIENT_ID`, `MICROSOFT_CLIENT_SECRET`, `MICROSOFT_TENANT_ID` |
| UChile | `OAUTH2` | OAuth2 genérico | `UCHILE_CLIENT_ID`, `UCHILE_CLIENT_SECRET`, `UCHILE_AUTH_URL`, `UCHILE_TOKEN_URL`, `UCHILE_USERINFO_URL` |
| Local (DB) | `LOCAL` | email + bcrypt | — |
| Todos | `` (vacío) | — | Todos los anteriores |

### 10.4 Nuevos Endpoints

| Método | Ruta | Descripción |
|---|---|---|
| GET | `/auth/mode` | Retorna `{"mode": AUTH_MODE}` para que el frontend configure la UI |
| POST | `/auth/login/local` | Login local: `{email, password}` → `TokenResponse` |
| GET | `/auth/login/{provider}` | Inicia flujo OAuth (redirect al proveedor) |
| GET | `/auth/callback/{provider}` | Callback OAuth, emite JWT, redirige al frontend |
| GET | `/auth/me` | Retorna usuario actual (requiere Bearer JWT) |

### 10.5 Módulos Creados

| Archivo | Responsabilidad |
|---|---|
| `app/auth/models.py` | Pydantic: `UserOut`, `TokenResponse`, `LoginRequest` |
| `app/auth/db.py` | Pool asyncpg (Cloud SQL o local), CRUD de usuarios |
| `app/auth/service.py` | JWT (python-jose), bcrypt (passlib), dep `get_current_user` |
| `app/auth/router.py` | Todos los endpoints `/auth/*` |
| `app/auth/middleware.py` | HTTP middleware FastAPI — protege `/api/*` |

### 10.6 Base de Datos de Usuarios

Soporta dos modos controlados por `DB_TYPE`:

- **`cloudsql`**: Conecta a Cloud SQL (PostgreSQL) en `peppy-ridge-493316-s3:southamerica-west1:asistiag` usando `cloud-sql-python-connector` con asyncpg. Requiere Application Default Credentials (ADC) o cuenta de servicio.
- **`local`**: PostgreSQL local/Docker vía `LOCAL_DB_HOST/PORT/USER/PASS/NAME`.

Tabla única `users`:

```sql
CREATE TABLE IF NOT EXISTS users (
  id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  email         TEXT UNIQUE NOT NULL,
  name          TEXT,
  provider      TEXT NOT NULL,   -- 'local' | 'google' | 'microsoft' | 'uchile'
  hashed_password TEXT,
  is_active     BOOLEAN DEFAULT TRUE,
  created_at    TIMESTAMPTZ DEFAULT NOW()
);
```

### 10.7 Variables de Entorno de Autenticación

| Variable | Descripción | Default |
|---|---|---|
| `AUTH_MODE` | Proveedor activo (`GOOGLE`/`MICROSOFT`/`OAUTH2`/`LOCAL`/vacío) | `` |
| `SECRET_KEY` | Clave HMAC para JWT y SessionMiddleware | — |
| `ACCESS_TOKEN_EXPIRE_MINUTES` | TTL del JWT en minutos | `480` |
| `FRONTEND_URL` | URL del frontend (redirect post-OAuth) | `http://localhost:5173` |
| `GOOGLE_CLIENT_ID` | OAuth2 Client ID de Google | — |
| `GOOGLE_CLIENT_SECRET` | OAuth2 Client Secret de Google | — |
| `MICROSOFT_CLIENT_ID` | OAuth2 Client ID de Microsoft | — |
| `MICROSOFT_CLIENT_SECRET` | OAuth2 Client Secret de Microsoft | — |
| `MICROSOFT_TENANT_ID` | Tenant de Azure AD | `common` |
| `UCHILE_CLIENT_ID` | OAuth2 Client ID de UChile | — |
| `UCHILE_CLIENT_SECRET` | OAuth2 Client Secret de UChile | — |
| `UCHILE_AUTH_URL` | Endpoint de autorización UChile | — |
| `UCHILE_TOKEN_URL` | Endpoint de token UChile | — |
| `UCHILE_USERINFO_URL` | Endpoint userinfo UChile | — |
| `DB_TYPE` | Tipo de BD: `cloudsql` o `local` | `local` |
| `CLOUDSQL_INSTANCE` | Connection name de Cloud SQL | — |
| `CLOUDSQL_DB_USER` | Usuario Cloud SQL | — |
| `CLOUDSQL_DB_PASS` | Contraseña Cloud SQL | — |
| `CLOUDSQL_DB_NAME` | Nombre de la base de datos | `rubricai_auth` |
| `LOCAL_DB_HOST` | Host PostgreSQL local | `localhost` |
| `LOCAL_DB_PORT` | Puerto PostgreSQL local | `5432` |
| `LOCAL_DB_USER` | Usuario PostgreSQL local | `postgres` |
| `LOCAL_DB_PASS` | Contraseña PostgreSQL local | — |
| `LOCAL_DB_NAME` | Nombre base de datos local | `rubricai_auth` |

### 10.8 Frontend

| Archivo | Responsabilidad |
|---|---|
| `frontend/src/contexts/AuthContext.jsx` | Estado global de auth, `patchFetch` para inyectar Bearer en `/api/*`, lectura de `?token=` post-OAuth |
| `frontend/src/pages/LoginPage.jsx` | UI de login: consulta `/auth/mode` y muestra los proveedores disponibles |
| `frontend/src/App.jsx` | `<AuthProvider>` wrapper + guard: si `!user` → `<LoginPage />`, si `user` → app principal con botón logout |
| `LANGSMITH_API_KEY` | API key de LangSmith (trazabilidad) | — |
| `LANGSMITH_PROJECT` | Nombre del proyecto LangSmith | `rubricas_qdrant_system` |
| `ORCHESTRATOR_HOST` | Host del servidor backend | `localhost` |
| `ORCHESTRATOR_PORT` | Puerto del servidor backend | `8000` |

---

## 8. Estructura de Directorios

```
rubricas/
├── app/
│   ├── server.py              # FastAPI + endpoints REST
│   ├── main_agent.py          # Root ADK Agent factory
│   ├── skill_loader.py        # Carga dinámica de skills
│   ├── qdrant_service.py      # Vector DB + Tool Registry
│   ├── ontology_extractor.py  # Extracción de ontología vía LLM
│   ├── rubric_repository.py   # CRUD de rúbricas en Qdrant
│   ├── batch_manager.py       # Estado de lotes en memoria
│   ├── docx_converter.py      # Markdown → DOCX, extracción DOCX
│   ├── domain.py              # Entidades, utilidades, caché
│   └── models.py              # Modelos Pydantic de API
├── common/
│   └── config.py              # Configuración, LangSmith, .env loader
├── skills/
│   ├── normativa-a-rubrica/SKILL.md
│   ├── evaluador-de-cumplimiento/SKILL.md
│   ├── asistente-de-redaccion/SKILL.md
│   └── repositorio-de-rubricas/SKILL.md
├── frontend/
│   ├── src/
│   │   ├── components/        # Componentes React
│   │   ├── App.jsx
│   │   └── main.jsx
│   ├── vite.config.js         # Proxy /api → :8000
│   └── package.json
├── docs/
│   └── ARCHITECTURE.md        # Este documento
├── .env                       # Variables de entorno (no versionado)
├── pyproject.toml             # Dependencias Python
└── package.json               # Scripts npm (dev:back + dev:front)
```

---

## 9. Arranque del Sistema

### Desarrollo (ambos servicios)

```bash
# Desde la raíz del proyecto
npm run dev
# Equivalente a:
#   uv run python -m app.server    → backend :8000
#   cd frontend && npm run dev     → frontend :5173
```

### Solo backend (con depurador VS Code)
Usar la configuración **"Backend: FastAPI (app.server)"** en `.vscode/launch.json`.

### Solo frontend
```bash
cd frontend && npm run dev
```
