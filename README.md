# RubricAI

Sistema de generación, evaluación y corrección de rúbricas de cumplimiento normativo usando Inteligencia Artificial (Google Gemini + GPT-4o-mini) y RAG sobre una base vectorial (Qdrant).

---

## Tabla de Contenidos

- [Arquitectura](#arquitectura)
- [Requisitos](#requisitos)
- [Instalación](#instalación)
- [Configuración](#configuración)
- [Primer Arranque y Setup Wizard](#primer-arranque-y-setup-wizard)
- [Uso](#uso)
- [Roles y Permisos](#roles-y-permisos)
- [Skills (Agentes)](#skills-agentes)
- [Estructura del Proyecto](#estructura-del-proyecto)
- [API REST](#api-rest)

---

## Arquitectura

El sistema corre en un único servidor FastAPI que usa el **Google ADK (Agent Development Kit)** para gestionar un agente orquestador y sus skills especializadas. El frontend es una SPA en React + Vite que se comunica con el backend vía proxy.

```
Navegador (React + Vite :5173)
        │  HTTP proxy /api/* → :8000
        ▼
FastAPI Server (:8000)
  ├── ADK Runner
  │     └── Orquestador
  │           ├── normativa-a-rubrica   (genera rúbricas desde PDFs)
  │           ├── evaluador-de-cumplimiento  (verifica documentos)
  │           ├── asistente-de-redaccion
  │           └── repositorio-de-rubricas
  ├── Auth (JWT + MongoDB Atlas)
  └── Tool Registry → Qdrant Vector DB
```

---

## Requisitos

| Herramienta | Versión mínima | Instalación |
|---|---|---|
| Python | 3.11+ | [python.org](https://www.python.org) |
| uv | cualquiera | `pip install uv` o [docs.astral.sh/uv](https://docs.astral.sh/uv/getting-started/installation/) |
| Node.js | 18+ | [nodejs.org](https://nodejs.org) |
| npm | 9+ | incluido con Node.js |

En Ubuntu, instalá las dependencias del sistema necesarias para `bcrypt` y drivers nativos:

```bash
sudo apt-get update
sudo apt-get install -y python3-dev build-essential libssl-dev
```

---

## Instalación

```bash
# 1. Clonar el repositorio
git clone <url-del-repo>
cd rubricas

# 2. Copiar y completar el archivo de entorno
cp .env.example .env
# Editá .env con tus claves (ver sección Configuración)

# 3. Instalar todas las dependencias (Python + Node)
npm run install:all
```

El comando `install:all` ejecuta `uv sync` para Python y `npm install` tanto en la raíz como en `frontend/`.

---

## Configuración

Todas las variables se configuran en el archivo `.env` en la raíz del proyecto. El archivo `.env.example` tiene todas las variables documentadas con sus valores por defecto.

### Variables obligatorias

| Variable | Descripción | Dónde obtenerla |
|---|---|---|
| `GOOGLE_API_KEY` | API key de Google Gemini | [aistudio.google.com](https://aistudio.google.com/app/apikey) |
| `OPENAI_API_KEY` | API key de OpenAI (GPT-4o-mini + embeddings) | [platform.openai.com](https://platform.openai.com/api-keys) |
| `MONGODB_URI` | URI de conexión a MongoDB Atlas | Atlas → Connect → Drivers |
| `SECRET_KEY` | Clave para firmar JWT (mínimo 32 chars) | `python -c "import secrets; print(secrets.token_hex(32))"` |

### Base de datos vectorial (Qdrant)

Controlada por `VECTOR_MODE`:

| Valor | Descripción | Variables adicionales |
|---|---|---|
| `server` | Qdrant Cloud (recomendado para producción) | `QDRANT_URL`, `QDRANT_API_KEY` |
| `disk` | Persistencia local en `./qdrant_data` | — |
| `memory` | Solo RAM, se pierde al reiniciar | — |

### MongoDB Atlas

1. Creá una cuenta en [cloud.mongodb.com](https://cloud.mongodb.com)
2. Creá un cluster gratuito (M0)
3. En **Database Access**: creá un usuario con rol `readWriteAnyDatabase`
4. En **Network Access**: agregá tu IP (o `0.0.0.0/0` para desarrollo)
5. En **Connect → Drivers**: copiá la URI y pegala en `MONGODB_URI`

La base de datos y las colecciones se crean automáticamente al primer arranque.

### Autenticación OAuth2 (opcional)

Por defecto el sistema usa login local (email + contraseña). Para habilitar proveedores externos:

**Google:**
1. Ir a [Google Cloud Console](https://console.cloud.google.com/apis/credentials)
2. Crear credenciales OAuth 2.0 → Aplicación web
3. Agregar `http://localhost:8000/auth/callback/google` como URI de redirección autorizada
4. Completar `GOOGLE_CLIENT_ID` y `GOOGLE_CLIENT_SECRET` en `.env`

**Microsoft:**
1. Ir a [Azure App Registrations](https://portal.azure.com/#view/Microsoft_AAD_RegisteredApps)
2. Registrar una aplicación nueva
3. Agregar `http://localhost:8000/auth/callback/microsoft` como URI de redirección
4. Completar `MICROSOFT_CLIENT_ID`, `MICROSOFT_CLIENT_SECRET` y `MICROSOFT_TENANT_ID`

---

## Primer Arranque y Setup Wizard

La primera vez que el sistema arranca con una base de datos vacía, el frontend muestra automáticamente el **Setup Wizard** en lugar del login.

El wizard te pide:

1. **Email y contraseña del administrador** — se crea el primer usuario con rol `admin`
2. **Nombre de la institución** — aparece en la interfaz
3. **Idioma por defecto** — español, portugués, gallego o inglés
4. **Logo e imagen de fondo** (opcional) — se sirven estáticamente desde el backend

Una vez completado, el wizard no vuelve a aparecer. Para resetear y volver a verlo, borrá la colección `users` y `app_settings` en MongoDB Atlas.

---

## Uso

### Arrancar el sistema

```bash
npm run dev
```

Esto lanza en paralelo:
- Backend en `http://localhost:8000`
- Frontend en `http://localhost:5173`

El frontend espera a que el backend esté listo antes de arrancar (health check automático).

### Scripts disponibles

| Comando | Descripción |
|---|---|
| `npm run dev` | Arranca backend + frontend en paralelo |
| `npm run dev:back` | Solo el backend |
| `npm run dev:front` | Solo el frontend (requiere backend corriendo) |
| `npm run install:all` | Instala dependencias Python y Node |

---

## Roles y Permisos

El sistema tiene tres roles de usuario que el administrador asigna desde el panel de configuración:

| Rol | Generar rúbricas | Repositorio | Asistente redacción | Evaluar documentos | Skills | Configuración |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| `admin` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `verificador` | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ |
| `rubricador` | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ |

### Gestión de usuarios

El administrador puede crear, editar y eliminar usuarios desde **Panel de Administración → Gestión de Usuarios**. También puede cambiar el rol de cualquier usuario excepto el propio.

Para crear usuarios por línea de comandos:

```bash
# Crear usuario normal
uv run python scripts/create_local_user_once.py

# Promover a admin
uv run python make_admin.py
```

---

## Skills (Agentes)

Las skills son agentes especializados que se cargan dinámicamente desde `skills/<nombre>/SKILL.md`. Cada skill es un archivo Markdown con frontmatter YAML:

```yaml
---
name: mi-skill
description: Descripción del agente
model: openai/gpt-4o-mini
tools:
  - buscar_contexto_qdrant
  - leer_documento_subido
---
# Instrucciones del agente en Markdown
Sos un experto en...
```

### Skills incluidas

| Skill | Descripción |
|---|---|
| `normativa-a-rubrica` | Genera rúbricas desde documentos normativos PDF |
| `evaluador-de-cumplimiento` | Evalúa documentos contra una rúbrica (solo `verificador` y `admin`) |
| `asistente-de-redaccion` | Ayuda a redactar documentos usando rúbricas como guía |
| `repositorio-de-rubricas` | Busca y gestiona rúbricas guardadas |

### Agregar una skill nueva

Desde la interfaz (solo admin): **Panel de Administración → Skills → Subir skill**

O manualmente: crear la carpeta `skills/<nombre>/SKILL.md` y reiniciar el servidor.

### Herramientas disponibles para skills

| Herramienta | Descripción |
|---|---|
| `buscar_contexto_qdrant` | Búsqueda semántica RAG en documentos indexados |
| `guardar_ontologia_en_qdrant` | Indexa entidades y relaciones extraídas de un PDF |
| `leer_rubrica_subida` | Lee el texto de una rúbrica cargada por el usuario |
| `leer_documento_subido` | Extrae texto de un PDF/DOCX subido |

---

## Estructura del Proyecto

```
rubricas/
├── app/
│   ├── auth/
│   │   ├── db.py           # CRUD de usuarios en MongoDB
│   │   ├── middleware.py   # Protección JWT de rutas /api/*
│   │   ├── models.py       # Modelos Pydantic de auth
│   │   ├── router.py       # Endpoints /auth/* (login, OAuth, /me)
│   │   └── service.py      # JWT, bcrypt, dependencias de rol
│   ├── server.py           # FastAPI + todos los endpoints REST
│   ├── main_agent.py       # Factory del agente raíz ADK
│   ├── skill_loader.py     # Carga dinámica de skills desde .md
│   ├── qdrant_service.py   # Vector DB + Tool Registry
│   ├── ontology_extractor.py  # Extracción de ontología vía LLM
│   ├── rubric_repository.py   # CRUD de rúbricas en Qdrant
│   ├── batch_manager.py    # Estado de lotes de extracción
│   ├── docx_converter.py   # Markdown → DOCX, extracción DOCX
│   └── models.py           # Modelos Pydantic de la API
├── common/
│   └── config.py           # Carga de .env, LangSmith
├── skills/
│   ├── normativa-a-rubrica/SKILL.md
│   ├── evaluador-de-cumplimiento/SKILL.md
│   ├── asistente-de-redaccion/SKILL.md
│   └── repositorio-de-rubricas/SKILL.md
├── frontend/
│   ├── src/
│   │   ├── components/     # ChatInterface, RubricGenerator, etc.
│   │   ├── contexts/       # AuthContext, LanguageContext
│   │   ├── pages/          # LoginPage, SettingsPage, SetupWizard
│   │   └── App.jsx
│   ├── vite.config.js      # Proxy /api → :8000
│   └── package.json
├── scripts/
│   └── create_local_user_once.py
├── docs/
│   ├── ARCHITECTURE.md     # Arquitectura detallada
│   └── setup.md
├── .env                    # Variables de entorno (no versionado)
├── .env.example            # Plantilla de variables de entorno
├── pyproject.toml          # Dependencias Python (uv)
├── package.json            # Scripts npm
└── _start.sh               # Script de arranque alternativo (bash)
```

---

## API REST

### Autenticación

| Método | Ruta | Auth | Descripción |
|---|---|---|---|
| `GET` | `/auth/mode` | No | Retorna el modo de auth configurado |
| `POST` | `/auth/login/local` | No | Login con email y contraseña |
| `GET` | `/auth/login/{provider}` | No | Inicia flujo OAuth (google, microsoft, uchile) |
| `GET` | `/auth/callback/{provider}` | No | Callback OAuth, emite JWT |
| `GET` | `/auth/me` | JWT | Datos del usuario actual |

### Sistema

| Método | Ruta | Auth | Descripción |
|---|---|---|---|
| `GET` | `/api/system/status` | No | Verifica si el sistema necesita setup inicial |
| `POST` | `/api/system/setup` | No | Ejecuta el setup inicial (solo si no hay admin) |
| `GET` | `/api/config` | Admin | Lee configuración del sistema |
| `PUT` | `/api/config/settings` | Admin | Actualiza configuración editable |
| `POST` | `/api/config/brand` | Admin | Sube logo e imagen de fondo |

### Usuarios

| Método | Ruta | Auth | Descripción |
|---|---|---|---|
| `GET` | `/api/users` | Admin | Lista todos los usuarios |
| `POST` | `/api/users` | Admin | Crea un usuario local |
| `PUT` | `/api/users/{email}` | Admin | Edita nombre, rol, contraseña o estado |
| `DELETE` | `/api/users/{email}` | Admin | Elimina un usuario |

### Chat y Generación

| Método | Ruta | Auth | Descripción |
|---|---|---|---|
| `POST` | `/api/chat` | JWT | Mensaje al orquestador ADK |
| `POST` | `/api/chat/reset` | JWT | Reinicia la sesión de chat |
| `POST` | `/api/upload` | JWT | Sube un PDF para procesamiento |
| `POST` | `/api/upload/batch` | JWT | Carga masiva de PDFs con extracción de ontología |
| `GET` | `/api/upload/status/{batch_id}` | JWT | Estado del lote de extracción |
| `POST` | `/api/generate` | JWT | Genera una rúbrica desde PDF(s) |
| `GET` | `/api/download/{filename}` | JWT | Descarga un DOCX generado |

### Evaluación (requiere rol `verificador` o `admin`)

| Método | Ruta | Auth | Descripción |
|---|---|---|---|
| `POST` | `/api/evaluate/upload_rubric` | Verificador | Sube una rúbrica para evaluación |
| `POST` | `/api/evaluate/upload_doc` | Verificador | Sube un documento a evaluar |
| `POST` | `/api/evaluate/run` | Verificador | Ejecuta la evaluación de cumplimiento |
| `POST` | `/api/evaluate/autocorrect` | Verificador | Autocorrige un documento según la rúbrica |

### Repositorio de Rúbricas

| Método | Ruta | Auth | Descripción |
|---|---|---|---|
| `GET` | `/api/rubrics` | JWT | Lista rúbricas guardadas (paginado + búsqueda) |
| `GET` | `/api/rubrics/{rubric_id}` | JWT | Detalle de una rúbrica |
| `DELETE` | `/api/rubrics/{rubric_id}` | JWT | Elimina una rúbrica |

### Skills (requiere rol `admin`)

| Método | Ruta | Auth | Descripción |
|---|---|---|---|
| `GET` | `/api/skills` | JWT | Lista skills cargadas |
| `GET` | `/api/skills/tools` | JWT | Lista herramientas disponibles |
| `POST` | `/api/skills/upload` | Admin | Sube una nueva skill `.md` |
| `DELETE` | `/api/skills/{skill_name}` | Admin | Elimina una skill |
| `GET` | `/api/skills/{skill_name}/download` | JWT | Descarga el `.md` de una skill |
