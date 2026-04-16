# Plan: Configuración Agnóstica y Setup Wizard Dinámico

Este documento detalla la arquitectura para soportar fluidamente despliegues 100% locales (sin dependencias cloud) y despliegues en la nube mediante SQLAlchemy y Qdrant Local/Cloud. Además, incluye la implementación de un asistente de configuración inicial ("First-Time Setup Wizard") que permita gestionar la cuenta base, almacenar variables de negocio, cargar multimedia (logo y fondo) y establecer el idioma predeterminado.

## Fases de Implementación

### Fase 1: Infraestructura Agnóstica
1. **Refactorización de Variables de Entorno (.env)**
   - Consolidar `DB_TYPE` permitiendo opciones como `sqlite`, `postgres-local`, y `postgres-cloud`.
   - Agregar `VECTOR_MODE` con opciones `memory`, `disk`, y `server` para definir el comportamiento de Qdrant.
   - Dejar en el `.env` estrictamente las credenciales y configuraciones de motor (URLs, contraseñas). Las variables dinámicas de la aplicación (Nombre de Institución, Auth Mode) se migrarán a la base de datos.

2. **Abstracción de Bases de Datos (SQLAlchemy & Qdrant)** *(depende de 1)*
   - Modificar las consultas en `app/auth/db.py`: Reemplazar el uso directo de `asyncpg` por **SQLAlchemy asíncrono** para permitir compatibilidad con múltiples motores.
   - Implementar **Alembic** para gestionar migraciones de base de datos independientemente del motor relacional.
   - Modificar `app/qdrant_service.py` para usar `location=":memory:"`, `path="./qdrant_data"`, o `url=...` según el valor de `VECTOR_MODE`.

### Fase 2: Persistencia Dinámica y Admin UI (Setup)
3. **Persistencia Dinámica del Negocio (Backend API)** *(depende de 2)*
   - Crear una nueva tabla usando Alembic llamada `app_settings` (con campos: key, value, updated_at).
   - Desarrollar endpoints públicos: 
     - `GET /api/system/status` (evalúa si existe configuración base y al menos un admin).
     - `POST /api/system/setup` (endpoint multi-part para recibir la inicialización, archivos e info básica).
   - Montar un directorio estático en FastAPI (ej. `app.mount("/uploads", ...)`) para servir nativamente el Logo y Fondo.
   - Refactorizar `/api/config` e implementar un `PUT /api/config` para actualizar configuraciones en caliente post-setup.

4. **First-Time Setup Wizard (React Frontend)** *(depende de 3)*
   - Desarrollar el componente `<SetupWizard />` en `frontend/src/pages/`.
   - Implementar Route Guard en `frontend/src/App.jsx` (AppShell) para interceptar: Si `/api/system/status` devuelve `setup_required: true`, redirigir al Wizard obligatoriamente.
   - **Contenido del Wizard:** 
     - Creación de cuenta Super Administrador inicial.
     - Nombre de Institución.
     - Idioma por defecto.
     - Subida de archivos visuales: Logo de la Institución e Imagen de Fondo.

5. **Panel de Administración en Caliente (React Frontend)** *(depende de 3)*
   - Refactorizar la actual página `SettingsPage.jsx` transformándola en un formulario reactivo conectado a `PUT /api/config`.
   - Habilitar la subida y sustitución de Logo y Fondo, así como la actualización del Idioma en el panel.
   - Modificar la capa base visual (el actual componente `GlobalBackground` en `App.jsx`) para que lea las URLs del logo y fondo dinámicamente desde la URL del backend en lugar de usar `import.meta.glob` empaquetado por Vite.

## Verificación de Éxito
1. **Configuración Inicial (Arranque Limpio):** 
   - Ejecutar la BD en blanco. 
   - El *Setup Wizard* debe aparecer, forzando a definir el idioma, crear admin, y subir el fondo y el logo. 
   - Al finalizar, el dashboard principal debe renderizar la marca subida y el idioma establecido de forma nativa.
2. **Edición Constante:** 
   - Ingresar como Admin al menú de "Configuración".
   - Sustituir logo y modificar idioma. 
   - Evaluar que, tras presionar "Guardar", la UI de la app adopta las modificaciones sin requerir recargar la página ni reiniciar procesos del servidor o Node.