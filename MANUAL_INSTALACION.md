# 📘 Manual de Instalación - RubricAI

## 📋 Tabla de Contenidos

1. [Requisitos Previos](#requisitos-previos)
2. [Instalación del Sistema](#instalación-del-sistema)
3. [Configuración Inicial](#configuración-inicial)
4. [Primer Ingreso como Administrador](#primer-ingreso-como-administrador)
5. [Gestión de Usuarios](#gestión-de-usuarios)
6. [Roles y Permisos](#roles-y-permisos)
7. [Skills Disponibles](#skills-disponibles)
8. [Solución de Problemas](#solución-de-problemas)

---

## 🔧 Requisitos Previos

### Software Necesario

- **Node.js**: versión 22.12+ o 20.19+
- **Python**: versión 3.11 o superior
- **uv**: gestor de paquetes Python (instalación: https://docs.astral.sh/uv/getting-started/installation/)
- **npm**: incluido con Node.js
- **Git**: para clonar el repositorio

### Verificar Instalaciones

```bash
node --version    # Debe mostrar v22.x o v20.19+
python --version  # Debe mostrar Python 3.11+
uv --version      # Debe mostrar uv 0.x.x
npm --version     # Debe mostrar 10.x+
```

---

## 📦 Instalación del Sistema

### 1. Clonar el Repositorio

```bash
git clone <url-del-repositorio>
cd rubricas
```

### 2. Instalar Dependencias

```bash
# Instalar dependencias de Node.js (root y frontend)
npm install
cd frontend && npm install && cd ..

# Instalar dependencias de Python
uv sync
```

---

## ⚙️ Configuración Inicial

### 1. Archivos de Configuración

Vas a recibir por correo electrónico dos archivos importantes:

- **`.env`**: Variables de entorno con claves de API y configuración
- **`credentials.json`**: Credenciales de Google Cloud (si usás Cloud SQL)

### 2. Ubicar los Archivos

1. **Archivo `.env`**:
   - Colocalo en la raíz del proyecto: `/rubricas/.env`
   - Este archivo contiene todas las claves de API necesarias

2. **Archivo `credentials.json`** (opcional, solo si usás Cloud SQL):
   - Colocalo en la raíz del proyecto: `/rubricas/credentials.json`
   - Este archivo contiene las credenciales de Google Cloud

### 3. Configurar MongoDB

En el archivo `.env`, buscá la línea:

```env
MONGODB_URI=mongodb+srv://root:<db_password>@cluster0.zf9fl.mongodb.net/?appName=Cluster0
```

**Reemplazá `<db_password>` con la contraseña real** que te llegó por correo.

Ejemplo:
```env
MONGODB_URI=mongodb+srv://root:MiPassword123@cluster0.zf9fl.mongodb.net/?appName=Cluster0
```

### 4. Verificar Configuración del `.env`

Asegurate de que estas variables estén configuradas:

```env
# Claves de IA (obligatorias)
GOOGLE_API_KEY=AIzaSy...
OPENAI_API_KEY=sk-proj-...

# Base de datos vectorial
QDRANT_URL=https://...
QDRANT_API_KEY=QKWSz2...

# MongoDB (completar con tu contraseña)
MONGODB_URI=mongodb+srv://root:TU_PASSWORD@cluster0.zf9fl.mongodb.net/?appName=Cluster0

# JWT (clave secreta para tokens)
SECRET_KEY=cambia-esto-por-una-clave-secreta-larga

# Servidor
ORCHESTRATOR_HOST=localhost
ORCHESTRATOR_PORT=8000
```

### 5. Generar Clave Secreta JWT (Recomendado)

Para mayor seguridad, generá una clave secreta única:

```bash
python -c "import secrets; print(secrets.token_hex(32))"
```

Copiá el resultado y reemplazá el valor de `SECRET_KEY` en el `.env`.

---

## 🚀 Iniciar el Sistema

### Arrancar Todos los Servicios

```bash
npm run dev
```

Este comando inicia:
- **Backend** (orchestrator + agents) en `http://localhost:8000`
- **Frontend** en `http://localhost:5173`

### Verificar que Todo Funciona

1. Abrí tu navegador en: `http://localhost:5173`
2. Deberías ver la pantalla de login

---

## 👤 Primer Ingreso como Administrador

### Credenciales por Defecto

El sistema crea automáticamente un usuario administrador:

- **Email**: `admin@rubricai.com`
- **Contraseña**: `admin123`

⚠️ **IMPORTANTE**: Cambiá esta contraseña inmediatamente después del primer ingreso.

### Pasos del Primer Ingreso

1. **Accedé al sistema**:
   - Abrí `http://localhost:5173`
   - Hacé clic en "Continuar con Email y Contraseña"

2. **Ingresá las credenciales**:
   - Email: `admin@rubricai.com`
   - Contraseña: `admin123`

3. **Cambiar la contraseña** (recomendado):
   - Una vez dentro, andá a **Configuración** (ícono de engranaje)
   - En la sección "Usuarios", buscá tu usuario admin
   - Hacé clic en "Editar" y cambiá la contraseña

---

## 👥 Gestión de Usuarios

### Crear Nuevos Usuarios

Solo los usuarios con rol **Admin** pueden crear y gestionar usuarios.

#### Pasos para Crear un Usuario

1. **Accedé a Configuración**:
   - Hacé clic en el ícono de engranaje (⚙️) en la esquina superior derecha
   - Seleccioná "Gestión de Usuarios"

2. **Crear Usuario**:
   - Hacé clic en el botón **"+ Crear Usuario"**
   - Completá el formulario:
     - **Email**: dirección de correo del usuario
     - **Nombre**: nombre completo
     - **Contraseña**: contraseña inicial (el usuario puede cambiarla después)
     - **Rol**: seleccioná el rol apropiado (ver sección siguiente)

3. **Guardar**:
   - Hacé clic en "Crear Usuario"
   - El usuario ya puede ingresar con sus credenciales

### Editar Usuarios Existentes

1. En la lista de usuarios, hacé clic en **"Editar"** junto al usuario
2. Modificá los campos necesarios (nombre, rol, contraseña)
3. Hacé clic en **"Guardar"**

### Eliminar Usuarios

1. En la lista de usuarios, hacé clic en **"Eliminar"** junto al usuario
2. Confirmá la eliminación

⚠️ **Nota**: No podés eliminar tu propio usuario mientras estés logueado.

---

## 🔐 Roles y Permisos

El sistema tiene tres roles con diferentes niveles de acceso:

### 1. 👑 Admin (Administrador)

**Permisos completos del sistema:**

- ✅ Acceso a **todos los skills**
- ✅ Gestión de usuarios (crear, editar, eliminar)
- ✅ Configuración del sistema
- ✅ Gestión de skills (activar/desactivar)
- ✅ Generación de rúbricas
- ✅ Evaluación de documentos
- ✅ Repositorio de rúbricas
- ✅ Asistente de redacción

**Casos de uso:**
- Administradores del sistema
- Personal de IT
- Coordinadores académicos con acceso total

---

### 2. 🔍 Verificador

**Acceso a todas las funcionalidades operativas:**

- ✅ Generación de rúbricas
- ✅ Evaluación de documentos
- ✅ Repositorio de rúbricas
- ✅ Asistente de redacción
- ❌ NO puede gestionar usuarios
- ❌ NO puede configurar el sistema
- ❌ NO puede gestionar skills

**Casos de uso:**
- Docentes evaluadores
- Personal académico que evalúa trabajos
- Coordinadores de cátedra

---

### 3. 📝 Rubricador

**Acceso limitado a generación de rúbricas:**

- ✅ Generación de rúbricas
- ✅ Repositorio de rúbricas (consulta)
- ✅ Asistente de redacción
- ❌ NO puede evaluar documentos
- ❌ NO puede gestionar usuarios
- ❌ NO puede configurar el sistema

**Casos de uso:**
- Docentes que solo crean rúbricas
- Asistentes académicos
- Personal de apoyo pedagógico

---

## 🛠️ Skills Disponibles

Los skills son funcionalidades específicas del sistema. Cada skill puede ser activado o desactivado por un administrador.

### 1. 📝 Generador de Rúbricas

**Descripción**: Genera rúbricas académicas a partir de documentos normativos.

**Quién puede usarlo**:
- ✅ Admin
- ✅ Verificador
- ✅ Rubricador

**Cómo usarlo**:
1. Hacé clic en "Generar rúbrica" en el chat
2. Subí un documento normativo (PDF)
3. Seleccioná el nivel académico
4. El sistema genera la rúbrica automáticamente
5. Descargá el resultado en formato DOCX

**Ejemplo de uso**:
- Subir un reglamento de trabajos finales
- Generar una rúbrica para evaluar tesis de grado

---

### 2. 🔍 Evaluador de Documentos

**Descripción**: Evalúa documentos académicos contra una rúbrica de referencia.

**Quién puede usarlo**:
- ✅ Admin
- ✅ Verificador
- ❌ Rubricador (sin acceso)

**Cómo usarlo**:
1. Hacé clic en "Evaluar documento" en el chat
2. Subí la rúbrica de referencia (TXT o MD)
3. Subí el documento del estudiante (PDF)
4. El sistema evalúa y genera un informe detallado
5. Descargá el resultado

**Ejemplo de uso**:
- Evaluar una tesis contra la rúbrica institucional
- Verificar cumplimiento de criterios académicos

---

### 3. 📁 Repositorio de Rúbricas

**Descripción**: Consulta y gestiona rúbricas almacenadas en el sistema.

**Quién puede usarlo**:
- ✅ Admin
- ✅ Verificador
- ✅ Rubricador

**Cómo usarlo**:
1. Hacé clic en "Repositorio" en el chat
2. Buscá rúbricas por tema o palabra clave
3. Visualizá rúbricas existentes
4. Descargá rúbricas para reutilizar

**Ejemplo de uso**:
- Buscar rúbricas de trabajos finales anteriores
- Consultar criterios de evaluación estándar

---

### 4. ✍️ Asistente de Redacción

**Descripción**: Ayuda a redactar documentos académicos siguiendo una rúbrica.

**Quién puede usarlo**:
- ✅ Admin
- ✅ Verificador
- ✅ Rubricador

**Cómo usarlo**:
1. Seleccioná una rúbrica del repositorio
2. El asistente te guía en la redacción
3. Recibís sugerencias basadas en los criterios
4. Verificás que tu documento cumple con la rúbrica

**Ejemplo de uso**:
- Redactar un informe siguiendo criterios institucionales
- Verificar que un documento cumple con todos los requisitos

---

## 🔧 Gestión de Skills (Solo Admin)

### Activar/Desactivar Skills

1. **Accedé a Configuración** (⚙️)
2. Buscá la sección **"Gestión de Skills"**
3. Hacé clic en el botón junto a cada skill para activarlo/desactivarlo
4. Los cambios se aplican inmediatamente

### ¿Por qué desactivar un skill?

- Mantenimiento temporal
- Restricción de funcionalidades
- Pruebas del sistema

---

## 🐛 Solución de Problemas

### Error: "Puerto 8000 ya en uso"

**Solución**:
```bash
# Matar el proceso en el puerto 8000
lsof -ti:8000 | xargs kill -9

# Reiniciar el sistema
npm run dev
```

---

### Error: "MongoDB connection failed"

**Causa**: Contraseña incorrecta en `MONGODB_URI`

**Solución**:
1. Abrí el archivo `.env`
2. Verificá que `<db_password>` esté reemplazado con la contraseña correcta
3. Guardá el archivo
4. Reiniciá el sistema: `npm run dev`

---

### Error: "Node.js version not supported"

**Causa**: Versión de Node.js antigua

**Solución**:
```bash
# Si usás nvm
nvm install 22
nvm use 22
nvm alias default 22

# Verificar
node --version  # Debe mostrar v22.x
```

---

### Error: "GOOGLE_API_KEY not found"

**Causa**: Archivo `.env` no configurado correctamente

**Solución**:
1. Verificá que el archivo `.env` esté en la raíz del proyecto
2. Verificá que contenga `GOOGLE_API_KEY=AIzaSy...`
3. Si falta, copiá el archivo que recibiste por correo

---

### No puedo crear usuarios

**Causa**: No tenés permisos de administrador

**Solución**:
- Solo usuarios con rol **Admin** pueden crear usuarios
- Contactá a un administrador del sistema

---

### El evaluador no aparece para mi usuario

**Causa**: Tu rol es "Rubricador"

**Solución**:
- Los rubricadores no tienen acceso al evaluador
- Contactá a un administrador para cambiar tu rol a "Verificador" si necesitás evaluar documentos

---

## 📞 Soporte

Si tenés problemas que no están cubiertos en este manual:

1. Revisá los logs del sistema en la terminal donde ejecutaste `npm run dev`
2. Contactá al equipo de soporte técnico
3. Enviá los logs de error para diagnóstico

---

## 🔄 Actualización del Sistema

Para actualizar el sistema a una nueva versión:

```bash
# Detener el sistema (Ctrl+C)

# Actualizar código
git pull origin main

# Reinstalar dependencias
npm install
cd frontend && npm install && cd ..
uv sync

# Reiniciar
npm run dev
```

---

## 📝 Notas Finales

- **Seguridad**: Cambiá las contraseñas por defecto inmediatamente
- **Backups**: MongoDB Atlas hace backups automáticos, pero considerá exportar rúbricas importantes
- **Permisos**: Asigná roles según las necesidades reales de cada usuario
- **Monitoreo**: Revisá los logs regularmente para detectar problemas

---

**Versión del Manual**: 1.0  
**Última Actualización**: Abril 2026  
**Sistema**: RubricAI v1.0.0
