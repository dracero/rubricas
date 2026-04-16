# Reglas de Implementación — RubricAI

Este documento define el proceso de trabajo obligatorio que el agente de desarrollo debe seguir para cada funcionalidad solicitada en el proyecto RubricAI.

---

## Proceso Obligatorio por Requerimiento

### 1. Checklist antes de codificar

Antes de escribir cualquier código, el agente debe descomponer el requerimiento en una lista de actividades atómicas y presentarla al usuario. El checklist debe cubrir todas las capas afectadas:

- [ ] Modelos / estructuras de datos (backend)
- [ ] Lógica de negocio / servicio (backend)
- [ ] Endpoint(s) API REST (backend)
- [ ] Herramientas ADK / Tool Registry (si aplica)
- [ ] Skill o instrucciones de agente (si aplica)
- [ ] Componente(s) React (frontend)
- [ ] Integración frontend ↔ API
- [ ] Variables de entorno nuevas (si aplica)
- [ ] Actualización de `docs/ARCHITECTURE.md`

### 2. Implementación actividad por actividad

- Cada actividad del checklist se implementa con la **porción mínima de código** necesaria para cumplirla.
- No se añade código especulativo, helpers genéricos ni refactorizaciones no solicitadas.
- Cada actividad se marca completada (`[x]`) inmediatamente al terminar.
- Se trabaja en orden: backend primero (modelos → servicio → API → herramientas/skills), luego frontend.

### 3. Cierre del requerimiento

Al finalizar todas las actividades del checklist:

1. **Verificar errores**: revisar diagnósticos de Pylance y eslint en los archivos modificados.
2. **Actualizar `docs/ARCHITECTURE.md`**: incorporar la funcionalidad nueva en la sección correspondiente (API REST, componentes, flujos, stack, etc.).
3. **Confirmar al usuario** que el requerimiento está completo con un resumen de lo implementado.

---

## Restricciones

- **No** agregar funcionalidades más allá de lo solicitado en el requerimiento.
- **No** generar archivos de documentación adicionales salvo los explícitamente pedidos.
- **No** modificar código existente que no sea necesario para el requerimiento en curso.
- Cada cambio en el backend debe ser compatible con el frontend existente y viceversa.
- Las skills ADK se deben mantener en `skills/<nombre>/SKILL.md` con frontmatter YAML válido.

---

## Referencia de Arquitectura

El estado actualizado de la arquitectura siempre se encuentra en [`docs/ARCHITECTURE.md`](./ARCHITECTURE.md).
