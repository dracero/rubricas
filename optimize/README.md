# optimize/ — Optimizador de Rúbricas con GEPA

Este módulo usa **DSPy GEPA** (Genetic-Pareto optimizer) para mejorar automáticamente las instrucciones del agente `rubricador` analizando por qué las rúbricas generadas son buenas o malas.

---

## ¿Qué hace GEPA?

En lugar de ajustar pesos (como RL), GEPA:
1. Genera una rúbrica con las instrucciones actuales
2. La evalúa con la métrica de calidad
3. Reflexiona sobre los errores usando un LLM "juez" (GPT-4o)
4. Propone instrucciones mejoradas
5. Repite hasta converger (~100 evaluaciones en modo `light`)

**Resultado**: instrucciones evolucionadas que se copian al `SKILL.md` del rubricador.

---

## Instalación

Las dependencias ya están en el `pyproject.toml`. Si necesitás reinstalar:

```bash
uv add dspy
```

Verificá que funciona:

```bash
cd /home/cetec/AIProjects/rubricas
.venv/bin/python -c "from dspy.teleprompt import GEPA; print('OK')"
```

---

## Uso rápido

```bash
# Modo liviano (~100 evaluaciones, ~5 min, recomendado para empezar)
.venv/bin/python -m optimize.run_gepa --auto light

# Modo medio (~300 evaluaciones, ~15 min, mejor calidad)
.venv/bin/python -m optimize.run_gepa --auto medium

# Modo pesado (~500 evaluaciones, ~30 min, máxima calidad)
.venv/bin/python -m optimize.run_gepa --auto heavy
```

Después de correr GEPA, evaluá la mejora:

```bash
.venv/bin/python -m optimize.evaluate_improvement
```

---

## Cómo ajustar el optimizador a tus necesidades

### 1. Agregar tus propias rúbricas de referencia (JSON)

Este es el paso más importante. Cuantos más ejemplos propios tengas, mejor va a optimizar.

**Creá un archivo** `optimize/mis_ejemplos.json`:

```json
[
  {
    "documento": "REGLAMENTO DE PRÁCTICAS PROFESIONALES\n\nArtículo 1. Requisitos de inscripción\nLos estudiantes que deseen realizar prácticas profesionales deben haber aprobado el 70% del total de las materias de la carrera y contar con aval del director de carrera.\n\nArtículo 2. Duración\nLas prácticas tienen una duración mínima de 200 horas reloj distribuidas en no más de 6 meses. El estudiante debe llevar un registro semanal de actividades firmado por el tutor empresarial.\n\nArtículo 3. Informe final\nAl concluir, el estudiante debe presentar un informe final de entre 20 y 30 páginas que incluya: descripción de la empresa, tareas realizadas, competencias adquiridas y conclusiones.",
    "ontologia": "entidades: requisito_inscripcion, duracion_practica, informe_final | relaciones: practica REQUIERE requisito_inscripcion",
    "rubrica": "• Institución: Facultad de Ingeniería\n• Ámbito de Aplicación: Prácticas profesionales de estudiantes de grado\n• Normativa de Referencia: Reglamento de Prácticas Profesionales\n• Nivel de Criticidad: Medio\n• Objetivos de la evaluación: Verificar que el estudiante cumpla con los requisitos académicos, de duración y de documentación establecidos para las prácticas profesionales\n\n| Área de Cumplimiento | Criterio de evaluación | Evidencias observables | Nivel mínimo aprobatorio |\n|---|---|---|---|\n| Inscripción | El estudiante tiene aprobado al menos el 70% de las materias y cuenta con aval del director (Art. 1) | Certificado analítico y nota de aval firmada | Al menos 70% de materias aprobadas con aval firmado por el director (ej: analítico con 42/60 materias y aval del Dr. Pérez fechado el 15/03) |\n| Duración | La práctica cumple mínimo 200 horas en no más de 6 meses (Art. 2) | Registro de asistencia y certificación de la empresa | Al menos 200 horas certificadas dentro del período (ej: planilla de asistencia con 210 horas entre el 01/03 y el 31/07) |\n| Registro de actividades | El estudiante lleva registro semanal firmado por el tutor empresarial (Art. 2) | Bitácora semanal con firma del tutor | Bitácora completa con firma en cada semana del período (ej: 24 entradas semanales firmadas por Ing. López) |\n| Informe final | El informe final tiene entre 20 y 30 páginas con todas las secciones requeridas (Art. 3) | Informe presentado con conteo de páginas verificado | Informe de 20-30 páginas con las 4 secciones presentes (ej: informe de 25 páginas con descripción de empresa, tareas, competencias y conclusiones) |"
  }
]
```

**Luego corrés GEPA con tus ejemplos:**

```bash
.venv/bin/python -m optimize.run_gepa --auto light --extra-data optimize/mis_ejemplos.json
```

### 2. Estructura del JSON — referencia completa

Cada objeto del JSON debe tener estas 3 claves:

| Clave | Tipo | Descripción |
|---|---|---|
| `documento` | string | Texto del documento normativo (puede ser el texto completo del PDF o un fragmento representativo) |
| `ontologia` | string | Resumen de entidades y relaciones (puede estar vacío `""`) |
| `rubrica` | string | Rúbrica ideal de referencia — **la que querés que el sistema aprenda a imitar** |

> **Tip**: El campo `rubrica` es el más importante. Asegurate de que sea exactamente como querés que salgan las rúbricas en producción: mismo formato, mismo nivel de detalle, mismos ejemplos entre paréntesis.

### 3. Ejemplo mínimo (5 campos en la rúbrica)

Si querés empezar con algo simple:

```json
[
  {
    "documento": "POLÍTICA DE USO DE EQUIPOS\nArt. 1 — El personal debe registrar el uso de equipos en el sistema.\nArt. 2 — Los equipos deben apagarse al finalizar el turno.",
    "ontologia": "",
    "rubrica": "• Institución: Empresa XYZ\n• Ámbito de Aplicación: Personal operativo con acceso a equipos\n• Normativa de Referencia: Política de Uso de Equipos\n• Nivel de Criticidad: Bajo\n• Objetivos de la evaluación: Verificar registro y apagado correcto de equipos\n\n| Área de Cumplimiento | Criterio de evaluación | Evidencias observables | Nivel mínimo aprobatorio |\n|---|---|---|---|\n| Registro | El personal registra el uso de equipos en el sistema (Art. 1) | Logs del sistema con usuario y horario | Al menos 1 registro por turno por equipo utilizado (ej: log del 20/03 con 3 entradas del operario García) |\n| Apagado | Los equipos se apagan al finalizar el turno (Art. 2) | Verificación de estado de equipos al cierre | 100% de equipos apagados al cierre del turno (ej: checklist firmado del 20/03 con 5 equipos verificados) |"
  }
]
```

### 4. Ajustar la métrica de calidad

Si querés cambiar qué se evalúa, editá `optimize/metric.py`.

Por ejemplo, para **exigir más criterios** (mínimo 6 en lugar de 4):

```python
# optimize/metric.py
MIN_CRITERIOS = 6          # Cambiar este valor
MIN_EJEMPLOS_PARENTESIS = 4  # Y este también si querés más ejemplos
```

Para **agregar un criterio nuevo** (ej: que incluya columna de peso/porcentaje):

```python
# Dentro de metrica_rubrica(), antes del return:
if "%" in rubrica or "Peso" in rubrica:
    score += 0.10
    ok_parts.append("✅ Incluye pesos/porcentajes")
else:
    feedback_parts.append("❌ Falta columna de peso/porcentaje.")
    # Acordate de ajustar los pesos para que sumen 1.0
```

### 5. Aplicar las instrucciones optimizadas al agente

Después de correr GEPA, el archivo `optimize/optimized_rubricador.json` contiene las instrucciones ganadoras. Para aplicarlas:

1. **Abrí** `optimize/optimized_rubricador.json`
2. Buscá el campo `"instructions"` dentro de `"generate.predict" > "signature"`
3. **Copiá** ese texto al `SKILL.md` del rubricador, reemplazando la sección `### Instrucciones` del sub-agente `rubricador`
4. Reiniciá el servidor: el agente ya usa las nuevas instrucciones

---

## Archivos del módulo

```
optimize/
├── __init__.py                  # Módulo Python
├── rubric_program.py            # Programa DSPy (el "estudiante" que GEPA optimiza)
├── metric.py                    # Métrica de calidad con feedback textual
├── trainset.py                  # 5 ejemplos de entrenamiento incorporados
├── run_gepa.py                  # Script principal de optimización
├── evaluate_improvement.py      # Comparación antes/después
└── optimized_rubricador.json    # Resultado de la última optimización (se genera al correr)
```

---

## Resultados de la última optimización

| Métrica | Valor |
|---|---|
| Score base (sin optimizar) | 0.800 |
| Score optimizado (GEPA light) | 0.800 |
| Modo | `light` (~388 evaluaciones) |
| Student LM | `gpt-4o-mini` |
| Reflection LM | `gpt-4o` |

**Interpretación**: GEPA confirmó que las instrucciones actuales ya son cercanas al óptimo para el trainset actual. Los 2 puntos débiles identificados (Información General faltante, ejemplos entre paréntesis incompletos) fueron corregidos directamente en el `SKILL.md`.

Para mejorar más el score, el próximo paso es agregar más ejemplos propios via `--extra-data`.

---

## Flujo completo

```
tus PDFs normativos
       ↓
mis_ejemplos.json  ──┐
                     ├──→  run_gepa.py  ──→  optimized_rubricador.json
trainset.py (base) ──┘         ↑
                           metric.py
                         (tus criterios)
                               ↓
                    Copiar instrucciones al SKILL.md
                               ↓
                    Agente mejorado en producción
```
