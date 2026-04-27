"""
optimize/metric.py
------------------
Métrica de calidad de rúbricas para GEPA.

GEPA espera que la métrica devuelva un dspy.Prediction con:
  - score: float entre 0.0 y 1.0
  - feedback: str explicando por qué la rúbrica es buena o mala

El feedback textual es lo que GEPA usa para reflexionar y proponer
instrucciones mejoradas — es la clave diferencial frente a métricas binarias.
"""

import re
import dspy

# ---------------------------------------------------------------------------
# Constantes de evaluación
# ---------------------------------------------------------------------------

REQUIRED_COLUMNS = [
    "Área de Cumplimiento",
    "Criterio de evaluación",
    "Evidencias observables",
    "Nivel mínimo aprobatorio",
]

MIN_CRITERIOS = 4          # mínimo de filas de criterios en la tabla
MIN_EJEMPLOS_PARENTESIS = 3  # mínimo de ejemplos "(ejemplo...)" en la rúbrica


# ---------------------------------------------------------------------------
# Métrica principal
# ---------------------------------------------------------------------------

def metrica_rubrica(gold: dspy.Example, pred: dspy.Prediction, trace=None, pred_name=None, pred_trace=None) -> dspy.Prediction:
    """
    Evalúa la calidad de una rúbrica generada.

    Criterios (con pesos):
      1. Tiene tabla Markdown              → 0.20
      2. Las 4 columnas correctas          → 0.20
      3. Información General presente      → 0.10
      4. Ejemplos entre paréntesis (≥ MIN) → 0.20
      5. Al menos MIN_CRITERIOS criterios  → 0.20
      6. No incluye secciones prohibidas   → 0.10

    Args:
        gold: Ejemplo de referencia (dspy.Example con campo `rubrica`).
        pred: Predicción del programa (dspy.Prediction con campo `rubrica`).
        trace: Traza de ejecución (inyectada por DSPy, no usada directamente).

    Returns:
        dspy.Prediction(score=float, feedback=str)
    """
    rubrica: str = getattr(pred, "rubrica", "") or ""
    score = 0.0
    feedback_parts: list[str] = []
    ok_parts: list[str] = []

    # --- 1. Tabla Markdown (0.20) ---
    if "|" in rubrica and "---" in rubrica:
        score += 0.20
        ok_parts.append("✅ Tiene tabla Markdown")
    else:
        feedback_parts.append("❌ No contiene tabla Markdown (usar | columna | columna |).")

    # --- 2. Columnas correctas (0.20) ---
    cols_presentes = sum(1 for c in REQUIRED_COLUMNS if c in rubrica)
    score += 0.20 * (cols_presentes / len(REQUIRED_COLUMNS))
    if cols_presentes == len(REQUIRED_COLUMNS):
        ok_parts.append("✅ Las 4 columnas correctas presentes")
    else:
        faltantes = [c for c in REQUIRED_COLUMNS if c not in rubrica]
        feedback_parts.append(
            f"❌ Columnas faltantes: {faltantes}. "
            "La tabla DEBE tener: Área de Cumplimiento | Criterio de evaluación | "
            "Evidencias observables | Nivel mínimo aprobatorio."
        )

    # --- 3. Información General (0.10) ---
    info_general_ok = (
        "Institución" in rubrica
        and "Ámbito de Aplicación" in rubrica
        and "Normativa de Referencia" in rubrica
    )
    if info_general_ok:
        score += 0.10
        ok_parts.append("✅ Información General completa")
    else:
        feedback_parts.append(
            "❌ Falta la sección Información General con viñetas (•) que incluya: "
            "Institución, Ámbito de Aplicación, Normativa de Referencia, "
            "Nivel de Criticidad, Objetivos."
        )

    # --- 4. Ejemplos entre paréntesis (0.20) ---
    ejemplos = re.findall(r'\([^)]{5,}\)', rubrica)  # paréntesis con ≥5 chars
    if len(ejemplos) >= MIN_EJEMPLOS_PARENTESIS:
        score += 0.20
        ok_parts.append(f"✅ {len(ejemplos)} ejemplos entre paréntesis")
    else:
        feedback_parts.append(
            f"❌ Solo hay {len(ejemplos)} ejemplos entre paréntesis. "
            f"La columna 'Nivel mínimo aprobatorio' DEBE tener un ejemplo concreto "
            f"entre paréntesis en CADA fila (ej: 'Al menos 3 sesiones registradas "
            f"(ej: actas firmadas del 10/03, 24/03 y 07/04)')."
        )

    # --- 5. Número de criterios (0.20) ---
    # Contar filas de tabla que no son encabezado ni separador
    filas_tabla = [
        line for line in rubrica.split("\n")
        if line.strip().startswith("|")
        and "---" not in line
        and not any(col in line for col in ["Área de Cumplimiento", "Criterio"])
    ]
    if len(filas_tabla) >= MIN_CRITERIOS:
        score += 0.20
        ok_parts.append(f"✅ {len(filas_tabla)} criterios en la tabla")
    else:
        feedback_parts.append(
            f"❌ Solo hay {len(filas_tabla)} criterios. "
            f"Se requieren al menos {MIN_CRITERIOS}. "
            "Revisá si el documento tiene más requisitos evaluables que no se cubrieron."
        )

    # --- 6. Sin secciones prohibidas (0.10) ---
    secciones_prohibidas = ["## Recomendaciones", "## Conclusiones", "## Cobertura", "## Áreas de"]
    encontradas = [s for s in secciones_prohibidas if s in rubrica]
    if not encontradas:
        score += 0.10
        ok_parts.append("✅ Sin secciones prohibidas")
    else:
        feedback_parts.append(
            f"❌ La rúbrica contiene secciones prohibidas: {encontradas}. "
            "La salida debe ser ÚNICAMENTE: Información General (viñetas) + Tabla Markdown."
        )

    # --- Componer feedback ---
    all_feedback = []
    if ok_parts:
        all_feedback.append("Lo que está bien: " + " | ".join(ok_parts))
    if feedback_parts:
        all_feedback.append("Lo que hay que mejorar: " + " | ".join(feedback_parts))
    feedback = " || ".join(all_feedback) if all_feedback else "Rúbrica evaluada."

    return dspy.Prediction(score=round(score, 3), feedback=feedback)


# ---------------------------------------------------------------------------
# Versión simplificada para uso sin GEPA (retorna solo el score)
# ---------------------------------------------------------------------------

def metrica_rubrica_simple(gold: dspy.Example, pred: dspy.Prediction, trace=None) -> float:
    """Versión simplificada que retorna solo el score (para BootstrapFewShot, etc.)."""
    result = metrica_rubrica(gold, pred, trace)
    return result.score
