"""
optimize/rubric_program.py
--------------------------
Programa DSPy que encapsula la generación de rúbricas de cumplimiento normativo.

Este módulo es el "estudiante" que GEPA va a optimizar: evoluciona las
instrucciones del ChainOfThought hasta que la métrica converja.
"""

import dspy


# ---------------------------------------------------------------------------
# Signature — define entradas y salidas del programa
# ---------------------------------------------------------------------------

class GenerarRubricaSignature(dspy.Signature):
    """
    Genera una rúbrica de cumplimiento normativo a partir de un documento.

    La rúbrica DEBE tener exactamente 2 secciones:
    1. INFORMACIÓN GENERAL — bloque con viñetas (Institución, Ámbito, Normativa,
       Nivel de Criticidad, Objetivos).
    2. MATRIZ DE EVALUACIÓN — tabla Markdown con columnas:
       | Área de Cumplimiento | Criterio de evaluación | Evidencias observables | Nivel mínimo aprobatorio |
       La columna "Nivel mínimo aprobatorio" SIEMPRE debe incluir un ejemplo concreto entre paréntesis.

    NO incluir ninguna otra sección (sin Recomendaciones, sin Conclusiones).
    SOLO extraer criterios que estén EXPLÍCITAMENTE en el documento (anti-alucinación).
    """

    documento_texto: str = dspy.InputField(
        desc="Texto completo del documento normativo del cual se extrae la rúbrica"
    )
    ontologia_resumen: str = dspy.InputField(
        desc="Resumen de entidades y relaciones extraídas del documento (puede estar vacío)"
    )
    rubrica: str = dspy.OutputField(
        desc=(
            "Rúbrica completa en formato Markdown. "
            "Empieza con viñetas de Información General, luego la tabla de evaluación. "
            "Sin secciones adicionales."
        )
    )


# ---------------------------------------------------------------------------
# Programa DSPy
# ---------------------------------------------------------------------------

class RubricadorDSPy(dspy.Module):
    """
    Módulo DSPy para generación de rúbricas.
    GEPA optimiza las instrucciones internas de `self.generate`.
    """

    def __init__(self):
        super().__init__()
        self.generate = dspy.ChainOfThought(GenerarRubricaSignature)

    def forward(self, documento_texto: str, ontologia_resumen: str = "") -> dspy.Prediction:
        """
        Args:
            documento_texto: Texto del PDF normativo.
            ontologia_resumen: Resumen de la ontología (opcional).

        Returns:
            dspy.Prediction con campo `rubrica`.
        """
        return self.generate(
            documento_texto=documento_texto,
            ontologia_resumen=ontologia_resumen,
        )
