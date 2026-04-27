"""
optimize/evaluate_improvement.py
---------------------------------
Compara la calidad de las rúbricas ANTES y DESPUÉS de la optimización GEPA.

Uso:
    cd /home/cetec/AIProjects/rubricas
    python -m optimize.evaluate_improvement

Muestra:
  - Score promedio antes (programa base)
  - Score promedio después (programa optimizado)
  - Tabla de comparación por ejemplo
  - Feedback detallado del último ejemplo
"""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

import dspy

from optimize.rubric_program import RubricadorDSPy
from optimize.metric import metrica_rubrica
from optimize.trainset import get_trainset


OPTIMIZED_PATH = Path(__file__).parent / "optimized_rubricador.json"


def evaluar_programa(programa: RubricadorDSPy, ejemplos: list[dspy.Example], label: str) -> list[float]:
    """Corre el programa sobre todos los ejemplos y retorna lista de scores."""
    scores = []
    print(f"\n{'='*50}")
    print(f"Evaluando: {label}")
    print(f"{'='*50}")

    for i, ejemplo in enumerate(ejemplos, 1):
        try:
            pred = programa(
                documento_texto=ejemplo.documento_texto,
                ontologia_resumen=ejemplo.ontologia_resumen,
            )
            resultado = metrica_rubrica(ejemplo, pred)
            scores.append(resultado.score)
            status = "✅" if resultado.score >= 0.7 else "⚠️" if resultado.score >= 0.4 else "❌"
            print(f"  Ejemplo {i}: {status} score={resultado.score:.3f}")
            if i == len(ejemplos):
                print(f"\n  Feedback del último ejemplo:")
                print(f"  {resultado.feedback}")
        except Exception as e:
            print(f"  Ejemplo {i}: ❌ Error — {e}")
            scores.append(0.0)

    avg = sum(scores) / len(scores) if scores else 0.0
    print(f"\n  Promedio: {avg:.3f}  ({len(scores)} ejemplos)")
    return scores


def main():
    openai_key = os.environ.get("OPENAI_API_KEY")
    if not openai_key:
        sys.exit("❌ OPENAI_API_KEY no encontrada en .env")

    lm = dspy.LM("openai/gpt-4o-mini", api_key=openai_key, max_tokens=4096)
    dspy.configure(lm=lm)

    # Usar el valset para evaluar (no visto durante el entrenamiento)
    _, valset = get_trainset()
    print(f"\n📊 Evaluando sobre {len(valset)} ejemplos del valset")

    # --- ANTES: programa sin optimizar ---
    programa_base = RubricadorDSPy()
    scores_antes = evaluar_programa(programa_base, valset, "ANTES (base sin optimizar)")

    # --- DESPUÉS: programa optimizado ---
    if not OPTIMIZED_PATH.exists():
        print(f"\n⚠️  No se encontró el programa optimizado en {OPTIMIZED_PATH}")
        print("   Primero corré: python -m optimize.run_gepa")
        sys.exit(1)

    programa_opt = RubricadorDSPy()
    programa_opt.load(str(OPTIMIZED_PATH))
    scores_despues = evaluar_programa(programa_opt, valset, "DESPUÉS (optimizado con GEPA)")

    # --- Resumen ---
    avg_antes = sum(scores_antes) / len(scores_antes) if scores_antes else 0.0
    avg_despues = sum(scores_despues) / len(scores_despues) if scores_despues else 0.0
    mejora = avg_despues - avg_antes
    mejora_pct = (mejora / avg_antes * 100) if avg_antes > 0 else 0.0

    print("\n" + "=" * 50)
    print("📈 RESUMEN DE MEJORA")
    print("=" * 50)
    print(f"  Score promedio ANTES:   {avg_antes:.3f}")
    print(f"  Score promedio DESPUÉS: {avg_despues:.3f}")
    mejora_emoji = "🚀" if mejora > 0 else "⚠️" if mejora == 0 else "📉"
    print(f"  Mejora absoluta:        {mejora:+.3f} {mejora_emoji}")
    print(f"  Mejora relativa:        {mejora_pct:+.1f}%")

    if avg_despues >= 0.8:
        print("\n✅ Las instrucciones optimizadas están listas para producción.")
        print("   Copiá el contenido de optimized_rubricador.json al SKILL.md del rubricador.")
    elif mejora > 0:
        print("\n⚠️  Hay mejora pero el score todavía puede subir más.")
        print("   Considerá correr con --auto medium o agregar más ejemplos al trainset.")
    else:
        print("\n❌ No hubo mejora. Revisá la métrica y el trainset.")


if __name__ == "__main__":
    main()
