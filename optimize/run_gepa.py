"""
optimize/run_gepa.py
--------------------
Script principal de optimización GEPA.

Uso:
    cd /home/cetec/AIProjects/rubricas
    python -m optimize.run_gepa [--auto light|medium|heavy] [--output PATH]

El script:
  1. Configura los LMs (student y reflection).
  2. Carga el trainset/valset.
  3. Corre GEPA con la métrica de calidad.
  4. Guarda el programa optimizado en JSON.
  5. Imprime las instrucciones evolucionadas para copiarlas al SKILL.md.
"""

import argparse
import json
import os
import sys
from pathlib import Path

# Cargar variables de entorno desde .env
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

import dspy
from dspy.teleprompt import GEPA

from optimize.rubric_program import RubricadorDSPy
from optimize.metric import metrica_rubrica
from optimize.trainset import get_trainset, load_from_json


# ---------------------------------------------------------------------------
# Configuración
# ---------------------------------------------------------------------------

DEFAULT_OUTPUT = Path(__file__).parent / "optimized_rubricador.json"
DEFAULT_AUTO = "light"   # light ≈ 100 eval | medium ≈ 300 | heavy ≈ 500


def build_lms() -> tuple[dspy.LM, dspy.LM]:
    """
    Construye los LMs de student y reflection.

    student_lm  : gpt-4o-mini  — el que genera la rúbrica (se optimiza)
    reflection_lm: gpt-4o      — el "juez" que reflexiona sobre los errores

    Ambas claves deben estar en el .env del proyecto.
    """
    openai_key = os.environ.get("OPENAI_API_KEY")
    if not openai_key:
        sys.exit("❌ OPENAI_API_KEY no encontrada en .env")

    student_lm = dspy.LM(
        model="openai/gpt-4o-mini",
        api_key=openai_key,
        max_tokens=4096,
        temperature=0.7,
    )

    reflection_lm = dspy.LM(
        model="openai/gpt-4o",
        api_key=openai_key,
        max_tokens=4096,
        temperature=0.3,   # reflexión más determinista
    )

    return student_lm, reflection_lm


# ---------------------------------------------------------------------------
# Script principal
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Optimiza el prompt del rubricador con DSPy GEPA"
    )
    parser.add_argument(
        "--auto",
        choices=["light", "medium", "heavy"],
        default=DEFAULT_AUTO,
        help=f"Intensidad de la optimización (default: {DEFAULT_AUTO})"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Ruta de salida del JSON optimizado (default: {DEFAULT_OUTPUT})"
    )
    parser.add_argument(
        "--extra-data",
        type=Path,
        default=None,
        dest="extra_data",
        help="Ruta a un JSON con ejemplos adicionales de entrenamiento"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Semilla para reproducibilidad del split train/val"
    )
    args = parser.parse_args()

    print("=" * 60)
    print("🧬 GEPA — Optimizador de Rúbricas")
    print("=" * 60)

    # 1. Configurar LMs
    student_lm, reflection_lm = build_lms()
    dspy.configure(lm=student_lm)
    print(f"✅ Student LM:     {student_lm.model}")
    print(f"✅ Reflection LM:  {reflection_lm.model}")

    # 2. Cargar datos
    extra = load_from_json(args.extra_data) if args.extra_data else None
    trainset, valset = get_trainset(seed=args.seed, extra_ejemplos=None)
    # Si hay extra, los agrega al train
    if extra:
        trainset = extra + trainset
    print(f"\n📊 Trainset: {len(trainset)} ejemplos")
    print(f"📊 Valset:   {len(valset)} ejemplos")

    # 3. Crear programa y optimizador
    programa = RubricadorDSPy()

    optimizer = GEPA(
        metric=metrica_rubrica,
        reflection_lm=reflection_lm,
        auto=args.auto,
    )

    print(f"\n🚀 Iniciando optimización (auto={args.auto})...")
    print("   Esto puede tardar varios minutos dependiendo del modo.\n")

    # 4. Compilar
    compiled = optimizer.compile(
        student=programa,
        trainset=trainset,
        valset=valset,
    )

    # 5. Guardar
    args.output.parent.mkdir(parents=True, exist_ok=True)
    compiled.save(str(args.output))
    print(f"\n✅ Programa optimizado guardado en: {args.output}")

    # 6. Mostrar instrucciones evolucionadas
    _print_optimized_instructions(compiled, args.output)


def _print_optimized_instructions(compiled: RubricadorDSPy, output_path: Path):
    """Extrae y muestra las instrucciones evolucionadas."""
    print("\n" + "=" * 60)
    print("📋 INSTRUCCIONES OPTIMIZADAS (copiar al SKILL.md):")
    print("=" * 60)

    try:
        # Las instrucciones están en el predictor interno
        instrucciones = compiled.generate.extended_signature.instructions
        print(instrucciones)
    except AttributeError:
        # Fallback: leer del JSON guardado
        try:
            with open(output_path) as f:
                data = json.load(f)
            print(json.dumps(data, indent=2, ensure_ascii=False))
        except Exception as e:
            print(f"⚠️ No se pudieron extraer las instrucciones: {e}")
            print("   Revisá el archivo JSON guardado manualmente.")

    print("\n" + "=" * 60)
    print("💡 Pasos siguientes:")
    print("   1. Copiá las instrucciones de arriba al SKILL.md del rubricador")
    print("   2. Corré: python -m optimize.evaluate_improvement")
    print("   3. Comparás la rúbrica antes vs. después")
    print("=" * 60)


if __name__ == "__main__":
    main()
