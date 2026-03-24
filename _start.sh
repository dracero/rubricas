#!/bin/bash

# Función para cerrar todos los procesos hijos al presionar Ctrl+C
cleanup() {
    echo "🛑 Cerrando todos los servicios..."
    # Mata todos los procesos en segundo plano (jobs) del script actual
    kill $(jobs -p) 2>/dev/null
    exit
}

# Capturamos las señales de interrupción para ejecutar el cleanup
trap cleanup SIGINT SIGTERM

echo "========================================="
echo "🚀 Iniciando Sistema Multi-Agente RubricAI"
echo "========================================="

echo "🤖 1/3 Iniciando Agentes (Generador, Evaluador, Greeter, Corrector)..."
(cd agents/generator && uv run python -m app --port 10001) &
(cd agents/evaluator && uv run python -m app --port 10002) &
(cd agents/greeter && uv run python -m app --port 10003) &
(cd agents/corrector && uv run python -m app --port 10005) &

echo "⏳ Esperando 10 segundos para asegurar que todos los agentes estén escuchando..."
sleep 10

echo "🧠 2/3 Iniciando Orquestador Central..."
uv run python -m hosts.orchestrator &

echo "⏳ Esperando 5 segundos para que el Orquestador descubra los agentes e inicie la API..."
sleep 5

echo "💻 3/3 Iniciando Frontend interactivo..."
(cd frontend && npm run dev) &

echo "✅ Todos los servicios han sido iniciados."
echo "Presiona Ctrl+C para detener todo el sistema de una sola vez."

# Mantenemos el script en ejecución observando los procesos de fondo
wait
