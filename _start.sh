#!/bin/bash

# Función para cerrar todos los procesos hijos al presionar Ctrl+C
cleanup() {
    echo "🛑 Cerrando todos los servicios..."
    kill $(jobs -p) 2>/dev/null
    exit
}

trap cleanup SIGINT SIGTERM

echo "========================================="
echo "🚀 Iniciando RubricAI (Skills Architecture)"
echo "========================================="

echo "🧠 1/2 Iniciando Servidor Backend (Agente + Skills)..."
uv run python -m app.server &

echo "⏳ Esperando 5 segundos para que el servidor arranque..."
sleep 5

echo "💻 2/2 Iniciando Frontend interactivo..."
(cd frontend && npm run dev) &

echo "✅ Todos los servicios han sido iniciados."
echo "   - Backend: http://localhost:8000"
echo "   - Frontend: http://localhost:5173"
echo ""
echo "📂 Skills cargados desde: ./skills/"
echo "Presiona Ctrl+C para detener todo el sistema."

wait
