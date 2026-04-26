#!/bin/bash
set -euo pipefail

# ── Limpieza al presionar Ctrl+C ──────────────────────────────────────────────
cleanup() {
    echo ""
    echo "🛑 Cerrando todos los servicios..."
    # Matar todos los procesos hijos del script
    kill $(jobs -p) 2>/dev/null || true
    exit 0
}
trap cleanup SIGINT SIGTERM

# ── Variables de entorno ──────────────────────────────────────────────────────
export PYTHONUTF8=1
export PYTHONIOENCODING=utf-8
export UV_LINK_MODE=copy

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "========================================="
echo "🚀 Iniciando RubricAI (Skills Architecture)"
echo "========================================="

# ── 1/2 Backend ───────────────────────────────────────────────────────────────
echo "🧠 1/2 Iniciando Servidor Backend..."
uv run python -m app.server &
BACKEND_PID=$!

# Esperar a que el backend responda (hasta 30 segundos)
echo "⏳ Esperando a que el backend arranque..."
MAX_WAIT=30
WAITED=0
BACKEND_READY=false

while [ $WAITED -lt $MAX_WAIT ]; do
    sleep 2
    WAITED=$((WAITED + 2))
    if curl -sf --max-time 2 http://localhost:8000/ > /dev/null 2>&1; then
        echo "✅ Backend listo en ${WAITED}s."
        BACKEND_READY=true
        break
    else
        echo "   Esperando backend... (${WAITED}s)"
    fi
done

if [ "$BACKEND_READY" = false ]; then
    echo "⚠️  ADVERTENCIA: El backend no respondió en ${MAX_WAIT}s. Revisá los logs."
fi

# ── 2/2 Frontend ──────────────────────────────────────────────────────────────
echo "💻 2/2 Iniciando Frontend..."
(cd "$ROOT_DIR/frontend" && npm run dev) &
FRONTEND_PID=$!

echo ""
echo "✅ Todos los servicios han sido iniciados."
echo "   - Backend:  http://localhost:8000"
echo "   - Frontend: http://localhost:5173"
echo ""
echo "📂 Skills cargados desde: ./skills/"
echo "Presiona Ctrl+C para detener todo el sistema."
echo ""

# Esperar a que alguno de los dos procesos termine
wait $BACKEND_PID $FRONTEND_PID
