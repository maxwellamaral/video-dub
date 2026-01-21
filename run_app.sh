#!/bin/bash

echo "=================================================="
echo "   DUBBLER PRO WEB (Backend + Frontend)"
echo "=================================================="

CURRENT_DIR=$(pwd)
LOG_DIR="$CURRENT_DIR/logs"

# Criar diretório de logs se não existir
mkdir -p "$LOG_DIR"

BACKEND_LOG="$LOG_DIR/backend.log"
FRONTEND_LOG="$LOG_DIR/frontend.log"

echo "[1/2] Iniciando Backend FastAPI (Porta 8000)..."
echo "       Log: $BACKEND_LOG"

# Iniciar backend em background com logs
cd "$CURRENT_DIR"
nohup uv run uvicorn src.backend.app:app --reload --host 0.0.0.0 --port 8000 > "$BACKEND_LOG" 2>&1 &
BACKEND_PID=$!
echo "       PID: $BACKEND_PID"

echo "[2/2] Iniciando Frontend Vue.js (Porta 5173)..."
echo "       Log: $FRONTEND_LOG"

# Iniciar frontend em background com logs
cd "$CURRENT_DIR/src/frontend"
nohup npm run dev > "$FRONTEND_LOG" 2>&1 &
FRONTEND_PID=$!
echo "       PID: $FRONTEND_PID"

cd "$CURRENT_DIR"

echo ""
echo "=================================================="
echo "   ACESSE NO NAVEGADOR: http://localhost:5173"
echo "=================================================="
echo ""
echo "💡 Dica: Os logs estão sendo salvos em:"
echo "   - Backend:  logs/backend.log"
echo "   - Frontend: logs/frontend.log"
echo ""
echo "⚠️  Para parar os serviços, execute:"
echo "   kill $BACKEND_PID $FRONTEND_PID"
echo ""
echo "Ou use o script de parada:"
echo "   ./stop_app.sh"
echo ""

# Salvar PIDs em arquivo para facilitar o stop
echo "$BACKEND_PID" > "$LOG_DIR/backend.pid"
echo "$FRONTEND_PID" > "$LOG_DIR/frontend.pid"

# Aguardar alguns segundos para os serviços iniciarem
sleep 3

# Verificar se os processos estão rodando
if ps -p $BACKEND_PID > /dev/null && ps -p $FRONTEND_PID > /dev/null; then
    echo "✅ Serviços iniciados com sucesso!"
else
    echo "❌ Erro ao iniciar serviços. Verifique os logs."
fi
