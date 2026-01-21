#!/bin/bash

echo "Parando serviços..."

LOG_DIR="$(pwd)/logs"
BACKEND_PID_FILE="$LOG_DIR/backend.pid"
FRONTEND_PID_FILE="$LOG_DIR/frontend.pid"

if [ -f "$BACKEND_PID_FILE" ]; then
    BACKEND_PID=$(cat "$BACKEND_PID_FILE")
    if ps -p $BACKEND_PID > /dev/null 2>&1; then
        kill $BACKEND_PID
        echo "✅ Backend parado (PID: $BACKEND_PID)"
    else
        echo "⚠️  Backend já estava parado"
    fi
    rm -f "$BACKEND_PID_FILE"
fi

if [ -f "$FRONTEND_PID_FILE" ]; then
    FRONTEND_PID=$(cat "$FRONTEND_PID_FILE")
    if ps -p $FRONTEND_PID > /dev/null 2>&1; then
        kill $FRONTEND_PID
        echo "✅ Frontend parado (PID: $FRONTEND_PID)"
    else
        echo "⚠️  Frontend já estava parado"
    fi
    rm -f "$FRONTEND_PID_FILE"
fi

# Garantir que as portas foram liberadas
echo ""
echo "Verificando portas..."
if lsof -ti:8000 > /dev/null 2>&1; then
    echo "⚠️  Porta 8000 ainda em uso. Matando processo..."
    kill -9 $(lsof -ti:8000) 2>/dev/null
fi

if lsof -ti:5173 > /dev/null 2>&1; then
    echo "⚠️  Porta 5173 ainda em uso. Matando processo..."
    kill -9 $(lsof -ti:5173) 2>/dev/null
fi

echo ""
echo "✅ Todos os serviços foram parados!"
