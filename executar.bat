@echo off
setlocal
set PYTHONIOENCODING=utf-8

echo ==========================================
echo   Iniciando Pipeline de Dublagem
echo ==========================================

if not exist ".venv" (
    echo [!] Ambiente virtual .venv nao encontrado.
    echo [!] Tentando criar com uv...
    uv venv
)

echo [+] Ativando ambiente virtual...
call .venv\Scripts\activate.bat

echo [+] Verificando dependencias...
uv sync

echo [+] Executando script...
python pipeline_dublagem.py

echo.
echo ==========================================
echo   Execucao finalizada.
echo ==========================================
pause
