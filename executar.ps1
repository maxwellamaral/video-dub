# Configurar codificação para UTF-8
$OutputEncoding = [Console]::InputEncoding = [Console]::OutputEncoding = [System.Text.Encoding]::UTF8

Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "   Iniciando Pipeline de Dublagem (PS)" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan

# Caminho do script
$scriptPath = Join-Path $PSScriptRoot "pipeline_dublagem.py"

# Verificar ambiente virtual
if (-not (Test-Path ".venv")) {
    Write-Host "[!] Ambiente virtual .venv não encontrado." -ForegroundColor Yellow
    Write-Host "[!] Tentando criar com uv..." -ForegroundColor Yellow
    uv venv
}

# Sincronizar dependências
Write-Host "[+] Verificando dependências..." -ForegroundColor Green
uv sync

# Ativar e rodar
Write-Host "[+] Executando script..." -ForegroundColor Green
& .venv\Scripts\python.exe $scriptPath

Write-Host "`n==========================================" -ForegroundColor Cyan
Write-Host "   Execução finalizada." -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan
Pause
