Write-Host "==================================================" -ForegroundColor Cyan
Write-Host "   DUBBLER PRO WEB (Backend + Frontend)"
Write-Host "==================================================" -ForegroundColor Cyan

$CurrentDir = Get-Location
$LogDir = Join-Path $CurrentDir "logs"

# Criar diretório de logs se não existir
if (-not (Test-Path $LogDir)) {
    New-Item -ItemType Directory -Path $LogDir | Out-Null
}

$BackendLog = Join-Path $LogDir "backend.log"
$FrontendLog = Join-Path $LogDir "frontend.log"

Write-Host "[1/2] Iniciando Backend FastAPI (Porta 8000)..." -ForegroundColor Yellow
Write-Host "       Log: $BackendLog" -ForegroundColor DarkGray
$BackendCmd = "Set-Location '$CurrentDir'; Start-Transcript -Path '$BackendLog' -Force; uv run uvicorn src.backend.app:app --reload --host 0.0.0.0 --port 8000"
Start-Process powershell -ArgumentList "-NoProfile", "-NoExit", "-Command", $BackendCmd -WindowStyle Minimized

Write-Host "[2/2] Iniciando Frontend Vue.js (Porta 5173)..." -ForegroundColor Yellow
Write-Host "       Log: $FrontendLog" -ForegroundColor DarkGray
$FrontendCmd = "Set-Location '$CurrentDir\src\frontend'; Start-Transcript -Path '$FrontendLog' -Force; npm run dev"
Start-Process powershell -ArgumentList "-NoProfile", "-NoExit", "-Command", $FrontendCmd -WindowStyle Minimized

Write-Host ""
Write-Host "==================================================" -ForegroundColor Green
Write-Host "   ACESSE NO NAVEGADOR: http://localhost:5173"
Write-Host "==================================================" -ForegroundColor Green
Write-Host ""
Write-Host "💡 Dica: Os logs estão sendo salvos em:" -ForegroundColor Cyan
Write-Host "   - Backend:  logs\backend.log" -ForegroundColor White
Write-Host "   - Frontend: logs\frontend.log" -ForegroundColor White
Write-Host ""
Write-Host "⚠️  Para parar os serviços, feche as janelas minimizadas do PowerShell" -ForegroundColor Yellow
Write-Host ""
Read-Host -Prompt "Pressione Enter para sair..."
