Write-Host "==================================================" -ForegroundColor Cyan
Write-Host "   DUBBLER PRO WEB (Backend + Frontend)"
Write-Host "==================================================" -ForegroundColor Cyan

$CurrentDir = Get-Location

Write-Host "[1/2] Iniciando Backend FastAPI (Porta 8000)..." -ForegroundColor Yellow
Start-Process powershell -ArgumentList "-NoExit", "-Command", "Set-Location '$CurrentDir'; & .\.venv\Scripts\Activate.ps1; uvicorn src.backend.app:app --reload --host 0.0.0.0 --port 8000" -WindowStyle Minimized

Write-Host "[2/2] Iniciando Frontend Vue.js (Porta 5173)..." -ForegroundColor Yellow
Start-Process powershell -ArgumentList "-NoExit", "-Command", "Set-Location '$CurrentDir\src\frontend'; npm run dev" -WindowStyle Minimized

Write-Host ""
Write-Host "==================================================" -ForegroundColor Green
Write-Host "   ACESSE NO NAVEGADOR: http://localhost:5173"
Write-Host "==================================================" -ForegroundColor Green
Write-Host ""
Read-Host -Prompt "Pressione Enter para sair..."
