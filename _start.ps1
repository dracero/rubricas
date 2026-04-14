$ErrorActionPreference = "Continue"
$root = $PSScriptRoot

Write-Host "========================================="
Write-Host "Iniciando RubricAI (Skills Architecture)"
Write-Host "========================================="

# Set UTF-8 for LiteLLM compatibility
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
$env:UV_LINK_MODE = "copy"

Write-Host "1/2 Iniciando Servidor Backend..."
$backend = Start-Process -NoNewWindow -PassThru -FilePath "uv" -ArgumentList "run", "python", "-m", "app.server" -WorkingDirectory $root

Write-Host "Esperando a que el backend arranque..."
$maxWait = 30
$waited = 0
while ($waited -lt $maxWait) {
    Start-Sleep -Seconds 2
    $waited += 2
    try {
        $response = Invoke-WebRequest -Uri "http://localhost:8000/" -UseBasicParsing -TimeoutSec 2 -ErrorAction SilentlyContinue
        if ($response.StatusCode -eq 200) {
            Write-Host "Backend listo en $waited segundos."
            break
        }
    } catch {
        Write-Host "  Esperando backend... ($waited s)"
    }
}

if ($waited -ge $maxWait) {
    Write-Host "ADVERTENCIA: El backend no respondio en $maxWait segundos. Revisa los logs."
}

Write-Host "2/2 Iniciando Frontend..."
$frontend = Start-Process -NoNewWindow -PassThru -FilePath "npm" -ArgumentList "run", "dev" -WorkingDirectory "$root\frontend"

Write-Host ""
Write-Host "Todos los servicios han sido iniciados."
Write-Host "   - Backend: http://localhost:8000"
Write-Host "   - Frontend: http://localhost:5173"
Write-Host ""
Write-Host "Presiona Ctrl+C para detener."

try {
    while (-not $backend.HasExited -or -not $frontend.HasExited) {
        Start-Sleep -Seconds 2
    }
} finally {
    Write-Host "Cerrando todos los servicios..."
    if (-not $backend.HasExited) { Stop-Process -Id $backend.Id -Force -ErrorAction SilentlyContinue }
    if (-not $frontend.HasExited) { Stop-Process -Id $frontend.Id -Force -ErrorAction SilentlyContinue }
}
