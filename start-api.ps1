# Start the Khmer Subtitle API (FastAPI + Uvicorn) on http://localhost:8000
# Run from anywhere: .\start-api.ps1
# ASCII only: Windows PowerShell 5.1 misreads emoji in scripts saved without a BOM.

Set-Location $PSScriptRoot
Write-Host "Starting Khmer Subtitle API on http://localhost:8000" -ForegroundColor Cyan

# Always use the project's py-venv (GPU PyTorch), even if another environment is active.
$python = Join-Path $PSScriptRoot "py-venv\Scripts\python.exe"
if (Test-Path $python) {
    Write-Host "Using $python" -ForegroundColor Green
} else {
    Write-Host "No py-venv found; using the Python on PATH (create it: see README)." -ForegroundColor Yellow
    $python = "python"
}

Write-Host "Checking dependencies from requirements.txt..." -ForegroundColor Cyan
& $python -m pip install -r requirements.txt -q

# No --reload: a reload would restart the server and drop running transcription jobs.
& $python -m uvicorn api.main:app --host 0.0.0.0 --port 8000
