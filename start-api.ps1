# Start the Khmer Subtitle API (FastAPI + Uvicorn)
# Run from the workspace root: .\start-api.ps1

Write-Host "🚀 Starting Khmer Subtitle API on http://localhost:8000" -ForegroundColor Cyan

# Activate the existing Python venv
$venv = Join-Path $PSScriptRoot "py-venv\Scripts\Activate.ps1"
if (Test-Path $venv) {
    . $venv
    Write-Host "✅ Virtual environment activated." -ForegroundColor Green
} else {
    Write-Host "⚠️  No venv found at py-venv\. Using system Python." -ForegroundColor Yellow
}

# Install all dependencies from main requirements.txt
Write-Host "📦 Installing dependencies from requirements.txt…" -ForegroundColor Cyan
pip install -r requirements.txt -q

# Start uvicorn
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
