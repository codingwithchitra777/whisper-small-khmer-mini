# Start the Khmer Subtitle Web App (Next.js)
# Run from the workspace root: .\start-webapp.ps1

Write-Host "🌐 Starting Khmer Subtitle Web App on http://localhost:3000" -ForegroundColor Cyan

$webappDir = Join-Path $PSScriptRoot "webapp"

if (-not (Test-Path (Join-Path $webappDir "node_modules"))) {
    Write-Host "📦 Installing Node.js dependencies..." -ForegroundColor Yellow
    Push-Location $webappDir
    npm install
    Pop-Location
}

Push-Location $webappDir
npm run dev
Pop-Location
