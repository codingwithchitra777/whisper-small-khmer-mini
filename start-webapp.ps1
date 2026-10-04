# Start the Khmer Subtitle Web App (Next.js) on http://localhost:3000
# Run from anywhere: .\start-webapp.ps1
# ASCII only: Windows PowerShell 5.1 misreads emoji in scripts saved without a BOM.

Write-Host "Starting Khmer Subtitle Web App on http://localhost:3000" -ForegroundColor Cyan

$webappDir = Join-Path $PSScriptRoot "webapp"

if (-not (Test-Path (Join-Path $webappDir "node_modules"))) {
    Write-Host "Installing Node.js dependencies..." -ForegroundColor Yellow
    Push-Location $webappDir
    npm install
    Pop-Location
}

Push-Location $webappDir
npm run dev
Pop-Location
