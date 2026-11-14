param([int]$Port = 8000, [switch]$WithML)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $projectRoot
if (-not (Test-Path -LiteralPath '.venv/Scripts/python.exe')) {
    python -c "import sys; sys.exit(0 if sys.version_info >= (3,11) else 1)"
    if ($LASTEXITCODE -ne 0) { throw 'Python 3.11 or newer is required for patched security dependencies.' }
    python -m venv .venv
    if ($LASTEXITCODE -ne 0) { throw 'Virtual environment creation failed.' }
}
$projectPython = Join-Path $projectRoot '.venv/Scripts/python.exe'
$packageSpec = if ($WithML) { '.[ml,test]' } else { '.[test]' }
& $projectPython -m pip install -c constraints-security.txt -e $packageSpec
if ($LASTEXITCODE -ne 0) { throw 'Python dependency installation failed.' }
Push-Location -LiteralPath (Join-Path $projectRoot 'web')
try {
    npm ci
    if ($LASTEXITCODE -ne 0) { throw 'Frontend dependency installation failed.' }
    npm run build
    if ($LASTEXITCODE -ne 0) { throw 'Frontend build failed.' }
} finally { Pop-Location }
Write-Host "Sodium Twin is starting at http://127.0.0.1:$Port. Training is disabled."
& $projectPython -m sodium_twin --port $Port
