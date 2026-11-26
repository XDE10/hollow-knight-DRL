[CmdletBinding()]
param(
    [string]$Model = "models/sisters_champion/sisters_champion.bin",
    [string]$GameRoot = "",
    [ValidateRange(1, 1000)][int]$Episodes = 1,
    [int]$Seed = 19900001,
    [ValidateRange(0, 65535)][int]$Port = 0,
    [ValidateRange(0, 4)][int]$SaveSlot = 0,
    [switch]$Headless,
    [switch]$AutoInstallMod
)

$ErrorActionPreference = "Stop"
$repo = (Resolve-Path (Join-Path $PSScriptRoot "../..")).Path

# Prefer active venv/conda python, then local .venv, then PATH python
$py = $null
if ($env:VIRTUAL_ENV -and (Test-Path (Join-Path $env:VIRTUAL_ENV "Scripts/python.exe"))) {
    $py = Join-Path $env:VIRTUAL_ENV "Scripts/python.exe"
} elseif (Test-Path (Join-Path $repo ".venv/Scripts/python.exe")) {
    $py = Join-Path $repo ".venv/Scripts/python.exe"
} else {
    $py = "python"
}

$pyArgs = @(
    (Join-Path $repo "scripts/sisters/play.py"),
    "--model", $Model,
    "--episodes", [string]$Episodes,
    "--seed", [string]$Seed
)
if ($Port -gt 0) { $pyArgs += @("--port", [string]$Port) }
if ($SaveSlot -gt 0) { $pyArgs += @("--save-slot", [string]$SaveSlot) }
if ($GameRoot) { $pyArgs += @("--game-root", $GameRoot) }
if ($Headless) { $pyArgs += "--headless" }
if ($AutoInstallMod) { $pyArgs += "--auto-install-mod" }

Push-Location $repo
try {
    & $py @pyArgs
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}
finally {
    Pop-Location
}
