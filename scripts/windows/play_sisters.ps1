[CmdletBinding()]
param(
    [string]$Model = "models/sisters_champion/sisters_champion.bin",
    [string]$GameRoot = "",
    [ValidateRange(1, 1000)][int]$Episodes = 1,
    [int]$Seed = 19900001,
    [ValidateRange(1, 65535)][int]$Port = 5555,
    [ValidateRange(1, 4)][int]$SaveSlot = 3,
    [switch]$Headless
)

$ErrorActionPreference = "Stop"
$repo = (Resolve-Path (Join-Path $PSScriptRoot "../..")).Path
$py = Join-Path $repo ".venv/Scripts/python.exe"
if (-not (Test-Path -LiteralPath $py)) { $py = "python" }

$pyArgs = @(
    (Join-Path $repo "scripts/sisters/play.py"),
    "--model", $Model,
    "--episodes", [string]$Episodes,
    "--seed", [string]$Seed,
    "--port", [string]$Port,
    "--save-slot", [string]$SaveSlot
)
if ($GameRoot) { $pyArgs += @("--game-root", $GameRoot) }
if ($Headless) { $pyArgs += "--headless" }

Push-Location $repo
try {
    & $py @pyArgs
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}
finally {
    Pop-Location
}
