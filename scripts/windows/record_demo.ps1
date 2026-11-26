[CmdletBinding()]
param(
    [ValidateSet("all", "untrained", "champion")][string]$Mode = "all",
    [string]$Model = "models/sisters_champion/sisters_champion.bin",
    [string]$GameRoot = "",
    [string]$OutputDir = "runs/demo_videos",
    [ValidateRange(0, 65535)][int]$Port = 0,
    [ValidateRange(0, 4)][int]$SaveSlot = 0,
    [int]$Fps = 30,
    [double]$UntrainedSpeed = 2.0,
    [double]$ChampionSpeed = 4.0,
    [double]$UntrainedGifSpeed = 1.5,
    [double]$ChampionGifSpeed = 5.0,
    [int]$UntrainedClips = 3,
    [double]$UntrainedTailSec = 10.0,
    [int]$GifWidth = 720,
    [int]$GifFps = 12,
    [int]$MaxAttempts = 6,
    [switch]$NoAudio,
    [switch]$UntrainedFullVideo,
    [switch]$AllowAnyWin,
    [switch]$KeepRaw
)

$ErrorActionPreference = "Stop"
$repo = (Resolve-Path (Join-Path $PSScriptRoot "../..")).Path
$py = $null
if ($env:VIRTUAL_ENV -and (Test-Path (Join-Path $env:VIRTUAL_ENV "Scripts/python.exe"))) {
    $py = Join-Path $env:VIRTUAL_ENV "Scripts/python.exe"
} elseif (Test-Path (Join-Path $repo ".venv/Scripts/python.exe")) {
    $py = Join-Path $repo ".venv/Scripts/python.exe"
} else {
    $py = "python"
}

$pyArgs = @(
    (Join-Path $repo "scripts/sisters/record_demo.py"),
    "--mode", $Mode,
    "--model", $Model,
    "--output-dir", $OutputDir,
    "--fps", [string]$Fps,
    "--untrained-speed", [string]$UntrainedSpeed,
    "--champion-speed", [string]$ChampionSpeed,
    "--untrained-gif-speed", [string]$UntrainedGifSpeed,
    "--champion-gif-speed", [string]$ChampionGifSpeed,
    "--untrained-clips", [string]$UntrainedClips,
    "--untrained-tail-sec", [string]$UntrainedTailSec,
    "--gif-width", [string]$GifWidth,
    "--gif-fps", [string]$GifFps,
    "--max-attempts", [string]$MaxAttempts
)
if ($Port -gt 0) { $pyArgs += @("--port", [string]$Port) }
if ($SaveSlot -gt 0) { $pyArgs += @("--save-slot", [string]$SaveSlot) }
if ($GameRoot) { $pyArgs += @("--game-root", $GameRoot) }
if ($NoAudio) { $pyArgs += "--no-audio" }
if ($UntrainedFullVideo) { $pyArgs += "--untrained-full-video" }
if ($AllowAnyWin) { $pyArgs += "--allow-any-win" }
if ($KeepRaw) { $pyArgs += "--keep-raw" }

Push-Location $repo
try {
    & $py @pyArgs
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}
finally {
    Pop-Location
}
