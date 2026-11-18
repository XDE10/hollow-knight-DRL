param([switch]$CheckOnly)
$ErrorActionPreference = 'Stop'
$repo = (Resolve-Path (Join-Path $PSScriptRoot '../..')).Path
$activePath = Join-Path $repo 'runs/training/active.json'
if (Test-Path $activePath) {
    $active = Get-Content -Raw $activePath | ConvertFrom-Json
    if (Get-Process -Id $active.pid -ErrorAction SilentlyContinue) {
        throw 'Training is controlling the game. Stop training and wait before resetting the arena.'
    }
}
. (Join-Path $PSScriptRoot 'game_runtime.ps1')
$settings = Initialize-HKRLGameRuntime -RepoRoot $repo
$python = Join-Path $repo '.venv/Scripts/python.exe'
if (-not (Test-Path $python)) { throw "Project Python is missing: $python" }
if (-not (Get-Process steam -ErrorAction SilentlyContinue)) {
    $steamPath = (Get-ItemProperty 'HKCU:\Software\Valve\Steam').SteamPath
    Start-Process (Join-Path $steamPath 'steam.exe') -ArgumentList '-silent'
    throw 'Steam is starting. Wait until it is signed in, then run this launcher again.'
}
if (-not (Get-Process hollow_knight -ErrorAction SilentlyContinue)) {
    Start-Process -FilePath (Join-Path $settings.game_root 'hollow_knight.exe') `
        -WorkingDirectory $settings.game_root `
        -ArgumentList '-screen-fullscreen', '0', '-screen-width', '1280', '-screen-height', '720', '-force-d3d11'
}

$ready = $false
for ($attempt = 0; $attempt -lt 60; $attempt++) {
    $socket = New-Object Net.Sockets.TcpClient
    try {
        $socket.Connect('127.0.0.1', 5555)
        $ready = $true
    } catch { Start-Sleep -Milliseconds 1000 }
    finally { $socket.Dispose() }
    if ($ready) { break }
}
if (-not $ready) { throw 'The game mod did not open its local endpoint within 60 seconds.' }

if ($CheckOnly) {
    $arguments = @('scripts/check_env.py', '--config', 'configs/train/local_cpu.yaml',
        '--task', 'configs/tasks/sisters_of_battle.yaml', '--timeout', '10')
} else {
    $arguments = @('scripts/live_action_demo.py', '--config', 'configs/train/local_cpu.yaml',
        '--task', 'configs/tasks/sisters_of_battle.yaml', '--reset-timeout', '90', '--command', 'status')
}
$process = Start-Process -FilePath $python -ArgumentList $arguments -WorkingDirectory $repo `
    -NoNewWindow -Wait -PassThru
if ($process.ExitCode -ne 0) { throw "Game setup check failed: $($process.ExitCode)" }
Write-Host 'Sisters of Battle / save slot 3 is ready.'
