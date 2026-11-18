param(
    [ValidateSet('Start', 'Stop', 'Status', 'Eval')][string]$Mode = 'Start',
    [int]$Updates = 20,
    [int]$Episodes = 10,
    [string]$RunDir = 'runs/training/sisters-gru-seed0'
)
$ErrorActionPreference = 'Stop'
$repo = (Resolve-Path (Join-Path $PSScriptRoot '../..')).Path
. (Join-Path $PSScriptRoot 'game_runtime.ps1')
$settings = Initialize-HKRLGameRuntime -RepoRoot $repo
$root = if ([IO.Path]::IsPathRooted($RunDir)) { $RunDir } else { Join-Path $repo $RunDir }
$statusPath = Join-Path $root 'status.json'
$activePath = Join-Path $repo 'runs/training/active.json'
if ($Mode -eq 'Status') {
    if (Test-Path $statusPath) { Get-Content $statusPath } else { Write-Host 'No training session yet.' }
    return
}
if ($Mode -eq 'Stop') {
    if (-not (Test-Path $activePath)) { Write-Host 'No active training process.'; return }
    $active = Get-Content -Raw $activePath | ConvertFrom-Json
    if ($active.mode -eq 'evaluation') { throw 'Evaluation is bounded by episode count; wait for it to finish.' }
    if (-not (Get-Process -Id $active.pid -ErrorAction SilentlyContinue)) {
        Write-Host 'Training is not running.'; return
    }
    New-Item -ItemType File -Path (Join-Path $active.run_dir 'STOP') -Force | Out-Null
    Write-Host 'Stop requested. Wait for this rollout and checkpoint to finish; keep the game open.'
    return
}
if (Test-Path $activePath) {
    $active = Get-Content -Raw $activePath | ConvertFrom-Json
    if (Get-Process -Id $active.pid -ErrorAction SilentlyContinue) {
        throw 'Training already owns this game. Use the stop launcher and wait before evaluation.'
    }
}
if ($Mode -eq 'Start') {
    $dll = Join-Path $settings.game_root 'hollow_knight_Data/Managed/Mods/HKRLEnvMod/HKRLEnvMod.dll'
    if (-not $settings.training_mod_sha256 -or
        (Get-FileHash $dll -Algorithm SHA256).Hash.ToLower() -ne $settings.training_mod_sha256) {
        throw 'The verified pause-safe training mod is missing or changed. Rebuild/deploy before training.'
    }
    & (Join-Path $PSScriptRoot 'start_sisters.ps1') -CheckOnly
    $arguments = @('-u', 'scripts/train.py', '--config', 'configs/train/sisters_cpu_gru.yaml',
        '--task', 'configs/tasks/sisters_of_battle.yaml', '--run-dir', ('"' + $root + '"'),
        '--updates', $Updates, '--checkpoint-every', '5', '--reset-timeout', '90', '--save-rollouts')
} else {
    if (-not (Test-Path (Join-Path $root 'checkpoints/index.jsonl'))) { throw 'No checkpoint to evaluate.' }
    $arguments = @('-u', 'scripts/run_eval.py', '--policy', 'model',
        '--train-config', 'configs/train/sisters_cpu_gru.yaml',
        '--tasks', 'configs/tasks/sisters_of_battle.yaml',
        '--checkpoint-dir', ('"' + (Join-Path $root 'checkpoints') + '"'),
        '--episodes', $Episodes, '--seeds', '10001', '10002', '10003',
        '--local-session', ('"' + $root + '"'),
        '--output', ('"' + (Join-Path $root 'eval-latest.json') + '"'))
}
$process = Start-Process -FilePath (Join-Path $repo '.venv/Scripts/python.exe') `
    -ArgumentList $arguments -WorkingDirectory $repo -NoNewWindow -Wait -PassThru
if ($process.ExitCode -ne 0) { throw "Training/evaluation failed (exit $($process.ExitCode)); see status.json." }
