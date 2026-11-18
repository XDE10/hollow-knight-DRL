param(
    [ValidateSet('Start', 'Stop', 'Status', 'Eval')][string]$Mode = 'Start',
    [int]$Updates = 20,
    [int]$Episodes = 10,
    [string]$RunDir = 'runs/training/sisters-counter-v2-seed0',
    [string]$TrainConfig = 'configs/train/sisters_counter_v2.yaml',
    [string]$Task = 'configs/experiments/sisters_counter.yaml'
)
$ErrorActionPreference = 'Stop'
$repo = (Resolve-Path (Join-Path $PSScriptRoot '../..')).Path
. (Join-Path $PSScriptRoot 'game_runtime.ps1')
$settings = Initialize-HKRLGameRuntime -RepoRoot $repo
$root = if ([IO.Path]::IsPathRooted($RunDir)) { $RunDir } else { Join-Path $repo $RunDir }
$statusPath = Join-Path $root 'status.json'
$activePath = Join-Path $repo 'runs/training/active.json'
if ($Mode -eq 'Status') {
    $active = Get-Content -Raw $activePath -ErrorAction SilentlyContinue | ConvertFrom-Json
    if ($active -and (Get-Process -Id $active.pid -ErrorAction SilentlyContinue)) {
        if (-not $PSBoundParameters.ContainsKey('RunDir') -or $root -eq $active.run_dir) {
            if ($active.mode -eq 'evaluation') {
                [ordered]@{ status = 'evaluating'; run_dir = $active.run_dir; pid = $active.pid } | ConvertTo-Json
                return
            }
            $statusPath = Join-Path $active.run_dir 'status.json'
        }
    }
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
    $arguments = @('-u', 'scripts/train.py', '--config', $TrainConfig,
        '--task', $Task, '--run-dir', ('"' + $root + '"'),
        '--updates', $Updates, '--checkpoint-every', '1', '--reset-timeout', '90', '--save-rollouts')
    if (-not (Test-Path (Join-Path $root 'checkpoints/index.jsonl'))) {
        if ([IO.Path]::GetFileName($TrainConfig) -eq 'sisters_counter_v2.yaml') {
            $initial = Join-Path $repo 'runs/training/sisters-gru-bc-ppo-seed0/checkpoints/checkpoint_v000021.pt'
            $demos = Join-Path $root 'demos'
            if (-not (Test-Path $initial) -or -not (Test-Path $demos)) {
                throw 'This profile starts from the offensive checkpoint and counterattack demonstrations. See docs/sisters_survival.md.'
            }
            $arguments += @('--init-weights', ('"' + $initial + '"'), '--demo-replay', ('"' + $demos + '"'))
        } else {
            $initial = Join-Path $root 'imitation/bc-weights.pt'
            if (Test-Path $initial) { $arguments += @('--init-weights', ('"' + $initial + '"')) }
        }
    }
} else {
    if (-not (Test-Path (Join-Path $root 'checkpoints/index.jsonl'))) { throw 'No checkpoint to evaluate.' }
    $replay = Join-Path $root ("eval-steps-" + (Get-Date -Format 'yyyyMMdd-HHmmss-fff') + ".jsonl")
    $arguments = @('-u', 'scripts/run_eval.py', '--policy', 'model',
        '--train-config', $TrainConfig,
        '--tasks', $Task,
        '--checkpoint-dir', ('"' + (Join-Path $root 'checkpoints') + '"'),
        '--episodes', $Episodes, '--seeds', '10001', '10002', '10003',
        '--local-session', ('"' + $root + '"'),
        '--replay-jsonl', ('"' + $replay + '"'),
        '--output', ('"' + (Join-Path $root 'eval-latest.json') + '"'))
}
$code = Invoke-HKRLConsole (Join-Path $repo '.venv/Scripts/python.exe') $arguments $repo
exit $code
