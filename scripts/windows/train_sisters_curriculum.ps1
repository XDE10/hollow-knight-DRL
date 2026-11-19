param(
    [string]$RunDir = 'runs/training/sisters-curriculum-1119',
    [string]$Initial = 'runs/training/sisters-overnight-1119/frozen/selected-follow-201/checkpoint.pt'
)
$ErrorActionPreference = 'Stop'
$repo = (Resolve-Path (Join-Path $PSScriptRoot '../..')).Path
Set-Location $repo
. (Join-Path $PSScriptRoot 'game_runtime.ps1')
$null = Initialize-HKRLGameRuntime $repo
$py = Join-Path $repo '.venv/Scripts/python.exe'
$cfg = 'configs/train/sisters_platform.yaml'
$task = 'configs/experiments/sisters_counter.yaml'
$job = Join-Path $RunDir 'train'
New-Item -ItemType Directory -Force $RunDir | Out-Null

function Invoke-Python([string[]]$Cli) {
    $quoted = @($Cli | ForEach-Object { '"' + $_ + '"' })
    $code = Invoke-HKRLConsole $py (@('-u') + $quoted) $repo
    if ($code -ne 0) { throw "Python failed with exit code $code." }
}

function Evaluate-Checkpoint([string]$Weights, [int]$Port, [int]$Count, [string]$Tag, [int]$FirstSeed) {
    $seeds = @($FirstSeed..($FirstSeed + $Count - 1) | ForEach-Object { [string]$_ })
    $prefix = Join-Path $RunDir $Tag
    Invoke-Python (@('scripts/run_eval.py', '--policy', 'model', '--checkpoint', $Weights,
        '--train-config', $cfg, '--tasks', $task, '--port', [string]$Port, '--episodes', [string]$Count,
        '--reset-timeout', '90', '--reset-retries', '2', '--local-session', $RunDir,
        '--output', "$prefix.json", '--replay-jsonl', "$prefix.jsonl", '--seeds') + $seeds)
    if ($Count -eq 10) {
        Invoke-Python @('scripts/score_sisters_eval.py', "$prefix.json", "$prefix.jsonl", '--output', "$prefix-score.json")
    }
}

if (!(Test-Path -LiteralPath (Join-Path $RunDir 'pilot-passed'))) {
    $tag = 'pilot-' + (Get-Date -Format 'yyyyMMdd-HHmmss')
    Evaluate-Checkpoint $Initial 5557 4 $tag 33610
    $rows = Get-Content (Join-Path $RunDir "$tag.jsonl") | ForEach-Object { ConvertFrom-Json $_ }
    $advanced = @($rows | Where-Object observed_boss_count -GE 2 | Select-Object -ExpandProperty episode -Unique)
    if ($advanced.Count -eq 0) { throw 'Sixfold curriculum pilot did not reach the simultaneous phase.' }
    Set-Content -LiteralPath (Join-Path $RunDir 'pilot-passed') -Value "$tag advanced=$($advanced.Count)/4"
    Write-Output "CURRICULUM_READY $($advanced.Count)/4"
}

foreach ($mark in @(100, 240, 400)) {
    $score = Join-Path $RunDir "original-$mark-score.json"
    if (Test-Path -LiteralPath $score) { continue }
    $statusPath = Join-Path $job 'status.json'
    $status = if (Test-Path -LiteralPath $statusPath) { Get-Content -Raw $statusPath | ConvertFrom-Json }
    if (!$status -or $status.update -lt $mark) {
        Invoke-Python @('scripts/train.py', '--config', $cfg, '--task', $task, '--run-dir', $job,
            '--updates', [string]$mark, '--ports', '5555', '5556', '5557', '--checkpoint-every', '10',
            '--batch-inference', '--reset-timeout', '90', '--reset-retries', '2', '--init-weights', $Initial)
        $status = Get-Content -Raw $statusPath | ConvertFrom-Json
    }
    if ($status.update -ne $mark) { throw "Training stopped at update $($status.update)." }
    $tag = "original-$mark"
    if (Test-Path -LiteralPath (Join-Path $RunDir "$tag.jsonl")) {
        $archived = "$tag-" + (Get-Date -Format 'yyyyMMdd-HHmmss') + '.partial.jsonl'
        Rename-Item -LiteralPath (Join-Path $RunDir "$tag.jsonl") -NewName $archived
    }
    Evaluate-Checkpoint $status.last_checkpoint 5555 10 $tag 33410
    Write-Output "ORIGINAL_EVALUATION_COMPLETE $($status.total_steps)"
}
