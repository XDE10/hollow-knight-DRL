param(
    [ValidateSet("Check", "Smoke")][string]$Mode = "Check",
    [ValidateSet("mantis_lords", "sisters_of_battle", "hornet_protector", "gruz_mother")]
    [string]$Task = "sisters_of_battle",
    [int]$Port = 5555
)
$ErrorActionPreference = "Stop"
$Repo = (Resolve-Path (Join-Path $PSScriptRoot "../..")).Path
. (Join-Path $PSScriptRoot "game_runtime.ps1")
$null = Initialize-HKRLGameRuntime -RepoRoot $Repo
$Python = Join-Path $Repo ".venv/Scripts/python.exe"
if (-not (Test-Path $Python)) { throw "Project Python is missing: $Python" }
$env:OMP_NUM_THREADS = "1"
$env:MKL_NUM_THREADS = "1"
Push-Location $Repo
try {
    & $Python scripts/check_env.py --config configs/train/local_cpu.yaml `
        --task "configs/tasks/$Task.yaml" --host 127.0.0.1 --port $Port --timeout 3
    if ($LASTEXITCODE -ne 0) {
        throw "Start Hollow Knight with HKRLEnvMod and a Godhome save first."
    }
    if ($Mode -eq "Smoke") {
        $Stamp = Get-Date -Format "yyyyMMdd-HHmmss"
        & $Python scripts/train.py --config configs/train/local_cpu.yaml `
            --task "configs/tasks/$Task.yaml" --host 127.0.0.1 --port $Port `
            --smoke --steps 128 --metrics "runs/game-smoke-$Stamp.jsonl"
        if ($LASTEXITCODE -ne 0) { throw "Live game smoke failed." }
    }
} finally { Pop-Location }
