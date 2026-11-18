param([switch]$Tests)
$ErrorActionPreference = "Stop"
$Repo = (Resolve-Path (Join-Path $PSScriptRoot "../..")).Path
$Python = Join-Path $Repo ".venv/Scripts/python.exe"
if (-not (Test-Path $Python)) { throw "Project Python is missing: $Python" }
$env:OMP_NUM_THREADS = "2"
$env:MKL_NUM_THREADS = "2"
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD = "1"
$env:NO_PROXY = "127.0.0.1,localhost,::1"
Push-Location $Repo
try {
    if ($Tests) {
        Push-Location (Join-Path $Repo "python")
        try {
            & $Python -m pytest tests -q -m "not integration" `
                --ignore=tests/test_linux_launch_scripts.py `
                --basetemp=../runs/pytest-local `
                --junitxml=../runs/validation/windows-tests.xml
            if ($LASTEXITCODE -ne 0) { throw "Python tests failed." }
        } finally { Pop-Location }
    }
    Write-Host "Synthetic compute check only. This does not run Hollow Knight."
    & $Python scripts/benchmark_compute.py --device cpu --output runs/validation/compute_cpu.json
    if ($LASTEXITCODE -ne 0) { throw "Compute check failed." }
    Write-Host "Results: runs/validation/compute_cpu.json"
} finally { Pop-Location }
