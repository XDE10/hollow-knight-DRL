# Shared local game setup. Machine paths remain in ignored runs/game-setup.
function Initialize-HKRLGameRuntime {
    param([string]$RepoRoot)
    $settingsPath = Join-Path $RepoRoot "runs/game-setup/local-game.json"
    if (-not (Test-Path $settingsPath)) { throw "Local game setup is missing: $settingsPath" }
    $settings = Get-Content -Raw $settingsPath | ConvertFrom-Json
    $configPath = Join-Path $settings.game_root "hollow_knight_Data/Managed/Mods/HKRLEnvMod/hkrl-runtime.conf"
    $values = @{}
    foreach ($line in Get-Content $configPath) {
        if ($line -match '^\s*(HKRL_[A-Z_]+)\s*=(.*)$') {
            $values[$Matches[1]] = $Matches[2].Trim()
        }
    }
    if ($values['HKRL_SAVE_SLOT'] -ne '3') { throw "This project launcher requires save slot 3." }
    if ($values['HKRL_HOST'] -ne '127.0.0.1') { throw "The local game endpoint must be loopback." }
    $env:HKRL_AUTH_TOKEN = $values['HKRL_AUTH_TOKEN']
    $env:HKRL_SAVE_SLOT = '3'
    $env:OMP_NUM_THREADS = '1'
    $env:MKL_NUM_THREADS = '1'
    # The game is licensed through the running Steam client.
    $env:SteamAppId = '367520'
    $env:SteamGameId = '367520'
    return $settings
}

# Start a console executable without allocating a Windows Terminal window.
# Callers supply the same quoted native arguments as Start-Process -ArgumentList.
function Invoke-HKRLConsole {
    param([string]$Executable, [string[]]$Arguments, [string]$WorkingDirectory)
    $info = New-Object System.Diagnostics.ProcessStartInfo
    $info.FileName = $Executable
    $info.Arguments = $Arguments -join ' '
    $info.WorkingDirectory = $WorkingDirectory
    $info.UseShellExecute = $false
    $info.CreateNoWindow = $true
    $info.RedirectStandardOutput = $true
    $info.RedirectStandardError = $true
    $child = [System.Diagnostics.Process]::Start($info)
    try {
        $errorText = $child.StandardError.ReadToEndAsync()
        while ($null -ne ($line = $child.StandardOutput.ReadLine())) {
            [Console]::Out.WriteLine($line)
        }
        $child.WaitForExit()
        [Console]::Error.Write($errorText.GetAwaiter().GetResult())
        return $child.ExitCode
    }
    finally { $child.Dispose() }
}
