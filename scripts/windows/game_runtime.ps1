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
