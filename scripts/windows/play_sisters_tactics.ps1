[CmdletBinding()]
param(
    [Parameter(Mandatory=$true)][string]$Checkpoint,
    [string]$GameRoot,
    [ValidateRange(1,65535)][int]$Port=5555,
    [ValidateRange(1,4)][int]$SaveSlot=3,
    [ValidateRange(1,1000)][int]$Episodes=1
)
$ErrorActionPreference='Stop'
$repo=(Resolve-Path (Join-Path $PSScriptRoot '../..')).Path
$checkpointPath=(Resolve-Path -LiteralPath $Checkpoint).Path
. (Join-Path $PSScriptRoot 'game_runtime.ps1')
$settings=Initialize-HKRLGameRuntime -RepoRoot $repo
if (!$GameRoot) { $GameRoot=$settings.game_root }
$game=(Resolve-Path -LiteralPath (Join-Path $GameRoot 'hollow_knight.exe')).Path
$runtime=Join-Path $GameRoot 'hollow_knight_Data/Managed/Mods/HKRLEnvMod/HKRLEnvMod.dll'
$manifest=Get-Content -Raw -LiteralPath (Join-Path $checkpointPath 'frozen.json') | ConvertFrom-Json
if (!(Test-Path -LiteralPath $runtime)) { throw "HKRLEnvMod is missing from $GameRoot" }
if ((Get-Process -Name hollow_knight -ErrorAction SilentlyContinue | Where-Object Path -eq $game) -or
    (Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue)) {
    throw "The game copy or port $Port is in use. The selected game must start with the checkpoint enabled."
}
$session=Join-Path $repo ("runs/tactics-play/"+[guid]::NewGuid().ToString("N"))
$null=New-Item -ItemType Directory -Path $session
$policy=Join-Path $session 'model.txt'
Copy-Item -LiteralPath (Join-Path $checkpointPath 'model.txt') -Destination $policy
if ((Get-FileHash -LiteralPath $policy).Hash -ne $manifest.policy_sha256) { throw 'Policy digest differs.' }
if ($manifest.head_sha256) {
    Copy-Item -LiteralPath (Join-Path $checkpointPath 'head.weights') -Destination ($policy+'.weights')
    if ((Get-FileHash -LiteralPath ($policy+'.weights')).Hash -ne $manifest.head_sha256) { throw 'Movement weight digest differs.' }
}
if ($manifest.world_sha256) {
    Copy-Item -LiteralPath (Join-Path $checkpointPath 'world.weights') -Destination ($policy+'.wm.weights')
    if ((Get-FileHash -LiteralPath ($policy+'.wm.weights')).Hash -ne $manifest.world_sha256) { throw 'World student digest differs.' }
}
$previous=@{}
foreach($name in 'HKRL_HOST','HKRL_PORT','HKRL_SAVE_SLOT','HKRL_TACTICS_POLICY','HKRL_TACTICS_TRACE','HKRL_SISTERS_NAIL_MULTIPLIER'){
    $previous[$name]=[Environment]::GetEnvironmentVariable($name,'Process')
}
$child=$null
Push-Location $repo
try {
    $env:HKRL_HOST='127.0.0.1'
    $env:HKRL_PORT=[string]$Port
    $env:HKRL_SAVE_SLOT=[string]$SaveSlot
    $env:HKRL_TACTICS_POLICY=$policy
    $env:HKRL_TACTICS_TRACE='0'
    $env:HKRL_SISTERS_NAIL_MULTIPLIER='1'
    # The visible game is the requested playback window.
    $child=Start-Process -FilePath $game -WorkingDirectory $GameRoot -ArgumentList @(
        '-screen-fullscreen','0','-screen-width','960','-screen-height','540'
    ) -PassThru
    $ready=$false
    for($i=0;$i -lt 90;$i++){
        if($child.HasExited){throw 'The selected game closed before starting its local actor.'}
        $listener=Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue
        if($listener){
            if($listener.OwningProcess -ne $child.Id -or $listener.LocalAddress -ne '127.0.0.1'){throw 'Unexpected game listener.'}
            $ready=$true
            break
        }
        Start-Sleep -Seconds 1
    }
    if(!$ready){throw 'The selected game did not open its local endpoint.'}
    $output=Join-Path $session 'results.json'
    & (Join-Path $repo '.venv/Scripts/python.exe') 'scripts/eval_sisters_tactics.py' --checkpoint $checkpointPath --policy-file $policy --port $Port --episodes $Episodes --output $output
    if($LASTEXITCODE -ne 0){throw "Playback failed with exit code $LASTEXITCODE"}
    Write-Output $output
}
finally {
    if($child -and !$child.HasExited){$null=$child.CloseMainWindow()}
    Pop-Location
    foreach($name in $previous.Keys){[Environment]::SetEnvironmentVariable($name,$previous[$name],'Process')}
}
