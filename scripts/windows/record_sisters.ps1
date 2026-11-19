param([ValidateRange(1,100)][int]$Episodes=1)
$ErrorActionPreference='Stop'
$repo=(Resolve-Path (Join-Path $PSScriptRoot '../..')).Path
. (Join-Path $PSScriptRoot 'game_runtime.ps1')
$s=Initialize-HKRLGameRuntime $repo
$dir=Join-Path $s.game_root 'hollow_knight_Data/Managed/Mods/HKRLEnvMod'
if (!(Get-Process hollow_knight -ErrorAction SilentlyContinue)) {
    & (Join-Path $PSScriptRoot 'start_sisters.ps1') -CheckOnly
}
$code=Invoke-HKRLConsole (Join-Path $repo '.venv/Scripts/python.exe') @(
    '-u','scripts/record_sisters_human.py','--output','runs/human-sisters',
    '--episodes',"$Episodes",'--mod-dir',('"' + $dir + '"')) $repo
exit $code
