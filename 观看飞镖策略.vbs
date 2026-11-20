Option Explicit
Dim shell, files, root, q, command, copyRoot
Set shell = CreateObject("WScript.Shell")
Set files = CreateObject("Scripting.FileSystemObject")
root = files.GetParentFolderName(WScript.ScriptFullName)
shell.CurrentDirectory = root
q = Chr(34)
copyRoot = files.BuildPath(root, "runs\game-setup\collectors\5557")
command = q & root & "\.venv\Scripts\pythonw.exe" & q & " scripts\windows\quiet_run.pyw runs\launcher\return-guard-playback.log powershell.exe -NoProfile -NonInteractive -ExecutionPolicy Bypass -File scripts\windows\play_sisters_tactics.ps1 -Checkpoint models\sisters-return-guard-v3 -VerifyRuntime -SaveSlot 3 -Episodes 1"
If files.FileExists(files.BuildPath(copyRoot, "hollow_knight.exe")) Then
    command = command & " -GameRoot " & q & copyRoot & q & " -Port 5557"
End If
shell.Run command, 0, False
