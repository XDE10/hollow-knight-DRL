Option Explicit
Dim shell, files, root, q, command, copyRoot
Set shell = CreateObject("WScript.Shell")
Set files = CreateObject("Scripting.FileSystemObject")
root = files.GetParentFolderName(WScript.ScriptFullName)
shell.CurrentDirectory = root
q = Chr(34)
copyRoot = files.BuildPath(root, "runs\game-setup\collectors\5556")
command = q & root & "\.venv\Scripts\pythonw.exe" & q & " scripts\windows\quiet_run.pyw runs\launcher\world-playback.log powershell.exe -NoProfile -NonInteractive -ExecutionPolicy Bypass -File scripts\windows\play_sisters_tactics.ps1 -Checkpoint models\sisters-world-v2 -SaveSlot 3 -Episodes 1"
If files.FileExists(files.BuildPath(copyRoot, "hollow_knight.exe")) Then
    command = command & " -GameRoot " & q & copyRoot & q & " -Port 5556"
End If
shell.Run command, 0, False
