Option Explicit
Dim shell, root, q, command
Set shell = CreateObject("WScript.Shell")
root = CreateObject("Scripting.FileSystemObject").GetParentFolderName(WScript.ScriptFullName)
shell.CurrentDirectory = root
q = Chr(34)
command = q & root & "\.venv\Scripts\pythonw.exe" & q & " scripts\windows\quiet_run.pyw runs\launcher\arena.log powershell.exe -NoProfile -NonInteractive -ExecutionPolicy Bypass -File scripts\windows\start_sisters.ps1"
shell.Run command, 0, False
