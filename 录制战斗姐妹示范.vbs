Option Explicit
Dim shell, root, q, command
Set shell = CreateObject("WScript.Shell")
root = CreateObject("Scripting.FileSystemObject").GetParentFolderName(WScript.ScriptFullName)
shell.CurrentDirectory = root
q = Chr(34)
command = q & root & "\.venv\Scripts\pythonw.exe" & q & " scripts\windows\quiet_run.pyw runs\launcher\human-demo.log powershell.exe -NoProfile -NonInteractive -ExecutionPolicy Bypass -File scripts\windows\record_sisters.ps1"
shell.Run command, 0, False
