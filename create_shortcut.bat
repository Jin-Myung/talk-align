@echo off
set "TARGET=%~dp0run_win.bat"
set "ICON=%~dp0icon.png"
set "SHORTCUT=%UserProfile%\Desktop\Talk Align.lnk"
set "PWS=%~dp0create_shortcut.ps1"

echo Creating shortcut...
echo $WshShell = New-Object -comObject WScript.Shell > "%PWS%"
echo $Shortcut = $WshShell.CreateShortcut("%SHORTCUT%") >> "%PWS%"
echo $Shortcut.TargetPath = "%TARGET%" >> "%PWS%"
echo $Shortcut.IconLocation = "%ICON%" >> "%PWS%"
echo $Shortcut.WindowStyle = 7 >> "%PWS%"
echo $Shortcut.Save() >> "%PWS%"

powershell -ExecutionPolicy Bypass -File "%PWS%"
del "%PWS%"

echo Shortcut created on Desktop!
pause
