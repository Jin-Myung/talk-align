@echo off
set "TARGET=%~dp0run_win.bat"
set "ICON_PNG=%~dp0icon.png"
set "ICON_ICO=%~dp0icon.ico"
set "PWS=%~dp0create_shortcut.ps1"

echo Checking for icon...
if exist "%ICON_ICO%" goto :CreateShortcut

echo Converting icon.png to icon.ico...
powershell -ExecutionPolicy Bypass -File "%~dp0convert_icon.ps1" -InputPng "%ICON_PNG%" -OutputIco "%ICON_ICO%"

:CreateShortcut
echo Creating shortcut...
echo $WshShell = New-Object -comObject WScript.Shell > "%PWS%"
echo $DesktopPath = [System.Environment]::GetFolderPath('Desktop') >> "%PWS%"
echo $ShortcutPath = Join-Path $DesktopPath "Talk Align.lnk" >> "%PWS%"
echo $Shortcut = $WshShell.CreateShortcut($ShortcutPath) >> "%PWS%"
echo $Shortcut.TargetPath = "%TARGET%" >> "%PWS%"
echo $Shortcut.IconLocation = "%ICON_ICO%" >> "%PWS%"
echo $Shortcut.WindowStyle = 7 >> "%PWS%"
echo $Shortcut.Save() >> "%PWS%"

powershell -ExecutionPolicy Bypass -File "%PWS%"
del "%PWS%"

echo Shortcut created on Desktop!
pause
