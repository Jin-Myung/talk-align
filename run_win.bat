@echo off
cd /d "%~dp0"

IF EXIST venv (
    call venv\Scripts\activate.bat
) ELSE (
    echo Error: 'venv' directory not found. Please run installation steps first.
    pause
    exit /b 1
)

echo Starting Web Server...
start "Talk Align Server" /min uvicorn server:app --host 0.0.0.0 --port 8000

timeout /t 2 /nobreak >nul

echo Opening Webpages...
start http://localhost:8000/public/operator.html
start http://localhost:8000/public/audience.html

echo Starting Main Program...
echo Press Ctrl+C to exit.
python main.py

echo Stopping Web Server...
taskkill /FI "WINDOWTITLE eq Talk Align Server" /T /F

pause
