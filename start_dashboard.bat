@echo off
echo ====================================
echo PUBG Player Behavior Analysis
echo Model Performance Dashboard
echo ====================================
echo.

cd /d "%~dp0backend"

echo Checking Python installation...
python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python is not installed or not in PATH
    echo Please install Python 3.7+ and try again
    pause
    exit /b 1
)

echo.
echo Installing required packages...
python -m pip install fastapi uvicorn[standard] pydantic numpy --quiet

echo.
echo Starting PUBG Analysis Dashboard...
echo.
echo Dashboard will be available at:
echo   - Dashboard: http://localhost:8000/static/index.html
echo   - API Docs:  http://localhost:8000/docs
echo.
echo Press Ctrl+C to stop the server
echo.

python fastapi_app.py

pause
