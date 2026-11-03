@echo off
setlocal
cd /d "%~dp0"

rem Prefer a local virtual environment, then an explicit PYTHON variable, then PATH.
set "PY="
if exist ".venv\Scripts\python.exe" set "PY=.venv\Scripts\python.exe"
if not defined PY if defined PYTHON set "PY=%PYTHON%"
if not defined PY set "PY=python"

"%PY%" -c "import streamlit, torch, torchvision, PIL, pandas" >nul 2>nul
if errorlevel 1 (
    echo.
    echo [ERROR] Missing Python dependencies.
    echo Create a virtual environment and install them first:
    echo.
    echo     python -m venv .venv
    echo     .venv\Scripts\python -m pip install -r requirements.txt
    echo.
    pause
    exit /b 1
)

echo Starting Neural Style Studio...
echo The browser will open http://127.0.0.1:8501
echo Close this window to stop the server.
echo.
"%PY%" -m streamlit run app.py --server.port 8501
pause
