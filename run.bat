@echo off
rem MOSS-Image LoRA training UI (local web page at http://127.0.0.1:7865/)
rem Usage: run.bat [--port N] [--no_browser]
rem Closing this window stops the UI only; a running training keeps going and is re-attached next time.
setlocal
set PYTHONUTF8=1
set "PY=%~dp0sd-scripts\venv\Scripts\python.exe"
if not exist "%PY%" (
  echo Not installed yet. Run install.bat first.
  pause
  exit /b 1
)
"%PY%" "%~dp0moss_ui.py" %*
if errorlevel 1 pause
