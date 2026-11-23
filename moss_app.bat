@echo off
rem MOSS-Image LoRA training UI in its own app window (pywebview + WebView2). In a web browser instead: run.bat
rem Usage: moss_app.bat [--port N]
rem Closing the window stops the UI only; a running training keeps going and is re-attached next time.
rem Errors go to ui_logs\app.log (no console window).
rem The venv's pythonw.exe (uv launcher) starts the base python.exe with a console, so the base pythonw.exe from
rem pyvenv.cfg "home" is used and moss_app.py adds the venv's site-packages itself.
setlocal
set PYTHONUTF8=1
set "VENV=%~dp0sd-scripts\venv"
if not exist "%VENV%\pyvenv.cfg" (
    echo Not installed yet. Run install.bat first.
    pause
    exit /b 1
)
set "PYHOME="
for /f "usebackq tokens=1,* delims==" %%a in ("%VENV%\pyvenv.cfg") do (
    for /f "tokens=*" %%k in ("%%a") do if /i "%%k"=="home " set "PYHOME=%%b"
    for /f "tokens=*" %%k in ("%%a") do if /i "%%k"=="home" set "PYHOME=%%b"
)
if defined PYHOME for /f "tokens=*" %%h in ("%PYHOME%") do set "PYHOME=%%h"
set "PYW=%PYHOME%\pythonw.exe"
if not exist "%PYW%" set "PYW=%VENV%\Scripts\pythonw.exe"
start "" "%PYW%" "%~dp0moss_app.py" %*
