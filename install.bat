@echo off
rem One-click installer. Downloads everything into this folder.
setlocal
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\install.ps1"
set "RC=%ERRORLEVEL%"
if not defined NOPAUSE pause
exit /b %RC%
