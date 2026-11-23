@echo off
rem MOSS-Image LoRA training launcher (console, without the UI)
rem Usage: train_moss.bat [config.toml] [extra args...]
rem   Without a config, moss_lora_32gb.toml next to this file is used.
rem   Extra args override the config, e.g. train_moss.bat my.toml --max_train_epochs 20
setlocal
set PYTHONUTF8=1
rem Persist Triton autotune results (convrot_int8) so later runs skip the per-bucket tuning
if not defined TRITON_CACHE_AUTOTUNING set TRITON_CACHE_AUTOTUNING=1
set "ROOT=%~dp0sd-scripts"
set "CFG=%~1"
if "%CFG%"=="" (set "CFG=%~dp0moss_lora_32gb.toml") else shift

set "EXTRA="
:collect
if "%~1"=="" goto run
set "EXTRA=%EXTRA% %1"
shift
goto collect

:run
cd /d "%ROOT%"
"%ROOT%\venv\Scripts\python.exe" -m accelerate.commands.launch --num_cpu_threads_per_process 1 --num_processes 1 --num_machines 1 --mixed_precision bf16 --dynamo_backend no ^
  "%~dp0moss_train.py" --config_file "%CFG%" %EXTRA%
set "RC=%ERRORLEVEL%"
echo.
echo exit code: %RC%
if not defined NOPAUSE pause
exit /b %RC%
