@echo off
REM Qwen3-VL-2B LoRA for the MicroDuck circle task. Forwards every argument to the PowerShell script.
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0run_microduck_vision.ps1" %*
exit /b %ERRORLEVEL%
