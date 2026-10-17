@echo off
setlocal EnableExtensions
cd /d "%~dp0.."

if exist ".venv\Scripts\activate.bat" call ".venv\Scripts\activate.bat"

set "PYTHONPATH=%CD%\src;%PYTHONPATH%"
set "CONFIG=%~1"
if "%CONFIG%"=="" set "CONFIG=configs\default.yaml"

echo Evaluating with: %CONFIG%
python -m swm.eval.evaluate --config "%CONFIG%"
exit /b %ERRORLEVEL%
