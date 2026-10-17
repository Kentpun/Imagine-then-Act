@echo off
setlocal EnableExtensions
cd /d "%~dp0.."

if exist ".venv\Scripts\activate.bat" call ".venv\Scripts\activate.bat"

set "PYTHONPATH=%CD%\src;%PYTHONPATH%"
set "CONFIG=%~1"
if "%CONFIG%"=="" set "CONFIG=configs\default.yaml"

echo Training with config: %CONFIG%
python -m swm.training.train --config "%CONFIG%"
set "ERR=%ERRORLEVEL%"
if not "%ERR%"=="0" (
  echo Train failed with exit code %ERR%
  exit /b %ERR%
)
echo.
echo Checkpoint should be under outputs\  ^(see paths.output_dir in the config^)
endlocal
