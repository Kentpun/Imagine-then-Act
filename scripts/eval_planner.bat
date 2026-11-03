@echo off
setlocal EnableExtensions
cd /d "%~dp0.."

if exist ".venv\Scripts\activate.bat" call ".venv\Scripts\activate.bat"

set "PYTHONPATH=%CD%\src;%PYTHONPATH%"
echo Planner eval...
python -m swm.eval.planner_eval %*
exit /b %ERRORLEVEL%
