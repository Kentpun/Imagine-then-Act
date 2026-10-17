@echo off
setlocal EnableExtensions
cd /d "%~dp0.."

if exist ".venv\Scripts\activate.bat" call ".venv\Scripts\activate.bat"

set "PYTHONPATH=%CD%\src;%PYTHONPATH%"
echo Downloading LIBERO ^(lerobot/libero^) ...
python scripts\download_data.py --repo_id lerobot/libero
exit /b %ERRORLEVEL%
