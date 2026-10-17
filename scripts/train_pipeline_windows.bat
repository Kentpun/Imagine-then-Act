@echo off
setlocal EnableExtensions
cd /d "%~dp0.."

set "CONFIG=%~1"
if "%CONFIG%"=="" set "CONFIG=configs\default.yaml"
set "SKIP_DOWNLOAD=%~2"

echo ========================================
echo  Imagine-then-Act Windows train pipeline
echo  Config: %CONFIG%
echo ========================================
echo.

if not exist ".venv\Scripts\python.exe" (
  echo venv missing - running setup_windows.bat ...
  call scripts\setup_windows.bat
  if errorlevel 1 exit /b 1
)

call ".venv\Scripts\activate.bat"
set "PYTHONPATH=%CD%\src;%PYTHONPATH%"

where nvidia-smi >nul 2>&1
if errorlevel 1 (
  echo WARNING: nvidia-smi not found. Training needs NVIDIA GPU + CUDA PyTorch.
  echo.
)

if /I not "%SKIP_DOWNLOAD%"=="SKIP_DOWNLOAD" (
  echo [1/3] Download LIBERO...
  python scripts\download_data.py --repo_id lerobot/libero
  if errorlevel 1 exit /b 1
) else (
  echo [1/3] Skip download
)

echo [2/3] Prepare labeled transitions...
python scripts\prepare_dataset.py --config "%CONFIG%"
if errorlevel 1 exit /b 1

echo [3/3] Train LoRA...
python -m swm.training.train --config "%CONFIG%"
if errorlevel 1 exit /b 1

echo.
echo Done. Next run: scripts\eval.bat %CONFIG%
endlocal
