@echo off
setlocal EnableExtensions
cd /d "%~dp0.."

echo === Imagine-then-Act: Windows env setup ===
echo.

where python >nul 2>&1
if errorlevel 1 (
  echo ERROR: Python not found. Install Python 3.10+ and add it to PATH.
  exit /b 1
)

if not exist ".venv\Scripts\python.exe" (
  echo Creating venv...
  python -m venv .venv
  if errorlevel 1 exit /b 1
)

call ".venv\Scripts\activate.bat"
python -m pip install -U pip
pip install -e .
if errorlevel 1 exit /b 1

echo.
echo Next: install CUDA PyTorch from https://pytorch.org then run:
echo   pip install -r requirements.txt
echo.
echo Or auto-install requirements.txt now with:
echo   scripts\setup_windows.bat INSTALL_REQ
echo.

if /I "%~1"=="INSTALL_REQ" (
  echo Installing requirements.txt ...
  pip install -r requirements.txt
  if errorlevel 1 exit /b 1
)

echo Done. Activate with: .venv\Scripts\activate.bat
endlocal
