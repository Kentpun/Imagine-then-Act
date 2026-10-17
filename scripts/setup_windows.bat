@echo off
setlocal EnableExtensions
cd /d "%~dp0.."

echo === Imagine-then-Act: Windows env setup ===
echo.

where python >nul 2>&1
if errorlevel 1 (
  echo ERROR: Python not found. Install Python 3.10+ from https://www.python.org/downloads/
  echo        and check "Add python.exe to PATH".
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
echo IMPORTANT: Install PyTorch for YOUR CUDA version first if training on GPU:
echo   https://pytorch.org
echo   Example CUDA 12.1:
echo   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
echo.
echo Then install the rest:
echo   pip install -r requirements.txt
echo.
set /p INSTALL_REQ=Install requirements.txt now? [Y/N]: 
if /I "%INSTALL_REQ%"=="Y" (
  pip install -r requirements.txt
  if errorlevel 1 exit /b 1
)

echo.
echo Done. Activate later with:
echo   .venv\Scripts\activate.bat
endlocal
