# One-time Windows setup for Qwen3-VL-2B LoRA World Model on NVIDIA RTX 3060 Ti (8GB VRAM, 32GB RAM)
# Run in PowerShell from semantic-world-model:
#   powershell -ExecutionPolicy Bypass -File .\scripts\setup_windows_3060ti.ps1
$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

$VenvDir = Join-Path $RepoRoot ".venv_win"
if (-not (Test-Path $VenvDir)) {
    Write-Host "Creating Python virtual environment at .venv_win ..."
    python -m venv $VenvDir
}

$Pip = Join-Path $VenvDir "Scripts\pip.exe"
$Py = Join-Path $VenvDir "Scripts\python.exe"

& $Py -m pip install --upgrade pip wheel setuptools

Write-Host "Installing PyTorch + CUDA 12.4 wheels for RTX 3060 Ti (Ampere SM 8.6)..."
& $Pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

Write-Host "Installing Qwen3-VL + LoRA + MQTT runtime dependencies..."
& $Pip install `
    "transformers>=4.57.0" `
    "accelerate>=0.33.0" `
    "peft>=0.12.0" `
    "bitsandbytes>=0.43.0" `
    "huggingface_hub>=0.24.0" `
    "safetensors>=0.4.0" `
    "sentencepiece>=0.2.0" `
    "protobuf>=4.25.0" `
    "pillow>=10.0.0" `
    "numpy>=2.0.0" `
    "pyyaml>=6.0" `
    "tqdm>=4.66.0" `
    "einops>=0.8.0" `
    "qwen-vl-utils>=0.0.8" `
    "paho-mqtt>=2.1.0"

& $Pip install --no-deps -e .

Write-Host ""
Write-Host "Verifying CUDA & RTX 3060 Ti VRAM..."
& $Py -c "import torch; assert torch.cuda.is_available(), 'CUDA not detected! Check NVIDIA driver.'; p = torch.cuda.get_device_properties(0); print(f'GPU Ready: {p.name} | VRAM: {p.total_memory / (1024**3):.2f} GB | bf16={torch.cuda.is_bf16_supported()}')"

Write-Host ""
Write-Host "Setup complete! Copy your checkpoint folder to:"
Write-Host "  outputs\microduck_semantic\checkpoint-final\"
Write-Host "Then run Terminal 2 with:"
Write-Host "  .\scripts\run_microduck_raw_demo.ps1 shuttle_ab"
