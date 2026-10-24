# Load the Qwen3-VL-2B general action LoRA and publish its generic action command.
#
# The simulator can stay on the Mac. On the Mac, start it with:
#   MICRODUCK_SKIP_VLM=1 ./scripts/run_simulator.sh --vision-ball
# Then point -Frame at the simulator's zone.ppm (a shared folder or a copy of the repo).
#
# Examples:
#   powershell -ExecutionPolicy Bypass -File .\scripts\run_microduck_vision.ps1
#   powershell -ExecutionPolicy Bypass -File .\scripts\run_microduck_vision.ps1 -Frame \\mac\microduck\zone.ppm
#   powershell -ExecutionPolicy Bypass -File .\scripts\run_microduck_vision.ps1 -LoraPython C:\venv\Scripts\python.exe

param(
    [string]$Frame = "",
    [string]$SessionEnv = "",
    [string]$Checkpoint = "",
    [string]$LoraPython = ""
)

$ErrorActionPreference = "Stop"
$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$DemoVision = Join-Path (Split-Path $RepoRoot -Parent) "microduck_demo_part1\.demo\vision\zone.ppm"

if (-not $Frame) { $Frame = $DemoVision }
if (-not $SessionEnv) {
    $localSession = Join-Path $RepoRoot ".demo\session.env"
    $demoSession = Join-Path (Split-Path $RepoRoot -Parent) "microduck_demo_part1\.demo\session.env"
    if (Test-Path $localSession) { $SessionEnv = $localSession }
    elseif (Test-Path $demoSession) { $SessionEnv = $demoSession }
    else { $SessionEnv = $localSession }
}
if (-not $Checkpoint) {
    $Checkpoint = Join-Path $RepoRoot "outputs\superpod_action\checkpoint-final"
}

foreach ($path in @($SessionEnv, $Checkpoint)) {
    if (-not (Test-Path $path)) {
        Write-Error "Missing $path"
    }
}

function Test-LoraCandidate([string]$Exe, [string[]]$Prefix) {
    if ($Exe -ne "py" -and $Exe -ne "python" -and -not (Test-Path $Exe)) {
        return $false
    }
    & $Exe @Prefix -c "import peft, torch" 2>$null
    return ($LASTEXITCODE -eq 0)
}

$pythonExe = $null
$pythonPrefix = @()
if ($LoraPython) {
    $pythonExe = $LoraPython
} elseif ($env:LORA_PYTHON) {
    $pythonExe = $env:LORA_PYTHON
} elseif (Test-LoraCandidate "py" @("-3.12")) {
    $pythonExe = "py"
    $pythonPrefix = @("-3.12")
} elseif (Test-LoraCandidate "python" @()) {
    $pythonExe = "python"
}

if (-not $pythonExe) {
    Write-Error "No Python with peft and torch found. Pass -LoraPython, or install them in the 3060 Ti environment."
}

& $pythonExe @pythonPrefix -c "import torch; print('CUDA available:' , torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')"
if (-not $?) {
    Write-Error "Could not import torch with $pythonExe"
}

$env:PYTHONPATH = "$(Join-Path $RepoRoot 'src');$env:PYTHONPATH"
Write-Host "Qwen3-VL-2B LoRA checkpoint: $Checkpoint"
Write-Host "MQTT session: $SessionEnv"
Write-Host "Overhead frame: $Frame"
Write-Host "The duck stands until the first generic action command is converted by MicroDuck."

& $pythonExe @pythonPrefix -m swm.microduck.vision_loop `
    --frame $Frame `
    --session-env $SessionEnv `
    --checkpoint $Checkpoint
exit $LASTEXITCODE
