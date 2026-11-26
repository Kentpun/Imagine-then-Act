# Windows Qwen3-VL-2B Imagine-then-Act Planner Launcher (Terminal 2 — RTX 3060 Ti 8GB VRAM)
# Usage:
#   .\scripts\run_microduck_raw_demo.ps1 [shuttle_ab|walk_ab|keep_zone|conditional|kick_goal] [--4bit]
$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot
$env:PYTHONPATH = "$RepoRoot\src;$env:PYTHONPATH"
$env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True"

$PyBin = "python"
if (Test-Path ".venv_win\Scripts\python.exe") {
    $PyBin = ".venv_win\Scripts\python.exe"
} elseif (Test-Path ".venv\Scripts\python.exe") {
    $PyBin = ".venv\Scripts\python.exe"
}

$Preset = "push_ab"
$ExtraArgs = @()
if ($args.Count -gt 0 -and -not ($args[0].ToString().StartsWith("--"))) {
    $Preset = $args[0].ToString()
    if ($args.Count -gt 1) {
        $ExtraArgs = $args[1..($args.Count - 1)]
    }
} else {
    $ExtraArgs = $args
}

$Ckpt = if ($env:CHECKPOINT) { $env:CHECKPOINT } else { "outputs\microduck_semantic\checkpoint-final" }
if (-not (Test-Path $Ckpt)) {
    $Ckpt = "outputs\superpod\checkpoint-final"
    Write-Host "MicroDuck LoRA missing; falling back to LIBERO LoRA at $Ckpt"
}

$Config = if ($env:CONFIG) { $env:CONFIG } else { "configs\microduck_demo_win3060ti.yaml" }

$Part1Session = Join-Path (Split-Path -Parent $RepoRoot) "microduck_demo_part1\.demo\session.env"
$SessionArgs = @()
if (Test-Path $Part1Session) {
    $SessionArgs = @("--session-env", $Part1Session)
}

$DetArgs = @()
if (Test-Path "outputs\microduck_detector\best.pt") {
    $DetArgs = @("--detector-weights", "outputs\microduck_detector\best.pt")
}

& $PyBin -m swm.microduck.vision_loop `
    --mode imagine `
    --raw-control `
    --step-and-think `
    --task-preset $Preset `
    --checkpoint $Ckpt `
    --config $Config `
    @SessionArgs `
    @DetArgs `
    @ExtraArgs
