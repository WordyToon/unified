<#
================================================================================
 Script: scripts/02_compile_models.ps1
 Purpose: Compile all 4 Vision Neural Networks (LeNet-5, ResNet-50, MobileNetV2,
          EfficientNet-B0) with real pre-trained weights for AMD ZCU106
 Outputs: baremetal/data/*.bin, baremetal/src/model_microcodes.h
================================================================================
#>
param(
    [ValidateSet("8bit", "16bit", "both")]
    [string]$Precision = "both"
)

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$BaseDir   = Split-Path -Parent $ScriptDir
$CompileScript = Join-Path $BaseDir "models\compile_all.py"

Write-Host "================================================================================" -ForegroundColor Cyan
Write-Host "  STEP 02: COMPILE MODELS WITH PRE-TRAINED WEIGHTS & GENERATE ASSETS" -ForegroundColor Cyan
Write-Host "  Target Precision: $Precision" -ForegroundColor White
Write-Host "  Script:           $CompileScript" -ForegroundColor White
Write-Host "================================================================================" -ForegroundColor Cyan

# Locate Python
$PythonExe = (Get-Command "python" -ErrorAction SilentlyContinue).Source
if (-not $PythonExe) {
    if (Test-Path "C:\ProgramData\anaconda3\python.exe") {
        $PythonExe = "C:\ProgramData\anaconda3\python.exe"
    } else {
        throw "Python executable not found in PATH or standard anaconda installation!"
    }
}
Write-Host "[*] Using Python: $PythonExe" -ForegroundColor Green

# Execute Compiler
& $PythonExe "$CompileScript" --precision $Precision
if ($LASTEXITCODE -ne 0) {
    throw "Model compilation failed with exit code $LASTEXITCODE!"
}

# Verify Generated Binaries
$DataDir = Join-Path $BaseDir "baremetal\data"
Write-Host "`n[*] Verifying Generated Hardware Assets in $DataDir:" -ForegroundColor Yellow
$RequiredFiles = @(
    "mnist_microcode.bin", "mnist_weights.bin",
    "resnet50_8bit_microcode.bin", "resnet50_8bit_weights.bin",
    "mobilenetv2_8bit_microcode.bin", "mobilenetv2_8bit_weights.bin",
    "efficientnetb0_8bit_microcode.bin", "efficientnetb0_8bit_weights.bin"
)

if ($Precision -eq "16bit" -or $Precision -eq "both") {
    $RequiredFiles += @(
        "resnet50_16bit_microcode.bin", "resnet50_16bit_weights.bin",
        "mobilenetv2_16bit_microcode.bin", "mobilenetv2_16bit_weights.bin",
        "efficientnetb0_16bit_microcode.bin", "efficientnetb0_16bit_weights.bin"
    )
}

$AllFound = $true
foreach ($rf in $RequiredFiles) {
    $fPath = Join-Path $DataDir $rf
    if (Test-Path $fPath) {
        $len = (Get-Item $fPath).Length
        Write-Host "  [OK] $rf ($("{0:N0}" -f $len) bytes)" -ForegroundColor Green
    } else {
        Write-Host "  [MISSING] $rf" -ForegroundColor Red
        $AllFound = $false
    }
}

if (-not $AllFound) {
    throw "One or more required hardware assets were not generated!"
}

$HeaderPath = Join-Path $BaseDir "baremetal\src\model_microcodes.h"
if (Test-Path $HeaderPath) {
    Write-Host "  [OK] baremetal/src/model_microcodes.h updated successfully!" -ForegroundColor Green
}

Write-Host "`n[+] All Models Compiled and Assets Verified Successfully!" -ForegroundColor Green
Write-Host "================================================================================" -ForegroundColor Cyan

