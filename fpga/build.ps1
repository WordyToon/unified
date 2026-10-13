# ==============================================================================
# Script: build.ps1
# Description: Automated Vivado build script for Unified c_engine on ZCU104
# ==============================================================================

param (
    [string]$Precision = "8bit",
    [string]$Board = "zcu106",
    [switch]$PackageOnly,
    [switch]$BdOnly,
    [double]$FreqMHz = 75.0,
    [switch]$Clean
)

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ScriptDir

# Normalize precision
if ($Precision -match "16") {
    $Precision = "16bit"
} else {
    $Precision = "8bit"
}
$env:PRECISION = $Precision
$env:BOARD = $Board
$env:TARGET_FREQ_MHZ = "$FreqMHz"

Write-Host "================================================================================" -ForegroundColor Cyan
Write-Host "  UNIFIED C_ENGINE VIVADO BUILD PIPELINE FOR AMD ZYNQ ULTRASCALE+ $( $Board.ToUpper() )" -ForegroundColor Cyan
Write-Host "  Board:           $Board" -ForegroundColor Cyan
Write-Host "  Precision:       $Precision" -ForegroundColor Cyan
Write-Host "  Clock Frequency: $FreqMHz MHz" -ForegroundColor Cyan
Write-Host "================================================================================" -ForegroundColor Cyan

# Set active RTL header
$HwConfigSrc = "..\rtl\hw_config_${Precision}.vh"
$HwConfigDst = "..\rtl\hw_config.vh"
if (Test-Path $HwConfigSrc) {
    Write-Host "[*] Configuring RTL header: $HwConfigSrc -> $HwConfigDst" -ForegroundColor Green
    Copy-Item -Path $HwConfigSrc -Destination $HwConfigDst -Force
}

if ($Clean) {
    Write-Host "[*] Cleaning previous build artifacts for $Precision..." -ForegroundColor Yellow
    Remove-Item -Path "vivado_prj", "ip_repo", "export/$Precision", "*.log", "*.jou" -Recurse -Force -ErrorAction SilentlyContinue
}

# Find Vivado
$VivadoCmd = Get-Command "vivado" -ErrorAction SilentlyContinue
if (-not $VivadoCmd) {
    $VivadoPaths = @(
        "C:\Xilinx\Vivado\2023.2\bin\vivado.bat",
        "C:\Xilinx\Vivado\2023.1\bin\vivado.bat",
        "C:\Xilinx\Vivado\2022.2\bin\vivado.bat",
        "D:\Xilinx\Vivado\2023.2\bin\vivado.bat"
    )
    foreach ($p in $VivadoPaths) {
        if (Test-Path $p) {
            $VivadoBin = $p
            break
        }
    }
} else {
    $VivadoBin = "vivado"
}

if (-not $VivadoBin) {
    Write-Error "Vivado executable not found in PATH or standard installation locations."
}

Write-Host "[*] Using Vivado: $VivadoBin" -ForegroundColor Green

if ($PackageOnly) {
    Write-Host "[*] Packaging IP Core..." -ForegroundColor Green
    & $VivadoBin -mode batch -source "package_ip.tcl" -notrace
    exit $LASTEXITCODE
}

if ($BdOnly) {
    Write-Host "[*] Creating Block Design at $FreqMHz MHz..." -ForegroundColor Green
    & $VivadoBin -mode batch -source "package_ip.tcl" -notrace
    $TclInit = "set target_freq_mhz $FreqMHz"
    & $VivadoBin -mode batch -source "create_bd.tcl" -tclargs $TclInit -notrace
    exit $LASTEXITCODE
}

# Full Flow (Packaging, Block Design, Synthesis, Implementation & Export)
$env:TARGET_FREQ_MHZ = "$FreqMHz"
& $VivadoBin -mode batch -source "build_bitstream.tcl" -notrace

Write-Host "================================================================================" -ForegroundColor Cyan
Write-Host "  BUILD COMPLETE FOR AMD ZCU106 ($Precision)!" -ForegroundColor Green
Write-Host "  Export Bitstream: fpga/export/$Precision/c_engine.bit" -ForegroundColor Green
Write-Host "  Export Platform:  fpga/export/$Precision/c_engine.xsa" -ForegroundColor Green
Write-Host "  Baremetal Target: baremetal/export/$Precision/c_engine.xsa" -ForegroundColor Green
Write-Host "================================================================================" -ForegroundColor Cyan

