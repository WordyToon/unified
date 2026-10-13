<#
================================================================================
 Script: scripts/03_package_ip.ps1
 Purpose: Package Unified c_engine DMA Top RTL as Vivado IP Core for AMD ZCU106
 Target:  AMD Zynq UltraScale+ ZCU106 (xczu7ev-ffvc1156-2-e)
================================================================================
#>
param(
    [ValidateSet("8bit", "16bit")]
    [string]$Precision = "8bit"
)

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$BaseDir   = Split-Path -Parent $ScriptDir
$FpgaDir   = Join-Path $BaseDir "fpga"

Write-Host "================================================================================" -ForegroundColor Cyan
Write-Host "  STEP 03: PACKAGE UNIFIED C_ENGINE DMA TOP IP" -ForegroundColor Cyan
Write-Host "  Precision: $Precision" -ForegroundColor White
Write-Host "================================================================================" -ForegroundColor Cyan

# Set Active Configuration Header
$HwCfgSrc = Join-Path $BaseDir "rtl\hw_config_${Precision}.vh"
$HwCfgDst = Join-Path $BaseDir "rtl\hw_config.vh"
if (Test-Path $HwCfgSrc) {
    Copy-Item $HwCfgSrc $HwCfgDst -Force
    Write-Host "[*] Active RTL configuration set to: $HwCfgSrc" -ForegroundColor Green
}

# Locate Vivado
$Vivado = "C:\Xilinx\Vivado\2023.2\bin\vivado.bat"
if (-not (Test-Path $Vivado)) {
    $VivadoCmd = (Get-Command "vivado" -ErrorAction SilentlyContinue).Source
    if ($VivadoCmd) { $Vivado = $VivadoCmd }
    else { throw "Vivado 2023.2 executable not found!" }
}

Write-Host "[*] Launching Vivado IP Packager..." -ForegroundColor Yellow
$Env:PRECISION = $Precision
$Env:BOARD = "zcu106"

# Run packaging directly in fpga directory
& $Vivado -mode batch -source (Join-Path $FpgaDir "package_ip.tcl") -notrace
if ($LASTEXITCODE -ne 0) {
    throw "Vivado IP packaging failed with exit code $LASTEXITCODE!"
}

$IpXml = Join-Path $FpgaDir "ip_repo\c_engine_dma_top\component.xml"
if (Test-Path $IpXml) {
    Write-Host "[+] Successfully packaged IP at: $FpgaDir\ip_repo\c_engine_dma_top" -ForegroundColor Green
} else {
    throw "IP component.xml not found after packaging!"
}
Write-Host "================================================================================" -ForegroundColor Cyan

