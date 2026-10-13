<#
================================================================================
 Script: scripts/05_build_baremetal.ps1
 Purpose: Compile Standalone Baremetal Applications for AMD ZCU106 Cortex-A53
 Toolchain: aarch64-none-elf-gcc (Vitis 2023.2)
================================================================================
#>
param(
    [ValidateSet("8bit", "16bit", "all")]
    [string]$Target = "all"
)

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$BaseDir   = Split-Path -Parent $ScriptDir
$BaremetalBuildScript = Join-Path $BaseDir "baremetal\build_elf.ps1"

Write-Host "================================================================================" -ForegroundColor Cyan
Write-Host "  STEP 05: BUILD STANDALONE BAREMETAL APPLICATION (CORTEX-A53)" -ForegroundColor Cyan
Write-Host "  Target: $Target" -ForegroundColor White
Write-Host "================================================================================" -ForegroundColor Cyan

& powershell -ExecutionPolicy Bypass -File "$BaremetalBuildScript" -Target $Target
if ($LASTEXITCODE -ne 0) {
    throw "Baremetal ELF build failed with exit code $LASTEXITCODE!"
}

Write-Host "`n[+] Standalone Baremetal Application Built Successfully!" -ForegroundColor Green
Write-Host "================================================================================" -ForegroundColor Cyan

