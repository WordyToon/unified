<#
================================================================================
 Script: scripts/01_clean_all.ps1
 Purpose: Clean all generated build artifacts and temporary files
 Target: AMD ZCU106 Baremetal & Unified RTL Accelerator Project
================================================================================
#>
param(
    [switch]$Force,
    [switch]$KeepExports
)

$ErrorActionPreference = "Continue"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$BaseDir   = Split-Path -Parent $ScriptDir

Write-Host "================================================================================" -ForegroundColor Cyan
Write-Host "  STEP 01: CLEAN ALL GENERATED BUILD ARTIFACTS" -ForegroundColor Cyan
Write-Host "  Base Directory: $BaseDir" -ForegroundColor White
Write-Host "================================================================================" -ForegroundColor Cyan

$ItemsToClean = @(
    (Join-Path $BaseDir "fpga\vivado_prj"),
    (Join-Path $BaseDir "fpga\ip_repo"),
    (Join-Path $BaseDir "fpga\.Xil"),
    (Join-Path $BaseDir "baremetal\build"),
    (Join-Path $BaseDir "sim\xsim.dir"),
    (Join-Path $BaseDir ".Xil"),
    (Join-Path $BaseDir "xsim.dir")
)

if (-not $KeepExports) {
    # Keep directory structure, remove contents if requested
    Write-Host "[*] Cleaning logs and temporary journal files..." -ForegroundColor Yellow
}

foreach ($item in $ItemsToClean) {
    if (Test-Path $item) {
        Write-Host "  [-] Removing: $item" -ForegroundColor Gray
        Remove-Item -Path $item -Recurse -Force -ErrorAction SilentlyContinue
    }
}

# Clean dangling logs and journals in project roots
$LogFiles = Get-ChildItem -Path $BaseDir -Include "*.log", "*.jou", "*.pb", "*.wdb", "*.str" -File -Recurse -ErrorAction SilentlyContinue |
    Where-Object { $_.FullName -notmatch "baremetal\\vitis_ws" }

foreach ($f in $LogFiles) {
    Remove-Item -Path $f.FullName -Force -ErrorAction SilentlyContinue
}

Write-Host "[+] Clean completed successfully!" -ForegroundColor Green
Write-Host "================================================================================" -ForegroundColor Cyan

