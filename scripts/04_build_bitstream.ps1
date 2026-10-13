<#
================================================================================
 Script: scripts/04_build_bitstream.ps1
 Purpose: Synthesize, Implement, and Generate Hardware Bitstream (.bit) and
          Platform (.xsa) for AMD ZCU106 Evaluation Kit
 Target:  AMD Zynq UltraScale+ ZCU106 (xczu7ev-ffvc1156-2-e)
================================================================================
#>
param(
    [ValidateSet("8bit", "16bit", "both")]
    [string]$Precision = "8bit",
    [double]$FreqMHz = 75.0,
    [switch]$Clean
)

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$BaseDir   = Split-Path -Parent $ScriptDir
$FpgaDir   = Join-Path $BaseDir "fpga"

$TargetList = if ($Precision -eq "both") { @("8bit", "16bit") } else { @($Precision) }

# Locate Vivado
$Vivado = "C:\Xilinx\Vivado\2023.2\bin\vivado.bat"
if (-not (Test-Path $Vivado)) {
    $VivadoCmd = (Get-Command "vivado" -ErrorAction SilentlyContinue).Source
    if ($VivadoCmd) { $Vivado = $VivadoCmd }
    else { throw "Vivado 2023.2 executable not found!" }
}

foreach ($targetPrec in $TargetList) {
    Write-Host "================================================================================" -ForegroundColor Cyan
    Write-Host "  STEP 04: BUILD HARDWARE BITSTREAM & XSA FOR AMD ZCU106 ($targetPrec)" -ForegroundColor Cyan
    Write-Host "  Board:           AMD ZCU106 (xczu7ev-ffvc1156-2-e)" -ForegroundColor White
    Write-Host "  Precision:       $targetPrec" -ForegroundColor White
    Write-Host "  Target Clock:    $FreqMHz MHz" -ForegroundColor White
    Write-Host "================================================================================" -ForegroundColor Cyan

    # 1. Set RTL header
    $HwCfgSrc = Join-Path $BaseDir "rtl\hw_config_${targetPrec}.vh"
    $HwCfgDst = Join-Path $BaseDir "rtl\hw_config.vh"
    if (Test-Path $HwCfgSrc) {
        Copy-Item $HwCfgSrc $HwCfgDst -Force
        Write-Host "[*] Active RTL configuration set to: $HwCfgSrc" -ForegroundColor Green
    }

    # 2. Package IP
    Write-Host "[*] Packaging IP core with ${targetPrec} RTL..." -ForegroundColor Yellow
    $Env:PRECISION = $targetPrec
    $clkFreq = if ($targetPrec -eq "16bit" -and $FreqMHz -eq 75.0) { 70.0 } else { $FreqMHz }
    $Env:TARGET_FREQ_MHZ = "$clkFreq"

    # 3. Run Build Bitstream
    Write-Host "[*] Launching Vivado Synthesis, Implementation & Bitstream Generation..." -ForegroundColor Yellow
    & $Vivado -mode batch -source (Join-Path $FpgaDir "build_bitstream.tcl") -notrace
    if ($LASTEXITCODE -ne 0) {
        throw "Hardware build failed for $targetPrec with exit code $LASTEXITCODE!"
    }

    # Verify Outputs
    $BitFile = Join-Path $FpgaDir "export\$targetPrec\c_engine.bit"
    $XsaFile = Join-Path $FpgaDir "export\$targetPrec\c_engine.xsa"

    if ((Test-Path $BitFile) -and (Test-Path $XsaFile)) {
        Write-Host "[+] Hardware Artifacts Generated for ${targetPrec}:" -ForegroundColor Green
        Write-Host "    Bitstream: $BitFile ($("{0:N0}" -f (Get-Item $BitFile).Length) bytes)" -ForegroundColor Green
        Write-Host "    Platform:  $XsaFile ($("{0:N0}" -f (Get-Item $XsaFile).Length) bytes)" -ForegroundColor Green
    } else {
        throw "Failed to locate generated .bit or .xsa files for $targetPrec!"
    }
}

Write-Host "`n[+] Hardware Bitstream & XSA Generation Complete!" -ForegroundColor Green
Write-Host "================================================================================" -ForegroundColor Cyan

