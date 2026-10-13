<#
================================================================================
 Script: scripts/reproduce_all.ps1
 Purpose: Master End-to-End Orchestrator for AMD ZCU106 Accelerator Workflow
 Target:  AMD Zynq UltraScale+ ZCU106 (xczu7ev-ffvc1156-2-e)
 Precisions: 8bit, 16bit, or both
 Models:  MNIST LeNet-5, ResNet-50, MobileNetV2, EfficientNet-B0
 Steps Executed:
   01: Clean Workspace (Optional with -Clean)
   02: Compile Neural Network Models (Weights, Microcode, Test Images)
   03: Package Vivado IP (c_engine_dma_top)
   04: Build Bitstream & Export Hardware (.xsa)
   05: Compile Standalone Baremetal Applications (Cortex-A53 ELF)
   06: Run RTL Behavioral Simulation (XSim)
   07: Deploy to Physical Hardware (Optional with -Deploy)
================================================================================
#>
param(
    [ValidateSet("8bit", "16bit", "both")]
    [string]$Precision = "8bit",

    [switch]$Clean,
    [switch]$SkipSynthesis,
    [switch]$SkipSim,
    [switch]$Deploy,
    [string]$Model = "all"
)

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$BaseDir   = Split-Path -Parent $ScriptDir
Set-Location $BaseDir

Write-Host "================================================================================" -ForegroundColor Magenta
Write-Host "  AMD ZCU106 FULL TURNKEY WORKFLOW ORCHESTRATOR" -ForegroundColor Magenta
Write-Host "  Platform:  AMD Zynq UltraScale+ ZCU106 (xczu7ev-ffvc1156-2-e)" -ForegroundColor White
Write-Host "  Precision: $Precision" -ForegroundColor White
Write-Host "  Model:     $Model" -ForegroundColor White
Write-Host "================================================================================" -ForegroundColor Magenta

$PrecisionsToProcess = if ($Precision -eq "both") { @("8bit", "16bit") } else { @($Precision) }

# STEP 01: Clean
if ($Clean) {
    Write-Host "`n[>>>] Running Step 01: Clean Artifacts..." -ForegroundColor Cyan
    & "$ScriptDir\01_clean_all.ps1"
}

# STEP 02: Compile Models
Write-Host "`n[>>>] Running Step 02: Model Compilation..." -ForegroundColor Cyan
& "$ScriptDir\02_compile_models.ps1" -Precision $Precision

foreach ($prec in $PrecisionsToProcess) {
    Write-Host "`n********************************************************************************" -ForegroundColor Yellow
    Write-Host "  PROCESSING ARCHITECTURE: $prec" -ForegroundColor Yellow
    Write-Host "********************************************************************************" -ForegroundColor Yellow

    # STEP 03: Package IP
    Write-Host "`n[>>>] Running Step 03: Vivado IP Packaging ($prec)..." -ForegroundColor Cyan
    & "$ScriptDir\03_package_ip.ps1" -Precision $prec

    # STEP 04: Build Bitstream & Export XSA
    if (-not $SkipSynthesis) {
        Write-Host "`n[>>>] Running Step 04: Hardware Synthesis, Implementation & Bitstream ($prec)..." -ForegroundColor Cyan
        & "$ScriptDir\04_build_bitstream.ps1" -Precision $prec -Jobs 8
    } else {
        Write-Host "`n[*] Skipping Step 04 (Synthesis) as requested." -ForegroundColor DarkGray
    }

    # STEP 05: Build Baremetal Application
    Write-Host "`n[>>>] Running Step 05: Build Baremetal ELF ($prec)..." -ForegroundColor Cyan
    & "$ScriptDir\05_build_baremetal.ps1" -Target $prec

    # STEP 06: Run Simulation
    if (-not $SkipSim) {
        Write-Host "`n[>>>] Running Step 06: RTL Behavioral Simulation ($prec, Model: $Model)..." -ForegroundColor Cyan
        & "$ScriptDir\06_run_simulations.ps1" -Model $Model -Precision $prec
    } else {
        Write-Host "`n[*] Skipping Step 06 (Simulation) as requested." -ForegroundColor DarkGray
    }

    # STEP 07: Deploy to Hardware
    if ($Deploy) {
        Write-Host "`n[>>>] Running Step 07: Hardware Deployment on AMD ZCU106 ($prec)..." -ForegroundColor Cyan
        & "$ScriptDir\07_deploy_zcu106.ps1" -Model $Model -Precision $prec
    }
}

Write-Host "`n================================================================================" -ForegroundColor Green
Write-Host "  ALL WORKFLOW STEPS COMPLETED SUCCESSFULLY!" -ForegroundColor Green
Write-Host "================================================================================" -ForegroundColor Green

