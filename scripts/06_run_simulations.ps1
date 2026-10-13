<#
================================================================================
 Script: scripts/06_run_simulations.ps1
 Purpose: Run cycle-accurate RTL behavioral simulation via Vivado XSim for:
          1. MNIST LeNet-5
          2. ResNet-50
          3. MobileNetV2
          4. EfficientNet-B0
 Target:  AMD ZCU106 RTL Systolic Array Accelerator (256 PEs @ 75 MHz)
================================================================================
#>
param(
    [ValidateSet("all", "mnist", "resnet50", "mobilenetv2", "efficientnetb0")]
    [string]$Model = "all",
    [ValidateSet("8bit", "16bit")]
    [string]$Precision = "8bit"
)

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$BaseDir   = Split-Path -Parent $ScriptDir
$SimDir    = Join-Path $BaseDir "sim"

# Locate Vivado tools
$VivadoBin = "C:\Xilinx\Vivado\2023.2\bin"
if (-not (Test-Path "$VivadoBin\xvlog.bat")) {
    $XvlogCmd = (Get-Command "xvlog" -ErrorAction SilentlyContinue).Source
    if ($XvlogCmd) { $VivadoBin = Split-Path -Parent $XvlogCmd }
    else { throw "Vivado simulator tools (xvlog/xelab/xsim) not found!" }
}

$ModelsToRun = if ($Model -eq "all") { @("mnist", "mobilenetv2", "efficientnetb0", "resnet50") } else { @($Model) }

Write-Host "================================================================================" -ForegroundColor Cyan
Write-Host "  STEP 06: RUN CYCLE-ACCURATE RTL SIMULATION (VIVADO XSIM)" -ForegroundColor Cyan
Write-Host "  Model:     $Model" -ForegroundColor White
Write-Host "  Precision: $Precision" -ForegroundColor White
Write-Host "  Simulator: Vivado 2023.2 XSim" -ForegroundColor White
Write-Host "================================================================================" -ForegroundColor Cyan

# Set active RTL configuration header
$HwCfgSrc = Join-Path $BaseDir "rtl\hw_config_${Precision}.vh"
$HwCfgDst = Join-Path $BaseDir "rtl\hw_config.vh"
if (Test-Path $HwCfgSrc) {
    Copy-Item $HwCfgSrc $HwCfgDst -Force
    Write-Host "[*] Active RTL configuration set to: $HwCfgSrc" -ForegroundColor Green
}

Push-Location $SimDir
try {
    foreach ($m in $ModelsToRun) {
        Write-Host "`n--------------------------------------------------------------------------------" -ForegroundColor Yellow
        Write-Host "  Running Simulation for: $($m.ToUpper()) ($Precision)" -ForegroundColor Yellow
        Write-Host "--------------------------------------------------------------------------------" -ForegroundColor Yellow

    # Copy model weights hex files
    if ($m -eq "mobilenetv2") {
        $simMacro = "SIM_MOBILENETV2"
        $wHex = Join-Path $SimDir "mobilenetv2_weights_${Precision}.hex"
        $gHex = Join-Path $SimDir "mobilenetv2_weights_gemv_${Precision}.hex"
    } elseif ($m -eq "efficientnetb0") {
        $simMacro = "SIM_EFFICIENTNETB0"
        $wHex = Join-Path $SimDir "efficientnetb0_weights_${Precision}.hex"
        $gHex = Join-Path $SimDir "efficientnetb0_weights_gemv_${Precision}.hex"
    } elseif ($m -eq "mnist") {
        $simMacro = "SIM_MNIST"
        $wHex = Join-Path $SimDir "weights_${Precision}.hex"
        $gHex = Join-Path $SimDir "weights_gemv_${Precision}.hex"
    } else {
        $simMacro = "SIM_RESNET50"
        $wHex = Join-Path $SimDir "weights_${Precision}.hex"
        $gHex = Join-Path $SimDir "weights_gemv_${Precision}.hex"
    }

    if (Test-Path $wHex) { Copy-Item $wHex (Join-Path $SimDir "weights.hex") -Force }
    if (Test-Path $gHex) { Copy-Item $gHex (Join-Path $SimDir "weights_gemv.hex") -Force }

    $RtlFiles = @(
        (Join-Path $BaseDir "rtl\act_lut_rom.v"),
        (Join-Path $BaseDir "rtl\config_mem.v"),
        (Join-Path $BaseDir "rtl\processing_element.v"),
        (Join-Path $BaseDir "rtl\c_engine.v"),
        (Join-Path $BaseDir "rtl\multibank_dualport_mem.v"),
        (Join-Path $BaseDir "rtl\requantization_block.v"),
        (Join-Path $BaseDir "rtl\c_engine_controller.v"),
        (Join-Path $BaseDir "rtl\c_engine_top.v"),
        (Join-Path $BaseDir "rtl\axi_lite_slave.v"),
        (Join-Path $BaseDir "rtl\axi4_master_rd.v"),
        (Join-Path $BaseDir "rtl\axi4_master_wr.v"),
        (Join-Path $BaseDir "rtl\dma_sram_bridge.v"),
        (Join-Path $BaseDir "rtl\c_engine_dma_top.v"),
        (Join-Path $SimDir "axi4_ddr_model.v"),
        (Join-Path $SimDir "tb_unified_c_engine.v")
    )

    $RtlInc = Join-Path $BaseDir "rtl"

    # 1. xvlog
    Write-Host "[1/3] Compiling RTL with xvlog..." -ForegroundColor Gray
    & "$VivadoBin\xvlog.bat" -nolog -d $simMacro -i "$RtlInc" -i "$SimDir" @RtlFiles
    if ($LASTEXITCODE -ne 0) {
        throw "xvlog compilation failed for $m!"
    }

    # 2. xelab
    Write-Host "[2/3] Elaborating design with xelab..." -ForegroundColor Gray
    & "$VivadoBin\xelab.bat" -nolog -debug typical work.tb_unified_c_engine -s "sim_snapshot_$m"
    if ($LASTEXITCODE -ne 0) {
        throw "xelab elaboration failed for $m!"
    }

    # 3. xsim
    Write-Host "[3/3] Executing simulation with xsim..." -ForegroundColor Gray
    $SimOut = & "$VivadoBin\xsim.bat" -nolog "sim_snapshot_$m" -runall 2>&1
    
    # Display simulation log highlights
    $SimOut | Where-Object { $_ -match "PASS|FAIL|Logit|Top|Cycle|Latency|MATCH|Result" } | ForEach-Object {
        if ($_ -match "PASS|MATCH") {
            Write-Host "  $_" -ForegroundColor Green
        } elseif ($_ -match "FAIL|MISMATCH") {
            Write-Host "  $_" -ForegroundColor Red
        } else {
            Write-Host "  $_" -ForegroundColor White
        }
    }

    Write-Host "[+] Simulation finished for $m!" -ForegroundColor Green
    }
} finally {
    Pop-Location
}

Write-Host "`n================================================================================" -ForegroundColor Cyan
Write-Host "  ALL RTL SIMULATIONS COMPLETED SUCCESSFULLY!" -ForegroundColor Green
Write-Host "================================================================================" -ForegroundColor Cyan

