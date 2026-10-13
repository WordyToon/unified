<#
================================================================================
 Script: baremetal/run_all.ps1
 Description: Automated One-Click Runner for All 4 Models on AMD ZCU106
 Target: AMD Zynq UltraScale+ ZCU106 (xczu7ev-ffvc1156-2-e)
 Models Evaluated:
   1. MNIST LeNet-5 (10 digits)
   2. ResNet-50 Full Backbone (1,000 ImageNet categories)
   3. MobileNetV2 Inverted Residual (1,000 ImageNet categories)
   4. EfficientNet-B0 MBConv (1,000 ImageNet categories)
================================================================================
#>
param (
    [ValidateSet("8bit", "16bit")]
    [string]$Precision = "8bit",

    [ValidateSet("dog", "car", "cat", "aeroplane")]
    [string]$Image = "dog"
)

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$BaseDir = Split-Path -Parent $ScriptDir
Set-Location $BaseDir

Write-Host "================================================================================" -ForegroundColor Cyan
Write-Host "  AMD ZCU106 UNIFIED BAREMETAL RUNNER - ALL 4 VISION MODELS" -ForegroundColor Cyan
Write-Host "  Board:            AMD Zynq UltraScale+ ZCU106 (xczu7ev-ffvc1156-2-e)" -ForegroundColor White
Write-Host "  Target Precision: $Precision" -ForegroundColor White
Write-Host "  Test Image:       $Image" -ForegroundColor White
Write-Host "  Memory Layout:    Concurrent Non-Overlapping DDR Preload (Option 1B)" -ForegroundColor White
Write-Host "================================================================================" -ForegroundColor Cyan

# Locate xsdb
$XsdbCmd = Get-Command "xsdb" -ErrorAction SilentlyContinue | Select-Object -ExpandProperty Source -ErrorAction SilentlyContinue
if (-not $XsdbCmd) {
    $Candidates = @(
        "C:\Xilinx\Vitis\2023.2\bin\xsdb.bat",
        "C:\Xilinx\SDK\2019.1\bin\xsdb.bat",
        "C:\Xilinx\Vitis_HLS\2023.2\bin\xsdb.bat"
    )
    foreach ($c in $Candidates) {
        if (Test-Path $c) {
            $XsdbCmd = $c
            break
        }
    }
}

if (-not $XsdbCmd) {
    Write-Error "[-] xsdb not found in PATH or standard Xilinx directories! Please ensure Vitis 2023.2 environment is initialized."
}

Write-Host "[*] Using XSDB: $XsdbCmd" -ForegroundColor Yellow
Write-Host "[*] Launching Automated JTAG Deployment (Option 1B)..." -ForegroundColor Yellow
Write-Host "    1. Programming Bitstream to ZCU106 PL" -ForegroundColor Gray
Write-Host "    2. Running PSU Initialization (psu_init)" -ForegroundColor Gray
Write-Host "    3. JTAG Preloading all 4 model weights + test images to DDR" -ForegroundColor Gray
Write-Host "    4. Downloading and executing Baremetal ELF on Cortex-A53 #0" -ForegroundColor Gray
# Detect Silicon Labs Quad CP2108 Interface 0 (PS UART0)
$DetectedPort = Get-CimInstance Win32_SerialPort -ErrorAction SilentlyContinue | Where-Object { $_.Description -match "Interface 0" } | Select-Object -ExpandProperty DeviceID -ErrorAction SilentlyContinue
if (-not $DetectedPort) { $DetectedPort = "COM7" }

Write-Host "================================================================================" -ForegroundColor DarkGray
Write-Host "  AMD ZCU106 SERIAL CONSOLE DETAILS:" -ForegroundColor Green
Write-Host "  Active COM Port:  $DetectedPort (Silicon Labs CP2108 Interface 0 -> PS UART0)" -ForegroundColor Yellow
Write-Host "  Baud Rate:        115200 baud, 8-N-1 (No Parity, 1 Stop Bit)" -ForegroundColor Yellow
Write-Host "  Note:             Connect PuTTY / TeraTerm to $DetectedPort before running." -ForegroundColor Cyan
Write-Host "================================================================================" -ForegroundColor DarkGray
Write-Host ""

& $XsdbCmd .\baremetal\scripts\run_xsdb.tcl all $Precision $Image

if ($LASTEXITCODE -eq 0) {
    Write-Host "`n(+) Execution launched successfully on AMD ZCU106!" -ForegroundColor Green
} else {
    Write-Error "`n[-] XSDB execution failed with exit code $LASTEXITCODE."
}

