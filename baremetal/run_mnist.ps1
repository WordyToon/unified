<#
================================================================================
 Script: baremetal/run_mnist.ps1
 Description: Automated Runner for MNIST LeNet-5 Digit Classification on AMD ZCU106
 Target: AMD Zynq UltraScale+ ZCU106 (xczu7ev-ffvc1156-2-e)
 Architecture: 5-Stage LeNet-5 (Conv1 -> Conv2 -> FC1 -> FC2 -> FC3)
================================================================================
#>
param (
    [ValidateSet("8bit", "16bit")]
    [string]$Precision = "8bit",

    [ValidateRange(0, 9)]
    [int]$Digit = 7
)

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$BaseDir = Split-Path -Parent $ScriptDir
Set-Location $BaseDir

Write-Host "================================================================================" -ForegroundColor Cyan
Write-Host "  AMD ZCU106 BAREMETAL RUNNER - MNIST LENET-5 DIGIT CLASSIFIER" -ForegroundColor Cyan
Write-Host "  Board:            AMD Zynq UltraScale+ ZCU106 (xczu7ev-ffvc1156-2-e)" -ForegroundColor White
Write-Host "  Target Precision: $Precision" -ForegroundColor White
Write-Host "  Test Digit:       Digit $Digit (digit${Digit}_${Precision}.bin)" -ForegroundColor White
Write-Host "  Model Mailbox:    Code 1 (MNIST Only)" -ForegroundColor White
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
Write-Host "    3. JTAG Preloading MNIST weights (0x30000000) and digit $Digit (0x20000000)" -ForegroundColor Gray
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

$ImageName = "digit$Digit"
& $XsdbCmd .\baremetal\scripts\run_xsdb.tcl mnist $Precision $ImageName

if ($LASTEXITCODE -eq 0) {
    Write-Host "`n(+) MNIST inference launched successfully on AMD ZCU106!" -ForegroundColor Green
} else {
    Write-Error "`n[-] XSDB execution failed with exit code $LASTEXITCODE."
}

