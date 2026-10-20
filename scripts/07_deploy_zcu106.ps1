<#
================================================================================
 Script: scripts/07_deploy_zcu106.ps1
 Purpose: Turnkey JTAG Baremetal Deployment on AMD Zynq UltraScale+ ZCU106
 Platform: AMD ZCU106 (xczu7ev-ffvc1156-2-e)
 Models Supported:
   1. mnist          - LeNet-5 (10-Digit Handwritten Classification)
   2. resnet50       - Full 50-Layer Backbone (1,000 ImageNet Classes)
   3. mobilenetv2    - Inverted Residual Network (1,000 ImageNet Classes)
   4. efficientnetb0 - MBConv Architecture (1,000 ImageNet Classes)
   5. all            - All 4 models preloaded and executed sequentially
================================================================================
#>
param(
    [ValidateSet("all", "mnist", "resnet50", "resnet50_head", "mobilenetv2", "efficientnetb0")]
    [string]$Model = "all",

    [ValidateSet("8bit", "16bit")]
    [string]$Precision = "8bit",

    [ValidateSet("auto", "dog", "car", "cat", "aeroplane",
                 "digit0", "digit1", "digit2", "digit3", "digit4",
                 "digit5", "digit6", "digit7", "digit8", "digit9",
                 "0", "1", "2", "3", "4", "5", "6", "7", "8", "9")]
    [string]$Image = "auto",

    [string]$Port = ""
)

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$BaseDir   = Split-Path -Parent $ScriptDir
Set-Location $BaseDir

# Resolve test image
$TestImage = $Image
if ($TestImage -eq "auto") {
    if ($Model -eq "mnist") {
        $TestImage = "digit7"
    } else {
        $TestImage = "dog"
    }
} elseif ($TestImage -match '^[0-9]$') {
    $TestImage = "digit$TestImage"
}

Write-Host "================================================================================" -ForegroundColor Cyan
Write-Host "  STEP 07: DEPLOY BAREMETAL INFERENCE TO AMD ZCU106 (VIA JTAG / XSDB)" -ForegroundColor Cyan
Write-Host "  Board:            AMD Zynq UltraScale+ ZCU106 (xczu7ev-ffvc1156-2-e)" -ForegroundColor White
Write-Host "  Model Selection:  $Model" -ForegroundColor White
Write-Host "  Target Precision: $Precision" -ForegroundColor White
Write-Host "  Test Image:       $TestImage" -ForegroundColor White
Write-Host "================================================================================" -ForegroundColor Cyan

# 1. Verify Bitstream
$BitFile = Join-Path $BaseDir "fpga\export\$Precision\c_engine.bit"
if (-not (Test-Path $BitFile)) {
    Write-Warning "[-] Bitstream not found at $BitFile!"
    Write-Warning "    Please run .\scripts\04_build_bitstream.ps1 -Precision $Precision first."
} else {
    Write-Host "[+] Found Bitstream: $BitFile ($("{0:N2} MB" -f ((Get-Item $BitFile).Length / 1MB)))" -ForegroundColor Green
}

# 2. Verify ELF Application
$ElfFile = Join-Path $BaseDir "baremetal\build\baremetal_c_engine_${Precision}.elf"
if (-not (Test-Path $ElfFile)) {
    Write-Warning "[-] Baremetal ELF not found at $ElfFile!"
    Write-Warning "    Please run .\scripts\05_build_baremetal.ps1 -Target $Precision first."
} else {
    Write-Host "[+] Found Baremetal ELF: $ElfFile" -ForegroundColor Green
}

# 3. Locate XSDB
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
    throw "xsdb utility not found in PATH or standard Xilinx Vitis 2023.2 directories!"
}
Write-Host "[+] Using XSDB: $XsdbCmd" -ForegroundColor Green

# 4. Auto-detect COM Port for Silicon Labs CP2108 Quad UART (Interface 0 -> PS UART0)
$DetectedPort = $Port
if (-not $DetectedPort) {
    $DetectedPort = Get-CimInstance Win32_SerialPort -ErrorAction SilentlyContinue |
        Where-Object { $_.Description -match "Interface 0" } |
        Select-Object -ExpandProperty DeviceID -ErrorAction SilentlyContinue
    if (-not $DetectedPort) { $DetectedPort = "COM7" }
}

Write-Host "--------------------------------------------------------------------------------" -ForegroundColor DarkGray
Write-Host "  AMD ZCU106 UART CONSOLE:" -ForegroundColor Yellow
Write-Host "  Port:      $DetectedPort" -ForegroundColor White
Write-Host "  Settings:  115200 baud, 8 data bits, no parity, 1 stop bit (8-N-1)" -ForegroundColor White
Write-Host "  Tip:       Open PuTTY / TeraTerm / Serial Monitor on $DetectedPort to view live logs." -ForegroundColor White
Write-Host "--------------------------------------------------------------------------------" -ForegroundColor DarkGray

# 5. Launch XSDB Deployment
$XsdbScript = Join-Path $BaseDir "baremetal\scripts\run_xsdb.tcl"
Write-Host "[*] Executing XSDB deployment script: $XsdbScript..." -ForegroundColor Yellow

& $XsdbCmd "$XsdbScript" $Model $Precision $TestImage

if ($LASTEXITCODE -ne 0) {
    throw "XSDB deployment failed with exit code $LASTEXITCODE!"
}

Write-Host "`n[+] Hardware Deployment Succeeded on AMD ZCU106!" -ForegroundColor Green
Write-Host "================================================================================" -ForegroundColor Cyan

