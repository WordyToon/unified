<#
================================================================================
 Script: baremetal/build_elf.ps1
 Description: Compiles 8-bit and 16-bit baremetal ELFs for AMD ZCU106 Cortex-A53
 Toolchain: aarch64-none-elf-gcc (Vitis 2023.2)
================================================================================
#>
param(
    [ValidateSet("all", "8bit", "16bit")]
    [string]$Target = "all"
)

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$BaseDir = Split-Path -Parent $ScriptDir
$SrcDir = Join-Path $ScriptDir "src"
$BuildDir = Join-Path $ScriptDir "build"
$Lscript = Join-Path $ScriptDir "vitis_ws\baremetal_c_engine\src\lscript.ld"

if (-not (Test-Path $BuildDir)) {
    New-Item -ItemType Directory -Path $BuildDir -Force | Out-Null
}

$Gcc = "C:\Xilinx\Vitis\2023.2\gnu\aarch64\nt\aarch64-none\bin\aarch64-none-elf-gcc.exe"
if (-not (Test-Path $Gcc)) {
    $GccCmd = Get-Command "aarch64-none-elf-gcc" -ErrorAction SilentlyContinue | Select-Object -ExpandProperty Source
    if ($GccCmd) { $Gcc = $GccCmd }
    else { throw "aarch64-none-elf-gcc compiler not found! Please check Vitis 2023.2 installation." }
}

Write-Host "================================================================================" -ForegroundColor Cyan
Write-Host "  AMD ZCU106 BAREMETAL ELF BUILDER" -ForegroundColor Cyan
Write-Host "  Compiler: $Gcc" -ForegroundColor White
Write-Host "  Target:   $Target" -ForegroundColor White
Write-Host "================================================================================" -ForegroundColor Cyan

if ($Target -eq "all" -or $Target -eq "8bit") {
    $BspInc8 = Join-Path $ScriptDir "vitis_ws\zcu106_c_engine_8bit\export\zcu106_c_engine_8bit\sw\zcu106_c_engine_8bit\standalone_domain\bspinclude\include"
    $BspLib8 = Join-Path $ScriptDir "vitis_ws\zcu106_c_engine_8bit\export\zcu106_c_engine_8bit\sw\zcu106_c_engine_8bit\standalone_domain\bsplib\lib"
    
    Write-Host "[*] Building 8-bit baremetal ELF..." -ForegroundColor Yellow
    & $Gcc -DTARGET_PRECISION=8 -Wall -O2 "-I$SrcDir" "-I$BspInc8" -c "$SrcDir\main.c" -o "$BuildDir\main_8bit.o"
    & $Gcc -DTARGET_PRECISION=8 -Wall -O2 "-I$SrcDir" "-I$BspInc8" -c "$SrcDir\c_engine_driver.c" -o "$BuildDir\driver_8bit.o"
    & $Gcc "-Wl,-T" "-Wl,$Lscript" "-L$BspLib8" -o "$BuildDir\baremetal_c_engine_8bit.elf" "$BuildDir\main_8bit.o" "$BuildDir\driver_8bit.o" -lm "-Wl,--start-group,-lxil,-lgcc,-lc,--end-group"
    Copy-Item "$BuildDir\baremetal_c_engine_8bit.elf" "$BuildDir\baremetal_c_engine.elf" -Force
    Write-Host "[+] Successfully created baremetal_c_engine_8bit.elf" -ForegroundColor Green
}

if ($Target -eq "all" -or $Target -eq "16bit") {
    $BspInc16 = Join-Path $ScriptDir "vitis_ws\zcu106_c_engine_16bit\export\zcu106_c_engine_16bit\sw\zcu106_c_engine_16bit\standalone_domain\bspinclude\include"
    $BspLib16 = Join-Path $ScriptDir "vitis_ws\zcu106_c_engine_16bit\export\zcu106_c_engine_16bit\sw\zcu106_c_engine_16bit\standalone_domain\bsplib\lib"
    
    Write-Host "[*] Building 16-bit baremetal ELF..." -ForegroundColor Yellow
    & $Gcc -DTARGET_PRECISION=16 -Wall -O2 "-I$SrcDir" "-I$BspInc16" -c "$SrcDir\main.c" -o "$BuildDir\main_16bit.o"
    & $Gcc -DTARGET_PRECISION=16 -Wall -O2 "-I$SrcDir" "-I$BspInc16" -c "$SrcDir\c_engine_driver.c" -o "$BuildDir\driver_16bit.o"
    & $Gcc "-Wl,-T" "-Wl,$Lscript" "-L$BspLib16" -o "$BuildDir\baremetal_c_engine_16bit.elf" "$BuildDir\main_16bit.o" "$BuildDir\driver_16bit.o" -lm "-Wl,--start-group,-lxil,-lgcc,-lc,--end-group"
    Write-Host "[+] Successfully created baremetal_c_engine_16bit.elf" -ForegroundColor Green
}

Get-Item "$BuildDir\*.elf" | Select-Object Name, Length, LastWriteTime | Format-Table -AutoSize

