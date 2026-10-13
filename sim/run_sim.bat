@echo off
REM ============================================================================
REM Batch Script: unified/sim/run_sim.bat
REM Description: Compiles and simulates the Unified 256-PE c_engine AXI4 DMA
REM              Systolic Accelerator with Vivado Simulator.
REM Target: AMD Zynq UltraScale+ ZCU106
REM Usage:
REM   run_sim.bat mnist
REM   run_sim.bat bottleneck
REM   run_sim.bat resnet50 [8bit|16bit]
REM   run_sim.bat gemv     [8bit|16bit]
REM ============================================================================

set VIVADO_BIN=C:\Xilinx\Vivado\2023.2\bin
pushd "%~dp0"
if not exist "%VIVADO_BIN%\xvlog.bat" (
    where xvlog.bat >nul 2>nul
    if %ERRORLEVEL% NEQ 0 (
        echo [ERROR] Vivado 2023.2 tools not found on PATH or at %VIVADO_BIN%!
        exit /b 1
    )
    set VIVADO_BIN=
)

echo ================================================================================
echo   AMD ZCU106 - UNIFIED 256-PE RTL SIMULATION
echo ================================================================================

echo [0/3] Cleaning previous simulation artifacts...
if exist xsim.dir rmdir /s /q xsim.dir
del /f /q *.wdb *.jou *.backup.jou *.pb *.log 2>nul

set PRECISION=8bit
if /i "%2"=="16" set PRECISION=16bit
if /i "%2"=="16bit" set PRECISION=16bit
if /i "%2"=="8" set PRECISION=8bit
if /i "%2"=="8bit" set PRECISION=8bit

if /i "%PRECISION%"=="16bit" (
    echo [*] Configuring Hardware RTL for 16-BIT PRECISION...
    copy /y "..\rtl\hw_config_16bit.vh" "..\rtl\hw_config.vh" >nul
) else (
    echo [*] Configuring Hardware RTL for 8-BIT PRECISION...
    copy /y "..\rtl\hw_config_8bit.vh" "..\rtl\hw_config.vh" >nul
)

set SIM_DEF=-d SIM_RESNET50
set MODEL_NAME=Full ResNet-50 %PRECISION%
set TB_TOP=tb_unified_c_engine
set SNAPSHOT=sim_unified_256pe

if /i "%1"=="1" set SIM_DEF=-d SIM_MNIST& set MODEL_NAME=MNIST LeNet-5& set TB_TOP=tb_unified_c_engine& set SNAPSHOT=sim_unified_256pe
if /i "%1"=="mnist" set SIM_DEF=-d SIM_MNIST& set MODEL_NAME=MNIST LeNet-5& set TB_TOP=tb_unified_c_engine& set SNAPSHOT=sim_unified_256pe
if /i "%1"=="2" set SIM_DEF=-d SIM_MOBILENETV2& set MODEL_NAME=MobileNetV2 %PRECISION%& set TB_TOP=tb_unified_c_engine& set SNAPSHOT=sim_unified_256pe
if /i "%1"=="mobilenetv2" set SIM_DEF=-d SIM_MOBILENETV2& set MODEL_NAME=MobileNetV2 %PRECISION%& set TB_TOP=tb_unified_c_engine& set SNAPSHOT=sim_unified_256pe
if /i "%1"=="3" set SIM_DEF=-d SIM_EFFICIENTNETB0& set MODEL_NAME=EfficientNet-B0 %PRECISION%& set TB_TOP=tb_unified_c_engine& set SNAPSHOT=sim_unified_256pe
if /i "%1"=="efficientnetb0" set SIM_DEF=-d SIM_EFFICIENTNETB0& set MODEL_NAME=EfficientNet-B0 %PRECISION%& set TB_TOP=tb_unified_c_engine& set SNAPSHOT=sim_unified_256pe
if /i "%1"=="4" set SIM_DEF=-d SIM_RESNET50& set MODEL_NAME=Full ResNet-50 %PRECISION%& set TB_TOP=tb_unified_c_engine& set SNAPSHOT=sim_unified_256pe
if /i "%1"=="resnet50" set SIM_DEF=-d SIM_RESNET50& set MODEL_NAME=Full ResNet-50 %PRECISION%& set TB_TOP=tb_unified_c_engine& set SNAPSHOT=sim_unified_256pe
if /i "%1"=="resnet50_full" set SIM_DEF=-d SIM_RESNET50& set MODEL_NAME=Full ResNet-50 %PRECISION%& set TB_TOP=tb_unified_c_engine& set SNAPSHOT=sim_unified_256pe

if /i "%SIM_DEF%"=="-d SIM_MOBILENETV2" (
    if exist "mobilenetv2_weights_gemv_%PRECISION%.hex" copy /y "mobilenetv2_weights_gemv_%PRECISION%.hex" "weights_gemv.hex" >nul
    if exist "mobilenetv2_weights_%PRECISION%.hex" copy /y "mobilenetv2_weights_%PRECISION%.hex" "weights.hex" >nul
) else if /i "%SIM_DEF%"=="-d SIM_EFFICIENTNETB0" (
    if exist "efficientnetb0_weights_gemv_%PRECISION%.hex" copy /y "efficientnetb0_weights_gemv_%PRECISION%.hex" "weights_gemv.hex" >nul
    if exist "efficientnetb0_weights_%PRECISION%.hex" copy /y "efficientnetb0_weights_%PRECISION%.hex" "weights.hex" >nul
) else (
    if exist "weights_%PRECISION%.hex" copy /y "weights_%PRECISION%.hex" "weights.hex" >nul
    if exist "weights_gemv_%PRECISION%.hex" copy /y "weights_gemv_%PRECISION%.hex" "weights_gemv.hex" >nul
)

echo [1/3] Compiling RTL and Testbench with xvlog [Target: %MODEL_NAME%]...
if "%VIVADO_BIN%"=="" (
    call xvlog.bat -nolog %SIM_DEF% -i ..\rtl ..\rtl\act_lut_rom.v ..\rtl\config_mem.v ..\rtl\processing_element.v ..\rtl\c_engine.v ..\rtl\multibank_dualport_mem.v ..\rtl\requantization_block.v ..\rtl\c_engine_controller.v ..\rtl\c_engine_top.v ..\rtl\axi_lite_slave.v ..\rtl\axi4_master_rd.v ..\rtl\axi4_master_wr.v ..\rtl\dma_sram_bridge.v ..\rtl\c_engine_dma_top.v axi4_ddr_model.v %TB_TOP%.v
) else (
    call "%VIVADO_BIN%\xvlog.bat" -nolog %SIM_DEF% -i ..\rtl ..\rtl\act_lut_rom.v ..\rtl\config_mem.v ..\rtl\processing_element.v ..\rtl\c_engine.v ..\rtl\multibank_dualport_mem.v ..\rtl\requantization_block.v ..\rtl\c_engine_controller.v ..\rtl\c_engine_top.v ..\rtl\axi_lite_slave.v ..\rtl\axi4_master_rd.v ..\rtl\axi4_master_wr.v ..\rtl\dma_sram_bridge.v ..\rtl\c_engine_dma_top.v axi4_ddr_model.v %TB_TOP%.v
)
if %ERRORLEVEL% NEQ 0 (
    echo [-] RTL Compilation failed!
    popd
    exit /b %ERRORLEVEL%
)

echo [2/3] Elaborating design with xelab...
if "%VIVADO_BIN%"=="" (
    call xelab.bat -nolog -debug typical work.%TB_TOP% -s %SNAPSHOT%
) else (
    call "%VIVADO_BIN%\xelab.bat" -nolog -debug typical work.%TB_TOP% -s %SNAPSHOT%
)
if not exist xsim.dir\%SNAPSHOT%\xsimk.exe (
    echo [-] Design Elaboration failed!
    popd
    exit /b 1
)

echo [3/3] Running simulation with xsim...
if "%VIVADO_BIN%"=="" (
    call xsim.bat -nolog %SNAPSHOT% -runall
) else (
    call "%VIVADO_BIN%\xsim.bat" -nolog %SNAPSHOT% -runall
)
if %ERRORLEVEL% NEQ 0 (
    echo [-] Simulation execution failed!
    popd
    exit /b %ERRORLEVEL%
)

echo [OK] Unified 256-PE RTL simulation for %MODEL_NAME% completed successfully!
popd
exit /b 0
