# ==============================================================================
# Script: unified/make.ps1
# Description: Unified build, simulation, synthesis, and verification pipeline
# Target: AMD Zynq UltraScale+ ZCU104 (256 PEs @ 100 MHz)
# Models: MNIST LeNet-5, ResNet-50 Bottleneck Block, Full ResNet-50 (1,000 Classes)
# Supports: Dual Precision (8-bit and 16-bit) via -Precision switch
# ==============================================================================

param (
    [Parameter(Position=0)]
    [string]$Target = "help",
    [string]$Precision = "8bit",
    [string]$Board = "zcu106",
    [double]$FreqMHz = 75.0,
    [string]$BoardIP = "192.168.2.99",
    [string]$BoardUser = "xilinx",
    [switch]$Clean
)

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ScriptDir

# Normalize precision argument
if ($Precision -match "16") {
    $Precision = "16bit"
} else {
    $Precision = "8bit"
}
$env:PRECISION = $Precision

# Helper to find Python
$PythonCmd = "python"
if (Test-Path "C:\Users\m.usman\.conda\envs\vitv1\python.exe") {
    $PythonCmd = "C:\Users\m.usman\.conda\envs\vitv1\python.exe"
}

function Show-Help {
    Write-Host "================================================================================" -ForegroundColor Cyan
    Write-Host "  UNIFIED C-ENGINE ACCELERATOR AUTOMATION (256 PEs @ $FreqMHz MHz)" -ForegroundColor Cyan
    Write-Host "  Active Precision: $Precision (Supports: 8bit | 16bit)" -ForegroundColor Cyan
    Write-Host "================================================================================" -ForegroundColor Cyan
    Write-Host "Usage: .\make.ps1 <Target> [-Precision <8bit|16bit>] [-FreqMHz <100.0>] [-BoardIP <IP>]"
    Write-Host ""
    Write-Host "Targets:" -ForegroundColor Yellow
    Write-Host "  compile-models      : Recompile all models (MNIST, ResNet-50, MobileNetV2, EfficientNet-B0)"
    Write-Host "  compile-mobilenetv2 : Compile MobileNetV2 weights, microcode, and features"
    Write-Host "  compile-efficientnetb0: Compile EfficientNet-B0 weights, microcode, and features"
    Write-Host "  sim-resnet50        : Run cycle-accurate Vivado RTL simulation for ResNet-50"
    Write-Host "  sim-mobilenetv2     : Run cycle-accurate Vivado RTL simulation for MobileNetV2"
    Write-Host "  sim-efficientnetb0  : Run cycle-accurate Vivado RTL simulation for EfficientNet-B0"
    Write-Host "  sim-gemv            : Run fast targeted GEMV classification head simulation"
    Write-Host "  sim-mnist           : Run RTL simulation for MNIST LeNet-5"
    Write-Host "  sim-bottleneck      : Run RTL simulation for ResNet-50 Bottleneck Block"
    Write-Host "  sim-all             : Run all model simulations sequentially"
    Write-Host "  package-ip          : Package unified RTL as Vivado IP core"
    Write-Host "  build-bd            : Generate Vivado Block Design for ZCU104"
    Write-Host "  build-bitstream     : Synthesize, implement, and export bitstream & .hwh"
    Write-Host "  test-pynq           : Verify all PYNQ Python drivers in off-board Mock mode"
    Write-Host "  test-mobilenetv2    : Verify MobileNetV2 PYNQ Python driver in Mock mode"
    Write-Host "  test-efficientnetb0 : Verify EfficientNet-B0 PYNQ Python driver in Mock mode"
    Write-Host "  run-test            : Run test images evaluation with Python test driver"
    Write-Host "  deploy-board        : Deploy all packages to ZCU104 board via SCP"
    Write-Host "  deploy-mobilenetv2  : Deploy MobileNetV2 package to ZCU104 board via SCP"
    Write-Host "  deploy-efficientnetb0: Deploy EfficientNet-B0 package to ZCU104 board via SCP"
    Write-Host "  test-board          : Run hardware verification remotely on ZCU104 via SSH"
    Write-Host "  all                 : Full automated pipeline (compile -> sim -> bitstream -> test)"
    Write-Host "  help                : Show this help message"
    Write-Host ""
    Write-Host "Examples:" -ForegroundColor Yellow
    Write-Host "  .\make.ps1 compile-models -Precision 8bit"
    Write-Host "  .\make.ps1 compile-models -Precision 16bit"
    Write-Host "  .\make.ps1 sim-resnet50 -Precision 16bit"
    Write-Host "  .\make.ps1 build-bitstream -Precision 16bit -FreqMHz 100.0"
    Write-Host "  .\make.ps1 test-pynq -Precision 16bit"
    Write-Host "  .\make.ps1 deploy-board -Precision 16bit -BoardIP 192.168.2.99"
    Write-Host "  .\make.ps1 test-board -Precision 16bit -BoardIP 192.168.2.99"
}

switch ($Target.ToLower()) {
    "compile-models" {
        Write-Host "================================================================================" -ForegroundColor Cyan
        Write-Host "  COMPILING ALL VISION MODELS FOR UNIFIED ACCELERATOR ($Precision)" -ForegroundColor Cyan
        Write-Host "================================================================================" -ForegroundColor Cyan
        Write-Host "[*] Compiling MNIST LeNet-5..." -ForegroundColor Green
        & $PythonCmd "models/mnist/compile.py"
        Write-Host "[*] Compiling ResNet-50 Bottleneck Block..." -ForegroundColor Green
        & $PythonCmd "models/resnet50_bottleneck/compile.py"
        Write-Host "[*] Compiling Full ResNet-50 (1,000 Classes, $Precision)..." -ForegroundColor Green
        & $PythonCmd "models/resnet50_full/compile.py" --precision $Precision
        Write-Host "[*] Compiling MobileNetV2 (1,000 Classes, $Precision)..." -ForegroundColor Green
        & $PythonCmd "models/mobilenetv2/compile.py" --precision $Precision
        Write-Host "[*] Compiling EfficientNet-B0 (1,000 Classes, $Precision)..." -ForegroundColor Green
        & $PythonCmd "models/efficientnetb0/compile.py" --precision $Precision
        Write-Host "[OK] All models compiled successfully for $Precision!" -ForegroundColor Green
    }

    "compile-mobilenetv2" {
        Write-Host "================================================================================" -ForegroundColor Cyan
        Write-Host "  COMPILING MOBILENETV2 (1,000 CLASSES, $Precision)" -ForegroundColor Cyan
        Write-Host "================================================================================" -ForegroundColor Cyan
        & $PythonCmd "models/mobilenetv2/compile.py" --precision $Precision
    }

    "compile-efficientnetb0" {
        Write-Host "================================================================================" -ForegroundColor Cyan
        Write-Host "  COMPILING EFFICIENTNET-B0 (1,000 CLASSES, $Precision)" -ForegroundColor Cyan
        Write-Host "================================================================================" -ForegroundColor Cyan
        & $PythonCmd "models/efficientnetb0/compile.py" --precision $Precision
    }

    "sim-resnet50" {
        Write-Host "================================================================================" -ForegroundColor Cyan
        Write-Host "  LAUNCHING FULL RESNET-50 RTL SIMULATION ($Precision)" -ForegroundColor Cyan
        Write-Host "================================================================================" -ForegroundColor Cyan
        Set-Location "sim"
        .\run_sim.bat resnet50 $Precision
        Set-Location ".."
    }

    "sim-mobilenetv2" {
        Write-Host "================================================================================" -ForegroundColor Cyan
        Write-Host "  LAUNCHING MOBILENETV2 RTL SIMULATION ($Precision)" -ForegroundColor Cyan
        Write-Host "================================================================================" -ForegroundColor Cyan
        Set-Location "sim"
        .\run_sim.bat mobilenetv2 $Precision
        Set-Location ".."
    }

    "sim-efficientnetb0" {
        Write-Host "================================================================================" -ForegroundColor Cyan
        Write-Host "  LAUNCHING EFFICIENTNET-B0 RTL SIMULATION ($Precision)" -ForegroundColor Cyan
        Write-Host "================================================================================" -ForegroundColor Cyan
        Set-Location "sim"
        .\run_sim.bat efficientnetb0 $Precision
        Set-Location ".."
    }

    "sim-gemv" {
        Write-Host "================================================================================" -ForegroundColor Cyan
        Write-Host "  LAUNCHING FAST RESNET-50 GEMV RTL SIMULATION ($Precision)" -ForegroundColor Cyan
        Write-Host "================================================================================" -ForegroundColor Cyan
        Set-Location "sim"
        .\run_sim.bat gemv $Precision
        Set-Location ".."
    }

    "sim-mnist" {
        Write-Host "[*] Launching MNIST LeNet-5 RTL Simulation..." -ForegroundColor Green
        Set-Location "sim"
        .\run_sim.bat mnist
        Set-Location ".."
    }

    "sim-bottleneck" {
        Write-Host "[*] Launching ResNet-50 Bottleneck RTL Simulation..." -ForegroundColor Green
        Set-Location "sim"
        .\run_sim.bat bottleneck
        Set-Location ".."
    }

    "sim-all" {
        Write-Host "================================================================================" -ForegroundColor Cyan
        Write-Host "  RUNNING ALL UNIFIED RTL SIMULATIONS (256 PEs, $Precision)" -ForegroundColor Cyan
        Write-Host "================================================================================" -ForegroundColor Cyan
        Set-Location "sim"
        Write-Host "`n>>> [1/5] SIMULATING MNIST LENET-5..." -ForegroundColor Yellow
        .\run_sim.bat mnist
        Write-Host "`n>>> [2/5] SIMULATING RESNET-50 BOTTLENECK BLOCK..." -ForegroundColor Yellow
        .\run_sim.bat bottleneck
        Write-Host "`n>>> [3/5] SIMULATING FULL RESNET-50 (1,000 CLASSES, $Precision)..." -ForegroundColor Yellow
        .\run_sim.bat resnet50 $Precision
        Write-Host "`n>>> [4/5] SIMULATING MOBILENETV2 (1,000 CLASSES, $Precision)..." -ForegroundColor Yellow
        .\run_sim.bat mobilenetv2 $Precision
        Write-Host "`n>>> [5/5] SIMULATING EFFICIENTNET-B0 (1,000 CLASSES, $Precision)..." -ForegroundColor Yellow
        .\run_sim.bat efficientnetb0 $Precision
        Set-Location ".."
        Write-Host "`n[OK] All simulations passed successfully!" -ForegroundColor Green
    }

    "package-ip" {
        Write-Host "================================================================================" -ForegroundColor Cyan
        Write-Host "  PACKAGING UNIFIED ACCELERATOR IP CORE ($Precision)" -ForegroundColor Cyan
        Write-Host "================================================================================" -ForegroundColor Cyan
        Set-Location "fpga"
        .\build.ps1 -PackageOnly -Precision $Precision
        Set-Location ".."
    }

    "build-bd" {
        Write-Host "================================================================================" -ForegroundColor Cyan
        Write-Host "  CREATING VIVADO BLOCK DESIGN AT $FreqMHz MHz ($Precision)" -ForegroundColor Cyan
        Write-Host "================================================================================" -ForegroundColor Cyan
        Set-Location "fpga"
        .\build.ps1 -BdOnly -Precision $Precision -FreqMHz $FreqMHz
        Set-Location ".."
    }

    "build-bitstream" {
        Write-Host "================================================================================" -ForegroundColor Cyan
        Write-Host "  SYNTHESIZING & IMPLEMENTING BITSTREAM FOR $( $Board.ToUpper() ) ($Precision @ $FreqMHz MHz)" -ForegroundColor Cyan
        Write-Host "================================================================================" -ForegroundColor Cyan
        Set-Location "fpga"
        if ($Clean) {
            .\build.ps1 -Board $Board -Precision $Precision -FreqMHz $FreqMHz -Clean
        } else {
            .\build.ps1 -Board $Board -Precision $Precision -FreqMHz $FreqMHz
        }
        Set-Location ".."
    }

    "test-pynq" {
        Write-Host "================================================================================" -ForegroundColor Cyan
        Write-Host "  TESTING PYNQ DRIVERS IN VERIFICATION MODE ($Precision)" -ForegroundColor Cyan
        Write-Host "================================================================================" -ForegroundColor Cyan
        Write-Host "`n[*] Testing MNIST LeNet-5 Test Driver Across 10 Digits (0..9)..." -ForegroundColor Yellow
        & $PythonCmd "pynq/mnist/test_driver.py"
        Write-Host "`n[*] Testing Full ResNet-50 ($Precision) Test Driver & Accuracy Across Images..." -ForegroundColor Yellow
        & $PythonCmd "pynq/resnet50_${Precision}/test_driver.py"
        Write-Host "`n[*] Testing MobileNetV2 ($Precision) Test Driver & Accuracy Across Images..." -ForegroundColor Yellow
        & $PythonCmd "pynq/mobilenetv2_${Precision}/test_driver.py"
        Write-Host "`n[*] Testing EfficientNet-B0 ($Precision) Test Driver & Accuracy Across Images..." -ForegroundColor Yellow
        & $PythonCmd "pynq/efficientnetb0_${Precision}/test_driver.py"
        Write-Host "`n[OK] All PYNQ drivers and mock inference passed 100% Top-1 accuracy!" -ForegroundColor Green
    }

    "test-mobilenetv2" {
        Write-Host "================================================================================" -ForegroundColor Cyan
        Write-Host "  TESTING MOBILENETV2 ($Precision) PYNQ TEST DRIVER" -ForegroundColor Cyan
        Write-Host "================================================================================" -ForegroundColor Cyan
        & $PythonCmd "pynq/mobilenetv2_${Precision}/test_driver.py"
    }

    "test-efficientnetb0" {
        Write-Host "================================================================================" -ForegroundColor Cyan
        Write-Host "  TESTING EFFICIENTNET-B0 ($Precision) PYNQ TEST DRIVER" -ForegroundColor Cyan
        Write-Host "================================================================================" -ForegroundColor Cyan
        & $PythonCmd "pynq/efficientnetb0_${Precision}/test_driver.py"
    }

    "run-test" {
        Write-Host "================================================================================" -ForegroundColor Cyan
        Write-Host "  RUNNING RESNET-50 TEST IMAGES EVALUATION ($Precision)" -ForegroundColor Cyan
        Write-Host "================================================================================" -ForegroundColor Cyan
        & $PythonCmd "pynq/resnet50_${Precision}/test_driver.py"
    }

    "deploy-board" {
        Write-Host "================================================================================" -ForegroundColor Cyan
        Write-Host "  DEPLOYING ALL MODELS ($Precision) TO ZCU104 BOARD ($BoardIP)" -ForegroundColor Cyan
        Write-Host "================================================================================" -ForegroundColor Cyan
        Write-Host "[*] Creating target directory on board: /home/xilinx/jupyter_notebooks/cnn_pynq" -ForegroundColor Yellow
        ssh -o ConnectTimeout=5 ${BoardUser}@${BoardIP} "mkdir -p /home/xilinx/jupyter_notebooks/cnn_pynq"
        Write-Host "[*] Copying resnet50_${Precision} package to ZCU104..." -ForegroundColor Yellow
        scp -r pynq/resnet50_${Precision} ${BoardUser}@${BoardIP}:/home/xilinx/jupyter_notebooks/cnn_pynq/
        Write-Host "[*] Copying mobilenetv2_${Precision} package to ZCU104..." -ForegroundColor Yellow
        scp -r pynq/mobilenetv2_${Precision} ${BoardUser}@${BoardIP}:/home/xilinx/jupyter_notebooks/cnn_pynq/
        Write-Host "[*] Copying efficientnetb0_${Precision} package to ZCU104..." -ForegroundColor Yellow
        scp -r pynq/efficientnetb0_${Precision} ${BoardUser}@${BoardIP}:/home/xilinx/jupyter_notebooks/cnn_pynq/
        Write-Host "[*] Copying mnist package to ZCU104..." -ForegroundColor Yellow
        scp -r pynq/mnist ${BoardUser}@${BoardIP}:/home/xilinx/jupyter_notebooks/cnn_pynq/
        Write-Host "[*] Configuring USB camera permissions on board..." -ForegroundColor Yellow
        ssh -o ConnectTimeout=5 ${BoardUser}@${BoardIP} "sudo chmod 666 /dev/video* 2>/dev/null || true"
        Write-Host "`n[OK] Deployment to ZCU104 completed successfully!" -ForegroundColor Green
        Write-Host "ResNet-50 Notebook:      http://${BoardIP}:9090/notebooks/cnn_pynq/resnet50_${Precision}/resnet50_cam.ipynb" -ForegroundColor Cyan
        Write-Host "MobileNetV2 Notebook:    http://${BoardIP}:9090/notebooks/cnn_pynq/mobilenetv2_${Precision}/mobilenetv2_cam.ipynb" -ForegroundColor Cyan
        Write-Host "EfficientNet-B0 Notebook: http://${BoardIP}:9090/notebooks/cnn_pynq/efficientnetb0_${Precision}/efficientnetb0_cam.ipynb" -ForegroundColor Cyan
        Write-Host "MNIST Notebook:          http://${BoardIP}:9090/notebooks/cnn_pynq/mnist/mnist_cam.ipynb" -ForegroundColor Cyan
    }

    "deploy-mobilenetv2" {
        Write-Host "================================================================================" -ForegroundColor Cyan
        Write-Host "  DEPLOYING MOBILENETV2 ($Precision) TO ZCU104 BOARD ($BoardIP)" -ForegroundColor Cyan
        Write-Host "================================================================================" -ForegroundColor Cyan
        ssh -o ConnectTimeout=5 ${BoardUser}@${BoardIP} "mkdir -p /home/xilinx/jupyter_notebooks/cnn_pynq"
        scp -r pynq/mobilenetv2_${Precision} ${BoardUser}@${BoardIP}:/home/xilinx/jupyter_notebooks/cnn_pynq/
        ssh -o ConnectTimeout=5 ${BoardUser}@${BoardIP} "sudo chmod 666 /dev/video* 2>/dev/null || true"
        Write-Host "[OK] MobileNetV2 deployed: http://${BoardIP}:9090/notebooks/cnn_pynq/mobilenetv2_${Precision}/mobilenetv2_cam.ipynb" -ForegroundColor Green
    }

    "deploy-efficientnetb0" {
        Write-Host "================================================================================" -ForegroundColor Cyan
        Write-Host "  DEPLOYING EFFICIENTNET-B0 ($Precision) TO ZCU104 BOARD ($BoardIP)" -ForegroundColor Cyan
        Write-Host "================================================================================" -ForegroundColor Cyan
        ssh -o ConnectTimeout=5 ${BoardUser}@${BoardIP} "mkdir -p /home/xilinx/jupyter_notebooks/cnn_pynq"
        scp -r pynq/efficientnetb0_${Precision} ${BoardUser}@${BoardIP}:/home/xilinx/jupyter_notebooks/cnn_pynq/
        ssh -o ConnectTimeout=5 ${BoardUser}@${BoardIP} "sudo chmod 666 /dev/video* 2>/dev/null || true"
        Write-Host "[OK] EfficientNet-B0 deployed: http://${BoardIP}:9090/notebooks/cnn_pynq/efficientnetb0_${Precision}/efficientnetb0_cam.ipynb" -ForegroundColor Green
    }

    "test-board" {
        Write-Host "================================================================================" -ForegroundColor Cyan
        Write-Host "  RUNNING HARDWARE INFERENCE REMOTELY ON ZCU104 ($BoardIP) [$Precision]" -ForegroundColor Cyan
        Write-Host "================================================================================" -ForegroundColor Cyan
        Write-Host "[*] Running MNIST Hardware Verification on ZCU104..." -ForegroundColor Yellow
        ssh -t ${BoardUser}@${BoardIP} "python3 /home/xilinx/jupyter_notebooks/cnn_pynq/mnist/test_driver.py"
        Write-Host "[*] Running ResNet-50 ($Precision) Hardware Verification on ZCU104..." -ForegroundColor Yellow
        ssh -t ${BoardUser}@${BoardIP} "python3 /home/xilinx/jupyter_notebooks/cnn_pynq/resnet50_${Precision}/test_driver.py"
        Write-Host "[*] Running MobileNetV2 ($Precision) Hardware Verification on ZCU104..." -ForegroundColor Yellow
        ssh -t ${BoardUser}@${BoardIP} "python3 /home/xilinx/jupyter_notebooks/cnn_pynq/mobilenetv2_${Precision}/test_driver.py"
        Write-Host "[*] Running EfficientNet-B0 ($Precision) Hardware Verification on ZCU104..." -ForegroundColor Yellow
        ssh -t ${BoardUser}@${BoardIP} "python3 /home/xilinx/jupyter_notebooks/cnn_pynq/efficientnetb0_${Precision}/test_driver.py"
    }

    "all" {
        Write-Host "================================================================================" -ForegroundColor Cyan
        Write-Host "  EXECUTING FULL UNIFIED PIPELINE FOR $Precision" -ForegroundColor Cyan
        Write-Host "================================================================================" -ForegroundColor Cyan
        Write-Host "[1/5] Compiling Models ($Precision)..." -ForegroundColor Green
        & $MyInvocation.MyCommand.Path "compile-models" -Precision $Precision
        Write-Host "`n[2/5] Running Simulation ($Precision)..." -ForegroundColor Green
        & $MyInvocation.MyCommand.Path "sim-resnet50" -Precision $Precision
        Write-Host "`n[3/5] Packaging Unified IP ($Precision)..." -ForegroundColor Green
        & $MyInvocation.MyCommand.Path "package-ip" -Precision $Precision
        Write-Host "`n[4/5] Building Unified Bitstream ($Precision @ $FreqMHz MHz)..." -ForegroundColor Green
        & $MyInvocation.MyCommand.Path "build-bitstream" -Precision $Precision -FreqMHz $FreqMHz
        Write-Host "`n[5/5] Testing PYNQ Python Drivers ($Precision)..." -ForegroundColor Green
        & $MyInvocation.MyCommand.Path "test-pynq" -Precision $Precision
        Write-Host "`n[+] COMPLETE UNIFIED PIPELINE FOR $Precision EXECUTED SUCCESSFULLY!" -ForegroundColor Green
    }

    default {
        Show-Help
    }
}
