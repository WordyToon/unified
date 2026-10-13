# Unified 256-PE Systolic Neural Network Accelerator (AMD ZCU106)

An ultra-efficient, parameterized systolic array neural network accelerator executing deep convolutional architectures **100% inside FPGA Programmable Logic (PL)** with **zero CPU convolution/GEMM fallback**.

Targeting the **AMD Zynq UltraScale+ ZCU106 Evaluation Kit** (`xczu7ev-ffvc1156-2-e`), a single unified hardware bitstream ($R=8, C=32 = 256\text{ PEs}$) with **guaranteed physical timing closure** executes diverse vision models ranging from lightweight digit recognition to deep 50-layer residual networks using **official pre-trained weights**:

1. **MNIST LeNet-5**: 5-stage convolutional digit classifier (10 classes: `0`–`9`).
2. **ResNet-50 Full Backbone**: Full 50-layer architecture (53 Conv stages + 4 GEMV sub-stages) generating all **1,000 ImageNet classification logits** directly in hardware.
3. **MobileNetV2**: 18 inverted residual blocks with hardware depthwise separable convolutions (1,000 ImageNet classes).
4. **EfficientNet-B0**: 16 MBConv blocks with hardware depthwise convolutions and fused **Swish (SiLU) LUT** non-linearities (1,000 ImageNet classes).

---

## Key Hardware Highlights

- **100% Hardware Math Execution**: Every layer computation (Standard 2D Conv, Depthwise Separable Conv, Matrix-Vector GEMV Projections, ReLU, Swish LUT activations, spatial Max/Average Pooling, and fused residual addition skips) is executed by the 256-PE systolic grid. The PS ARM Cortex-A53 CPU performs **zero neural network math**; it acts purely as an orchestrator triggering DMAs and reading final classification logits from DDR.
- **Single Reusable Bitstream / Independent Python Compilation**: The PL systolic array is synthesized once per precision. Each neural network compiles independently in Python to emit layer microcodes and contiguous weights preloaded into DDR.
- **Maximized On-Chip UltraRAM Utilization**:
  - **8-Bit (INT8)**: Uses all **96 UltraRAM288 blocks** (100.0% utilization) on the XCZU7EV chip, scaling on-chip SRAM to **3.93 MB** (15,360 rows $\times$ 256 B/row) for 100% on-chip intermediate activations (zero DRAM activation spilling).
  - **16-Bit (INT16)**: Uses **64 UltraRAM288 blocks** (66.7% utilization) providing **2.10 MB** (4,096 rows $\times$ 512 B/row) with zero BRAM spilling and over $2\times$ headroom for all models.
- **Dual Precision with Full Physical Timing Closure**:
  - **8-Bit Mode @ 75.0 MHz**: Worst Negative Slack $WNS = \mathbf{+0.281\text{ ns}}$ (Timing Closed, Met).
  - **16-Bit Mode @ 70.0 MHz**: Worst Negative Slack $WNS = \mathbf{+0.079\text{ ns}}$ (Timing Closed, Met).

---

## Quickstart Guide for Beginners

### Prerequisites
Before running simulations or board-level testing, ensure the following tools are installed on your Windows host machine:
1. **AMD Vivado & Vitis 2023.2** (Installed to `C:\Xilinx\Vivado\2023.2` and `C:\Xilinx\Vitis\2023.2`).
2. **Python 3.10+** with required packages:
   ```powershell
   pip install torch torchvision numpy pillow
   ```
3. **Serial Terminal Software** (PuTTY, TeraTerm, or VS Code Serial Monitor).

---

### Option A: Board-Level Testing on AMD ZCU106 (Physical Hardware)

Prebuilt bitstreams (`.bit`), hardware platforms (`.xsa`), baremetal ELFs, pre-trained weights, and test images are already generated and ready to run immediately.

#### Step 1: Connect the ZCU106 Hardware
1. Connect the **AMD ZCU106 Evaluation Board** to your PC using two micro-USB cables:
   - **USB-JTAG port (`J2`)** (for bitstream and data programming).
   - **USB-UART port (`J83`)** (for serial console output).
2. Set configuration DIP switches `SW6` to **JTAG Mode: `0000` (all 4 switches ON)**.
3. Power on the board using power switch `SW1`.

#### Step 2: Open the Serial Console
Open PuTTY or TeraTerm and connect to the Silicon Labs Quad CP2108 USB-to-UART port (typically Interface 0 / **COM7**):
- **Baud Rate**: `115200`
- **Data Bits**: `8`
- **Parity**: `None`
- **Stop Bits**: `1`
- **Flow Control**: `None`

#### Step 3: Run Inference with One Command
Open PowerShell in the `unified/` project root directory and run:

```powershell
# 1. Run All 4 Models Sequentially (MNIST -> ResNet-50 -> MobileNetV2 -> EfficientNet-B0)
.\scripts\07_deploy_zcu106.ps1 -Model all -Precision 8bit

# 2. Run MNIST LeNet-5 Digit Classification (Tests digit 7)
.\scripts\07_deploy_zcu106.ps1 -Model mnist -Precision 8bit -Image digit7

# 3. Run Full ResNet-50 on ImageNet Image "dog" (8-bit)
.\scripts\07_deploy_zcu106.ps1 -Model resnet50 -Precision 8bit -Image dog

# 4. Run MobileNetV2 on ImageNet Image "car" (16-bit)
.\scripts\07_deploy_zcu106.ps1 -Model mobilenetv2 -Precision 16bit -Image car

# 5. Run EfficientNet-B0 on ImageNet Image "cat" (8-bit)
.\scripts\07_deploy_zcu106.ps1 -Model efficientnetb0 -Precision 8bit -Image cat
```

The script will automatically program the FPGA bitstream, initialize the PS DDR4, download weights and test images via JTAG, execute the baremetal application on Cortex-A53 Core #0, and stream real-time classification results to your UART terminal.

---

### Option B: RTL Simulation Verification (No Board Required)

You can verify the exact cycle-accurate hardware execution inside the Vivado XSim simulator without any FPGA hardware connected:

```powershell
# Run 8-bit cycle-accurate simulation for MNIST LeNet-5:
.\scripts\06_run_simulations.ps1 -Model mnist -Precision 8bit

# Run 16-bit cycle-accurate simulation for MNIST LeNet-5:
.\scripts\06_run_simulations.ps1 -Model mnist -Precision 16bit

# Run ResNet-50 simulation:
.\scripts\06_run_simulations.ps1 -Model resnet50 -Precision 8bit

# Run MobileNetV2 simulation:
.\scripts\06_run_simulations.ps1 -Model mobilenetv2 -Precision 8bit

# Run EfficientNet-B0 simulation:
.\scripts\06_run_simulations.ps1 -Model efficientnetb0 -Precision 8bit

# Run all model simulations sequentially:
.\scripts\06_run_simulations.ps1 -Model all -Precision 8bit
```

During simulation, the testbench writes inputs to an emulated DDR4 slave, triggers the hardware accelerator via AXI-Lite, executes the systolic array, and reads final classification logits from DDR to verify top-1 predictions.

---

## Turnkey Reproduction Workflow (`scripts/`)

For users who want to rebuild or reproduce the entire project from scratch, the `scripts/` directory contains numbered, turnkey PowerShell scripts:

```
scripts/
├── 01_clean_all.ps1          # Clean generated build artifacts & temporary files
├── 02_compile_models.ps1     # Compile weights, microcode, & stimulus from PyTorch
├── 03_package_ip.ps1         # Package RTL into Vivado IP repository
├── 04_build_bitstream.ps1    # Synthesize, implement, & export .bit and .xsa
├── 05_build_baremetal.ps1    # Build standalone Cortex-A53 baremetal ELFs
├── 06_run_simulations.ps1    # Run cycle-accurate RTL simulations in XSim
├── 07_deploy_zcu106.ps1      # Turnkey JTAG deployment via XSDB onto ZCU106
└── reproduce_all.ps1         # Master orchestrator running all steps end-to-end
```

### Step-by-Step Script Reference

#### Step 01: Clean Workspace
Cleans temporary build files, project databases, and compiled outputs:
```powershell
.\scripts\01_clean_all.ps1
```

#### Step 02: Compile Neural Network Models
Downloads pre-trained PyTorch weights (`DEFAULT`), folds BatchNorm, extracts kernel tensors, quantizes weights, and generates microcode sequences for all 4 models:
```powershell
# Compile both 8-bit and 16-bit models:
.\scripts\02_compile_models.ps1 -Precision both

# Or compile only 8-bit:
.\scripts\02_compile_models.ps1 -Precision 8bit
```
*Generated Files*: `baremetal/data/*_weights.bin`, `baremetal/data/*_microcode.bin`, `baremetal/src/model_microcodes.h`.

#### Step 03: Package Vivado IP Core
Packages the parameterized Verilog RTL into the Vivado IP repository (`fpga/ip_repo/c_engine_dma_top`):
```powershell
.\scripts\03_package_ip.ps1 -Precision 8bit
.\scripts\03_package_ip.ps1 -Precision 16bit
```

#### Step 04: Build Bitstream & Export Hardware Platform
Creates the Block Design, synthesizes the design, runs placement and routing with timing closure, and exports the `.bit` bitstream and `.xsa` hardware platform:
```powershell
# Build 8-bit hardware (75.0 MHz, 96 UltraRAMs, WNS = +0.625 ns):
.\scripts\04_build_bitstream.ps1 -Precision 8bit

# Build 16-bit hardware (70.0 MHz, 64 UltraRAMs, WNS = +0.079 ns):
.\scripts\04_build_bitstream.ps1 -Precision 16bit
```
*Exported Artifacts*:
- `fpga/export/8bit/c_engine.bit`, `fpga/export/8bit/c_engine.xsa`
- `fpga/export/16bit/c_engine.bit`, `fpga/export/16bit/c_engine.xsa`
- Synchronized to `baremetal/export/`

#### Step 05: Build Baremetal Cortex-A53 Applications
Compiles the standalone baremetal C application using `aarch64-none-elf-gcc` from Vitis 2023.2:
```powershell
# Build both 8-bit and 16-bit baremetal ELFs:
.\scripts\05_build_baremetal.ps1 -Target all
```
*Generated Binaries*:
- `baremetal/build/baremetal_c_engine_8bit.elf` (266 KB)
- `baremetal/build/baremetal_c_engine_16bit.elf` (266 KB)

#### Step 06: Run Simulation Verification
Executes cycle-accurate RTL simulation in Vivado XSim (see Option B above).

#### Step 07: Deploy to AMD ZCU106 Board
Deploys bitstream, preloads weights and test images to DDR, and launches baremetal inference on Cortex-A53 Core #0 (see Option A above).

---

### Master One-Click Orchestrator (`reproduce_all.ps1`)

To execute the entire pipeline end-to-end automatically:

```powershell
# Rebuild models, IP, and baremetal applications, and run simulation:
.\scripts\reproduce_all.ps1 -Precision 8bit -SkipSynthesis

# Run full flow including Vivado synthesis, implementation, and simulation:
.\scripts\reproduce_all.ps1 -Precision 8bit

# Run full flow and automatically deploy to physical board:
.\scripts\reproduce_all.ps1 -Precision 8bit -Deploy
```

---

## Dedicated DDR Memory Layout (4 GB PS DDR4)

All models and datasets reside in distinct, non-overlapping regions in physical DDR memory, allowing concurrent preloading via JTAG without collisions:

| Memory Base Address | Size | Description |
| :--- | :--- | :--- |
| `0x0000_0000` | 512 MB | Standalone Baremetal ELF (`baremetal_c_engine.elf`) |
| `0x2000_0000` | 1 MB | MNIST Digit Image Buffer (`digit0`..`digit9`, 1 KB 8-bit / 2 KB 16-bit) |
| `0x2010_0000` | 1 MB | ImageNet Input Image Buffer (`dog`, `car`, `cat`, `aeroplane`) |
| `0x2100_0000` | 64 KB | Microcode Instruction uops Buffer (`microcode.bin`) |
| `0x21FF_0000` | 4 Bytes | **Model Selection Mailbox** (0=All, 1=MNIST, 2=ResNet-50, 3=MobileNetV2, 4=EfficientNet-B0) |
| `0x3000_0000` | 16 MB | MNIST LeNet-5 Trained Weights (32 KB 8-bit / 64 KB 16-bit) |
| `0x3100_0000` | 64 MB | ResNet-50 Full Backbone Trained Weights (23.5 MB 8-bit / 47.1 MB 16-bit) |
| `0x3500_0000` | 16 MB | MobileNetV2 Trained Weights (2.27 MB 8-bit / 4.54 MB 16-bit) |
| `0x3600_0000` | 16 MB | EfficientNet-B0 Trained Weights (3.41 MB 8-bit / 6.82 MB 16-bit) |
| `0x3800_0000` | 1 MB | Accelerator Output Logits Buffer |
| `0xA000_0000` | 64 KB | Accelerator AXI-Lite MMIO Control Slave |

---

## Directory Structure

```text
unified/
├── baremetal/                   # Standalone baremetal C runtime & deployment
│   ├── build/                   # Compiled Cortex-A53 ELFs (8bit and 16bit)
│   ├── data/                    # Preloaded weights (.bin), microcodes, test images
│   ├── export/                  # Exported hardware handoff (.xsa)
│   ├── scripts/                 # Automated XSDB JTAG deployment scripts (run_xsdb.tcl)
│   ├── src/                     # C driver (c_engine_driver.c), registers, main.c
│   ├── run_all.ps1              # Run all 4 models sequentially on ZCU106
│   ├── run_mnist.ps1            # Run MNIST digit classification
│   ├── run_resnet50.ps1         # Run ResNet-50 inference
│   ├── run_mobilenetv2.ps1      # Run MobileNetV2 inference
│   └── run_efficientnetb0.ps1   # Run EfficientNet-B0 inference
├── fpga/                        # Vivado synthesis, block design, and bitstreams
│   ├── export/                  # Exported bitstreams (.bit) and platforms (.xsa)
│   │   ├── 8bit/                # 8-bit bitstream, XSA, utilization & timing reports
│   │   └── 16bit/               # 16-bit bitstream, XSA, utilization & timing reports
│   ├── ip_repo/                 # Packaged c_engine_dma_top IP core
│   ├── build_bitstream.tcl      # Full synthesis & implementation script
│   ├── create_bd.tcl            # ZCU106 block design generation script
│   └── package_ip.tcl           # Vivado IP packaging script
├── models/                      # Model definitions, PyTorch compilers & weight exporters
│   ├── compile_all.py           # Master script compiling all 4 models
│   ├── mnist/                   # LeNet-5 compiler & weight extraction
│   ├── resnet50_full/           # Full ResNet-50 compiler & weight extraction
│   ├── mobilenetv2/             # MobileNetV2 compiler & weight extraction
│   └── efficientnetb0/          # EfficientNet-B0 compiler & weight extraction
├── rtl/                         # Verilog RTL source files (256-PE systolic array)
│   ├── hw_config_8bit.vh        # 8-bit config (75 MHz, 3.93 MB SRAM, 96 UltraRAMs)
│   ├── hw_config_16bit.vh       # 16-bit config (70 MHz, 2.10 MB SRAM, 64 UltraRAMs)
│   ├── c_engine_top.v           # 256-PE Systolic Array Compute Core
│   ├── c_engine_dma_top.v       # Complete accelerator top with dual 128-bit DMAs
│   ├── c_engine_controller.v    # Autonomous sequencer FSM & 14-bit addressing
│   ├── multibank_dualport_mem.v # True dual-port multi-bank UltraRAM memory
│   └── requantization_block.v   # Requantization, residual adders, & LUT activations
├── scripts/                     # Numbered turnkey PowerShell reproduction suite
│   ├── 01_clean_all.ps1 .. 07_deploy_zcu106.ps1
│   └── reproduce_all.ps1
├── sim/                         # Vivado XSim cycle-accurate testbenches
│   ├── tb_unified_c_engine.v    # Unified verification testbench for all models
│   └── axi4_ddr_model.v         # Dual-port 128-bit AXI4 DDR memory model
└── README.md                    # This master documentation file
```

---

## Beginner's Troubleshooting Guide

| Issue | Cause | Solution |
| :--- | :--- | :--- |
| **`xsdb not found`** | Vitis 2023.2 is not in your system `PATH`. | The scripts automatically search `C:\Xilinx\Vitis\2023.2\bin\xsdb.bat`. Ensure Vitis 2023.2 is installed at that location or add its `bin` folder to your system `PATH`. |
| **`Cannot connect to hw_server`** | Hardware server not started or USB disconnected. | Make sure the micro-USB cable is securely connected to the **JTAG port (`J2`)** on the ZCU106 and the power switch is ON. `run_xsdb.tcl` auto-connects to `127.0.0.1:3121`. |
| **No output on UART Terminal** | Wrong COM port or baud rate mismatch. | In Device Manager, check under *Ports (COM & LPT)* for **Silicon Labs CP2108 Quad USB to UART**. Select **Interface 0** (typically `COM7`). Ensure your terminal is configured for **115200 baud, 8 data bits, no parity, 1 stop bit (8-N-1)**. |
| **FPGA DONE LED does not turn green** | Board DIP switches not in JTAG mode. | Verify that switch `SW6` is configured for **JTAG mode: `0000` (all 4 pins ON)**. |
| **Python ModuleNotFoundError (`torch` / `torchvision`)** | PyTorch is missing in the active Python environment. | Run `pip install torch torchvision numpy pillow` in your terminal before running `02_compile_models.ps1`. |
| **Simulation timeout in XSim** | Simulation macro missing or incorrect model. | Use [`scripts/06_run_simulations.ps1`](scripts/06_run_simulations.ps1), which automatically defines `-d SIM_MNIST`, `-d SIM_RESNET50`, etc., and sets the active `hw_config.vh` header. |
