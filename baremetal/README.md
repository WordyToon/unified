# AMD ZCU106 Baremetal Neural Accelerator Guide

This folder contains the complete standalone baremetal C application and automated deployment tooling for driving the **Unified 256-PE Systolic Compute Engine** on the **AMD Zynq UltraScale+ ZCU106 Evaluation Board** (`xczu7ev-ffvc1156-2-e`).

---

## 1. System Architecture & DDR Memory Map

The application executes baremetal on ARM Cortex-A53 Core #0. The processor's primary roles are:
1. **Preload Model Assets**: JTAG download via XSDB (Option 1B) pushes model weights, microcode, and test images into dedicated non-overlapping DDR4 locations.
2. **Mailbox Control**: Writes target model execution selection (0=All, 1=MNIST, 2=ResNet-50, 3=MobileNetV2, 4=EfficientNet-B0) to `0x21FF_0000`.
3. **Accelerator Trigger & Synchronization**: Controls AXI-Lite MMIO (`0xA000_0000`), initiates DMA transfers, and polls engine execution sticky flags.
4. **Logits Extraction & Display**: Reads output logits from DDR `0x3800_0000`, applies Softmax, computes Top-5 predictions, and prints cycle counts, latency, and FPS to the UART console.

### Dedicated Non-Overlapping DDR Memory Map (4 GB PS DDR4 Low Space)
| Memory Base | Size | Description |
| :--- | :--- | :--- |
| `0x0000_0000` | 512 MB | Standalone Baremetal ELF (`baremetal_c_engine.elf`) |
| `0x2000_0000` | 1 MB | MNIST Digit Image Buffer (`digit0`..`digit9`, 1 KB 8-bit / 2 KB 16-bit) |
| `0x2010_0000` | 1 MB | ImageNet Input Image Buffer (`dog`, `car`, `cat`, 150.5 KB 8-bit / 301 KB 16-bit) |
| `0x2100_0000` | 64 KB | Microcode Instruction uops Buffer (`microcode.bin`) |
| `0x21FF_0000` | 4 Bytes | **Model Selection Control Mailbox** (0=All, 1=MNIST, 2=ResNet-50, 3=MobileNetV2, 4=EfficientNet-B0) |
| `0x3000_0000` | 16 MB | MNIST LeNet-5 Trained Weights (32 KB 8-bit / 64 KB 16-bit, preloaded to SRAM) |
| `0x3100_0000` | 64 MB | ResNet-50 Full Backbone Trained Weights (23.5 MB 8-bit / 47.1 MB 16-bit) |
| `0x3500_0000` | 16 MB | MobileNetV2 Trained Weights (2.27 MB 8-bit / 4.54 MB 16-bit) |
| `0x3600_0000` | 16 MB | EfficientNet-B0 Trained Weights (3.41 MB 8-bit / 6.82 MB 16-bit) |
| `0x3800_0000` | 1 MB | Accelerator Output Logits Buffer |
| `0xA000_0000` | 64 KB | C-Engine Accelerator AXI-Lite MMIO Slave |

---

## 2. Directory Structure

```text
baremetal/
├── build/
│   ├── baremetal_c_engine_8bit.elf     # Compiled Cortex-A53 ELF for 8-bit @ 75 MHz
│   ├── baremetal_c_engine_16bit.elf    # Compiled Cortex-A53 ELF for 16-bit @ 70 MHz
│   └── baremetal_c_engine.elf          # Active ELF symlink/copy
├── data/
│   ├── digit0_8bit.bin .. digit9_8bit.bin    # Pre-formatted MNIST 32x32 test digits (1 KB each)
│   ├── digit0_16bit.bin .. digit9_16bit.bin  # Pre-formatted MNIST 32x32 test digits (2 KB each)
│   ├── dog_8bit.bin, car_8bit.bin, etc.      # Pre-formatted ImageNet 196-chunk images (150.5 KB)
│   ├── dog_16bit.bin, car_16bit.bin, etc.    # Pre-formatted ImageNet 196-chunk images (301 KB)
│   ├── mnist_weights.bin, mnist_16bit_weights.bin
│   ├── resnet50_8bit_weights.bin, resnet50_16bit_weights.bin
│   ├── mobilenetv2_8bit_weights.bin, mobilenetv2_16bit_weights.bin
│   └── efficientnetb0_8bit_weights.bin, efficientnetb0_16bit_weights.bin
├── export/
│   ├── 8bit/c_engine.xsa               # Hardware handoff platform (75.0 MHz)
│   └── 16bit/c_engine.xsa              # Hardware handoff platform (70.0 MHz)
├── scripts/
│   ├── build_app.tcl                   # Automated XSCT build script
│   └── run_xsdb.tcl                    # Automated XSDB JTAG preloader and runner (Option 1B)
├── src/
│   ├── c_engine_regs.h                 # Register map & DDR memory layout
│   ├── c_engine_driver.h / .c          # Production baremetal driver with cache management
│   ├── main.c                          # Multi-model evaluation application
│   └── model_microcodes.h              # Built-in microcode ROM tables
├── run_all.ps1                         # PowerShell script: Run all 4 models sequentially
├── run_mnist.ps1                       # PowerShell script: Run MNIST with configurable digit
├── run_resnet50.ps1                    # PowerShell script: Run ResNet-50 with configurable image
├── run_mobilenetv2.ps1                 # PowerShell script: Run MobileNetV2
└── run_efficientnetb0.ps1              # PowerShell script: Run EfficientNet-B0
```

---

## 3. How to Run on AMD ZCU106 Board

### Physical Board Setup
1. Connect the **AMD ZCU106 Evaluation Kit** to your Windows host machine using two micro-USB cables:
   - USB-JTAG port (`J2`)
   - USB-UART port (`J83`)
2. Set configuration DIP switches `SW6` to **JTAG Mode: `0000` (all ON)**.
3. Power on the board with the power switch (`SW1`).
4. Open a serial terminal (PuTTY, TeraTerm, or VS Code Serial Monitor) pointing to the Silicon Labs CP2108 USB to Quad UART port (typically Interface 0):
   - **Baud Rate**: `115200`
   - **Data Bits**: `8`
   - **Parity**: `None`
   - **Stop Bits**: `1`
   - **Flow Control**: `None`

---

### One-Click PowerShell Execution

Open PowerShell in the project directory:

#### 1. Run All 4 Models (MNIST, ResNet-50, MobileNetV2, EfficientNet-B0)
```powershell
# 8-Bit Unified Engine @ 75.0 MHz (Default)
.\baremetal\run_all.ps1 -Precision 8bit -Image dog

# 16-Bit Unified Engine @ 70.0 MHz
.\baremetal\run_all.ps1 -Precision 16bit -Image dog
```

#### 2. Run MNIST LeNet-5 Digit Classification
```powershell
# Classify Digit 7
.\baremetal\run_mnist.ps1 -Precision 8bit -Digit 7

# Classify Digit 3 in 16-bit
.\baremetal\run_mnist.ps1 -Precision 16bit -Digit 3
```

#### 3. Run ResNet-50 Full Backbone (1,000 Classes)
```powershell
.\baremetal\run_resnet50.ps1 -Precision 8bit -Image dog
.\baremetal\run_resnet50.ps1 -Precision 16bit -Image car
```

#### 4. Run MobileNetV2 (Depthwise Mode = 2'b11)
```powershell
.\baremetal\run_mobilenetv2.ps1 -Precision 8bit -Image dog
```

#### 5. Run EfficientNet-B0 (MBConv + LUT Swish)
```powershell
.\baremetal\run_efficientnetb0.ps1 -Precision 8bit -Image dog
```

---

## 4. Rebuilding the Baremetal ELFs (Optional)

If modifications are made to `baremetal/src/`:
```powershell
# Rebuild both 8-Bit and 16-Bit ELFs using the standalone gcc toolchain:
.\scripts\05_build_baremetal.ps1 -Target all

# Or directly via baremetal builder:
powershell -ExecutionPolicy Bypass -File .\baremetal\build_elf.ps1 -Target all
```
Both outputs will automatically update `baremetal/build/baremetal_c_engine_8bit.elf` and `baremetal_c_engine_16bit.elf`.
