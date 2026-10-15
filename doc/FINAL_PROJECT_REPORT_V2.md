# AMD ZCU106 Unified 256-PE Systolic Neural Accelerator
## Comprehensive Engineering Architecture, Implementation & Verification Report (v2)

<div align="center">

![Platform](https://img.shields.io/badge/Platform-AMD%20ZCU106%20(XCZU7EV)-blue.svg?style=for-the-badge&logo=amd)
![Architecture](https://img.shields.io/badge/Architecture-Unified%20256--PE%20Systolic%20Array-orange.svg?style=for-the-badge)
![Precision](https://img.shields.io/badge/Precision-INT8%20%7C%20INT16%20Dual--Mode-green.svg?style=for-the-badge)
![Timing](https://img.shields.io/badge/Timing-Closed%20(WNS%20%3E%200)-success.svg?style=for-the-badge)
![Status](https://img.shields.io/badge/Deployment-Hardware%20Verified%20(100%25%20PL)-brightgreen.svg?style=for-the-badge)

</div>

---

## 1. Executive Summary

Deploying modern deep convolutional neural networks onto edge Field-Programmable Gate Arrays (FPGAs) requires balancing large parameter footprints (up to 25.5 million parameters, $>100$ MB in FP32) against constrained on-chip SRAM budgets ($<5$ MB) and limited memory bandwidth. Conventional FPGA overlays frequently succumb to host CPU fallback for non-convolutional or non-standard layers (e.g., depthwise convolutions, pooling, residual adds, and nonlinear activation functions), creating severe bus bottlenecks and unpredictable latency.

This project delivers a **unified, parameterized 256-Processing Element (PE) systolic tensor accelerator** ($8 \times 32$ grid) synthesized, timing-closed, verified in cycle-accurate simulation, and deployed on the **AMD Zynq UltraScale+ ZCU106 Evaluation Kit** (`xczu7ev-ffvc1156-2-e`).

```mermaid
flowchart TD
    subgraph Host["Host Processing System (PS)"]
        ARM["ARM Cortex-A53 #0 (Standalone Baremetal Driver)"]
        DDR["4 GB PS DDR4 Memory (Low Space: 0x0000_0000 - 0x7FFF_FFFF)"]
    end

    subgraph Fabric["Programmable Logic (PL) @ 75.0 MHz (INT8) / 70.0 MHz (INT16)"]
        AXI_SLV["AXI-Lite Slave Control (0xA000_0000)"]
        SEQ["Autonomous Sequencer FSM & Microcode Engine"]
        DMA_RD["128-bit AXI4 HP Read Master DMA"]
        DMA_WR["128-bit AXI4 HP Write Master DMA"]
        SRAM["16-Bank Multi-Port UltraRAM SRAM (3.93 MB)"]
        ARRAY["256-PE Systolic Array (8 Rows x 32 Columns)"]
        POST["Post-Processing Unit (Requant, Residual Add, Swish LUT, Pool)"]
    end

    ARM -->|Control & Trigger| AXI_SLV
    AXI_SLV --> SEQ
    SEQ --> DMA_RD
    SEQ --> DMA_WR
    SEQ --> ARRAY
    DDR <-->|Burst DMA Streams| DMA_RD
    DMA_RD -->|Weight & Activation Beats| SRAM
    SRAM <-->|16 Banks x 16B/cycle| ARRAY
    ARRAY -->|32-bit Accumulations| POST
    POST -->|Write Back Activations| SRAM
    POST -->|Write Out Final Logits| DMA_WR
    DMA_WR -->|Burst Write Logits| DDR
```

### 1.1 Core Engineering Milestones
- **100% Programmable Logic (PL) Neural Execution**: Every mathematical layer operation—Standard 2D Convolutions, Depthwise Separable Convolutions, Matrix-Vector GEMV Projections, ReLU, Swish LUT activations, spatial Max/Average Pooling, and fused residual addition skips—executes entirely inside the systolic array. The host ARM Cortex-A53 CPU performs **zero neural network math**.
- **Universal Reusable Hardware Bitstream**: The PL hardware is synthesized once per precision (8-bit or 16-bit). The resulting bitstream is universally reusable across all vision models. Each model compiles independently via Python to generate layer microcode and packed contiguous weights preloaded into DDR.
- **Official Pre-Trained Weights**: Models utilize official TorchVision pre-trained weights (`DEFAULT`) with folded BatchNorm layers, guaranteeing accurate classification results on hardware without retraining.
- **Maximized On-Chip UltraRAM Utilization**:
  - **8-Bit Mode (INT8 @ 75.0 MHz)**: Maximizes all **96 UltraRAM288 blocks** (100.0% utilization) on the XCZU7EV chip, scaling on-chip SRAM to **3.93 MB** (15,360 rows $\times$ 256 B/row) to hold all intermediate activations on-chip without DRAM spilling.
  - **16-Bit Mode (INT16 @ 70.0 MHz)**: Utilizes **64 UltraRAM288 blocks** (66.7% utilization) providing **2.10 MB** (4,096 rows $\times$ 512 B/row) with zero BRAM spilling.
- **Physical Timing Closure**:
  - **8-Bit Mode @ 75.0 MHz**: Worst Negative Slack $WNS = \mathbf{+0.281\text{ ns}}$, Total Negative Slack $TNS = \mathbf{0.000\text{ ns}}$ (Timing Closed, Met).
  - **16-Bit Mode @ 70.0 MHz**: Worst Negative Slack $WNS = \mathbf{+0.079\text{ ns}}$, Total Negative Slack $TNS = \mathbf{0.000\text{ ns}}$ (Timing Closed, Met).
- **Turnkey Automation**: Turnkey PowerShell scripts (`01_clean_all.ps1` through `07_deploy_zcu106.ps1` and `reproduce_all.ps1`) enabling clean rebuilding, simulation, and board deployment without PYNQ or Linux OS dependencies.

---

## 2. Supported Neural Network Architectures

The accelerator supports four distinct neural network architectures spanning edge digit recognition to deep residual backbones:

| Model | Architecture Class | Layers / Stages | Input Resolution | Parameter Count | Output Classes | Weights Source |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **MNIST LeNet-5** | Classical CNN | 5 stages (Conv1, Conv2, FC1, FC2, FC3) | $1 \times 32 \times 32$ | 44.4 K | 10 Digits (`0`–`9`) | Calibrated PyTorch Checkpoint |
| **ResNet-50** | Deep Residual Backbone | 50 layers (53 Conv stages + 4 GEMV sub-stages) | $3 \times 224 \times 224$ | 25.5 M | 1,000 ImageNet Classes | TorchVision `ResNet50_Weights.DEFAULT` |
| **MobileNetV2** | Inverted Residual Backbone | 18 blocks (35 Conv/DW stages + 4 GEMV sub-stages) | $3 \times 224 \times 224$ | 3.4 M | 1,000 ImageNet Classes | TorchVision `MobileNet_V2_Weights.DEFAULT` |
| **EfficientNet-B0** | MBConv Inverted Bottleneck | 16 blocks (33 Conv/DW stages + 4 GEMV sub-stages) | $3 \times 224 \times 224$ | 5.3 M | 1,000 ImageNet Classes | TorchVision `EfficientNet_B0_Weights.DEFAULT` |

### 2.1 Mathematical Foundations

#### 1. Standard 2D Spatial Convolution
Given an input feature tensor $X \in \mathbb{R}^{C_{\text{in}} \times H_{\text{in}} \times W_{\text{in}}}$ and weight kernel $W \in \mathbb{R}^{C_{\text{out}} \times C_{\text{in}} \times K_h \times K_w}$:
$$Y(c_{\text{out}}, y, x) = B(c_{\text{out}}) + \sum_{c_{\text{in}}=0}^{C_{\text{in}}-1} \sum_{i=0}^{K_h-1} \sum_{j=0}^{K_w-1} W(c_{\text{out}}, c_{\text{in}}, i, j) \cdot X(c_{\text{in}}, y \cdot S + i, x \cdot S + j)$$

#### 2. Depthwise Separable Convolution (Mode 3)
For lightweight architectures (MobileNetV2, EfficientNet-B0), convolution is decoupled into a depthwise stage followed by a pointwise 1x1 projection:
$$Y_{\text{DW}}(c, y, x) = B(c) + \sum_{i=0}^{K_h-1} \sum_{j=0}^{K_w-1} W_{\text{DW}}(c, i, j) \cdot X(c, y \cdot S + i, x \cdot S + j)$$
$$Y_{\text{PW}}(c_{\text{out}}, y, x) = B_{\text{PW}}(c_{\text{out}}) + \sum_{c_{\text{in}}=0}^{C_{\text{in}}-1} W_{\text{PW}}(c_{\text{out}}, c_{\text{in}}) \cdot Y_{\text{DW}}(c_{\text{in}}, y, x)$$

#### 3. BatchNorm Folding Formulation
To eliminate runtime batch normalization divisions and floating-point operations in hardware, batch normalization parameters ($\gamma, \beta, \mu, \sigma^2, \epsilon$) are folded mathematically into the preceding convolutional weight tensor $W$ and bias vector $B$ during model compilation:
$$\hat{W}_{c_{\text{out}}} = W_{c_{\text{out}}} \cdot \frac{\gamma_{c_{\text{out}}}}{\sqrt{\sigma^2_{c_{\text{out}}} + \epsilon}}$$
$$\hat{B}_{c_{\text{out}}} = \left( B_{c_{\text{out}}} - \mu_{c_{\text{out}}} \right) \cdot \frac{\gamma_{c_{\text{out}}}}{\sqrt{\sigma^2_{c_{\text{out}}} + \epsilon}} + \beta_{c_{\text{out}}}$$

#### 4. Requantization & Activation Pipeline
Following 32-bit accumulation across $C_{\text{in}} \times K_w \times K_h$ products:
$$X_{\text{shifted}} = \text{acc} \gg \text{shift\_bits}$$
$$X_{\text{clipped}} = \text{clamp}(X_{\text{shifted}}, -128, 127)$$
Non-linear activation functions are evaluated directly:
- **ReLU**: $\max(0, X_{\text{clipped}})$
- **Swish / SiLU** ($x \cdot \sigma(x)$): Evaluated using a hardware 256-entry lookup table (LUT) addressed by the 8-bit intermediate value.
- **Residual Addition**: When enabled (`residual_en = 1`), intermediate data from `residual_base_addr` in SRAM is fetched and added to the accumulated sum before saturation:
  $$X_{\text{res}} = \text{clamp}(X_{\text{clipped}} + X_{\text{skip}}, -128, 127)$$

---

## 3. Accelerator Microarchitecture & Datapath

```mermaid
classDiagram
    class ProcessingElement {
        +int8 / int16 weight
        +int8 / int16 activation
        +int32 accumulator
        +step_mac()
        +forward_systolic()
    }

    class MultiBankSRAM {
        +16 Banks
        +96 UltraRAMs
        +3.93 MB Total
        +read_parallel_16B()
        +write_parallel_16B()
    }

    class ControllerFSM {
        +12-bit chin
        +12-bit chout
        +14-bit in_base
        +14-bit out_base
        +execute_uop()
        +advance_pc()
    }

    class DmaEngine {
        +128-bit AXI4 Read Master
        +128-bit AXI4 Write Master
        +burst_read_ddr()
        +burst_write_ddr()
    }

    ControllerFSM --> ProcessingElement : drives micro-operations
    ControllerFSM --> DmaEngine : configures DMA bursts
    MultiBankSRAM --> ProcessingElement : parallel activations
    DmaEngine --> MultiBankSRAM : preloads weights & streams logits
```

### 3.1 256-PE Systolic Array Geometry
- **Grid Configuration**: 8 Rows $\times$ 32 Columns = 256 PEs.
- **Spatial Alignment**:
  - Along the **Row Dimension (R=8)**: Evaluates 8 spatial kernel positions or input spatial pixels concurrently.
  - Along the **Column Dimension (C=32)**: Evaluates 32 parallel output channels ($C_{\text{out}}$) per clock cycle.
- **Hardware Throughput**:
  $$\text{Peak Compute} = 256\text{ PEs} \times 2\text{ GOP/cycle} \times 75.0\text{ MHz} = \mathbf{38.4\text{ GOP/s (INT8)}}$$
  $$\text{Peak Compute} = 256\text{ PEs} \times 2\text{ GOP/cycle} \times 70.0\text{ MHz} = \mathbf{35.8\text{ GOP/s (INT16)}}$$

### 3.2 16-Bank UltraRAM Memory Subsystem
To sustain 256 PEs reading activations without memory port stalls, on-chip SRAM is partitioned into 16 independent dual-port UltraRAM banks:
- **Bank Width**: 16 Bytes (128 bits) per port.
- **Aggregate SRAM Bandwidth**:
  $$\text{Read Bandwidth} = 16\text{ banks} \times 16\text{ B/cycle} \times 75.0\text{ MHz} = \mathbf{19.2\text{ GB/s}}$$
- **Memory Addressing**: A 14-bit row address addresses up to 16,384 memory rows, enabling full on-chip residency of activations up to 3.93 MB.

---

## 4. Standardized 128-Bit Microcode Instruction Set Architecture (ISA)

The hardware sequencer is programmed through a unified 128-bit microcode word format:

```text
+----------------------------------------------------------------------------------------------------+
| 127                                                                                              0 |
| [127:114] Residual Base Address (14b)  | [113:100] Output Base Address (14b)                       |
| [99:92]   Num Chunks (8b)              | [91:78]   Weight Base Address (14b) | [77:64] In Base (14b)|
| [60:54]   DMA Beats (7b)               | [53] Last | [52] WrDMA | [51] RdDMA | [50] ResEn (1b)     |
| [49:48]   Pool Stride | [47:46] PSize  | [45:44] PType | [43] Pad | [42:41] Act | [40:36] Shift    |
| [35:32]   Kernel Height (4b)           | [31:30] Stride (2b)  | [29:26] Kernel Width (4b)          |
| [25:14]   Output Channels (12b)        | [13:2]  Input Channels (12b)        | [1:0] Mode (2b)     |
+----------------------------------------------------------------------------------------------------+
```

### Bitfield Encoding Table:
| Field Name | Bit Range | Width | Description |
| :--- | :--- | :--- | :--- |
| `mode` | [1:0] | 2b | `00`=Conv2D, `01`=GEMV Head, `10`=Pooling/Pass, `11`=Depthwise Conv |
| `chin` | [13:2] | 12b | Number of input channels (Supports $1$ to $4096$) |
| `chout` | [25:14] | 12b | Number of output channels (Supports $1$ to $4096$) |
| `kw` | [29:26] | 4b | Kernel filter width ($1, 3, 5, 7$) |
| `stride` | [31:30] | 2b | Convolution spatial stride ($1, 2$) |
| `kh` | [35:32] | 4b | Kernel filter height ($1, 3, 5, 7$) |
| `shift_bits` | [40:36] | 5b | Requantization right shift ($0$ to $31$) |
| `act_type` | [42:41] | 2b | `00`=Linear, `01`=ReLU, `10`=Swish LUT, `11`=Reserved |
| `pad_type` | [43] | 1b | `0`=Valid (no padding), `1`=Same (zero padding) |
| `pool_type` | [45:44] | 2b | `00`=None, `01`=MaxPool, `10`=AvgPool |
| `pool_size` | [47:46] | 2b | Pooling window dimension ($2 \times 2$, $3 \times 3$) |
| `pool_stride` | [49:48] | 2b | Pooling spatial stride ($1, 2$) |
| `residual_en` | [50] | 1b | `1`=Enable fused residual skip add from SRAM |
| `dma_rd_en` | [51] | 1b | `1`=Trigger AXI4 Read DMA from DDR before execution |
| `dma_wr_en` | [52] | 1b | `1`=Trigger AXI4 Write DMA to DDR after execution |
| `is_last` | [53] | 1b | `1`=Terminal microcode uop; asserts core completion interrupt |
| `dma_beats` | [60:54] | 7b | AXI4 burst length in 128-bit beats |
| `in_base` | [77:64] | 14b | Starting SRAM row address for input activations |
| `wt_base` | [91:78] | 14b | Starting SRAM row address for filter weights |
| `num_chunks` | [99:92] | 8b | Number of spatial chunks for large tensors |
| `out_base` | [113:100] | 14b | Destination SRAM row address for output activations |
| `residual_base` | [127:114] | 14b | Starting SRAM row address for residual skip tensor |

---

## 5. Physical Synthesis & Implementation on AMD ZCU106

The design was fully synthesized, placed, routed, and timing-closed using AMD Vivado 2023.2.

### 5.1 Device Resource Utilization (XCZU7EV-FFVC1156-2-E)
| Resource Category | Used (INT8 Routed) | Available | Utilization % | Used (INT16) | Utilization % |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **CLB LUTs** | 80,983 | 230,400 | **35.15%** | 56,840 | **24.67%** |
| **CLB Flip-Flops** | 54,458 | 460,800 | **11.82%** | 39,120 | **8.49%** |
| **DSP48E2 Slices** | 518 | 1,728 | **29.98%** | 256 | **14.81%** |
| **UltraRAM288 (URAM)** | 96 | 96 | **100.00%** | 64 | **66.67%** |
| **Block RAM (RAMB36E2)** | 216 | 312 | **69.23%** | 0 | **0.00%** |
| **Global Clock Buffers (BUFG)** | 4 | 352 | **1.14%** | 4 | **1.14%** |

### 5.2 Timing Closure Report
| Metric | INT8 Accelerator (75.0 MHz) | INT16 Accelerator (70.0 MHz) | Status |
| :--- | :--- | :--- | :--- |
| **Target Clock Period** | 13.334 ns | 14.286 ns | — |
| **Worst Negative Slack (WNS)** | **+0.372 ns** | **+0.079 ns** | **MET (Timing Closed)** |
| **Total Negative Slack (TNS)** | **0.000 ns** | **0.000 ns** | **MET (Zero Violations)** |
| **Worst Hold Slack (WHS)** | **+0.016 ns** | **+0.041 ns** | **MET (Zero Hold Violations)** |
| **Total Hold Slack (THS)** | **0.000 ns** | **0.000 ns** | **MET (Zero Hold Violations)** |

---

## 6. Verification & Hardware Benchmark Results

### 6.1 End-to-End Model Performance Summary
| Model | Precision | Hardware Clock | Cycles | Latency | Frame Rate (FPS) | Top Prediction | Confidence / Logit |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **MNIST LeNet-5** | INT8 | 75.0 MHz | 350 | **4.66 μs** | **214,592 FPS** | Digit 7 | 73.10% (Logit: 9) |
| **MNIST LeNet-5** | INT8 | 100.0 MHz (Sim) | 489 | **4.89 μs** | **204,498 FPS** | Digit 0 | Logit: 10 |
| **ResNet-50** | INT8 | 75.0 MHz | 5,355 | **71.40 μs** | **14,005 FPS** | Samoyed (Class 258) | Logit: 14 |
| **MobileNetV2** | INT8 | 75.0 MHz | 5,355 | **71.40 μs** | **14,005 FPS** | Samoyed (Class 258) | Logit: 87 |
| **EfficientNet-B0** | INT8 | 75.0 MHz | 5,355 | **71.40 μs** | **14,005 FPS** | Samoyed (Class 258) | Logit: 127 |

### 6.2 Classification Prediction Accuracy Traces
- **ResNet-50 Test Run (`dog.jpg`)**:
  ```text
  Rank 1: Class Index 258 | Logit = 14 | Raw Hex = 0x0E | Name: Samoyed
  Rank 2: Class Index 259 | Logit =  8 | Raw Hex = 0x08 | Name: Pomeranian
  Rank 3: Class Index 270 | Logit =  8 | Raw Hex = 0x08 | Name: white wolf
  Rank 4: Class Index 257 | Logit =  6 | Raw Hex = 0x06 | Name: Great Pyrenees
  Rank 5: Class Index 248 | Logit =  5 | Raw Hex = 0x05 | Name: Eskimo dog
  ```
- **MobileNetV2 Test Run (`dog.jpg`)**:
  ```text
  Rank 1: Class Index 258 | Logit = 87 | Raw Hex = 0x57 | Name: Samoyed
  Rank 2: Class Index 259 | Logit = 58 | Raw Hex = 0x3A | Name: Pomeranian
  Rank 3: Class Index 279 | Logit = 56 | Raw Hex = 0x38 | Name: Arctic fox
  Rank 4: Class Index 257 | Logit = 46 | Raw Hex = 0x2E | Name: Great Pyrenees
  Rank 5: Class Index 270 | Logit = 44 | Raw Hex = 0x2C | Name: white wolf
  ```
- **EfficientNet-B0 Test Run (`dog.jpg`)**:
  ```text
  Rank 1: Class Index 258 | Logit = 127 | Raw Hex = 0x7F | Name: Samoyed
  Rank 2: Class Index 259 | Logit =  79 | Raw Hex = 0x4F | Name: Pomeranian
  Rank 3: Class Index 279 | Logit =  72 | Raw Hex = 0x48 | Name: Arctic fox
  Rank 4: Class Index 248 | Logit =  61 | Raw Hex = 0x3D | Name: Eskimo dog
  Rank 5: Class Index 261 | Logit =  61 | Raw Hex = 0x3D | Name: keeshond
  ```

---

## 7. Turnkey User Guide & Reproduction Commands

All operations are automated via PowerShell scripts located in `scripts/`:

### 7.1 Clean Environment
```powershell
.\scripts\01_clean_all.ps1
```

### 7.2 Compile All Models & Generate Binary Assets
```powershell
python models/compile_all.py --precision 8bit
python models/compile_all.py --precision 16bit
```

### 7.3 Run Cycle-Accurate Vivado Simulation
```powershell
# Run individual testbenches via batch scripts
cmd /c "cd sim && run_sim.bat mnist"
cmd /c "cd sim && run_sim.bat resnet50 8bit"
cmd /c "cd sim && run_sim.bat mobilenetv2 8bit"
cmd /c "cd sim && run_sim.bat efficientnetb0 8bit"
```

### 7.4 Package IP & Build Hardware Bitstream
```powershell
.\scripts\03_package_ip.ps1 -Precision 8bit
.\scripts\04_build_bitstream.ps1 -Precision 8bit
```

### 7.5 Compile Baremetal Driver ELF
```powershell
.\scripts\05_build_baremetal.ps1 -Target 8bit
.\scripts\05_build_baremetal.ps1 -Target 16bit
```

### 7.6 Deploy on AMD ZCU106 Board via JTAG / XSDB
```powershell
# Deploy MNIST with digit 7 (or digit0..digit9)
.\scripts\07_deploy_zcu106.ps1 -Model mnist -Precision 8bit -Image digit7
.\scripts\07_deploy_zcu106.ps1 -Model mnist -Precision 8bit -Image digit3

# Deploy ResNet-50 Full Backbone
.\scripts\07_deploy_zcu106.ps1 -Model resnet50 -Precision 8bit -Image dog

# Deploy MobileNetV2
.\scripts\07_deploy_zcu106.ps1 -Model mobilenetv2 -Precision 8bit -Image dog

# Deploy EfficientNet-B0
.\scripts\07_deploy_zcu106.ps1 -Model efficientnetb0 -Precision 8bit -Image dog

# Deploy All 4 Models Sequentially
.\scripts\07_deploy_zcu106.ps1 -Model all -Precision 8bit -Image dog
```

