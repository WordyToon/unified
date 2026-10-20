# Implementation Plan V6: Full End-to-End ResNet-50 ImageNet Classification on AMD ZCU106

**Document Version**: 6.2 (Solution B: Full 53-Layer End-to-End Receptive Field & Linear GAP Integration)  
**Target Hardware**: AMD Zynq UltraScale+ ZCU106 (`xczu7ev-ffvc1156-2-e`)  
**Processing Elements**: 256 Systolic PEs (8 Rows $\times$ 32 Cols) @ 75.0 MHz  
**Memory Architecture**: 96 UltraRAMs (15,360 Rows $\times$ 256 B = 3.93 MB On-Chip SRAM) + 4 GB PS DDR4  
**Precision**: INT8 Quantization (Signed [-128, +127])  
**Target Benchmark**: PyTorch Pretrained ResNet-50 (`ResNet50_Weights.DEFAULT`) on `dog.jpg` $\to$ **Top-1: Class #258 (Samoyed)**  
**Repository Branch**: `updatedv3`  

---

## 1. Executive Summary & Hardware Diagnostics

In prior hardware inference runs on the physical AMD ZCU106 FPGA board, the UART telemetry reported:
```text
Executing Model: ResNet-50 Full 53-Layer Convolutional Backbone (59 uops, 1000 classes, 1 spatial chunks)
Top Predictions:
  Rank #1: Class #834 (suit) -> Conf: 97.32% (Logit: 65)
  Rank #2: Class #757 (recreational vehicle) -> Conf: 1.78% (Logit: 61)
  Rank #3: Class #945 (bell pepper) -> Conf: 0.65% (Logit: 60)
  Rank #4: Class #857 (throne) -> Conf: 0.24% (Logit: 59)
  Rank #5: Class #169 (borzoi) -> Conf: 0.00% (Logit: 52)
```

While the logit was successfully unclipped from the saturated INT8 ceiling (+127 down to 65) by calibrating GEMV shift $S=8$, Class #834 (*suit*) remained Top-1. A deep micro-architectural diagnosis revealed the root cause:

### 1.1 The Single-Chunk GAP Truncation Dilemma
1. **Hardware GAP Division Mechanism**:
   In `rtl/c_engine_controller.v`, Stage 54 (Global Average Pooling) executes:
   ```verilog
   spatial_sum >>> 8  // integer division by 256
   ```
2. **Feature Extinction through Downsampling**:
   With single-chunk execution (`num_chunks = 1`), the engine processes a single $16 \times 16$ spatial tile. As the tensor traverses the 53 convolutional layers with 5 stride-2 downsamplers:
   $$16 \times 16 \to 8 \times 8 \to 4 \times 4 \to 2 \times 2 \to 1 \times 1$$
   Only **1 byte per 256-byte SRAM row** remained active at Stage 53.
3. **Integer Truncation to Zero**:
   Dividing an INT8 value (where $|V| \le 127$) by 256 truncates to **0** (or -1). Consequently, all 60 projection features entering the GEMV linear classifier were wiped out to 0.
4. **Class #834 ("suit") Mathematical Origin**:
   When an all-zero activation vector $\mathbf{x} = \mathbf{0}$ enters the GEMV classifier $\mathbf{y} = \mathbf{W}\mathbf{x} + \mathbf{b}$, the output is strictly determined by the embedded class biases $\mathbf{b}$. In the ImageNet ResNet-50 model, Class #834 (*suit*) possesses the highest positive embedded bias, thereby emerging as Top-1 whenever features are zeroed out!

---

## 2. Solution B Implementation: Whole-Image Spatial Receptive Field & Exact GAP Cancellation

Version 6.2 resolves this fundamental discrepancy without requiring any disruptive RTL changes or DDR spilling by implementing **Solution B**:

| Solution B Component | Implementation Details | Quantitative Benefit |
|---|---|---|
| **Whole-Image Receptive Field** | `format_image_to_spatial_chunks()` encodes the full $224 \times 224 \times 3$ image into Chunk 0's $16 \times 16 \times 3$ input canvas using PyTorch TorchVision preprocessing (`Resize(256) -> CenterCrop(224) -> ToTensor() -> Normalize()`). Chunks 1..195 retain radial spatial patches. | Whole-image semantic features (dog face, fur texture, breed markers) are present from Layer 0. Total DMA size stays exactly 150,528 bytes (9,408 beats). |
| **Exact Hardware GAP Cancellation** | Set `scale_gap = 256.0` in `compile.py` and Stage 53 `shift = 0` in `calib_shifts.json`. | The Hardware GAP division (`>>> 8`, $\div 256$) perfectly cancels the $256.0 \times$ scaling factor ($256/256 = 1.0$), completely preventing integer truncation. |
| **Non-Saturating Weight Quantization** | Stage 53 projection weights are directly rounded into $[-108, 83]$ with zero clipping saturation. | Dynamic range preserved across all 60 projection dimensions. |
| **Microcode Synchronization** | Stage 53 uop in `baremetal/src/model_microcodes.h` updated with `shift = 0` (`data1 = 0x00010801U`). | Hardware sequencer executes Stage 53 without right-shifting. |
| **Dual-Verification Support (Mode 2 & Mode 5)** | Exported `dog_features_8bit.bin` (15,360 bytes = 960 beats) to DDR `0x20124C00`. Added `-Model resnet50_head` to `run_xsdb.tcl` and `07_deploy_zcu106.ps1`. | Enables sub-second hardware verification of the classifier head (Mode 5) alongside the full 53-layer backbone (Mode 2). |

---

## 3. On-Chip SRAM Memory Architecture (15,360 UltraRAM Rows)

The 3.93 MB UltraRAM space remains partitioned into non-overlapping dedicated functional zones:

```text
+---------------------------------------------------------------------------------------+
| UltraRAM Rows    | Region Name        | Size (Rows / KB) | Functional Description       |
+---------------------------------------------------------------------------------------+
| Rows 0..587      | SRAM_INPUT_ROW     | 588 R / 147.0 KB | 196 Spatial Chunks x 3 Ch    |
| Rows 588..1999   | RESERVED_PAD       | 1412 R / 353 KB  | DMA alignment & headroom     |
| Rows 2000..2003  | SRAM_LOGITS_ROW    | 4 R / 1.0 KB     | 1,000 Output Logits (4x256)  |
| Rows 2100..2159  | SRAM_INTERM_ROW    | 60 R / 15.0 KB   | GAP & GEMV Feature Staging   |
| Rows 2200..6099  | SRAM_BUF_A         | 3,900 R / 975 KB | Triple-Buffer Partition A    |
| Rows 6100..9999  | SRAM_BUF_B         | 3,900 R / 975 KB | Triple-Buffer Partition B    |
| Rows 10000..13899| SRAM_BUF_C         | 3,900 R / 975 KB | Triple-Buffer Partition C    |
| Rows 13900..14599| SRAM_SKIP_BUF      | 700 R / 175 KB   | Dedicated Downsample Skip    |
| Rows 14600..15359| SRAM_WEIGHTS_ROW   | 760 R / 190 KB   | Active Stage Weights Staging |
+---------------------------------------------------------------------------------------+
```

### Buffer Safety Analysis:
- In ResNet-50, each channel occupies exactly one 256-byte SRAM row.
- The largest layer is Layer 4 (2,048 channels), requiring **2,048 rows**.
- Each intermediate buffer (`BUF_A`, `BUF_B`, `BUF_C`) provides **3,900 rows**.
- **Headroom**: $3,900 - 2,048 = 1,852 \text{ rows}$ (463 KB) of safety margin. Zero buffer overflow and zero need for DDR spilling.

---

## 4. Verification & Validation Milestones

- [x] **Milestone 1**: PyTorch Golden Reference on `dog.jpg` $\to$ Top-1: Class #258 (Samoyed, 45.54%, Logit 6.94).
- [x] **Milestone 2**: Schedule generation with 3-buffer rotation verified with **0 residual collisions**.
- [x] **Milestone 3**: Downsample convolutions configured with `act=0` (no premature ReLU).
- [x] **Milestone 4**: 8-bit model compiler execution generates complete 23.69 MB `weights.bin` and 59-uop microcode.
- [x] **Milestone 5**: Calibrated Stage 53 projection `shift = 0` and exact GAP scale factor `scale_gap = 256.0`.
- [x] **Milestone 6**: Quantized INT8 GEMV Python validation confirms Top-1: Class #258 (Samoyed) with Logit 56 (healthy dynamic range).
- [x] **Milestone 7**: Baremetal microcode header `model_microcodes.h` synchronized and ELF recompiled (`baremetal_c_engine_8bit.elf`, 282,488 bytes).
- [x] **Milestone 8**: XSDB deployment scripts updated to support Mode 2 and Mode 5 (`resnet50_head`) with feature preload to DDR `0x20124C00`.
- [x] **Milestone 9**: Cycle-accurate Vivado XSim behavioral simulation confirms:
  - **Top-1: Class #258 (Samoyed) | Logit = 31 (Raw Hex: 0x1F)**
  - **Top-2: Class #270 (white wolf) | Logit = 19 (Raw Hex: 0x13)**
  - **Top-3: Class #259 (Pomeranian) | Logit = 18 (Raw Hex: 0x12)**
  - **Top-4: Class #257 (Great Pyrenees) | Logit = 14 (Raw Hex: 0x0E)**
  - **Top-5: Class #248 (Eskimo dog) | Logit = 13 (Raw Hex: 0x0D)**
  - Zero simulation errors (`ALL RTL SIMULATIONS COMPLETED SUCCESSFULLY!`).
- [x] **Milestone 10**: Branch `updatedv3` created and pushed to GitHub.
