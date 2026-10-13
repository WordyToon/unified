# RTL Directory (`rtl/`)

This directory contains all synthesizable Verilog HDL source modules for the **Unified 256-PE Systolic CNN Accelerator** and its multi-bank dual-port memory subsystem on AMD ZCU104.

---

## File Manifest

| File | Module Name | Description |
| :--- | :--- | :--- |
| [`hw_config.vh`](hw_config.vh) | Header | Global compile-time configuration parameters ($R=8, C=32, F_{\text{clk}}=100\text{ MHz}, \text{Depth}=2048, \text{Banks}=16, \text{CfgDepth}=64$). |
| [`processing_element.v`](processing_element.v) | `processing_element` | Parameterized Processing Element (PE) with signed multiplier, true 16-bit signed bias input (`bias_in`), and accumulator registers (`Outf_map`, `HOR`, `VERT`, `ENG`). |
| [`c_engine.v`](c_engine.v) | `c_engine` | 2D Systolic Array grid ($8 \times 32 = 256\text{ PEs}$) with horizontal forwarding ($c-1 \to c$), vertical forwarding ($r-1 \to r$), and local engine loops. Pure MAC compute fabric. |
| [`multibank_dualport_mem.v`](multibank_dualport_mem.v) | `multibank_dualport_mem` | Multi-bank true dual-port RAM with 16 banks of 128-bit words ($16 \times 128 = 2048\text{ bits} = 256\text{ B/row}$) across 2,048 rows (512 KB total). Offers single-cycle full feature map access on Port B. |
| [`config_mem.v`](config_mem.v) | `config_mem` | 64-word $\times$ 96-bit distributed RAM (`(* ram_style = "distributed" *)`) storing layer descriptors (Config Words) with autonomous multi-layer sequencing. Eliminates routing congestion. |
| [`requantization_block.v`](requantization_block.v) | `requantization_block` | Post-Processing Unit (PPU): Dynamic shift, saturation clamping, activations (Linear, ReLU, Leaky-ReLU), boundary padding masking, strided multiplexer (Strides 1 & 2), $2 \times 2$ MaxPool, fused in-situ residual addition, and **clock-pipelined output register (`data_out_reg`)** for timing closure. |
| [`c_engine_controller.v`](c_engine_controller.v) | `c_engine_controller` | Autonomous FSM with instruction sequencer, autonomous DDR weight streamer via AXI4 Read DMA, 16-bit bias decoder, activation dispatcher, residual address decoder, and pipelined BRAM writeback. |
| [`c_engine_top.v`](c_engine_top.v) | `c_engine_top` | Top-level accelerator module integrating memory, config memory, controller, systolic array, PPU, and host override multiplexers. Defaults bound to `hw_config.vh`. |
| [`axi_lite_slave.v`](axi_lite_slave.v) | `axi_lite_slave` | 32-bit AXI4-Lite slave configuration register block: control/status, DMA start, base addresses, transfer beat lengths, W1C interrupt status clearing, and 64-bit hardware cycle counter. |
| [`axi4_master_rd.v`](axi4_master_rd.v) | `axi4_master_rd` | 128-bit AXI4 burst read master: autonomously streams image chunks and layer weights from external DDR memory via INCR bursts (up to 256 beats) with 4KB boundary splitting. |
| [`axi4_master_wr.v`](axi4_master_wr.v) | `axi4_master_wr` | 128-bit AXI4 burst write master: offloads classification logits from on-chip SRAM to DDR memory via INCR bursts with response checking. |
| [`dma_sram_bridge.v`](dma_sram_bridge.v) | `dma_sram_bridge` | Zero-latency 1-to-1 steering bridge mapping sequential 128-bit AXI bursts directly to individual multi-bank SRAM slices (`gen_banks[bank].mem[row]`). |
| [`c_engine_dma_top.v`](c_engine_dma_top.v) | `c_engine_dma_top` | Complete top-level accelerator IP core integrating the systolic engine (`c_engine_top`), AXI4 read master (`m0_axi`), write master (`m1_axi`), and AXI-Lite slave (`s_axi_lite`). |

---

## Global RTL Parameters (`hw_config.vh`)

| Parameter | Current Value | Description |
| :--- | :---: | :--- |
| `CFG_R` | `8` | Number of PE grid rows |
| `CFG_C` | `32` | Number of PE grid columns ($8 \times 32 = 256\text{ PEs}$) |
| `CFG_CLK_FREQ_MHZ` | `100` | Target clock frequency in MHz ($T_{\text{clk}} = 10.0\text{ ns}$) |
| `CFG_X_DATA_WIDTH` | `8` | Bit-width of input activations (signed INT8) |
| `CFG_W_DATA_WIDTH` | `8` | Bit-width of filter/matrix weights (signed INT8) |
| `CFG_B_DATA_WIDTH` | `16` | Bit-width of bias coefficients (signed INT16) |
| `CFG_Y_DATA_WIDTH` | `32` | Bit-width of internal accumulator / outputs (signed INT32) |
| `CFG_NUM_BANKS` | `16` | Number of parallel memory banks |
| `CFG_WORD_SIZE` | `128` | Bit-width per bank word ($16 \text{ bytes}$) |
| `CFG_MEM_DEPTH` | `2048` | Memory depth per bank ($2,048 \text{ rows} \times 256\text{ B/row} = 512\text{ KB}$) |
| `CFG_ADDR_WIDTH` | `11` | Address width ($\log_2(2048) = 11\text{ bits}$) |
| `CFG_MAX_WEIGHTS` | `2048` | Weight staging capacity for multi-channel conv |
| `CFG_CFG_WIDTH` | `96` | Config memory instruction word width |
| `CFG_CFG_DEPTH` | `64` | Maximum supported layer instructions |
| `CFG_CFG_ADDR_WIDTH`| `6` | Config address width ($\log_2(64) = 6\text{ bits}$) |
| `CFG_NUM_CHUNKS` | `196` | Total spatial chunks for $224 \times 224$ ImageNet input |

---

## 96-Bit Config Word Layout

The sequencer instruction is 96 bits wide, programmed via three consecutive 32-bit registers over AXI-Lite (`ADDR_CFG_DATA0`, `ADDR_CFG_DATA1`, `ADDR_CFG_DATA2`). Writing `DATA2` generates an automatic single-cycle write strobe:

```
+-----------+---------------+---------------------------------------------------------------------------------+
| Word      | Bit Range     | Field Definition & Description                                                  |
+-----------+---------------+---------------------------------------------------------------------------------+
| DATA0     | [1:0]         | Mode: 2'b00 = CONV2D, 2'b01 = GEMV / FC, 2'b10 = POOLING                        |
|           | [9:2]         | Input Channels (Ch_in)                                                          |
|           | [17:10]       | Output Channels (Ch_out / M Length)                                             |
|           | [21:18]       | Kernel Width (KW)                                                               |
|           | [25:22]       | Kernel Height (KH)                                                              |
|           | [30:26]       | Requantization Shift Bits (S: 0..31)                                             |
|           | [31]          | Padding Type: 1'b0 = VALID, 1'b1 = SAME                                         |
+-----------+---------------+---------------------------------------------------------------------------------+
| DATA1     | [42:32]       | Input Base Row Address in SRAM (11 bits: 0..2047)                               |
|           | [53:43]       | Weight Base Row Address in SRAM (11 bits: 0..2047)                              |
|           | [63:54]       | Output Base Row Address in SRAM (lower 10 bits of 11-bit row)                   |
+-----------+---------------+---------------------------------------------------------------------------------+
| DATA2     | [64]          | Output Base Row Address MSB (bit 10 of 11-bit row)                              |
|           | [66:65]       | Activation Type: 2'b00 = Linear, 2'b01 = ReLU, 2'b10 = Leaky-ReLU (0.125)       |
|           | [69:67]       | Convolution Stride: 3'b001 = Stride 1, 3'b010 = Stride 2                        |
|           | [71:70]       | Pooling Type: 2'b00 = Bypass, 2'b01 = 2x2 MaxPool                               |
|           | [72]          | Residual Enable: 1'b1 = Element-wise residual addition from Port B              |
|           | [83:73]       | Residual Base Row Address in SRAM (11 bits: 0..2047)                            |
|           | [84]          | Last Stage Flag (triggers Engine Done on final PC)                              |
|           | [95:85]       | Reserved / Zero-padded                                                          |
+-----------+---------------+---------------------------------------------------------------------------------+
```

---

## Post-Processing Unit (PPU) & Critical Path Optimization

The PPU (`requantization_block.v`) performs the full output consolidation pipeline:
1. **Dynamic Shift**: Arithmetic right-shift by `cfg_shift_bits`: `pe_val_shifted = pe_val_in >>> cfg_shift_bits`
2. **Saturation Clamp**: Clamped to signed 8-bit range `[-128, +127]`
3. **Fused Residual Addition**: When `cfg_residual_en == 1'b1`, adds the element-wise skip tensor from SRAM Port B with saturation clipping.
4. **Activation Function**:
   - `2'b00`: Linear passthrough
   - `2'b01`: ReLU (`max(0, x)`)
   - `2'b10`: Leaky ReLU ($x \ge 0 \ ?\ x : x >>> 3$, with $\alpha = 0.125$)
5. **Padding Masking**: Mask boundary PEs where $r < KH-1$ or $c < KW-1$ for VALID convolution.
6. **Strided Multiplexing & 2x2 Max Pooling**:
   - Strides 1 and 2 subsampling.
   - 2x2 MaxPool comparator trees with stride-2 compaction.
7. **Pipelined Output Register (`data_out_reg`)**:
   - Rather than driving BRAM write data directly through a 21.157 ns combinational path, `data_out_reg` captures the requantized result on clock edge.
   - The controller performs writeback across 2 cycles: Cycle 1 registers the output, Cycle 2 writes to BRAM with **zero combinational routing delay**.
   - Slashed the critical path to **< 4.0 ns**, ensuring timing closure at 100.0 MHz (**WNS = +0.163 ns**).
