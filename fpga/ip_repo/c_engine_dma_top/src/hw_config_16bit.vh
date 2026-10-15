// =============================================================================
// File: unified/rtl/hw_config_16bit.vh
// Description: Unified Hardware Configuration Header for c_engine RTL (INT16).
// Target: AMD Zynq UltraScale+ ZCU106 Evaluation Kit
// Specifications:
//   - Array Dimensions: 8 Rows x 32 Columns (256 Processing Elements)
//   - Clock Frequency: 70.0 MHz (14.29 ns clock period)
//   - Precision: INT16 Inputs/Weights, INT32 Biases, INT48 Accumulators
//   - Memory: 32 Banks, 128 bits/bank (512 B/row), Depth 4096 rows (2.10 MB, 64 URAMs)
//   - Microcode Capacity: Up to 256 128-bit instruction words
// =============================================================================

`ifndef HW_CONFIG_VH
`define HW_CONFIG_VH

`define CFG_PRECISION_16BIT

// Systolic Array Dimensions (256 PEs)
`define CFG_R                   8
`define CFG_C                   32

// Target Clock Frequency (70.0 MHz for guaranteed clean timing closure)
`define CFG_CLK_FREQ_MHZ        70

// Precision Bit-Widths (INT16)
`define CFG_X_DATA_WIDTH        16
`define CFG_W_DATA_WIDTH        16
`define CFG_B_DATA_WIDTH        32
`define CFG_Y_DATA_WIDTH        48
`define CFG_CLIP_MAX            32767
`define CFG_CLIP_MIN            -32768

// Multi-Bank Dual-Port SRAM Parameters (2.09 MB = 4096 rows x 512 B/row, 64 UltraRAMs on ZCU106)
`define CFG_NUM_BANKS           32
`define CFG_WORD_SIZE           128
`define CFG_MEM_DEPTH           4096
`define CFG_ADDR_WIDTH          12
`define CFG_MAX_WEIGHTS         4096

// Microcode Configuration Memory Geometry
`define CFG_CFG_WIDTH           128
`define CFG_CFG_DEPTH           256
`define CFG_CFG_ADDR_WIDTH      8

// Autonomous Spatial Tiling Parameters (224x224 -> 196 chunks of 8x32x3)
`define CFG_NUM_CHUNKS          196

`endif // HW_CONFIG_VH

