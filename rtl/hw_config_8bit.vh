// =============================================================================
// File: unified/rtl/hw_config_8bit.vh
// Description: Unified Hardware Configuration Header for c_engine RTL (INT8).
// Target: AMD Zynq UltraScale+ ZCU106 Evaluation Kit
// Specifications:
//   - Array Dimensions: 8 Rows x 32 Columns (256 Processing Elements)
//   - Clock Frequency: 75.0 MHz (13.33 ns clock period)
//   - Precision: INT8 Inputs/Weights, INT16 Biases, INT32 Accumulators
//   - Memory: 16 Banks, 128 bits/bank (256 B/row), Depth 15360 rows (3.93 MB, 96 URAMs)
//   - Microcode Capacity: Up to 256 128-bit instruction words
// =============================================================================

`ifndef HW_CONFIG_VH
`define HW_CONFIG_VH

`define CFG_PRECISION_8BIT

// Systolic Array Dimensions (256 PEs)
`define CFG_R                   8
`define CFG_C                   32

// Target Clock Frequency (75.0 MHz for guaranteed timing closure)
`define CFG_CLK_FREQ_MHZ        75

// Precision Bit-Widths (INT8)
`define CFG_X_DATA_WIDTH        8
`define CFG_W_DATA_WIDTH        8
`define CFG_B_DATA_WIDTH        16
`define CFG_Y_DATA_WIDTH        32
`define CFG_CLIP_MAX            127
`define CFG_CLIP_MIN            -128

// Multi-Bank Dual-Port SRAM Parameters (3.93 MB = 15360 rows x 256 B/row, 96 UltraRAMs on ZCU106)
`define CFG_NUM_BANKS           16
`define CFG_WORD_SIZE           128
`define CFG_MEM_DEPTH           15360
`define CFG_ADDR_WIDTH          14
`define CFG_MAX_WEIGHTS         4096

// Microcode Configuration Memory Geometry
`define CFG_CFG_WIDTH           128
`define CFG_CFG_DEPTH           256
`define CFG_CFG_ADDR_WIDTH      8

// Autonomous Spatial Tiling Parameters (224x224 -> 196 chunks of 8x32x3)
`define CFG_NUM_CHUNKS          196

`endif // HW_CONFIG_VH

