`timescale 1ns / 1ps
`include "hw_config.vh"
// =============================================================================
// Module: c_engine_top
// Description: Integrated top-level systolic accelerator system.
//              Contains:
//              1. c_engine: 2D systolic array (R x C PEs).
//              2. multibank_dualport_mem: Multi-bank dual-port SRAM.
//              3. c_engine_controller: Autonomous sequencing FSM.
//              Provides host memory access multiplexer for loading activations,
//              weights, and reading back results.
// =============================================================================

module c_engine_top #(
    parameter R                   = `CFG_R,
    parameter C                   = `CFG_C,
    parameter x_DATA_WIDTH        = `CFG_X_DATA_WIDTH,
    parameter w_DATA_WIDTH        = `CFG_W_DATA_WIDTH,
    parameter b_DATA_WIDTH        = `CFG_B_DATA_WIDTH,
    parameter y_DATA_WIDTH        = `CFG_Y_DATA_WIDTH,
    parameter NUM_BANKS           = `CFG_NUM_BANKS,
    parameter WORD_SIZE           = `CFG_WORD_SIZE,
    parameter MEM_DEPTH           = `CFG_MEM_DEPTH,
    parameter ADDR_WIDTH          = `CFG_ADDR_WIDTH,
    parameter MAX_WEIGHTS         = `CFG_MAX_WEIGHTS,
    parameter CFG_WIDTH           = `CFG_CFG_WIDTH,
    parameter CFG_DEPTH           = `CFG_CFG_DEPTH,
    parameter CFG_ADDR_WIDTH      = `CFG_CFG_ADDR_WIDTH,
    parameter NUM_CHUNKS          = `CFG_NUM_CHUNKS,
    parameter signed [x_DATA_WIDTH-1:0] CLIP_MAX = `CFG_CLIP_MAX,
    parameter signed [x_DATA_WIDTH-1:0] CLIP_MIN = `CFG_CLIP_MIN
)(
    input  wire                               clk,
    input  wire                               reset,

    // Accelerator Control Interface
    input  wire                               start,
    input  wire                               cfg_mode_select,     // 0: static ports, 1: config_mem instruction sequencing
    input  wire                               cfg_wr_en,           // Host write enable to config_mem
    input  wire [CFG_ADDR_WIDTH-1:0]          cfg_wr_addr,         // Host write address to config_mem
    input  wire [CFG_WIDTH-1:0]               cfg_wr_data,         // Host 80-bit config word
    input  wire                               mode,                // 0: 2D Conv, 1: GEMV
    input  wire [7:0]                         cfg_chin,            // Ch_in (Conv) / K (GEMV)
    input  wire [7:0]                         cfg_chout,           // Ch_out (Conv) / M (GEMV)
    input  wire [3:0]                         cfg_kw,              // Kernel width (default 3)
    input  wire [3:0]                         cfg_kh,              // Kernel height (default 3)
    input  wire [4:0]                         cfg_shift_bits,      // Re-quantization shift amount: 0 to 31
    input  wire [1:0]                         act_type,            // Static activation type (00: None, 01: ReLU, 10: Leaky ReLU, 11: LUT)
    input  wire [2:0]                         stride,              // Static stride (default 3'd1)
    input  wire                               pad_type,            // Static pad type (0: VALID, 1: SAME)
    input  wire [1:0]                         pool_type,           // Static pool type (00: None, 01: MaxPool, 10: AvgPool)
    input  wire [1:0]                         pool_size,           // Static pool size (00: 1x1, 01: 2x2, 10: 3x3)
    input  wire [2:0]                         pool_stride,         // Static pool stride (default 3'd2)
    input  wire                               cfg_residual_en,     // Static residual enable (1: fused addition)
    input  wire [ADDR_WIDTH-1:0]              cfg_residual_base_addr,// Static residual base address in RAM
    input  wire [ADDR_WIDTH-1:0]              cfg_input_base_addr, // e.g. 0
    input  wire [ADDR_WIDTH-1:0]              cfg_weight_base_addr,// e.g. 3
    input  wire [ADDR_WIDTH-1:0]              cfg_output_base_addr,// e.g. 5
    output wire                               busy,
    output wire                               done,

    // Autonomous DMA Trigger & Status Interface (pass-through from controller)
    input  wire [31:0]                        wt_ddr_addr,
    input  wire [15:0]                        reg_num_chunks,
    output wire                               ctrl_start_rd_dma,
    output wire                               ctrl_rd_is_weight,
    output wire [31:0]                        ctrl_rd_ddr_offset,
    output wire [ADDR_WIDTH-1:0]              ctrl_rd_sram_row,
    output wire [15:0]                        ctrl_rd_beats,
    input  wire                               rd_dma_busy,
    input  wire                               rd_dma_done,

    output wire                               ctrl_start_wr_dma,
    output wire [ADDR_WIDTH-1:0]              ctrl_wr_sram_row,
    output wire [15:0]                        ctrl_wr_beats,
    input  wire                               wr_dma_busy,
    input  wire                               wr_dma_done,

    output wire [7:0]                         active_chunk_idx,
    output wire [CFG_ADDR_WIDTH-1:0]          active_pc,
    output wire [4:0]                         active_fsm_state,

    // External Host Memory Access Interface (for loading/reading multi-bank RAM)
    input  wire                               ext_mem_override,    // 1: Host has Port A, 0: Controller has Port A
    input  wire                               ext_mem_en,
    input  wire [NUM_BANKS-1:0]               ext_mem_we,
    input  wire [ADDR_WIDTH-1:0]              ext_mem_addr,
    input  wire [NUM_BANKS*WORD_SIZE-1:0]     ext_mem_din,
    output wire [NUM_BANKS*WORD_SIZE-1:0]     ext_mem_dout,

    // Direct External GEMV Activation Feed (broadcast to all PEs during GEMV)
    input  wire signed [x_DATA_WIDTH-1:0]     ext_gemv_x_val,

    // Direct Full-Precision Outputs from Systolic Array
    output wire [y_DATA_WIDTH*R*C-1:0]        engdata_out
);

    // -------------------------------------------------------------------------
    // Interconnect Wires
    // -------------------------------------------------------------------------

    // Controller <-> Memory Port A
    wire [ADDR_WIDTH-1:0]              ctrl_mem_addr_a;
    wire                               ctrl_mem_en_a;
    wire [NUM_BANKS-1:0]               ctrl_mem_we_a;
    wire [NUM_BANKS*WORD_SIZE-1:0]     ctrl_mem_din_a;
    wire [NUM_BANKS*WORD_SIZE-1:0]     mem_dout_a;

    // Memory Port A Mux (Host vs. Controller)
    wire [ADDR_WIDTH-1:0]              mux_mem_addr_a = ext_mem_override ? ext_mem_addr : ctrl_mem_addr_a;
    wire                               mux_mem_en_a   = ext_mem_override ? ext_mem_en   : ctrl_mem_en_a;
    wire [NUM_BANKS-1:0]               mux_mem_we_a   = ext_mem_override ? ext_mem_we   : ctrl_mem_we_a;
    wire [NUM_BANKS*WORD_SIZE-1:0]     mux_mem_din_a  = ext_mem_override ? ext_mem_din  : ctrl_mem_din_a;
    assign ext_mem_dout = mem_dout_a;

    // Controller <-> Memory Port B
    wire [ADDR_WIDTH-1:0]              ctrl_mem_addr_b;
    wire                               ctrl_mem_en_b;
    wire [NUM_BANKS-1:0]               ctrl_mem_we_b;
    wire [NUM_BANKS*WORD_SIZE-1:0]     ctrl_mem_din_b;
    wire [NUM_BANKS*WORD_SIZE-1:0]     mem_dout_b;

    // Controller <-> c_engine Controls
    wire                               pe_enable;
    wire                               sel0;
    wire [1:0]                         sel1;
    wire                               reg_enable;
    wire                               hor_enable;
    wire                               vert_enable;
    wire                               eng_enable;
    wire                               bias_enable;
    wire signed [b_DATA_WIDTH-1:0]     ctrl_bias_out;
    wire [x_DATA_WIDTH*R*C-1:0]        ctrl_engdata_in;
    wire [w_DATA_WIDTH*R*C-1:0]        ctrl_weight_in;
    wire                               ctrl_mode;
    wire signed [x_DATA_WIDTH-1:0]     ctrl_gemv_x_val;

    // Broadcast ctrl_gemv_x_val to all PEs when in GEMV mode
    wire [x_DATA_WIDTH*R*C-1:0]        gemv_x_broadcast;
    genvar gi;
    generate
        for (gi = 0; gi < R*C; gi = gi + 1) begin : gen_gemv_bcast
            assign gemv_x_broadcast[(gi+1)*x_DATA_WIDTH-1 : gi*x_DATA_WIDTH] = ctrl_gemv_x_val;
        end
    endgenerate

    // Final engdata_in fed to c_engine
    wire [x_DATA_WIDTH*R*C-1:0] c_engine_din = (ctrl_mode == 1'b1) ? gemv_x_broadcast : ctrl_engdata_in;

    // -------------------------------------------------------------------------
    // Instance: Multi-Bank Dual-Port Memory
    // -------------------------------------------------------------------------
    multibank_dualport_mem #(
        .NUM_BANKS       (NUM_BANKS),
        .WORD_SIZE       (WORD_SIZE),
        .MEM_DEPTH       (MEM_DEPTH),
        .ADDR_WIDTH      (ADDR_WIDTH),
        .REGISTERED_READ (1) // Registered synchronous read for true dual-port Block RAM
    ) u_memory (
        // Port A
        .clk_a  (clk),
        .en_a   (mux_mem_en_a),
        .we_a   (mux_mem_we_a),
        .addr_a (mux_mem_addr_a),
        .din_a  (mux_mem_din_a),
        .dout_a (mem_dout_a),

        // Port B
        .clk_b  (clk),
        .en_b   (ctrl_mem_en_b),
        .we_b   (ctrl_mem_we_b),
        .addr_b (ctrl_mem_addr_b),
        .din_b  (ctrl_mem_din_b),
        .dout_b (mem_dout_b)
    );

    // -------------------------------------------------------------------------
    // Instance: 64-bit Configuration Memory for Layer Descriptors
    // -------------------------------------------------------------------------
    wire [CFG_ADDR_WIDTH-1:0] ctrl_cfg_rd_addr;
    wire [CFG_WIDTH-1:0]      ctrl_cfg_rd_data;

    config_mem #(
        .CFG_WIDTH      (CFG_WIDTH),
        .CFG_DEPTH      (CFG_DEPTH),
        .CFG_ADDR_WIDTH (CFG_ADDR_WIDTH)
    ) u_config_mem (
        .clk            (clk),
        .reset          (reset),
        .cfg_wr_en      (cfg_wr_en),
        .cfg_wr_addr    (cfg_wr_addr),
        .cfg_wr_data    (cfg_wr_data),
        .cfg_rd_addr    (ctrl_cfg_rd_addr),
        .cfg_rd_data    (ctrl_cfg_rd_data)
    );

    // -------------------------------------------------------------------------
    // Instance: Autonomous Controller FSM
    // -------------------------------------------------------------------------
    c_engine_controller #(
        .R                   (R),
        .C                   (C),
        .x_DATA_WIDTH        (x_DATA_WIDTH),
        .w_DATA_WIDTH        (w_DATA_WIDTH),
        .b_DATA_WIDTH        (b_DATA_WIDTH),
        .y_DATA_WIDTH        (y_DATA_WIDTH),
        .NUM_BANKS           (NUM_BANKS),
        .WORD_SIZE           (WORD_SIZE),
        .MEM_DEPTH           (MEM_DEPTH),
        .ADDR_WIDTH          (ADDR_WIDTH),
        .MAX_WEIGHTS         (MAX_WEIGHTS),
        .CFG_WIDTH           (CFG_WIDTH),
        .CFG_ADDR_WIDTH      (CFG_ADDR_WIDTH),
        .NUM_CHUNKS          (NUM_CHUNKS),
        .CLIP_MAX            (CLIP_MAX),
        .CLIP_MIN            (CLIP_MIN)
    ) u_controller (
        .clk                 (clk),
        .reset               (reset),
        .start               (start),
        .cfg_mode_select     (cfg_mode_select),
        .cfg_rd_addr         (ctrl_cfg_rd_addr),
        .cfg_rd_data         (ctrl_cfg_rd_data),
        .mode                (mode),
        .cfg_chin            (cfg_chin),
        .cfg_chout           (cfg_chout),
        .cfg_kw              (cfg_kw),
        .cfg_kh              (cfg_kh),
        .cfg_shift_bits      (cfg_shift_bits),
        .act_type            (act_type),
        .stride              (stride),
        .pad_type            (pad_type),
        .pool_type           (pool_type),
        .pool_size           (pool_size),
        .pool_stride         (pool_stride),
        .cfg_residual_en     (cfg_residual_en),
        .cfg_residual_base_addr(cfg_residual_base_addr),
        .cfg_input_base_addr (cfg_input_base_addr),
        .cfg_weight_base_addr(cfg_weight_base_addr),
        .cfg_output_base_addr(cfg_output_base_addr),
        .busy                (busy),
        .done                (done),

        .wt_ddr_addr         (wt_ddr_addr),
        .reg_num_chunks      (reg_num_chunks),
        .ctrl_start_rd_dma   (ctrl_start_rd_dma),
        .ctrl_rd_is_weight   (ctrl_rd_is_weight),
        .ctrl_rd_ddr_offset  (ctrl_rd_ddr_offset),
        .ctrl_rd_sram_row    (ctrl_rd_sram_row),
        .ctrl_rd_beats       (ctrl_rd_beats),
        .rd_dma_busy         (rd_dma_busy),
        .rd_dma_done         (rd_dma_done),

        .ctrl_start_wr_dma   (ctrl_start_wr_dma),
        .ctrl_wr_sram_row    (ctrl_wr_sram_row),
        .ctrl_wr_beats       (ctrl_wr_beats),
        .wr_dma_busy         (wr_dma_busy),
        .wr_dma_done         (wr_dma_done),

        .active_chunk_idx    (active_chunk_idx),
        .active_pc           (active_pc),
        .active_fsm_state    (active_fsm_state),

        // Memory Port A
        .mem_addr_a          (ctrl_mem_addr_a),
        .mem_en_a            (ctrl_mem_en_a),
        .mem_we_a            (ctrl_mem_we_a),
        .mem_din_a           (ctrl_mem_din_a),
        .mem_dout_a          (mem_dout_a),

        // Memory Port B
        .mem_addr_b          (ctrl_mem_addr_b),
        .mem_en_b            (ctrl_mem_en_b),
        .mem_we_b            (ctrl_mem_we_b),
        .mem_din_b           (ctrl_mem_din_b),
        .mem_dout_b          (mem_dout_b),

        // GEMV Interface
        .ext_gemv_x_val      (ext_gemv_x_val),
        .ctrl_mode           (ctrl_mode),
        .gemv_x_val          (ctrl_gemv_x_val),

        // c_engine Controls
        .pe_enable           (pe_enable),
        .sel0                (sel0),
        .sel1                (sel1),
        .reg_enable          (reg_enable),
        .hor_enable          (hor_enable),
        .vert_enable         (vert_enable),
        .eng_enable          (eng_enable),
        .bias_enable         (bias_enable),
        .bias_out            (ctrl_bias_out),
        .engdata_in          (ctrl_engdata_in),
        .weight_in           (ctrl_weight_in),
        .engdata_out         (engdata_out)
    );

    // -------------------------------------------------------------------------
    // Instance: 2D Systolic Compute Fabric (c_engine)
    // -------------------------------------------------------------------------
    c_engine #(
        .R            (R),
        .C            (C),
        .x_DATA_WIDTH (x_DATA_WIDTH),
        .w_DATA_WIDTH (w_DATA_WIDTH),
        .b_DATA_WIDTH (b_DATA_WIDTH),
        .y_DATA_WIDTH (y_DATA_WIDTH)
    ) u_c_engine (
        .clk          (clk),
        .reset        (reset),
        .pe_enable    (pe_enable),
        .sel0         (sel0),
        .sel1         (sel1),
        .reg_enable   (reg_enable),
        .hor_enable   (hor_enable),
        .vert_enable  (vert_enable),
        .eng_enable   (eng_enable),
        .bias_enable  (bias_enable),
        .bias_in      (ctrl_bias_out),
        .engdata_in   (c_engine_din),
        .weight_in    (ctrl_weight_in),
        .engdata_out  (engdata_out)
    );

endmodule
