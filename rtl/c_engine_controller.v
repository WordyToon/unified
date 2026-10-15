`timescale 1ns / 1ps
`include "hw_config.vh"
// =============================================================================
// Module: c_engine_controller
// Description: Autonomous multi-mode controller for c_engine systolic accelerator.
//              Features:
//              1. Dual Operation Modes:
//                 - Mode 0: 2D Spatial Convolution (arbitrary KW x KH, Ch_in, Ch_out).
//                 - Mode 1: GEMV / Dense Layer Matrix-Vector Multiplication.
//              2. Instruction Sequencer with Config Memory:
//                 - Supports 96-bit Config Word decoding and autonomous multi-layer
//                   pipelining across layers (Conv1 -> Conv2 -> Dense) until
//                   is_last flag triggers global done.
//                 - Single-pass spatial tiling across all 196 chunks (Stage 0).
//                 - Autonomous layer weight DMA streaming from physical DDR memory.
//              3. Dynamic Multi-Chunk Weight Loader (up to MAX_WEIGHTS = 2048).
//              4. Pipelined weight indexing (0 combinational multipliers in inner loop).
//              5. Saturated Re-Quantization with runtime dynamic shift.
// =============================================================================

module c_engine_controller #(
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
    parameter CFG_ADDR_WIDTH      = `CFG_CFG_ADDR_WIDTH,
    parameter NUM_CHUNKS          = `CFG_NUM_CHUNKS,
    parameter signed [x_DATA_WIDTH-1:0] CLIP_MAX = `CFG_CLIP_MAX,
    parameter signed [x_DATA_WIDTH-1:0] CLIP_MIN = `CFG_CLIP_MIN
)(
    input  wire                               clk,
    input  wire                               reset,

    // Command & Configuration Interface
    input  wire                               start,
    input  wire                               cfg_mode_select,     // 0: External static ports, 1: config_mem instruction sequencing
    output reg  [CFG_ADDR_WIDTH-1:0]          cfg_rd_addr,         // Program Counter to config_mem
    input  wire [CFG_WIDTH-1:0]               cfg_rd_data,         // 96-bit Config Word from config_mem

    // Static Control Registers (used when cfg_mode_select == 0)
    input  wire                               mode,                // 0: 2D Conv, 1: GEMV
    input  wire [7:0]                         cfg_chin,            // Ch_in for Conv, K length for GEMV
    input  wire [7:0]                         cfg_chout,           // Ch_out for Conv, output size for GEMV
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
    output reg                                busy,
    output reg                                done,

    // Autonomous DMA Trigger & Status Interface
    input  wire [31:0]                        wt_ddr_addr,
    input  wire [15:0]                        reg_num_chunks,
    output reg                                ctrl_start_rd_dma,
    output reg                                ctrl_rd_is_weight,
    output reg [31:0]                         ctrl_rd_ddr_offset,
    output reg [ADDR_WIDTH-1:0]               ctrl_rd_sram_row,
    output reg [15:0]                         ctrl_rd_beats,
    input  wire                               rd_dma_busy,
    input  wire                               rd_dma_done,

    output reg                                ctrl_start_wr_dma,
    output reg [ADDR_WIDTH-1:0]               ctrl_wr_sram_row,
    output reg [15:0]                         ctrl_wr_beats,
    input  wire                               wr_dma_busy,
    input  wire                               wr_dma_done,

    // Debug & Status Monitoring Outputs
    output wire [7:0]                         active_chunk_idx,
    output wire [CFG_ADDR_WIDTH-1:0]          active_pc,
    output wire [4:0]                         active_fsm_state,

    // Memory Port A Interface (Weights Loading & Output Feature Map Writeback)
    output reg  [ADDR_WIDTH-1:0]              mem_addr_a,
    output reg                                mem_en_a,
    output reg  [NUM_BANKS-1:0]               mem_we_a,
    output wire [NUM_BANKS*WORD_SIZE-1:0]     mem_din_a,
    input  wire [NUM_BANKS*WORD_SIZE-1:0]     mem_dout_a,

    // Memory Port B Interface (Single-Cycle Feature Map Access)
    output reg  [ADDR_WIDTH-1:0]              mem_addr_b,
    output reg                                mem_en_b,
    output wire [NUM_BANKS-1:0]               mem_we_b,
    output wire [NUM_BANKS*WORD_SIZE-1:0]     mem_din_b,
    input  wire [NUM_BANKS*WORD_SIZE-1:0]     mem_dout_b,

    // External GEMV override & outputs
    input  wire signed [x_DATA_WIDTH-1:0]     ext_gemv_x_val,
    output wire                               ctrl_mode,
    output wire signed [x_DATA_WIDTH-1:0]     gemv_x_val,

    // c_engine Control & Data Interface
    output reg                                pe_enable,
    output reg                                sel0,
    output reg  [1:0]                         sel1,
    output reg                                reg_enable,
    output reg                                hor_enable,
    output reg                                vert_enable,
    output reg                                eng_enable,
    output reg                                bias_enable,
    output wire signed [b_DATA_WIDTH-1:0]     bias_out,
    output wire [x_DATA_WIDTH*R*C-1:0]        engdata_in,
    output wire [w_DATA_WIDTH*R*C-1:0]        weight_in,
    input  wire [y_DATA_WIDTH*R*C-1:0]        engdata_out
);

    localparam NUM_PES = R * C;

    // Port B write is disabled (Port B is read-only)
    assign mem_we_b  = {NUM_BANKS{1'b0}};
    assign mem_din_b = {(NUM_BANKS*WORD_SIZE){1'b0}};
    assign engdata_in = mem_dout_b;

    // -------------------------------------------------------------------------
    // Program Counter & Decoded Instruction Registers
    // -------------------------------------------------------------------------
    reg [CFG_ADDR_WIDTH-1:0] pc;
    reg [1:0]                dec_mode;
    reg [11:0]               dec_chin;
    reg [11:0]               dec_chout;
    reg [3:0]                dec_kw;
    reg [3:0]                dec_kh;
    reg [4:0]                dec_shift_bits;
    reg [1:0]                dec_act_type;
    reg [2:0]                dec_stride;
    reg                      dec_pad_type;
    reg [1:0]                dec_pool_type;
    reg [1:0]                dec_pool_size;
    reg [2:0]                dec_pool_stride;
    reg                      dec_residual_en;
    reg [ADDR_WIDTH-1:0]     dec_residual_base_addr;
    reg [ADDR_WIDTH-1:0]     dec_input_base_addr;
    reg [ADDR_WIDTH-1:0]     dec_weight_base_addr;
    reg [ADDR_WIDTH-1:0]     dec_output_base_addr;
    reg                      dec_is_last;
    reg                      dec_dma_rd_en;
    reg                      dec_dma_wr_en;
    reg [6:0]                dec_dma_beats;
    reg [7:0]                dec_num_chunks;

    // Active Layer Configuration Multiplexers
    wire [1:0]            active_mode             = (cfg_mode_select == 1'b1) ? dec_mode             : {1'b0, mode};
    wire [11:0]           active_chin             = (cfg_mode_select == 1'b1) ? dec_chin             : {4'd0, cfg_chin};
    wire [11:0]           active_chout            = (cfg_mode_select == 1'b1) ? dec_chout            : {4'd0, cfg_chout};
    wire [3:0]            active_kw               = (cfg_mode_select == 1'b1) ? dec_kw               : cfg_kw;
    wire [3:0]            active_kh               = (cfg_mode_select == 1'b1) ? dec_kh               : cfg_kh;
    wire [4:0]            active_shift_bits       = (cfg_mode_select == 1'b1) ? dec_shift_bits       : cfg_shift_bits;
    wire [1:0]            active_act_type         = (cfg_mode_select == 1'b1) ? dec_act_type         : act_type;
    wire [2:0]            active_stride           = (cfg_mode_select == 1'b1) ? dec_stride           : (stride ? stride : 3'd1);
    wire                  active_pad_type         = (cfg_mode_select == 1'b1) ? dec_pad_type         : pad_type;
    wire [1:0]            active_pool_type        = (cfg_mode_select == 1'b1) ? dec_pool_type        : pool_type;
    wire [1:0]            active_pool_size        = (cfg_mode_select == 1'b1) ? dec_pool_size        : pool_size;
    wire [2:0]            active_pool_stride      = (cfg_mode_select == 1'b1) ? dec_pool_stride      : (pool_stride ? pool_stride : 3'd2);
    wire                  active_residual_en      = (cfg_mode_select == 1'b1) ? dec_residual_en      : cfg_residual_en;
    wire [ADDR_WIDTH-1:0] active_residual_base_addr=(cfg_mode_select == 1'b1) ? dec_residual_base_addr: cfg_residual_base_addr;
    wire [ADDR_WIDTH-1:0] active_input_base_addr  = (cfg_mode_select == 1'b1) ? dec_input_base_addr  : cfg_input_base_addr;
    wire [ADDR_WIDTH-1:0] active_weight_base_addr = (cfg_mode_select == 1'b1) ? dec_weight_base_addr : cfg_weight_base_addr;
    wire [ADDR_WIDTH-1:0] active_output_base_addr = (cfg_mode_select == 1'b1) ? dec_output_base_addr : cfg_output_base_addr;
    wire                  active_is_last          = (cfg_mode_select == 1'b1) ? dec_is_last          : 1'b1;
    wire [7:0]            active_num_chunks       = (cfg_mode_select == 1'b1 && dec_num_chunks != 8'd0) ? dec_num_chunks :
                                                    (reg_num_chunks[7:0] != 8'd0) ? reg_num_chunks[7:0] : 8'd1;

    // -------------------------------------------------------------------------
    // Internal Weight Staging Table
    // Dynamically loaded from memory at active_weight_base_addr
    // -------------------------------------------------------------------------
    reg signed [w_DATA_WIDTH-1:0] weight_table [0:MAX_WEIGHTS-1];

    // FSM State Encoding
    localparam S_IDLE            = 5'd0,
               S_FETCH_CFG       = 5'd1,
               S_LOAD_WEIGHTS    = 5'd2,
               S_LATCH_WEIGHTS   = 5'd3,
               S_CONV_START      = 5'd4,
               S_CONV_STEP_H     = 5'd5,
               S_CONV_ROW_DONE   = 5'd6,
               S_CONV_STEP_V     = 5'd7,
               S_CONV_FINAL_V    = 5'd8,
               S_CONV_WAIT_1X1   = 5'd9,
               S_CONV_ACCUM_ENG  = 5'd10,
               S_CONV_BIAS       = 5'd11,
               S_CONV_WB         = 5'd12,
               S_CONV_CLEAR_ENG  = 5'd13,
               S_GEMV_COMPUTE    = 5'd14,
               S_GEMV_WB         = 5'd15,
               S_POOL_READ       = 5'd16,
               S_POOL_WB         = 5'd17,
               S_DONE            = 5'd18,
               S_GEMV_WAIT       = 5'd19,
               S_CONV_RES_WAIT   = 5'd20,
               S_POOL_WAIT       = 5'd21,
               S_CONV_INIT_TAP   = 5'd22,
               S_DMA_RD_START    = 5'd23,
               S_DMA_RD_WAIT     = 5'd24,
               S_DMA_WR_START    = 5'd25,
               S_DMA_WR_WAIT     = 5'd26,
               S_GEMV_PIPE1      = 5'd27,
               S_GEMV_PIPE2      = 5'd28,
               S_DECODE_CFG      = 5'd29,
               S_CONV_PIPE       = 5'd30;

    reg [4:0] state;

    // Iteration counters
    reg [7:0]  oc_cnt;
    reg [7:0]  ch_cnt;
    reg [3:0]  v_cnt;
    reg [3:0]  h_cnt;
    reg [7:0]  k_cnt;
    reg [15:0] wt_chunk_cnt;

    // Autonomous spatial tiling counters and offsets
    reg [7:0]            chunk_idx;
    reg [ADDR_WIDTH-1:0] chunk_in_offset;
    reg [ADDR_WIDTH-1:0] chunk_out_offset;
    reg                  stage_wt_loaded;

    assign active_chunk_idx = chunk_idx;
    assign active_pc        = pc;
    assign active_fsm_state = state;

    assign ctrl_mode  = active_mode[0];
    assign gemv_x_val = (cfg_mode_select == 1'b1) ? mem_dout_b[k_cnt*x_DATA_WIDTH +: x_DATA_WIDTH] : ext_gemv_x_val;

    // Multi-pass Hardware Tiling Engine registers
    reg [7:0]  tile_ic_cnt;
    reg [7:0]  tile_oc_cnt;
    reg [31:0] current_wt_ddr_offset;
    reg [31:0] stage0_total_wt_bytes;

    // Hardware Global Average Pooling (GAP) Engine registers
    reg [7:0]                     gap_ch_idx;
    reg [ADDR_WIDTH-1:0]          last_conv_out_addr;
    reg                           gap_executed;
    reg [NUM_BANKS*WORD_SIZE-1:0] gap_row_buf;

    reg signed [17:0] sum_grp_reg [0:7];
    integer g_idx, p_sub_idx;
    reg signed [17:0] gap_tmp_sum;

    always @(posedge clk) begin
        if (reset) begin
            for (g_idx = 0; g_idx < 8; g_idx = g_idx + 1)
                sum_grp_reg[g_idx] <= 18'sd0;
        end else if (state == S_POOL_WAIT) begin
            for (g_idx = 0; g_idx < 8; g_idx = g_idx + 1) begin
                gap_tmp_sum = 18'sd0;
                for (p_sub_idx = 0; p_sub_idx < 32; p_sub_idx = p_sub_idx + 1) begin
                    if (w_DATA_WIDTH == 16) begin
                        if ((g_idx*32 + p_sub_idx) < 128)
                            gap_tmp_sum = gap_tmp_sum + $signed(mem_dout_b[(g_idx*32 + p_sub_idx)*16 +: 16]);
                    end else begin
                        gap_tmp_sum = gap_tmp_sum + $signed(mem_dout_b[(g_idx*32 + p_sub_idx)*8 +: 8]);
                    end
                end
                sum_grp_reg[g_idx] <= gap_tmp_sum;
            end
        end
    end

    wire signed [20:0] spatial_sum = sum_grp_reg[0] + sum_grp_reg[1] + sum_grp_reg[2] + sum_grp_reg[3] +
                                     sum_grp_reg[4] + sum_grp_reg[5] + sum_grp_reg[6] + sum_grp_reg[7];

    wire signed [x_DATA_WIDTH-1:0] spatial_mean = (w_DATA_WIDTH == 16) ?
        (spatial_sum >>> 7) : (spatial_sum >>> 8);

    // Tile dimension parameters
    wire [15:0] ext_chin  = {4'd0, active_chin};
    wire [15:0] ext_chout = {4'd0, active_chout};

    wire [7:0] tile_tin  = (active_mode == 2'b11) ? 8'd1 :
                           (active_kw >= 4'd7) ? 8'd3 : (active_kw >= 4'd3) ? 8'd8  : 8'd32;
    wire [7:0] tile_tout = (active_mode == 2'b11) ? 8'd32 :
                           (active_kw >= 4'd7) ? 8'd8 : (active_kw >= 4'd3) ? 8'd16 : 8'd32;

    wire [7:0] num_ic_tiles = (active_mode == 2'b11) ? 8'd1 :
                              (ext_chin <= {8'd0, tile_tin}) ? 8'd1 :
                              (active_kw >= 4'd7) ? ((ext_chin + 16'd2) / 16'd3) :
                              (active_kw >= 4'd3) ? ((ext_chin + 16'd7) >> 3) :
                              ((ext_chin + 16'd31) >> 5);

    wire [7:0] num_oc_tiles = (ext_chout <= {8'd0, tile_tout}) ? 8'd1 :
                              (active_mode == 2'b11) ? ((ext_chout + 16'd31) >> 5) :
                              (active_kw >= 4'd7) ? ((ext_chout + 16'd7) >> 3) :
                              (active_kw >= 4'd3) ? ((ext_chout + 16'd15) >> 4) :
                              ((ext_chout + 16'd31) >> 5);

    wire [ADDR_WIDTH-1:0] oc_offset = (active_mode == 2'b11) ? ({ {(ADDR_WIDTH-8){1'b0}}, tile_oc_cnt } << 5) :
                                      (active_kw >= 4'd7)   ? ({ {(ADDR_WIDTH-8){1'b0}}, tile_oc_cnt } << 3) :
                                      (active_kw >= 4'd3)   ? ({ {(ADDR_WIDTH-8){1'b0}}, tile_oc_cnt } << 4) :
                                      ({ {(ADDR_WIDTH-8){1'b0}}, tile_oc_cnt } << 5);

    wire [ADDR_WIDTH-1:0] ic_offset = (active_mode == 2'b11) ? oc_offset :
                                      (active_kw >= 4'd7)   ? {ADDR_WIDTH{1'b0}} :
                                      (active_kw >= 4'd3)   ? ({ {(ADDR_WIDTH-8){1'b0}}, tile_ic_cnt } << 3) :
                                      ({ {(ADDR_WIDTH-8){1'b0}}, tile_ic_cnt } << 5);

    wire [15:0] eff_chin_rem = (ext_chin > { {(16-ADDR_WIDTH){1'b0}}, ic_offset }) ?
                               (ext_chin - { {(16-ADDR_WIDTH){1'b0}}, ic_offset }) : 16'd0;
    wire [7:0] cur_tile_chin = (active_mode == 2'b11) ? 8'd1 :
                               (eff_chin_rem < {8'd0, tile_tin}) ? eff_chin_rem[7:0] : tile_tin;

    wire [15:0] eff_chout_rem = (ext_chout > { {(16-ADDR_WIDTH){1'b0}}, oc_offset }) ?
                                (ext_chout - { {(16-ADDR_WIDTH){1'b0}}, oc_offset }) : 16'd0;
    wire [7:0] cur_tile_chout = (eff_chout_rem < {8'd0, tile_tout}) ? eff_chout_rem[7:0] : tile_tout;

    // Weights per tile
    wire [15:0] cur_tile_weights = (active_mode == 2'b00) ?
        ({8'd0, cur_tile_chout} * {8'd0, cur_tile_chin} * {12'd0, active_kh * active_kw} +
         ((tile_ic_cnt == num_ic_tiles - 1) ? ({14'd0, cur_tile_chout} << 1) : 16'd0)) :
        (active_mode == 2'b11) ?
        ({8'd0, cur_tile_chout} * {12'd0, active_kh * active_kw} + ({14'd0, cur_tile_chout} << 1)) :
        ({8'd0, active_chout[7:0]} * {8'd0, active_chin[7:0]});

    localparam CHUNK_SIZE   = (NUM_BANKS * WORD_SIZE) / w_DATA_WIDTH;
    localparam BEAT_WEIGHTS = 128 / w_DATA_WIDTH; // 16 for INT8, 8 for INT16
    wire [15:0] cur_tile_beats = (active_mode == 2'b01) ?
        ({8'd0, active_chin[7:0]} * NUM_BANKS) :
        ((w_DATA_WIDTH == 8) ?
            ((cur_tile_weights + 16'd15) >> 4) :
            ((cur_tile_weights + 16'd7) >> 3));

    wire [15:0] cur_tile_chunks = (cur_tile_weights + 16'd255) >> 8;
    wire [15:0] num_weight_chunks = (active_mode == 2'b01) ?
        ((({20'd0, active_chout} * {20'd0, active_chin}) + 32'd255) >> 8) :
        cur_tile_chunks;

    // Registered stride and bias base for fast non-blocking weight indexing
    reg [7:0]  reg_kernel_area;
    reg [31:0] reg_ch_stride;
    reg [31:0] reg_bias_base;

    // Pipelined weight indexing (0 combinational multipliers in inner loop!)
    reg [31:0] oc_base_addr;
    reg [31:0] ch_base_addr;
    reg [31:0] v_base_addr;
    reg [10:0] conv_weight_addr;
    wire signed [w_DATA_WIDTH-1:0] current_conv_wt = weight_table[conv_weight_addr % MAX_WEIGHTS];
    reg signed [b_DATA_WIDTH-1:0] current_conv_bias;
    assign bias_out = current_conv_bias;

    // Universal 4D spatial tiling addresses across all layers
    wire [ADDR_WIDTH-1:0] eff_output_base_addr   = active_output_base_addr + chunk_out_offset;
    wire [ADDR_WIDTH-1:0] eff_residual_base_addr = active_residual_base_addr + chunk_out_offset;
    wire [ADDR_WIDTH-1:0] eff_input_base_addr    = (dec_dma_rd_en) ?
        active_input_base_addr :
        (active_input_base_addr + chunk_in_offset);

    // Replicate scalar weight across all R*C PEs for Conv broadcast
    wire [w_DATA_WIDTH*R*C-1:0] weight_broadcast_conv;
    genvar gi;
    generate
        for (gi = 0; gi < NUM_PES; gi = gi + 1) begin : gen_wt_bcast
            assign weight_broadcast_conv[(gi+1)*w_DATA_WIDTH-1 : gi*w_DATA_WIDTH] = current_conv_wt;
        end
    endgenerate

    // Combinational weight_in: In GEMV, feeds from mem_dout_a; In Conv, feeds broadcast weight
    assign weight_in = (active_mode == 2'b01) ? mem_dout_a : weight_broadcast_conv;

    // -------------------------------------------------------------------------
    // Re-Quantization, Saturated Clipping, Activation & Pooling Block
    // Drives mem_din_a for memory writeback.
    // -------------------------------------------------------------------------
    wire requant_residual_en = (active_mode == 2'b00) ?
        (active_residual_en | (tile_ic_cnt > 0)) :
        active_residual_en;

    wire [1:0] requant_act_type = (active_mode == 2'b00 && tile_ic_cnt < num_ic_tiles - 1) ?
        2'b00 : // Linear Passthrough for intermediate partial sums
        active_act_type;

    wire [NUM_BANKS*WORD_SIZE-1:0] requant_data_out;

    requantization_block #(
        .R(R),
        .C(C),
        .IN_DATA_WIDTH(y_DATA_WIDTH),
        .OUT_DATA_WIDTH(x_DATA_WIDTH),
        .CLIP_MAX(CLIP_MAX),
        .CLIP_MIN(CLIP_MIN)
    ) u_requant (
        .clk(clk),
        .reset(reset),
        .cfg_shift_bits(active_shift_bits),
        .cfg_act_type(requant_act_type),
        .cfg_stride(active_stride),
        .cfg_pad_type(active_pad_type),
        .cfg_kw(active_kw),
        .cfg_kh(active_kh),
        .cfg_pool_type(active_pool_type),
        .cfg_pool_size(active_pool_size),
        .cfg_pool_stride(active_pool_stride),
        .cfg_residual_en(requant_residual_en),
        .residual_in(mem_dout_b),
        .direct_in_sel(active_mode == 2'b10),
        .direct_in_8bit(mem_dout_b),
        .data_in(engdata_out),
        .data_out(requant_data_out)
    );

    assign mem_din_a = (state == S_POOL_WB || (state == S_POOL_READ && gap_ch_idx == active_chin - 1)) ? gap_row_buf : requant_data_out;

    integer idx;

    // -------------------------------------------------------------------------
    // Main Autonomous FSM
    // -------------------------------------------------------------------------
    always @(posedge clk or posedge reset) begin
        if (reset) begin
            state                <= S_IDLE;
            busy                 <= 1'b0;
            done                 <= 1'b0;
            pc                   <= {CFG_ADDR_WIDTH{1'b0}};
            cfg_rd_addr          <= {CFG_ADDR_WIDTH{1'b0}};
            dec_mode             <= 2'b00;
            dec_chin             <= 12'd0;
            dec_chout            <= 12'd0;
            dec_kw               <= 4'd0;
            dec_kh               <= 4'd0;
            dec_shift_bits       <= 5'd0;
            dec_act_type         <= 2'b00;
            dec_stride           <= 3'd1;
            dec_pad_type         <= 1'b1;
            dec_pool_type        <= 2'b00;
            dec_pool_size        <= 2'b00;
            dec_pool_stride      <= 3'd2;
            dec_residual_en      <= 1'b0;
            dec_residual_base_addr <= {ADDR_WIDTH{1'b0}};
            dec_input_base_addr  <= {ADDR_WIDTH{1'b0}};
            dec_weight_base_addr <= {ADDR_WIDTH{1'b0}};
            dec_output_base_addr <= {ADDR_WIDTH{1'b0}};
            dec_is_last          <= 1'b0;
            dec_dma_rd_en        <= 1'b0;
            dec_dma_wr_en        <= 1'b0;
            dec_dma_beats        <= 7'd0;
            ctrl_start_rd_dma    <= 1'b0;
            ctrl_rd_is_weight    <= 1'b0;
            ctrl_rd_ddr_offset   <= 32'd0;
            ctrl_rd_sram_row     <= {ADDR_WIDTH{1'b0}};
            ctrl_rd_beats        <= 16'd0;
            ctrl_start_wr_dma    <= 1'b0;
            ctrl_wr_sram_row     <= {ADDR_WIDTH{1'b0}};
            ctrl_wr_beats        <= 16'd0;
            chunk_idx            <= 8'd0;
            chunk_in_offset      <= {ADDR_WIDTH{1'b0}};
            chunk_out_offset     <= {ADDR_WIDTH{1'b0}};
            stage_wt_loaded      <= 1'b0;
            tile_ic_cnt          <= 8'd0;
            tile_oc_cnt          <= 8'd0;
            stage0_total_wt_bytes<= 32'd0;
            oc_cnt               <= 8'd0;
            ch_cnt               <= 8'd0;
            v_cnt                <= 4'd0;
            h_cnt                <= 4'd0;
            k_cnt                <= 8'd0;
            wt_chunk_cnt         <= 16'd0;
            reg_kernel_area      <= 8'd0;
            reg_ch_stride        <= 32'd0;
            reg_bias_base        <= 32'd0;
            oc_base_addr         <= 32'd0;
            ch_base_addr         <= 32'd0;
            v_base_addr          <= 32'd0;
            conv_weight_addr     <= 11'd0;
            current_conv_bias    <= {b_DATA_WIDTH{1'b0}};
            mem_addr_a           <= {ADDR_WIDTH{1'b0}};
            mem_en_a             <= 1'b0;
            mem_we_a             <= {NUM_BANKS{1'b0}};
            mem_addr_b           <= {ADDR_WIDTH{1'b0}};
            mem_en_b             <= 1'b0;
            pe_enable            <= 1'b0;
            sel0                 <= 1'b0;
            sel1                 <= 2'b00;
            reg_enable           <= 1'b0;
            hor_enable           <= 1'b0;
            vert_enable          <= 1'b0;
            eng_enable           <= 1'b0;
            bias_enable          <= 1'b0;
        end else begin
            case (state)
                // -------------------------------------------------------------
                // Idle / Standby
                // -------------------------------------------------------------
                S_IDLE: begin
                    done               <= 1'b0;
                    ctrl_start_rd_dma  <= 1'b0;
                    ctrl_rd_is_weight  <= 1'b0;
                    ctrl_rd_ddr_offset <= 32'd0;
                    ctrl_start_wr_dma  <= 1'b0;
                    chunk_idx          <= 8'd0;
                    chunk_in_offset    <= {ADDR_WIDTH{1'b0}};
                    chunk_out_offset   <= {ADDR_WIDTH{1'b0}};
                    stage_wt_loaded    <= 1'b0;
                    tile_ic_cnt        <= 8'd0;
                    tile_oc_cnt        <= 8'd0;
                    stage0_total_wt_bytes <= 32'd0;
                    gap_ch_idx         <= 8'd0;
                    gap_executed       <= 1'b0;
                    last_conv_out_addr <= {ADDR_WIDTH{1'b0}};
                    gap_row_buf        <= {(NUM_BANKS*WORD_SIZE){1'b0}};
                    mem_we_a           <= {NUM_BANKS{1'b0}};
                    mem_en_a           <= 1'b0;
                    mem_en_b           <= 1'b0;
                    pe_enable          <= 1'b0;
                    reg_enable         <= 1'b0;
                    hor_enable         <= 1'b0;
                    vert_enable        <= 1'b0;
                    eng_enable         <= 1'b0;
                    bias_enable        <= 1'b0;

                    if (start) begin
                        busy                  <= 1'b1;
                        gap_ch_idx            <= 8'd0;
                        gap_executed          <= 1'b0;
                        last_conv_out_addr    <= {ADDR_WIDTH{1'b0}};
                        gap_row_buf           <= {(NUM_BANKS*WORD_SIZE){1'b0}};
                        chunk_idx             <= 8'd0;
                        chunk_in_offset       <= {ADDR_WIDTH{1'b0}};
                        chunk_out_offset      <= {ADDR_WIDTH{1'b0}};
                        stage_wt_loaded       <= 1'b0;
                        current_wt_ddr_offset <= 32'd0;
                        tile_ic_cnt           <= 8'd0;
                        tile_oc_cnt           <= 8'd0;
                        if (cfg_mode_select == 1'b1) begin
                            pc          <= {CFG_ADDR_WIDTH{1'b0}};
                            cfg_rd_addr <= {CFG_ADDR_WIDTH{1'b0}};
                            state       <= S_FETCH_CFG;
                        end else begin
                            // Static register mode
                            oc_cnt <= 8'd0;
                            ch_cnt <= 8'd0;
                            v_cnt  <= 4'd0;
                            h_cnt  <= 4'd0;
                            k_cnt  <= 8'd0;
                            if (mode == 1'b0) begin
                                wt_chunk_cnt <= 16'd0;
                                mem_addr_a   <= cfg_weight_base_addr;
                                mem_en_a     <= 1'b1;
                                state        <= S_LOAD_WEIGHTS;
                            end else begin
                                pe_enable  <= 1'b1;
                                mem_addr_a <= cfg_weight_base_addr;
                                mem_en_a   <= 1'b1;
                                mem_addr_b <= cfg_input_base_addr;
                                mem_en_b   <= 1'b1;
                                sel0       <= 1'b0;
                                sel1       <= 2'b00;
                                eng_enable <= 1'b1;
                                state      <= S_GEMV_WAIT;
                            end
                        end
                    end
                end

                // -------------------------------------------------------------
                // Instruction Fetch from config_mem (128-bit Config Word)
                // -------------------------------------------------------------
                S_FETCH_CFG: begin
                    dec_mode               <= cfg_rd_data[1:0];
                    dec_chin               <= cfg_rd_data[13:2];
                    dec_chout              <= cfg_rd_data[25:14];
                    dec_kw                 <= cfg_rd_data[29:26];
                    dec_stride             <= (cfg_rd_data[31:30] == 2'd0) ? 3'd1 : {1'b0, cfg_rd_data[31:30]};

                    dec_kh                 <= cfg_rd_data[35:32];
                    dec_shift_bits         <= cfg_rd_data[40:36];
                    dec_act_type           <= cfg_rd_data[42:41];
                    dec_pad_type           <= cfg_rd_data[43];
                    dec_pool_type          <= cfg_rd_data[45:44];
                    dec_pool_size          <= cfg_rd_data[47:46];
                    dec_pool_stride        <= (cfg_rd_data[49:48] == 2'd0) ? 3'd2 : {1'b0, cfg_rd_data[49:48]};
                    dec_residual_en        <= cfg_rd_data[50];
                    dec_dma_rd_en          <= cfg_rd_data[51];
                    dec_dma_wr_en          <= cfg_rd_data[52];
                    dec_is_last            <= cfg_rd_data[53];
                    dec_dma_beats          <= cfg_rd_data[60:54];

                    dec_input_base_addr    <= cfg_rd_data[77:64];
                    dec_weight_base_addr   <= cfg_rd_data[91:78];
                    dec_num_chunks         <= (cfg_rd_data[99:92] == 8'd0) ? 8'd1 : cfg_rd_data[99:92];

                    dec_output_base_addr   <= cfg_rd_data[113:100];
                    dec_residual_base_addr <= cfg_rd_data[127:114];

                    oc_cnt                 <= 8'd0;
                    ch_cnt                 <= 8'd0;
                    v_cnt                  <= 4'd0;
                    h_cnt                  <= 4'd0;
                    k_cnt                  <= 8'd0;
                    wt_chunk_cnt           <= 16'd0;
                    chunk_idx              <= 8'd0;
                    chunk_in_offset        <= {ADDR_WIDTH{1'b0}};
                    chunk_out_offset       <= {ADDR_WIDTH{1'b0}};

                    state                  <= S_DECODE_CFG;
                end

                // -------------------------------------------------------------
                // Instruction Decode & Trigger: Registers & cur_tile_beats are 100% stable
                // -------------------------------------------------------------
                S_DECODE_CFG: begin
                    if (wt_ddr_addr != 32'd0 && !stage_wt_loaded) begin
                        // Autonomous DMA Read: Stream active stage/tile weights from DDR into SRAM
                        ctrl_start_rd_dma  <= 1'b1;
                        ctrl_rd_is_weight  <= 1'b1;
                        ctrl_rd_ddr_offset <= current_wt_ddr_offset;
                        ctrl_rd_sram_row   <= active_weight_base_addr;
                        ctrl_rd_beats      <= (cur_tile_beats == 16'd0) ? 16'd1 : cur_tile_beats;
                        state              <= S_DMA_RD_WAIT;
                    end else if (dec_dma_rd_en) begin
                        // Autonomous DMA Read: Stream input chunk
                        ctrl_start_rd_dma  <= 1'b1;
                        ctrl_rd_is_weight  <= 1'b0;
                        ctrl_rd_ddr_offset <= (active_num_chunks > 8'd1) ?
                            ({24'd0, chunk_idx} * ((x_DATA_WIDTH == 8) ? 32'd768 : 32'd1536)) : 32'd0;
                        ctrl_rd_sram_row   <= active_input_base_addr;
                        ctrl_rd_beats      <= (active_num_chunks > 8'd1) ?
                            ((x_DATA_WIDTH == 8) ? 16'd48 : 16'd96) :
                            ((dec_dma_beats == 7'd0) ? 16'd16 : {9'd0, dec_dma_beats});
                        state              <= S_DMA_RD_WAIT;
                    end else begin
                        if (active_mode == 2'b00 || active_mode == 2'b11) begin
                            // Mode 0: 2D Convolution / Mode 3: Depthwise Conv -> Start weight chunk loading
                            wt_chunk_cnt <= 16'd0;
                            mem_addr_a   <= active_weight_base_addr;
                            mem_en_a     <= 1'b1;
                            state        <= S_LOAD_WEIGHTS;
                        end else if (active_mode == 2'b01) begin
                            // Mode 1: GEMV -> Check if Hardware GAP is needed before first GEMV uop (multi-chunk models only)
                            if (active_num_chunks > 8'd1 && !gap_executed && (last_conv_out_addr != {ADDR_WIDTH{1'b0}})) begin
                                gap_ch_idx  <= 8'd0;
                                gap_row_buf <= {(NUM_BANKS*WORD_SIZE){1'b0}};
                                mem_addr_b  <= last_conv_out_addr;
                                mem_en_b    <= 1'b1;
                                state       <= S_POOL_WAIT;
                            end else begin
                                pe_enable  <= 1'b1;
                                mem_addr_a <= active_weight_base_addr;
                                mem_en_a   <= 1'b1;
                                mem_addr_b <= eff_input_base_addr;
                                mem_en_b   <= 1'b1;
                                sel0       <= 1'b0;
                                sel1       <= 2'b00;
                                eng_enable <= 1'b0;
                                reg_enable <= 1'b0;
                                state      <= S_GEMV_WAIT;
                            end
                        end else if (active_mode == 2'b10) begin
                            // Mode 2: Standalone Global Average Pooling
                            gap_ch_idx  <= 8'd0;
                            gap_row_buf <= {(NUM_BANKS*WORD_SIZE){1'b0}};
                            mem_addr_b  <= eff_input_base_addr;
                            mem_en_b    <= 1'b1;
                            state       <= S_POOL_WAIT;
                        end else begin
                            state      <= S_DONE;
                        end
                    end
                end

                // -------------------------------------------------------------
                // Autonomous DMA Inbound Fetch Wait State
                // -------------------------------------------------------------
                S_DMA_RD_WAIT: begin
                    ctrl_start_rd_dma <= 1'b0;
                    if (rd_dma_done) begin
                        if (ctrl_rd_is_weight) begin
                            ctrl_rd_is_weight <= 1'b0;
                            stage_wt_loaded   <= 1'b1;
                            current_wt_ddr_offset <= current_wt_ddr_offset + ({16'd0, cur_tile_beats} * 32'd16);
                            if (chunk_idx == 8'd0 && pc == 0) begin
                                stage0_total_wt_bytes <= current_wt_ddr_offset + ({16'd0, cur_tile_beats} * 32'd16);
                            end
                            if (dec_dma_rd_en) begin
                                // Weight load complete; proceed to input chunk fetch
                                ctrl_start_rd_dma  <= 1'b1;
                                ctrl_rd_is_weight  <= 1'b0;
                                ctrl_rd_ddr_offset <= (active_num_chunks > 8'd1) ?
                                    ({24'd0, chunk_idx} * ((x_DATA_WIDTH == 8) ? 32'd768 : 32'd1536)) : 32'd0;
                                ctrl_rd_sram_row   <= dec_input_base_addr;
                                ctrl_rd_beats      <= (active_num_chunks > 8'd1) ?
                                    ((x_DATA_WIDTH == 8) ? 16'd48 : 16'd96) :
                                    ((dec_dma_beats == 7'd0) ? 16'd16 : {9'd0, dec_dma_beats});
                                state              <= S_DMA_RD_WAIT;
                            end else begin
                                // Input is resident; proceed directly to execution
                                if (dec_mode == 2'b00 || dec_mode == 2'b11) begin
                                    wt_chunk_cnt <= 16'd0;
                                    mem_addr_a   <= dec_weight_base_addr;
                                    mem_en_a     <= 1'b1;
                                    state        <= S_LOAD_WEIGHTS;
                                end else if (dec_mode == 2'b01) begin
                                    pe_enable  <= 1'b1;
                                    mem_addr_a <= dec_weight_base_addr;
                                    mem_en_a   <= 1'b1;
                                    mem_addr_b <= eff_input_base_addr;
                                    mem_en_b   <= 1'b1;
                                    sel0       <= 1'b0;
                                    sel1       <= 2'b00;
                                    eng_enable <= 1'b0;
                                    reg_enable <= 1'b0;
                                    state      <= S_GEMV_WAIT;
                                end else if (dec_mode == 2'b10) begin
                                    mem_addr_b <= eff_input_base_addr;
                                    mem_en_b   <= 1'b1;
                                    state      <= S_POOL_WAIT;
                                end else begin
                                    state      <= S_DONE;
                                end
                            end
                        end else begin
                            // Activation chunk fetch complete; proceed to execution
                            if (dec_mode == 2'b00 || dec_mode == 2'b11) begin
                                wt_chunk_cnt <= 16'd0;
                                mem_addr_a   <= dec_weight_base_addr;
                                mem_en_a     <= 1'b1;
                                state        <= S_LOAD_WEIGHTS;
                            end else if (dec_mode == 2'b01) begin
                                pe_enable  <= 1'b1;
                                mem_addr_a <= dec_weight_base_addr;
                                mem_en_a   <= 1'b1;
                                mem_addr_b <= eff_input_base_addr;
                                mem_en_b   <= 1'b1;
                                sel0       <= 1'b0;
                                sel1       <= 2'b00;
                                eng_enable <= 1'b0;
                                reg_enable <= 1'b0;
                                state      <= S_GEMV_WAIT;
                            end else if (dec_mode == 2'b10) begin
                                mem_addr_b <= eff_input_base_addr;
                                mem_en_b   <= 1'b1;
                                state      <= S_POOL_WAIT;
                            end else begin
                                state      <= S_DONE;
                            end
                        end
                    end
                end

                // -------------------------------------------------------------
                // Scalable Multi-Chunk Weight Loading (Mode 0: 2D Convolution)
                // -------------------------------------------------------------
                S_LOAD_WEIGHTS: begin
                    reg_kernel_area  <= active_kh * active_kw;
                    reg_ch_stride    <= (active_mode == 2'b11) ? (active_kh * active_kw) : (cur_tile_chin * (active_kh * active_kw));
                    reg_bias_base    <= (active_mode == 2'b11) ? (cur_tile_chout * (active_kh * active_kw)) : (cur_tile_chout * cur_tile_chin * (active_kh * active_kw));
                    oc_base_addr     <= 32'd0;
                    ch_base_addr     <= 32'd0;
                    v_base_addr      <= 32'd0;
                    conv_weight_addr <= 11'd0;
                    state            <= S_LATCH_WEIGHTS;
                end

                S_LATCH_WEIGHTS: begin
                    case (wt_chunk_cnt[2:0])
                        3'd0: begin
                            for (idx = 0; idx < CHUNK_SIZE; idx = idx + 1) begin
                                if (idx < MAX_WEIGHTS)
                                    weight_table[idx] <= mem_dout_a[idx*w_DATA_WIDTH +: w_DATA_WIDTH];
                            end
                        end
                        3'd1: begin
                            for (idx = 0; idx < CHUNK_SIZE; idx = idx + 1) begin
                                if (CHUNK_SIZE + idx < MAX_WEIGHTS)
                                    weight_table[CHUNK_SIZE + idx] <= mem_dout_a[idx*w_DATA_WIDTH +: w_DATA_WIDTH];
                            end
                        end
                        3'd2: begin
                            for (idx = 0; idx < CHUNK_SIZE; idx = idx + 1) begin
                                if (2*CHUNK_SIZE + idx < MAX_WEIGHTS)
                                    weight_table[2*CHUNK_SIZE + idx] <= mem_dout_a[idx*w_DATA_WIDTH +: w_DATA_WIDTH];
                            end
                        end
                        3'd3: begin
                            for (idx = 0; idx < CHUNK_SIZE; idx = idx + 1) begin
                                if (3*CHUNK_SIZE + idx < MAX_WEIGHTS)
                                    weight_table[3*CHUNK_SIZE + idx] <= mem_dout_a[idx*w_DATA_WIDTH +: w_DATA_WIDTH];
                            end
                        end
                        3'd4: begin
                            for (idx = 0; idx < CHUNK_SIZE; idx = idx + 1) begin
                                if (4*CHUNK_SIZE + idx < MAX_WEIGHTS)
                                    weight_table[4*CHUNK_SIZE + idx] <= mem_dout_a[idx*w_DATA_WIDTH +: w_DATA_WIDTH];
                            end
                        end
                        3'd5: begin
                            for (idx = 0; idx < CHUNK_SIZE; idx = idx + 1) begin
                                if (5*CHUNK_SIZE + idx < MAX_WEIGHTS)
                                    weight_table[5*CHUNK_SIZE + idx] <= mem_dout_a[idx*w_DATA_WIDTH +: w_DATA_WIDTH];
                            end
                        end
                        3'd6: begin
                            for (idx = 0; idx < CHUNK_SIZE; idx = idx + 1) begin
                                if (6*CHUNK_SIZE + idx < MAX_WEIGHTS)
                                    weight_table[6*CHUNK_SIZE + idx] <= mem_dout_a[idx*w_DATA_WIDTH +: w_DATA_WIDTH];
                            end
                        end
                        3'd7: begin
                            for (idx = 0; idx < CHUNK_SIZE; idx = idx + 1) begin
                                if (7*CHUNK_SIZE + idx < MAX_WEIGHTS)
                                    weight_table[7*CHUNK_SIZE + idx] <= mem_dout_a[idx*w_DATA_WIDTH +: w_DATA_WIDTH];
                            end
                        end
                    endcase

                    if (wt_chunk_cnt + 16'd1 < num_weight_chunks && wt_chunk_cnt < 16'd7) begin
                        // Advance to next chunk
                        wt_chunk_cnt <= wt_chunk_cnt + 16'd1;
                        mem_addr_a   <= active_weight_base_addr + (wt_chunk_cnt[ADDR_WIDTH-1:0] + 1'b1);
                        mem_en_a     <= 1'b1;
                        state        <= S_LOAD_WEIGHTS;
                    end else begin
                        // All weight chunks loaded! Start convolution
                        mem_en_a         <= 1'b0;
                        mem_addr_b       <= (active_mode == 2'b11) ? (eff_input_base_addr + oc_offset) : (eff_input_base_addr + ic_offset);
                        mem_en_b         <= 1'b1;
                        conv_weight_addr <= 11'd0;
                        state            <= S_CONV_START;
                    end
                end

                // -------------------------------------------------------------
                // Mode 0: 2D Spatial Convolution Execution
                // -------------------------------------------------------------
                S_CONV_START: begin
                    pe_enable        <= 1'b1;
                    mem_addr_b       <= (active_mode == 2'b11) ? (eff_input_base_addr + oc_offset + oc_cnt) : (eff_input_base_addr + ic_offset + ch_cnt);
                    mem_en_b         <= 1'b1;
                    v_cnt            <= 4'd0;
                    h_cnt            <= 4'd0;
                    hor_enable       <= 1'b0;
                    vert_enable      <= 1'b0;
                    reg_enable       <= 1'b0;
                    eng_enable       <= 1'b0;
                    conv_weight_addr <= ch_base_addr[10:0];
                    state            <= S_CONV_INIT_TAP;
                end

                S_CONV_INIT_TAP: begin
                    if (active_kw == 4'd1 && active_kh == 4'd1) begin
                        // Pointwise 1x1: Tap 0 latches mult into Outf_map
                        sel0        <= 1'b0;
                        sel1        <= 2'b00;
                        hor_enable  <= 1'b0;
                        vert_enable <= 1'b0;
                        reg_enable  <= 1'b1;
                        state       <= S_CONV_WAIT_1X1;
                    end else begin
                        // KW x KH walk: Schedule Tap 0 (h=0) of row 0
                        sel0             <= 1'b0;
                        sel1             <= 2'b00; // mult + 0
                        hor_enable       <= 1'b1;
                        reg_enable       <= 1'b0;
                        vert_enable      <= 1'b0;
                        eng_enable       <= 1'b0;
                        h_cnt            <= 4'd0;
                        conv_weight_addr <= conv_weight_addr + 11'd1;
                        state            <= S_CONV_STEP_H;
                    end
                end

                S_CONV_STEP_H: begin
                    if (h_cnt < active_kw - 2) begin
                        h_cnt            <= h_cnt + 4'd1;
                        sel0             <= 1'b0;
                        sel1             <= 2'b01; // mult + hor_left
                        hor_enable       <= 1'b1;
                        reg_enable       <= 1'b0;
                        conv_weight_addr <= conv_weight_addr + 11'd1;
                    end else if (h_cnt == active_kw - 2) begin
                        h_cnt            <= active_kw - 4'd1;
                        sel0             <= 1'b0;
                        sel1             <= 2'b01; // mult + hor_left
                        hor_enable       <= 1'b0;
                        reg_enable       <= 1'b1; // latch row sum into Outf_map
                    end else begin
                        hor_enable <= 1'b0;
                        reg_enable <= 1'b0;
                        state      <= S_CONV_ROW_DONE;
                    end
                end

                S_CONV_ROW_DONE: begin
                    if (v_cnt == 4'd0) begin
                        sel0             <= 1'b1; // Outf_map feedback
                        sel1             <= 2'b00; // + 0
                        vert_enable      <= 1'b1;  // latch into VERT
                        reg_enable       <= 1'b0;
                        v_base_addr      <= ch_base_addr + {28'd0, active_kw};
                        conv_weight_addr <= (ch_base_addr + {28'd0, active_kw}) % MAX_WEIGHTS;
                        state            <= S_CONV_STEP_V;
                    end else if (v_cnt < active_kh - 1) begin
                        sel0             <= 1'b1; // Outf_map feedback
                        sel1             <= 2'b10; // + vert_up
                        vert_enable      <= 1'b1;  // latch into VERT
                        reg_enable       <= 1'b0;
                        v_base_addr      <= ch_base_addr + ({28'd0, v_cnt + 4'd1} * {28'd0, active_kw});
                        conv_weight_addr <= (ch_base_addr + ({28'd0, v_cnt + 4'd1} * {28'd0, active_kw})) % MAX_WEIGHTS;
                        state            <= S_CONV_STEP_V;
                    end else begin
                        sel0        <= 1'b1;  // Outf_map feedback
                        sel1        <= 2'b10; // + vert_up
                        vert_enable <= 1'b0;
                        reg_enable  <= 1'b1;  // Latch full 2D patch into Outf_map
                        state       <= S_CONV_FINAL_V;
                    end
                end

                S_CONV_STEP_V: begin
                    vert_enable      <= 1'b0;
                    v_cnt            <= v_cnt + 4'd1;
                    h_cnt            <= 4'd0;
                    sel0             <= 1'b0;
                    sel1             <= 2'b00;
                    hor_enable       <= 1'b1;
                    reg_enable       <= 1'b0;
                    conv_weight_addr <= conv_weight_addr + 11'd1;
                    state            <= S_CONV_STEP_H;
                end

                S_CONV_FINAL_V: begin
                    sel0       <= 1'b1;  // Outf_map
                    sel1       <= 2'b11; // + ENG
                    eng_enable <= 1'b1;  // Accumulate into ENG
                    reg_enable <= 1'b1;  // Latch into Outf_map
                    state      <= S_CONV_ACCUM_ENG;
                end

                S_CONV_WAIT_1X1: begin
                    sel0       <= 1'b1;  // Outf_map
                    sel1       <= 2'b11; // + ENG
                    eng_enable <= 1'b1;  // Accumulate into ENG
                    reg_enable <= 1'b1;  // Latch into Outf_map
                    state      <= S_CONV_ACCUM_ENG;
                end

                S_CONV_ACCUM_ENG: begin
                    eng_enable <= 1'b0;
                    if (ch_cnt < cur_tile_chin - 1) begin
                        ch_cnt           <= ch_cnt + 8'd1;
                        ch_base_addr     <= oc_base_addr + ({24'd0, ch_cnt + 8'd1} * {24'd0, reg_kernel_area});
                        v_base_addr      <= oc_base_addr + ({24'd0, ch_cnt + 8'd1} * {24'd0, reg_kernel_area});
                        conv_weight_addr <= (oc_base_addr + ({24'd0, ch_cnt + 8'd1} * {24'd0, reg_kernel_area})) % MAX_WEIGHTS;
                        mem_addr_b       <= eff_input_base_addr + ic_offset + (ch_cnt + 8'd1);
                        mem_en_b         <= 1'b1;
                        state            <= S_CONV_START;
                    end else begin
                        if (tile_ic_cnt > 0) begin
                            // Read previous partial sum from SRAM Port B for accumulation
                            mem_addr_b <= eff_output_base_addr + oc_offset + oc_cnt;
                            mem_en_b   <= 1'b1;
                        end else if (active_residual_en && (tile_ic_cnt == num_ic_tiles - 1)) begin
                            mem_addr_b <= active_residual_base_addr + oc_offset + oc_cnt;
                            mem_en_b   <= 1'b1;
                        end

                        if (tile_ic_cnt == num_ic_tiles - 1) begin
                            sel0              <= 1'b1;  // Outf_map
                            sel1              <= 2'b00; // + 16-bit Bias (bias_out)
                            bias_enable       <= 1'b1;  // Mux0 = bias_sign_ext
                            reg_enable        <= 1'b1;  // Latch into Outf_map
                            current_conv_bias <= {weight_table[(reg_bias_base + ({24'd0, oc_cnt} << 1) + 1) % MAX_WEIGHTS],
                                                  weight_table[(reg_bias_base + ({24'd0, oc_cnt} << 1)) % MAX_WEIGHTS]};
                        end else begin
                            bias_enable       <= 1'b0;
                            reg_enable        <= 1'b0;
                        end
                        state <= S_CONV_BIAS;
                    end
                end

                S_CONV_BIAS: begin
                    bias_enable <= 1'b0;
                    reg_enable  <= 1'b0;
                    state       <= S_CONV_PIPE;
                end

                S_CONV_PIPE: begin
                    mem_addr_a  <= eff_output_base_addr + oc_offset + oc_cnt;
                    mem_en_a    <= 1'b1;
                    mem_we_a    <= {NUM_BANKS{1'b1}};
                    state       <= S_CONV_WB;
                end

                S_CONV_WB: begin
                    mem_we_a  <= {NUM_BANKS{1'b0}};
                    mem_en_a  <= 1'b0;
                    mem_en_b  <= 1'b0;
                    pe_enable <= 1'b0; // Clear ENG for next filter
                    state     <= S_CONV_CLEAR_ENG;
                end

                S_CONV_CLEAR_ENG: begin
                    if (oc_cnt < cur_tile_chout - 1) begin
                        oc_cnt           <= oc_cnt + 8'd1;
                        ch_cnt           <= 8'd0;
                        oc_base_addr     <= ({24'd0, oc_cnt + 8'd1} * reg_ch_stride);
                        ch_base_addr     <= ({24'd0, oc_cnt + 8'd1} * reg_ch_stride);
                        v_base_addr      <= ({24'd0, oc_cnt + 8'd1} * reg_ch_stride);
                        conv_weight_addr <= ({24'd0, oc_cnt + 8'd1} * reg_ch_stride) % MAX_WEIGHTS;
                        pe_enable        <= 1'b1;
                        mem_addr_b       <= (active_mode == 2'b11) ? (eff_input_base_addr + oc_offset + (oc_cnt + 8'd1)) : (eff_input_base_addr + ic_offset);
                        mem_en_b         <= 1'b1;
                        state            <= S_CONV_START;
                    end else begin
                        // Completed all output channels in current tile!
                        oc_cnt <= 8'd0;
                        ch_cnt <= 8'd0;
                        if (chunk_idx < active_num_chunks - 8'd1) begin
                            // Advance to next spatial chunk of current layer using same resident weights
                            chunk_idx        <= chunk_idx + 8'd1;
                            chunk_in_offset  <= chunk_in_offset + { {(ADDR_WIDTH-12){1'b0}}, active_chin };
                            chunk_out_offset <= chunk_out_offset + { {(ADDR_WIDTH-12){1'b0}}, active_chout };
                            oc_base_addr     <= 32'd0;
                            ch_base_addr     <= 32'd0;
                            v_base_addr      <= 32'd0;
                            conv_weight_addr <= 11'd0;
                            if (dec_dma_rd_en) begin
                                // DMA stream next input chunk from DDR
                                ctrl_start_rd_dma  <= 1'b1;
                                ctrl_rd_is_weight  <= 1'b0;
                                ctrl_rd_ddr_offset <= (chunk_idx + 8'd1) * ((x_DATA_WIDTH == 8) ? 32'd768 : 32'd1536);
                                ctrl_rd_sram_row   <= active_input_base_addr;
                                ctrl_rd_beats      <= (dec_dma_beats == 7'd0) ? 16'd48 : {9'd0, dec_dma_beats};
                                state              <= S_DMA_RD_WAIT;
                            end else begin
                                // Input chunk is in SRAM, and weights are already in weight_table!
                                state              <= S_CONV_START;
                            end
                        end else begin
                            // All spatial chunks completed for current channel tile!
                            chunk_idx        <= 8'd0;
                            chunk_in_offset  <= {ADDR_WIDTH{1'b0}};
                            chunk_out_offset <= {ADDR_WIDTH{1'b0}};
                            oc_base_addr     <= 32'd0;
                            ch_base_addr     <= 32'd0;
                            v_base_addr      <= 32'd0;
                            conv_weight_addr <= 11'd0;
                            if (tile_ic_cnt < num_ic_tiles - 1) begin
                                tile_ic_cnt     <= tile_ic_cnt + 8'd1;
                                stage_wt_loaded <= 1'b0;
                                state           <= S_DECODE_CFG;
                            end else if (tile_oc_cnt < num_oc_tiles - 1) begin
                                tile_ic_cnt     <= 8'd0;
                                tile_oc_cnt     <= tile_oc_cnt + 8'd1;
                                stage_wt_loaded <= 1'b0;
                                state           <= S_DECODE_CFG;
                            end else begin
                                // Layer complete across all tiles and chunks!
                                tile_ic_cnt     <= 8'd0;
                                tile_oc_cnt     <= 8'd0;
                                stage_wt_loaded <= 1'b0;
                                last_conv_out_addr <= active_output_base_addr;
                                if (dec_dma_wr_en) begin
                                    ctrl_start_wr_dma <= 1'b1;
                                    ctrl_wr_sram_row  <= active_output_base_addr;
                                    ctrl_wr_beats     <= (dec_dma_beats == 7'd0) ? 16'd16 : {9'd0, dec_dma_beats};
                                    state             <= S_DMA_WR_WAIT;
                                end else if (cfg_mode_select == 1'b1) begin
                                    if (active_is_last) begin
                                        state <= S_DONE;
                                    end else begin
                                        pc          <= pc + { {(CFG_ADDR_WIDTH-1){1'b0}}, 1'b1 };
                                        cfg_rd_addr <= pc + { {(CFG_ADDR_WIDTH-1){1'b0}}, 1'b1 };
                                        state       <= S_FETCH_CFG;
                                    end
                                end else begin
                                    state <= S_DONE;
                                end
                            end
                        end
                    end
                end

                // -------------------------------------------------------------
                // Mode 1: GEMV / Dense Layer Execution
                // -------------------------------------------------------------
                S_GEMV_WAIT: begin
                    hor_enable  <= 1'b0;
                    vert_enable <= 1'b0;
                    bias_enable <= 1'b0;
                    sel0        <= 1'b0;
                    sel1        <= 2'b00; // mult + 0 for Beat 0
                    k_cnt       <= 8'd0;
                    if (active_chin == 8'd1) begin
                        eng_enable <= 1'b0;
                        reg_enable <= 1'b1;
                    end else begin
                        eng_enable <= 1'b1;
                        reg_enable <= 1'b0;
                        mem_addr_a <= active_weight_base_addr + 8'd1;
                    end
                    state <= S_GEMV_COMPUTE;
                end

                S_GEMV_COMPUTE: begin
                    hor_enable  <= 1'b0;
                    vert_enable <= 1'b0;
                    bias_enable <= 1'b0;

                    if (active_chin == 8'd1) begin
                        eng_enable <= 1'b0;
                        reg_enable <= 1'b0;
                        state      <= S_GEMV_PIPE1;
                    end else if (k_cnt < active_chin - 1) begin
                        k_cnt      <= k_cnt + 8'd1;
                        sel0       <= 1'b0;
                        sel1       <= 2'b11; // mult + ENG
                        eng_enable <= 1'b1;
                        reg_enable <= 1'b0;
                        mem_addr_a <= active_weight_base_addr + (k_cnt + 8'd2);
                    end else begin
                        // Final tap (k_cnt == active_chin - 1):
                        // Complete dot product across all active_chin features is on adder_out.
                        eng_enable <= 1'b0;
                        reg_enable <= 1'b1; // Latch final sum into Outf_map at next clock edge
                        state      <= S_GEMV_PIPE1;
                    end
                end

                S_GEMV_PIPE1: begin
                    // Outf_map now holds the complete sum. PPU comb_data_out is valid combinationally.
                    // Wait 1 cycle for PPU data_out_reg to latch comb_data_out.
                    reg_enable <= 1'b0;
                    eng_enable <= 1'b0;
                    state      <= S_GEMV_PIPE2;
                end

                S_GEMV_PIPE2: begin
                    // PPU data_out_reg now holds the stable quantized logits.
                    // Assert BRAM Port A write enable to commit to SRAM at next clock edge.
                    mem_addr_a <= eff_output_base_addr;
                    mem_en_a   <= 1'b1;
                    mem_we_a   <= {NUM_BANKS{1'b1}};
                    state      <= S_GEMV_WB;
                end

                S_GEMV_WB: begin
                    eng_enable <= 1'b0;
                    reg_enable <= 1'b0;
                    mem_we_a   <= {NUM_BANKS{1'b0}};
                    mem_en_a   <= 1'b0;
                    mem_en_b   <= 1'b0;
                    if (chunk_idx < active_num_chunks - 8'd1) begin
                        chunk_idx        <= chunk_idx + 8'd1;
                        chunk_in_offset  <= chunk_in_offset + { {(ADDR_WIDTH-12){1'b0}}, active_chin };
                        chunk_out_offset <= chunk_out_offset + { {(ADDR_WIDTH-12){1'b0}}, active_chout };
                        pe_enable        <= 1'b1;
                        mem_addr_a       <= active_weight_base_addr;
                        mem_en_a         <= 1'b1;
                        mem_addr_b       <= active_input_base_addr + chunk_in_offset + { {(ADDR_WIDTH-12){1'b0}}, active_chin };
                        mem_en_b         <= 1'b1;
                        sel0             <= 1'b0;
                        sel1             <= 2'b00;
                        state            <= S_GEMV_WAIT;
                    end else begin
                        chunk_idx        <= 8'd0;
                        chunk_in_offset  <= {ADDR_WIDTH{1'b0}};
                        chunk_out_offset <= {ADDR_WIDTH{1'b0}};
                        if (cfg_mode_select == 1'b1) begin
                            if (!active_is_last) begin
                                if (dec_dma_wr_en) begin
                                    ctrl_start_wr_dma <= 1'b1;
                                    ctrl_wr_sram_row  <= active_output_base_addr;
                                    ctrl_wr_beats     <= (dec_dma_beats == 7'd0) ? 16'd16 : {9'd0, dec_dma_beats};
                                    state             <= S_DMA_WR_WAIT;
                                end else begin
                                    pc              <= pc + { {(CFG_ADDR_WIDTH-1){1'b0}}, 1'b1 };
                                    cfg_rd_addr     <= pc + { {(CFG_ADDR_WIDTH-1){1'b0}}, 1'b1 };
                                    stage_wt_loaded <= 1'b0;
                                    state           <= S_FETCH_CFG;
                                end
                            end else begin
                                // Final Stage: GEMV Head Writeback (1,000 logits across Rows 2000..2003, or single row for MNIST)
                                ctrl_start_wr_dma <= 1'b1;
                                ctrl_wr_sram_row  <= (active_output_base_addr >= 11'd2000) ? 11'd2000 : active_output_base_addr;
                                ctrl_wr_beats     <= (active_output_base_addr >= 11'd2000) ? (4 * NUM_BANKS) : NUM_BANKS;
                                state             <= S_DMA_WR_WAIT;
                            end
                        end else begin
                            if (dec_dma_wr_en) begin
                                ctrl_start_wr_dma <= 1'b1;
                                ctrl_wr_sram_row  <= active_output_base_addr;
                                ctrl_wr_beats     <= (dec_dma_beats == 7'd0) ? 16'd16 : {9'd0, dec_dma_beats};
                                state             <= S_DMA_WR_WAIT;
                            end else begin
                                state <= S_DONE;
                            end
                        end
                    end
                end

                // -------------------------------------------------------------
                // Mode 2 / Auto-GAP: Hardware Global Average Pooling Engine
                // -------------------------------------------------------------
                S_POOL_WAIT: begin
                    state <= S_POOL_READ;
                end

                S_POOL_READ: begin
                    if (w_DATA_WIDTH == 16)
                        gap_row_buf[gap_ch_idx*16 +: 16] <= spatial_mean;
                    else
                        gap_row_buf[gap_ch_idx*8 +: 8] <= spatial_mean;

                    if (gap_ch_idx < active_chin - 1) begin
                        gap_ch_idx <= gap_ch_idx + 8'd1;
                        mem_addr_b <= ((active_mode == 2'b10) ? active_input_base_addr : last_conv_out_addr) + (gap_ch_idx + 8'd1);
                        mem_en_b   <= 1'b1;
                        state      <= S_POOL_WAIT;
                    end else begin
                        mem_en_b   <= 1'b0;
                        mem_addr_a <= active_input_base_addr;
                        mem_en_a   <= 1'b1;
                        mem_we_a   <= {NUM_BANKS{1'b1}};
                        state      <= S_POOL_WB;
                    end
                end

                S_POOL_WB: begin
                    mem_en_a     <= 1'b0;
                    mem_we_a     <= {NUM_BANKS{1'b0}};
                    gap_executed <= 1'b1;

                    if (active_mode == 2'b10) begin
                        if (cfg_mode_select == 1'b1 && !active_is_last) begin
                            pc              <= pc + { {(CFG_ADDR_WIDTH-1){1'b0}}, 1'b1 };
                            cfg_rd_addr     <= pc + { {(CFG_ADDR_WIDTH-1){1'b0}}, 1'b1 };
                            stage_wt_loaded <= 1'b0;
                            state           <= S_FETCH_CFG;
                        end else begin
                            state <= S_DONE;
                        end
                    end else begin
                        pe_enable  <= 1'b1;
                        mem_addr_a <= active_weight_base_addr;
                        mem_en_a   <= 1'b1;
                        mem_addr_b <= active_input_base_addr;
                        mem_en_b   <= 1'b1;
                        sel0       <= 1'b0;
                        sel1       <= 2'b00;
                        eng_enable <= 1'b0;
                        reg_enable <= 1'b0;
                        state      <= S_GEMV_WAIT;
                    end
                end

                // -------------------------------------------------------------
                // Autonomous DMA Outbound Spill Wait State
                // -------------------------------------------------------------
                S_DMA_WR_WAIT: begin
                    ctrl_start_wr_dma <= 1'b0;
                    if (wr_dma_done) begin
                        if ((cfg_mode_select == 1'b1) && !active_is_last) begin
                            pc              <= pc + { {(CFG_ADDR_WIDTH-1){1'b0}}, 1'b1 };
                            cfg_rd_addr     <= pc + { {(CFG_ADDR_WIDTH-1){1'b0}}, 1'b1 };
                            stage_wt_loaded <= 1'b0;
                            state           <= S_FETCH_CFG;
                        end else begin
                            state <= S_DONE;
                        end
                    end
                end

                // -------------------------------------------------------------
                // Finish
                // -------------------------------------------------------------
                S_DONE: begin
                    ctrl_start_rd_dma <= 1'b0;
                    ctrl_start_wr_dma <= 1'b0;
                    mem_we_a          <= {NUM_BANKS{1'b0}};
                    mem_en_a          <= 1'b0;
                    mem_en_b          <= 1'b0;
                    pe_enable         <= 1'b0;
                    reg_enable        <= 1'b0;
                    hor_enable        <= 1'b0;
                    vert_enable       <= 1'b0;
                    eng_enable        <= 1'b0;
                    bias_enable       <= 1'b0;
                    busy              <= 1'b0;
                    done              <= 1'b1;
                    state             <= S_IDLE;
                end

                default: state <= S_IDLE;
            endcase
        end
    end

endmodule
