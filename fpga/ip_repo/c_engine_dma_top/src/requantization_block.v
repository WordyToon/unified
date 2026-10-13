`timescale 1ns / 1ps
// =============================================================================
// Module: requantization_block
// Description: Parallel Re-Quantization, Saturation Clipping, 4-Mode Activation,
//              Zero-Padding Masking, 2D Spatial Pooling (MaxPool & AvgPool),
//              and Strided Output Compaction Post-Processing Unit (PPU).
//              Features:
//              1. Dynamic arithmetic right-shift (>>> cfg_shift_bits).
//              2. Saturated clipping to signed 8-bit range: [-128, +127].
//              3. Post-Processing Unit (PPU) activation function (cfg_act_type):
//                 - 2'b00: None (Linear / Passthrough)
//                 - 2'b01: ReLU (max(0, x))
//                 - 2'b10: Leaky ReLU (x >= 0 ? x : x >>> 3, alpha = 0.125)
//                 - 2'b11: LUT-based non-linear activation (256-entry GELU ROM)
//              4. Zero-Padding Masking (cfg_pad_type):
//                 - 1'b1: SAME padding (all 256 PEs are valid)
//                 - 1'b0: VALID padding (masks boundary PEs touching padding to 0)
//              5. 2D Spatial Pooling Unit (cfg_pool_type, cfg_pool_size, cfg_pool_stride):
//                 - cfg_pool_type: 2'b00 (Bypass/None), 2'b01 (MaxPool), 2'b10 (AvgPool)
//                 - cfg_pool_size: 2'b00 (1x1), 2'b01 (2x2), 2'b10 (3x3)
//                 - cfg_pool_stride: 3'd1 to 3'd7
//                 - Direct 8-bit streaming input (direct_in_sel, direct_in_8bit)
//                   for standalone memory-to-memory streaming.
//              6. Strided Compaction / Sub-Sampling (cfg_stride):
//                 - Supports strides S in [1..7] (satisfying S < min(R, C) = 8).
//                 - Sub-samples PE(r_out * S, c_out * S) and contiguously packs
//                   H_out x W_out elements into output byte indices [0 .. H_out*W_out-1],
//                   zeroing remaining bytes on the 2048-bit bus.
// =============================================================================

module requantization_block #(
    parameter R              = 8,
    parameter C              = 32,
    parameter IN_DATA_WIDTH  = 32,
    parameter OUT_DATA_WIDTH = 8,
    parameter signed [OUT_DATA_WIDTH-1:0] CLIP_MAX = (1 <<< (OUT_DATA_WIDTH-1)) - 1,
    parameter signed [OUT_DATA_WIDTH-1:0] CLIP_MIN = -(1 <<< (OUT_DATA_WIDTH-1))
)(
    input  wire                               clk,
    input  wire                               reset,
    input  wire [4:0]                         cfg_shift_bits,   // Dynamic shift amount: 0 to 31
    input  wire [1:0]                         cfg_act_type,     // 00: None, 01: ReLU, 10: Leaky ReLU, 11: LUT
    input  wire [2:0]                         cfg_stride,       // Stride amount: 3'd1 to 3'd7
    input  wire                               cfg_pad_type,     // 0: VALID, 1: SAME
    input  wire [3:0]                         cfg_kw,           // Kernel width (used for VALID mask)
    input  wire [3:0]                         cfg_kh,           // Kernel height (used for VALID mask)
    input  wire [1:0]                         cfg_pool_type,    // 00: None, 01: MaxPool, 10: AvgPool
    input  wire [1:0]                         cfg_pool_size,    // 00: 1x1, 01: 2x2, 10: 3x3
    input  wire [2:0]                         cfg_pool_stride,  // 3'd1 to 3'd7 (default 2 for 2x2)
    input  wire                               cfg_residual_en,  // 1: Fused residual tensor addition before activation
    input  wire [OUT_DATA_WIDTH*R*C-1:0]      residual_in,      // 2048-bit residual tensor from RAM Port B
    input  wire                               direct_in_sel,    // 0: PE 32-bit data_in, 1: direct_in_8bit
    input  wire [OUT_DATA_WIDTH*R*C-1:0]      direct_in_8bit,   // Direct 8-bit input from RAM Port B
    input  wire [IN_DATA_WIDTH*R*C-1:0]       data_in,          // 32-bit PE outputs (8192 bits)
    output wire [OUT_DATA_WIDTH*R*C-1:0]      data_out          // Re-quantized & compacted 8-bit outputs (2048 bits)
);

    localparam NUM_PES = R * C; // 256

    // Internal wire of un-registered post-processed activations
    wire [OUT_DATA_WIDTH*R*C-1:0]    comb_data_out;

    // Internal wire array of processed PE activations
    wire signed [OUT_DATA_WIDTH-1:0] pe_act_arr [0:NUM_PES-1];

    // 256-Entry Activation LUT for Swish / SiLU Function
    reg signed [7:0] act_lut [0:255];
    initial begin
        `include "act_lut_data.vh"
    end

    genvar i;
    generate
        for (i = 0; i < NUM_PES; i = i + 1) begin : gen_requant_pe
            // Slice 32-bit signed input from PE(i)
            wire signed [IN_DATA_WIDTH-1:0] pe_val_in;
            assign pe_val_in = data_in[(i+1)*IN_DATA_WIDTH-1 : i*IN_DATA_WIDTH];

            // Signed arithmetic right shift by runtime cfg_shift_bits
            wire signed [IN_DATA_WIDTH-1:0] pe_val_shifted;
            assign pe_val_shifted = pe_val_in >>> cfg_shift_bits;

            // Saturated clipping to [-128, 127]
            reg signed [OUT_DATA_WIDTH-1:0] pe_val_clipped;
            always @(*) begin
                if (pe_val_shifted > $signed({{ (IN_DATA_WIDTH-OUT_DATA_WIDTH){CLIP_MAX[OUT_DATA_WIDTH-1]} }, CLIP_MAX})) begin
                    pe_val_clipped = CLIP_MAX;
                end else if (pe_val_shifted < $signed({{ (IN_DATA_WIDTH-OUT_DATA_WIDTH){CLIP_MIN[OUT_DATA_WIDTH-1]} }, CLIP_MIN})) begin
                    pe_val_clipped = CLIP_MIN;
                end else begin
                    pe_val_clipped = pe_val_shifted[OUT_DATA_WIDTH-1:0];
                end
            end

            // 4-Mode Activation Post-Processing Unit (PPU)
            // Fused Saturated Residual Addition (Z_clipped + X_residual -> clamp to [-128, +127])
            wire signed [OUT_DATA_WIDTH-1:0] res_byte;
            assign res_byte = residual_in[(i+1)*OUT_DATA_WIDTH-1 : i*OUT_DATA_WIDTH];

            wire signed [OUT_DATA_WIDTH:0] pe_res_sum;
            assign pe_res_sum = $signed({{1{pe_val_clipped[OUT_DATA_WIDTH-1]}}, pe_val_clipped}) +
                                $signed({{1{res_byte[OUT_DATA_WIDTH-1]}}, res_byte});

            reg signed [OUT_DATA_WIDTH-1:0] pe_post_res;
            always @(*) begin
                if (cfg_residual_en) begin
                    if (pe_res_sum > $signed({{1{CLIP_MAX[OUT_DATA_WIDTH-1]}}, CLIP_MAX})) begin
                        pe_post_res = CLIP_MAX;
                    end else if (pe_res_sum < $signed({{1{CLIP_MIN[OUT_DATA_WIDTH-1]}}, CLIP_MIN})) begin
                        pe_post_res = CLIP_MIN;
                    end else begin
                        pe_post_res = pe_res_sum[OUT_DATA_WIDTH-1:0];
                    end
                end else begin
                    pe_post_res = pe_val_clipped;
                end
            end

            // 4-Mode Activation Post-Processing Unit (PPU) applied on post-residual result
            // Mode 00: Passthrough (Linear)
            // Mode 01: ReLU (zero out negative values)
            // Mode 10: Leaky ReLU (negative values multiplied by 0.125 via arithmetic right shift >>> 3)
            // Mode 11: LUT-based non-linear activation (Exact 256-entry Swish: x * sigmoid(x))
            reg signed [OUT_DATA_WIDTH-1:0] pe_val_act;
            always @(*) begin
                case (cfg_act_type)
                    2'b00:   pe_val_act = pe_post_res;
                    2'b01:   pe_val_act = pe_post_res[OUT_DATA_WIDTH-1] ? {OUT_DATA_WIDTH{1'b0}} : pe_post_res;
                    2'b10:   pe_val_act = pe_post_res[OUT_DATA_WIDTH-1] ? (pe_post_res >>> 3) : pe_post_res;
                    2'b11:   pe_val_act = (OUT_DATA_WIDTH == 8) ? act_lut[pe_post_res ^ 8'h80] :
                                          ((pe_post_res > 0) ? pe_post_res : {OUT_DATA_WIDTH{1'b0}});
                    default: pe_val_act = pe_post_res;
                endcase
            end

            // Padding Validity Check:
            // In VALID padding mode (cfg_pad_type == 0), pixels touching receptive field boundary zeros are zeroed.
            // In SAME padding mode (cfg_pad_type == 1), all pixels are valid.
            localparam integer PE_ROW = i / C;
            localparam integer PE_COL = i % C;

            wire is_valid_pixel;
            assign is_valid_pixel = (cfg_pad_type == 1'b1) ||
                                    ((PE_ROW >= (cfg_kh - 4'd1)) && (PE_COL >= (cfg_kw - 4'd1)));

            wire signed [OUT_DATA_WIDTH-1:0] conv_pixel;
            assign conv_pixel = is_valid_pixel ? pe_val_act : {OUT_DATA_WIDTH{1'b0}};

            // Direct input bypass for standalone memory-to-memory streaming
            wire signed [OUT_DATA_WIDTH-1:0] direct_pixel;
            assign direct_pixel = direct_in_8bit[(i+1)*OUT_DATA_WIDTH-1 : i*OUT_DATA_WIDTH];

            assign pe_act_arr[i] = direct_in_sel ? direct_pixel : conv_pixel;
        end
    endgenerate

    // Effective pooling stride: defaults to 2 for 2x2, else 1 if 0
    wire [2:0] eff_pool_stride;
    assign eff_pool_stride = (cfg_pool_stride != 3'd0) ? cfg_pool_stride :
                             (cfg_pool_size == 2'b01 ? 3'd2 : 3'd1);

    // Signed comparator helper function
    function signed [OUT_DATA_WIDTH-1:0] smax2;
        input signed [OUT_DATA_WIDTH-1:0] a;
        input signed [OUT_DATA_WIDTH-1:0] b;
        begin
            smax2 = (a > b) ? a : b;
        end
    endfunction

    // -------------------------------------------------------------------------
    // Unified Strided Compaction & 2D Spatial Pooling Output Multiplexers
    // -------------------------------------------------------------------------
    genvar k;
    generate
        for (k = 0; k < NUM_PES; k = k + 1) begin : gen_output_pack
            // Compile-time localparams for 2x2 pooling Stride 2 (R_out = R/2, C_out = C/2)
            localparam integer POOL_C_OUT_S2 = C / 2;
            localparam integer POOL_R_OUT_S2 = R / 2;
            localparam integer POOL_NUM_S2   = POOL_R_OUT_S2 * POOL_C_OUT_S2;
            localparam integer P2_R_S2       = (k / POOL_C_OUT_S2) * 2;
            localparam integer P2_C_S2       = (k % POOL_C_OUT_S2) * 2;
            localparam integer IDX00_2X2_S2  = (k < POOL_NUM_S2) ? (P2_R_S2 * C + P2_C_S2) : 0;
            localparam integer IDX01_2X2_S2  = (k < POOL_NUM_S2) ? (P2_R_S2 * C + P2_C_S2 + 1) : 0;
            localparam integer IDX10_2X2_S2  = (k < POOL_NUM_S2) ? ((P2_R_S2 + 1) * C + P2_C_S2) : 0;
            localparam integer IDX11_2X2_S2  = (k < POOL_NUM_S2) ? ((P2_R_S2 + 1) * C + P2_C_S2 + 1) : 0;

            // Stride 2 indices
            localparam integer S2_COLS = (C / 2 > 0) ? (C / 2) : 1;
            localparam integer S2_ROWS = (R / 2 > 0) ? (R / 2) : 1;
            localparam integer S2_COUNT = S2_COLS * S2_ROWS;
            localparam integer S2_IDX = (k < S2_COUNT) ? (((k / S2_COLS) * 2) * C + ((k % S2_COLS) * 2)) : 0;

            wire signed [OUT_DATA_WIDTH-1:0] w00_s2 = (k < POOL_NUM_S2) ? pe_act_arr[IDX00_2X2_S2] : {OUT_DATA_WIDTH{1'b0}};
            wire signed [OUT_DATA_WIDTH-1:0] w01_s2 = (k < POOL_NUM_S2) ? pe_act_arr[IDX01_2X2_S2] : {OUT_DATA_WIDTH{1'b0}};
            wire signed [OUT_DATA_WIDTH-1:0] w10_s2 = (k < POOL_NUM_S2) ? pe_act_arr[IDX10_2X2_S2] : {OUT_DATA_WIDTH{1'b0}};
            wire signed [OUT_DATA_WIDTH-1:0] w11_s2 = (k < POOL_NUM_S2) ? pe_act_arr[IDX11_2X2_S2] : {OUT_DATA_WIDTH{1'b0}};

            wire signed [OUT_DATA_WIDTH-1:0] max_2x2_s2 = smax2(smax2(w00_s2, w01_s2), smax2(w10_s2, w11_s2));

            reg signed [OUT_DATA_WIDTH-1:0] out_byte;
            always @(*) begin
                if (cfg_pool_type == 2'b00) begin
                    case (cfg_stride)
                        3'd2: out_byte = (k < S2_COUNT) ? pe_act_arr[S2_IDX] : {OUT_DATA_WIDTH{1'b0}};
                        default: out_byte = pe_act_arr[k];
                    endcase
                end else if (cfg_pool_type == 2'b01) begin
                    out_byte = (k < POOL_NUM_S2) ? max_2x2_s2 : {OUT_DATA_WIDTH{1'b0}};
                end else begin
                    out_byte = pe_act_arr[k];
                end
            end

            assign comb_data_out[(k+1)*OUT_DATA_WIDTH-1 : k*OUT_DATA_WIDTH] = out_byte;
        end
    endgenerate

    // -------------------------------------------------------------------------
    // Registered Output Stage (PPU Pipeline Register)
    // Eliminates long combinational timing path between PEs and BRAM write port.
    // -------------------------------------------------------------------------
    reg [OUT_DATA_WIDTH*R*C-1:0] data_out_reg;
    always @(posedge clk or posedge reset) begin
        if (reset) begin
            data_out_reg <= {(OUT_DATA_WIDTH*R*C){1'b0}};
        end else begin
            data_out_reg <= comb_data_out;
        end
    end

    assign data_out = data_out_reg;

endmodule
