`timescale 1ns / 1ps
// =============================================================================
// Module: processing_element
// Description: Parameterized signed Multiply-Accumulate Processing Element (PE)
//              with dynamic spatial operand steering and synchronous register
//              dynamics for systolic computing.
// =============================================================================

module processing_element #(
    parameter x_DATA_WIDTH = 8,
    parameter w_DATA_WIDTH = 8,
    parameter b_DATA_WIDTH = 16,
    parameter y_DATA_WIDTH = 32
)(
    input  wire                     clk,
    input  wire                     reset,       // Asynchronous active-high
    input  wire                     pe_enable,   // Synchronous master PE enable
    input  wire                     sel0,        // Mux1: 0=mult_out, 1=Outf_map feedback
    input  wire [1:0]               sel1,        // Mux2: 00=mux0, 01=hor, 10=vert, 11=ENG
    input  wire                     reg_enable,  // Clock enable for Outf_map
    input  wire                     hor_enable,  // Clock enable for HOR (clears to 0 when 0)
    input  wire                     vert_enable, // Clock enable for VERT (clears when eng_en=1)
    input  wire                     eng_enable,  // Clock enable for ENG local feedback
    input  wire                     bias_enable, // Bias mode: 1=bias_sign_ext, 0=0
    input  wire signed [x_DATA_WIDTH-1:0] inf_map, // Signed input activation
    input  wire signed [w_DATA_WIDTH-1:0] wt,      // Signed weight coefficient
    input  wire signed [b_DATA_WIDTH-1:0] bias_in, // Signed 16-bit bias input
    input  wire signed [y_DATA_WIDTH-1:0] hor,     // Horizontal input from left neighbor
    input  wire signed [y_DATA_WIDTH-1:0] vert,    // Vertical input from upper neighbor
    output reg  signed [y_DATA_WIDTH-1:0] Outf_map,// Primary output feature map register
    output reg  signed [y_DATA_WIDTH-1:0] HOR,     // Horizontal shift register (to right neighbor)
    output reg  signed [y_DATA_WIDTH-1:0] VERT,    // Vertical shift register (to bottom neighbor)
    output reg  signed [y_DATA_WIDTH-1:0] ENG      // Local private accumulation register
);

    // -------------------------------------------------------------------------
    // Combinational Datapath
    // -------------------------------------------------------------------------
    (* use_dsp = "yes" *)
    wire signed [x_DATA_WIDTH+w_DATA_WIDTH-1:0] mult_out;
    assign mult_out = inf_map * wt;

    // Sign extension of mult_out to y_DATA_WIDTH
    wire signed [y_DATA_WIDTH-1:0] mult_out_ext;
    generate
        if (y_DATA_WIDTH > (x_DATA_WIDTH + w_DATA_WIDTH)) begin : gen_mult_ext
            assign mult_out_ext = {{(y_DATA_WIDTH - (x_DATA_WIDTH + w_DATA_WIDTH)){mult_out[x_DATA_WIDTH + w_DATA_WIDTH - 1]}}, mult_out};
        end else begin : gen_mult_trunc
            assign mult_out_ext = mult_out[y_DATA_WIDTH-1:0];
        end
    endgenerate

    // Bias Mux (Mux0) - Directly sign-extends 16-bit bias without shifting
    wire signed [y_DATA_WIDTH-1:0] bias_sign_ext;
    generate
        if (y_DATA_WIDTH > b_DATA_WIDTH) begin : gen_bias_ext
            assign bias_sign_ext = {{(y_DATA_WIDTH - b_DATA_WIDTH){bias_in[b_DATA_WIDTH-1]}}, bias_in};
        end else begin : gen_bias_noext
            assign bias_sign_ext = bias_in[y_DATA_WIDTH-1:0];
        end
    endgenerate

    wire signed [y_DATA_WIDTH-1:0] mux0_out;
    assign mux0_out = (bias_enable == 1'b0) ? {y_DATA_WIDTH{1'b0}} : bias_sign_ext;

    // Mux1: selects between multiplier output and Outf_map feedback
    wire signed [y_DATA_WIDTH-1:0] mux1_out;
    assign mux1_out = (sel0 == 1'b0) ? mult_out_ext : Outf_map;

    // Mux2: selects between bias/zero, hor neighbor, vert neighbor, and ENG feedback
    reg signed [y_DATA_WIDTH-1:0] mux2_out;
    always @(*) begin
        case (sel1)
            2'b00: mux2_out = mux0_out;
            2'b01: mux2_out = hor;
            2'b10: mux2_out = vert;
            2'b11: mux2_out = ENG;
        endcase
    end

    // Adder
    (* use_dsp = "yes" *)
    wire signed [y_DATA_WIDTH-1:0] adder_out;
    assign adder_out = mux1_out + mux2_out;

    // -------------------------------------------------------------------------
    // Synchronous Register Dynamics
    // -------------------------------------------------------------------------
    always @(posedge clk) begin
        if (reset) begin
            Outf_map <= {y_DATA_WIDTH{1'b0}};
        end else if (reg_enable) begin
            Outf_map <= adder_out;
        end
        // Outf_map holds value when reg_enable is low

        if (reset || !pe_enable) begin
            HOR <= {y_DATA_WIDTH{1'b0}};
        end else if (hor_enable) begin
            HOR <= adder_out;
        end else begin
            HOR <= {y_DATA_WIDTH{1'b0}}; // Explicit synchronous clear when !hor_enable
        end

        if (reset || !pe_enable || eng_enable) begin
            VERT <= {y_DATA_WIDTH{1'b0}}; // Explicit clear when eng_enable is active or !pe_enable
        end else if (vert_enable) begin
            VERT <= adder_out;
        end
        // VERT holds when idle (!vert_enable && !eng_enable)

        if (reset || !pe_enable) begin
            ENG <= {y_DATA_WIDTH{1'b0}};
        end else if (eng_enable) begin
            ENG <= adder_out;
        end
        // ENG holds value when eng_enable is low
    end

endmodule
