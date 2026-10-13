`timescale 1ns / 1ps
// =============================================================================
// Module: c_engine
// Description: Fully parameterizable 2D systolic array compute fabric.
//              Organized as a rectangular grid of R rows by C columns of
//              identical processing elements (processing_element.v).
//              Features parallel spatial data broadcast with horizontal (HOR)
//              and vertical (VERT) systolic neighbor forwarding, and local
//              private self-feedback (ENG) accumulation.
// =============================================================================

module c_engine #(
    parameter R            = 8,
    parameter C            = 32,
    parameter x_DATA_WIDTH = 8,
    parameter w_DATA_WIDTH = 8,
    parameter b_DATA_WIDTH = 16,
    parameter y_DATA_WIDTH = 32
)(
    input  wire                               clk,
    input  wire                               reset,       // Asynchronous active-high
    input  wire                               pe_enable,   // Synchronous master enable
    input  wire                               sel0,        // Mux1 select broadcast
    input  wire [1:0]                         sel1,        // Mux2 select broadcast
    input  wire                               reg_enable,  // Clock enable for Outf_map
    input  wire                               hor_enable,  // Clock enable for HOR
    input  wire                               vert_enable, // Clock enable for VERT
    input  wire                               eng_enable,  // Clock enable for ENG
    input  wire                               bias_enable, // Bias enable broadcast
    input  wire signed [b_DATA_WIDTH-1:0]     bias_in,     // Broadcast 16-bit bias
    input  wire [x_DATA_WIDTH*R*C-1:0]        engdata_in,  // Parallel input activations
    input  wire [w_DATA_WIDTH*R*C-1:0]        weight_in,   // Parallel weights
    output wire [y_DATA_WIDTH*R*C-1:0]        engdata_out  // Parallel output activations
);

    // 2D internal interconnect wire matrices between PEs
    wire signed [y_DATA_WIDTH-1:0] hor_wires  [0:R-1][0:C-1];
    wire signed [y_DATA_WIDTH-1:0] vert_wires [0:R-1][0:C-1];
    wire signed [y_DATA_WIDTH-1:0] eng_wires  [0:R-1][0:C-1];

    genvar x, y;
    generate
        for (x = 0; x < R; x = x + 1) begin : gen_row
            for (y = 0; y < C; y = y + 1) begin : gen_col

                // Slice indexing for PE(x, y):
                // Linear index = x*C + y
                // Range: [(y+1)*WIDTH-1 + x*C*WIDTH : y*WIDTH + x*C*WIDTH]
                wire signed [x_DATA_WIDTH-1:0] pe_inf_map;
                wire signed [w_DATA_WIDTH-1:0] pe_wt;
                wire signed [y_DATA_WIDTH-1:0] pe_outf_map;
                wire signed [y_DATA_WIDTH-1:0] pe_hor_in;
                wire signed [y_DATA_WIDTH-1:0] pe_vert_in;

                assign pe_inf_map = engdata_in[(y+1)*x_DATA_WIDTH-1 + (x*C*x_DATA_WIDTH) : (y*x_DATA_WIDTH) + (x*C*x_DATA_WIDTH)];
                assign pe_wt      = weight_in  [(y+1)*w_DATA_WIDTH-1 + (x*C*w_DATA_WIDTH) : (y*w_DATA_WIDTH) + (x*C*w_DATA_WIDTH)];
                assign engdata_out[(y+1)*y_DATA_WIDTH-1 + (x*C*y_DATA_WIDTH) : (y*y_DATA_WIDTH) + (x*C*y_DATA_WIDTH)] = pe_outf_map;

                // Horizontal boundary: Col 0 receives 0, others receive from left neighbor (y-1)
                if (y == 0) begin : gen_hor_boundary
                    assign pe_hor_in = {y_DATA_WIDTH{1'b0}};
                end else begin : gen_hor_internal
                    assign pe_hor_in = hor_wires[x][y-1];
                end

                // Vertical boundary: Row 0 receives 0, others receive from upper neighbor (x-1)
                if (x == 0) begin : gen_vert_boundary
                    assign pe_vert_in = {y_DATA_WIDTH{1'b0}};
                end else begin : gen_vert_internal
                    assign pe_vert_in = vert_wires[x-1][y];
                end

                // Instantiate PE(x, y)
                processing_element #(
                    .x_DATA_WIDTH(x_DATA_WIDTH),
                    .w_DATA_WIDTH(w_DATA_WIDTH),
                    .b_DATA_WIDTH(b_DATA_WIDTH),
                    .y_DATA_WIDTH(y_DATA_WIDTH)
                ) u_pe (
                    .clk        (clk),
                    .reset      (reset),
                    .pe_enable  (pe_enable),
                    .sel0       (sel0),
                    .sel1       (sel1),
                    .reg_enable (reg_enable),
                    .hor_enable (hor_enable),
                    .vert_enable(vert_enable),
                    .eng_enable (eng_enable),
                    .bias_enable(bias_enable),
                    .inf_map    (pe_inf_map),
                    .wt         (pe_wt),
                    .bias_in    (bias_in),
                    .hor        (pe_hor_in),
                    .vert       (pe_vert_in),
                    .Outf_map   (pe_outf_map),
                    .HOR        (hor_wires[x][y]),
                    .VERT       (vert_wires[x][y]),
                    .ENG        (eng_wires[x][y])
                );

            end
        end
    endgenerate

endmodule
