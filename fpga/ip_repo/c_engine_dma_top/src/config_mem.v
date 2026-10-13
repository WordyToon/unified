`timescale 1ns / 1ps
// =============================================================================
// Module: config_mem
// Description: Dedicated 72-bit Configuration Memory for the systolic accelerator.
//              Stores layer instruction descriptors (Config Words) in the control
//              path to enable autonomous multi-layer CNN execution.
//              - Port A: Host write interface for programming layer instructions.
//              - Port B: Controller read interface addressed by Program Counter (PC).
// =============================================================================

module config_mem #(
    parameter CFG_WIDTH      = 128,
    parameter CFG_DEPTH      = 256,
    parameter CFG_ADDR_WIDTH = 8
)(
    input  wire                      clk,
    input  wire                      reset,

    // Host Write Interface
    input  wire                      cfg_wr_en,
    input  wire [CFG_ADDR_WIDTH-1:0] cfg_wr_addr,
    input  wire [CFG_WIDTH-1:0]      cfg_wr_data,

    // Controller Read Interface
    input  wire [CFG_ADDR_WIDTH-1:0] cfg_rd_addr,
    output wire [CFG_WIDTH-1:0]      cfg_rd_data
);

    // Distributed RAM for microcode instruction memory (64 x 96 bits)
    (* ram_style = "distributed" *)
    reg [CFG_WIDTH-1:0] mem [0:CFG_DEPTH-1];

    integer i;
    initial begin
        for (i = 0; i < CFG_DEPTH; i = i + 1) begin
            mem[i] = {CFG_WIDTH{1'b0}};
        end
    end

    // Synchronous Write
    always @(posedge clk) begin
        if (cfg_wr_en) begin
            mem[cfg_wr_addr] <= cfg_wr_data;
        end
    end

    // Asynchronous / Combinational Read
    assign cfg_rd_data = mem[cfg_rd_addr];

endmodule

