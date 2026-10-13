`timescale 1ns / 1ps
// =============================================================================
// Module: multibank_dualport_mem
// Description: Multi-bank true dual-port memory architecture.
//              Contains NUM_BANKS independent memory banks each of width
//              WORD_SIZE and depth MEM_DEPTH.
//              Allows parallel single-cycle access across all banks (2048-bit
//              wide parallel bus for 16 banks x 128-bit words).
//              Port A is typically used for Host loading / weights / writeback.
//              Port B is used for single-cycle accelerator feature map reads.
// =============================================================================

module multibank_dualport_mem #(
    parameter NUM_BANKS       = 16,
    parameter WORD_SIZE       = 128,
    parameter MEM_DEPTH       = 256,
    parameter ADDR_WIDTH      = 8,   // $clog2(MEM_DEPTH)
    parameter REGISTERED_READ = 1    // 1: registered BRAM read
)(
    // Port A (Host Load/Read/Writeback)
    input  wire                                    clk_a,
    input  wire                                    en_a,
    input  wire [NUM_BANKS-1:0]                    we_a,   // Byte/Bank-granular write enable
    input  wire [ADDR_WIDTH-1:0]                   addr_a,
    input  wire [NUM_BANKS*WORD_SIZE-1:0]          din_a,
    output reg  [NUM_BANKS*WORD_SIZE-1:0]          dout_a,

    // Port B (Accelerator Array Single-Cycle Read/Write)
    input  wire                                    clk_b,
    input  wire                                    en_b,
    input  wire [NUM_BANKS-1:0]                    we_b,
    input  wire [ADDR_WIDTH-1:0]                   addr_b,
    input  wire [NUM_BANKS*WORD_SIZE-1:0]          din_b,
    output reg  [NUM_BANKS*WORD_SIZE-1:0]          dout_b
);

    genvar b;
    generate
        for (b = 0; b < NUM_BANKS; b = b + 1) begin : gen_banks

            (* ram_style = "ultra" *)
            reg [WORD_SIZE-1:0] mem [0:MEM_DEPTH-1];

`ifndef SYNTHESIS
            integer init_m;
            initial begin
                for (init_m = 0; init_m < MEM_DEPTH; init_m = init_m + 1) begin
                    mem[init_m] = {WORD_SIZE{1'b0}};
                end
            end
`endif

            wire [WORD_SIZE-1:0] bank_din_a  = din_a[(b+1)*WORD_SIZE-1 : b*WORD_SIZE];
            wire [WORD_SIZE-1:0] bank_din_b  = din_b[(b+1)*WORD_SIZE-1 : b*WORD_SIZE];

            // Port A Access (UltraRAM NO_CHANGE write/read template)
            always @(posedge clk_a) begin
                if (en_a) begin
                    if (we_a[b]) begin
                        mem[addr_a] <= bank_din_a;
                    end else begin
                        dout_a[(b+1)*WORD_SIZE-1 : b*WORD_SIZE] <= mem[addr_a];
                    end
                end
            end

            // Port B Access (UltraRAM NO_CHANGE write/read template)
            always @(posedge clk_b) begin
                if (en_b) begin
                    if (we_b[b]) begin
                        mem[addr_b] <= bank_din_b;
                    end else begin
                        dout_b[(b+1)*WORD_SIZE-1 : b*WORD_SIZE] <= mem[addr_b];
                    end
                end
            end

        end
    endgenerate

endmodule
