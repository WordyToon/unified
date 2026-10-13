`timescale 1ns / 1ps
// =============================================================================
// Module: axi4_ddr_model
// Description: Behavioral Pure-Verilog 128-bit AXI4 Dual-Port DDR Memory Model.
//              Provides high-fidelity memory emulation for simulation:
//                - Port 0: Read Slave Channel (connects to M0_AXI_RD).
//                - Port 1: Write Slave Channel (connects to M1_AXI_WR).
//              Features:
//                - Realistic programmable read/write latencies.
//                - 16 MB memory space (1,048,576 words x 128 bits = 16,777,216 Bytes).
//                - Byte strobe handling for unaligned/partial writes.
// =============================================================================

module axi4_ddr_model #(
    parameter MEM_SIZE_WORDS    = 1048576, // 16 MB at 16 bytes/word
    parameter READ_LATENCY      = 2,       // Cycles before RVALID
    parameter C_AXI_ADDR_WIDTH  = 32,
    parameter C_AXI_DATA_WIDTH  = 128
)(
    input  wire                             aclk,
    input  wire                             aresetn,

    // Port 0: Read Slave Interface (from M0_AXI_RD)
    input  wire [3:0]                       s0_axi_arid,
    input  wire [C_AXI_ADDR_WIDTH-1:0]      s0_axi_araddr,
    input  wire [7:0]                       s0_axi_arlen,
    input  wire [2:0]                       s0_axi_arsize,
    input  wire [1:0]                       s0_axi_arburst,
    input  wire                             s0_axi_arvalid,
    output reg                              s0_axi_arready,

    output reg  [3:0]                       s0_axi_rid,
    output reg  [C_AXI_DATA_WIDTH-1:0]      s0_axi_rdata,
    output reg  [1:0]                       s0_axi_rresp,
    output reg                              s0_axi_rlast,
    output reg                              s0_axi_rvalid,
    input  wire                             s0_axi_rready,

    // Port 1: Write Slave Interface (from M1_AXI_WR)
    input  wire [3:0]                       s1_axi_awid,
    input  wire [C_AXI_ADDR_WIDTH-1:0]      s1_axi_awaddr,
    input  wire [7:0]                       s1_axi_awlen,
    input  wire [2:0]                       s1_axi_awsize,
    input  wire [1:0]                       s1_axi_awburst,
    input  wire                             s1_axi_awvalid,
    output reg                              s1_axi_awready,

    input  wire [C_AXI_DATA_WIDTH-1:0]      s1_axi_wdata,
    input  wire [(C_AXI_DATA_WIDTH/8)-1:0]  s1_axi_wstrb,
    input  wire                             s1_axi_wlast,
    input  wire                             s1_axi_wvalid,
    output reg                              s1_axi_wready,

    output reg  [3:0]                       s1_axi_bid,
    output reg  [1:0]                       s1_axi_bresp,
    output reg                              s1_axi_bvalid,
    input  wire                             s1_axi_bready
);

    // Memory array: 16 MB = 1M words of 128 bits
    reg [C_AXI_DATA_WIDTH-1:0] mem [0:MEM_SIZE_WORDS-1];

    integer init_m;
    initial begin
        for (init_m = 0; init_m < MEM_SIZE_WORDS; init_m = init_m + 1) begin
            mem[init_m] = {C_AXI_DATA_WIDTH{1'b0}};
        end
    end

    // Helper functions for tasks
    task write_word(input [C_AXI_ADDR_WIDTH-1:0] byte_addr, input [C_AXI_DATA_WIDTH-1:0] data);
        integer word_idx;
        begin
            word_idx = (byte_addr >> 4) % MEM_SIZE_WORDS;
            mem[word_idx] = data;
        end
    endtask

    function [C_AXI_DATA_WIDTH-1:0] read_word(input [C_AXI_ADDR_WIDTH-1:0] byte_addr);
        integer word_idx;
        begin
            word_idx = (byte_addr >> 4) % MEM_SIZE_WORDS;
            read_word = mem[word_idx];
        end
    endfunction

    // -------------------------------------------------------------------------
    // Read Channel Logic
    // -------------------------------------------------------------------------
    reg [C_AXI_ADDR_WIDTH-1:0] rd_addr;
    reg [7:0]                  rd_len;
    reg [7:0]                  rd_count;
    reg [3:0]                  rd_id;
    reg [3:0]                  rd_lat_counter;
    reg                        rd_active;

    always @(posedge aclk or negedge aresetn) begin
        if (!aresetn) begin
            s0_axi_arready <= 1'b1;
            s0_axi_rvalid  <= 1'b0;
            s0_axi_rdata   <= {C_AXI_DATA_WIDTH{1'b0}};
            s0_axi_rresp   <= 2'b00;
            s0_axi_rlast   <= 1'b0;
            s0_axi_rid     <= 4'd0;
            rd_active      <= 1'b0;
            rd_lat_counter <= 4'd0;
            rd_count       <= 8'd0;
        end else begin
            // Address Handshake
            if (s0_axi_arvalid && s0_axi_arready) begin
                rd_addr        <= s0_axi_araddr;
                rd_len         <= s0_axi_arlen;
                rd_id          <= s0_axi_arid;
                rd_count       <= 8'd0;
                rd_lat_counter <= READ_LATENCY;
                rd_active      <= 1'b1;
                s0_axi_arready <= 1'b0; // De-assert during burst
            end

            // Read Data Bursts with Latency
            if (rd_active) begin
                if (rd_lat_counter > 0) begin
                    rd_lat_counter <= rd_lat_counter - 1'b1;
                end else begin
                    // Emit Read Beat
                    if (!s0_axi_rvalid || s0_axi_rready) begin
                        s0_axi_rvalid <= 1'b1;
                        s0_axi_rid    <= rd_id;
                        s0_axi_rresp  <= 2'b00; // OKAY
                        s0_axi_rdata  <= mem[(rd_addr >> 4) % MEM_SIZE_WORDS];
                        s0_axi_rlast  <= (rd_count == rd_len);

                        rd_addr  <= rd_addr + 32'd16;
                        rd_count <= rd_count + 8'd1;

                        if (rd_count == rd_len) begin
                            rd_active <= 1'b0;
                        end
                    end
                end
            end else if (s0_axi_rvalid && s0_axi_rready) begin
                s0_axi_rvalid  <= 1'b0;
                s0_axi_rlast   <= 1'b0;
                s0_axi_arready <= 1'b1; // Ready for next burst
            end
        end
    end

    // -------------------------------------------------------------------------
    // Write Channel Logic
    // -------------------------------------------------------------------------
    reg [C_AXI_ADDR_WIDTH-1:0] wr_addr;
    reg [7:0]                  wr_len;
    reg [7:0]                  wr_count;
    reg [3:0]                  wr_id;
    reg                        wr_active;
    integer i;

    always @(posedge aclk or negedge aresetn) begin
        if (!aresetn) begin
            s1_axi_awready <= 1'b1;
            s1_axi_wready  <= 1'b0;
            s1_axi_bvalid  <= 1'b0;
            s1_axi_bresp   <= 2'b00;
            s1_axi_bid     <= 4'd0;
            wr_active      <= 1'b0;
            wr_count       <= 8'd0;
        end else begin
            // Address Handshake
            if (s1_axi_awvalid && s1_axi_awready) begin
                wr_addr        <= s1_axi_awaddr;
                wr_len         <= s1_axi_awlen;
                wr_id          <= s1_axi_awid;
                wr_count       <= 8'd0;
                wr_active      <= 1'b1;
                s1_axi_awready <= 1'b0;
                s1_axi_wready  <= 1'b1;
            end

            // Data Handshake
            if (wr_active && s1_axi_wvalid && s1_axi_wready) begin
                // Byte-strobe write
                for (i = 0; i < 16; i = i + 1) begin
                    if (s1_axi_wstrb[i]) begin
                        mem[(wr_addr >> 4) % MEM_SIZE_WORDS][(i*8)+:8] <= s1_axi_wdata[(i*8)+:8];
                    end
                end

                wr_addr  <= wr_addr + 32'd16;
                wr_count <= wr_count + 8'd1;

                if (s1_axi_wlast || (wr_count == wr_len)) begin
                    wr_active     <= 1'b0;
                    s1_axi_wready <= 1'b0;
                    s1_axi_bvalid <= 1'b1;
                    s1_axi_bid    <= wr_id;
                    s1_axi_bresp  <= 2'b00; // OKAY
                end
            end

            // Response Handshake
            if (s1_axi_bvalid && s1_axi_bready) begin
                s1_axi_bvalid  <= 1'b0;
                s1_axi_awready <= 1'b1;
            end
        end
    end

endmodule
