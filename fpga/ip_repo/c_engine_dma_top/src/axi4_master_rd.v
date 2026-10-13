`timescale 1ns / 1ps
// =============================================================================
// Module: axi4_master_rd
// Description: Autonomous 128-bit AXI4 Read Master for DDR Memory Fetch.
//              Streams data into the accelerator SRAM bridge without vendor IPs.
//              Handles:
//                - Configurable burst lengths up to 256 beats (AxLEN = 0..255).
//                - Mandatory 4KB AXI boundary splitting.
//                - Clean handshake with backpressure handling (RREADY/RVALID).
// =============================================================================

module axi4_master_rd #(
    parameter C_M_AXI_ADDR_WIDTH = 32,
    parameter C_M_AXI_DATA_WIDTH = 128
)(
    input  wire                                aclk,
    input  wire                                aresetn,

    // Control from Register File
    input  wire                                start,
    input  wire [C_M_AXI_ADDR_WIDTH-1:0]       base_addr,
    input  wire [15:0]                         total_beats,
    output reg                                 busy,
    output reg                                 done,

    // Stream Output to SRAM Bridge
    output wire [C_M_AXI_DATA_WIDTH-1:0]       stream_data,
    output wire                                stream_valid,
    input  wire                                stream_ready,

    // AXI4 Master Read Address Channel
    output wire [3:0]                          m_axi_arid,
    output reg  [C_M_AXI_ADDR_WIDTH-1:0]       m_axi_araddr,
    output reg  [7:0]                          m_axi_arlen,
    output wire [2:0]                          m_axi_arsize,
    output wire [1:0]                          m_axi_arburst,
    output wire                                m_axi_arlock,
    output wire [3:0]                          m_axi_arcache,
    output wire [2:0]                          m_axi_arprot,
    output wire [3:0]                          m_axi_arqos,
    output reg                                 m_axi_arvalid,
    input  wire                                m_axi_arready,

    // AXI4 Master Read Data Channel
    input  wire [3:0]                          m_axi_rid,
    input  wire [C_M_AXI_DATA_WIDTH-1:0]       m_axi_rdata,
    input  wire [1:0]                          m_axi_rresp,
    input  wire                                m_axi_rlast,
    input  wire                                m_axi_rvalid,
    output wire                                m_axi_rready
);

    // Static AXI parameters for 128-bit (16-byte) INCR bursts
    assign m_axi_arid    = 4'd0;
    assign m_axi_arsize  = 3'b100;  // 2^4 = 16 bytes = 128 bits
    assign m_axi_arburst = 2'b01;   // INCR
    assign m_axi_arlock  = 1'b0;
    assign m_axi_arcache = 4'b0011; // Normal, non-cacheable bufferable
    assign m_axi_arprot  = 3'b000;
    assign m_axi_arqos   = 4'b0000;

    // FSM States
    localparam S_IDLE       = 2'd0;
    localparam S_AR_ADDR    = 2'd1;
    localparam S_READ_BURST = 2'd2;
    localparam S_DONE       = 2'd3;

    reg [1:0] state;

    reg [C_M_AXI_ADDR_WIDTH-1:0] current_addr;
    reg [15:0]                   beats_remaining;
    reg [7:0]                    current_burst_beats; // 1..256
    reg [8:0]                    next_len;

    // 4KB boundary calculation
    wire [11:0] page_offset_bytes = current_addr[11:0];
    wire [12:0] bytes_to_4k       = 13'd4096 - {1'b0, page_offset_bytes};
    wire [8:0]  max_beats_4k      = bytes_to_4k[12:4];

    // Stream connection
    assign stream_data  = m_axi_rdata;
    assign stream_valid = (state == S_READ_BURST) && m_axi_rvalid;
    assign m_axi_rready = (state == S_READ_BURST) && stream_ready;

    always @(posedge aclk or negedge aresetn) begin
        if (!aresetn) begin
            state               <= S_IDLE;
            busy                <= 1'b0;
            done                <= 1'b0;
            m_axi_arvalid       <= 1'b0;
            m_axi_araddr        <= {C_M_AXI_ADDR_WIDTH{1'b0}};
            m_axi_arlen         <= 8'd0;
            current_addr        <= {C_M_AXI_ADDR_WIDTH{1'b0}};
            beats_remaining     <= 16'd0;
            current_burst_beats <= 8'd0;
            next_len            <= 9'd0;
        end else begin
            done <= 1'b0;

            case (state)
                S_IDLE: begin
                    busy <= 1'b0;
                    if (start) begin
                        if (total_beats > 0) begin
                            busy            <= 1'b1;
                            current_addr    <= base_addr;
                            beats_remaining <= total_beats;
                            state           <= S_AR_ADDR;
                        end else begin
                            done  <= 1'b1;
                            state <= S_IDLE;
                        end
                    end
                end

                S_AR_ADDR: begin
                    busy <= 1'b1;
                    if (!m_axi_arvalid) begin
                        next_len = (beats_remaining < 16'd256) ? beats_remaining[8:0] : 9'd256;
                        if (next_len > max_beats_4k) begin
                            next_len = max_beats_4k;
                        end
                        if (next_len == 9'd0) next_len = 9'd1;

                        current_burst_beats <= next_len[7:0];
                        m_axi_araddr        <= current_addr;
                        m_axi_arlen         <= next_len[7:0] - 8'd1;
                        m_axi_arvalid       <= 1'b1;
                    end else if (m_axi_arvalid && m_axi_arready) begin
                        m_axi_arvalid <= 1'b0;
                        state         <= S_READ_BURST;
                    end
                end

                S_READ_BURST: begin
                    busy <= 1'b1;
                    if (m_axi_rvalid && m_axi_rready) begin
                        beats_remaining <= beats_remaining - 16'd1;
                        current_addr    <= current_addr + 32'd16;

                        if (m_axi_rlast) begin
                            if (beats_remaining == 16'd1) begin
                                state <= S_DONE;
                            end else begin
                                state <= S_AR_ADDR;
                            end
                        end
                    end
                end

                S_DONE: begin
                    busy  <= 1'b0;
                    done  <= 1'b1;
                    state <= S_IDLE;
                end
            endcase
        end
    end

endmodule
