`timescale 1ns / 1ps
// =============================================================================
// Module: axi4_master_wr
// Description: Autonomous 128-bit AXI4 Write Master for DDR Memory Writeback.
//              Offloads results from the accelerator SRAM bridge without vendor IPs.
//              Handles:
//                - Configurable burst lengths up to 256 beats (AxLEN = 0..255).
//                - Mandatory 4KB AXI boundary splitting.
//                - Proper write strobe generation and BVALID/BREADY response.
// =============================================================================

module axi4_master_wr #(
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

    // Stream Input from SRAM Bridge
    input  wire [C_M_AXI_DATA_WIDTH-1:0]       stream_data,
    input  wire                                stream_valid,
    output wire                                stream_ready,

    // AXI4 Master Write Address Channel
    output wire [3:0]                          m_axi_awid,
    output reg  [C_M_AXI_ADDR_WIDTH-1:0]       m_axi_awaddr,
    output reg  [7:0]                          m_axi_awlen,
    output wire [2:0]                          m_axi_awsize,
    output wire [1:0]                          m_axi_awburst,
    output wire                                m_axi_awlock,
    output wire [3:0]                          m_axi_awcache,
    output wire [2:0]                          m_axi_awprot,
    output wire [3:0]                          m_axi_awqos,
    output reg                                 m_axi_awvalid,
    input  wire                                m_axi_awready,

    // AXI4 Master Write Data Channel
    output wire [C_M_AXI_DATA_WIDTH-1:0]       m_axi_wdata,
    output wire [(C_M_AXI_DATA_WIDTH/8)-1:0]   m_axi_wstrb,
    output wire                                m_axi_wlast,
    output wire                                m_axi_wvalid,
    input  wire                                m_axi_wready,

    // AXI4 Master Write Response Channel
    input  wire [3:0]                          m_axi_bid,
    input  wire [1:0]                          m_axi_bresp,
    input  wire                                m_axi_bvalid,
    output reg                                 m_axi_bready
);

    // Static AXI parameters for 128-bit (16-byte) INCR bursts
    assign m_axi_awid    = 4'd0;
    assign m_axi_awsize  = 3'b100;  // 2^4 = 16 bytes = 128 bits
    assign m_axi_awburst = 2'b01;   // INCR
    assign m_axi_awlock  = 1'b0;
    assign m_axi_awcache = 4'b0011; // Normal, non-cacheable bufferable
    assign m_axi_awprot  = 3'b000;
    assign m_axi_awqos   = 4'b0000;
    assign m_axi_wstrb   = 16'hFFFF;

    // FSM States
    localparam S_IDLE        = 3'd0;
    localparam S_AW_ADDR     = 3'd1;
    localparam S_WRITE_BURST = 3'd2;
    localparam S_WAIT_RESP   = 3'd3;
    localparam S_DONE        = 3'd4;

    reg [2:0] state;

    reg [C_M_AXI_ADDR_WIDTH-1:0] current_addr;
    reg [15:0]                   beats_remaining;
    reg [7:0]                    burst_beat_count;
    reg [7:0]                    target_burst_len;
    reg [8:0]                    next_len;

    // 4KB boundary calculation
    wire [11:0] page_offset_bytes = current_addr[11:0];
    wire [12:0] bytes_to_4k       = 13'd4096 - {1'b0, page_offset_bytes};
    wire [8:0]  max_beats_4k      = bytes_to_4k[12:4];

    // Stream to Write Data connection (Strict Single-Cycle Synchronized Handshake)
    assign m_axi_wdata  = stream_data;
    assign m_axi_wvalid = (state == S_WRITE_BURST) && stream_valid;
    assign stream_ready = (state == S_WRITE_BURST) && m_axi_wready;
    assign m_axi_wlast  = (burst_beat_count == target_burst_len);

    always @(posedge aclk or negedge aresetn) begin
        if (!aresetn) begin
            state            <= S_IDLE;
            busy             <= 1'b0;
            done             <= 1'b0;
            m_axi_awvalid    <= 1'b0;
            m_axi_awaddr     <= {C_M_AXI_ADDR_WIDTH{1'b0}};
            m_axi_awlen      <= 8'd0;
            m_axi_bready     <= 1'b0;
            current_addr     <= {C_M_AXI_ADDR_WIDTH{1'b0}};
            beats_remaining  <= 16'd0;
            burst_beat_count <= 8'd0;
            target_burst_len <= 8'd0;
            next_len         <= 9'd0;
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
                            state           <= S_AW_ADDR;
                        end else begin
                            done  <= 1'b1;
                            state <= S_IDLE;
                        end
                    end
                end

                S_AW_ADDR: begin
                    busy <= 1'b1;
                    if (!m_axi_awvalid) begin
                        next_len = (beats_remaining < 16'd256) ? beats_remaining[8:0] : 9'd256;
                        if (next_len > max_beats_4k) begin
                            next_len = max_beats_4k;
                        end
                        if (next_len == 9'd0) next_len = 9'd1;

                        target_burst_len <= next_len[7:0] - 8'd1;
                        m_axi_awaddr     <= current_addr;
                        m_axi_awlen      <= next_len[7:0] - 8'd1;
                        m_axi_awvalid    <= 1'b1;
                    end else if (m_axi_awvalid && m_axi_awready) begin
                        m_axi_awvalid    <= 1'b0;
                        burst_beat_count <= 8'd0;
                        state            <= S_WRITE_BURST;
                    end
                end

                S_WRITE_BURST: begin
                    busy <= 1'b1;
                    if (m_axi_wvalid && m_axi_wready) begin
                        current_addr     <= current_addr + 32'd16;
                        beats_remaining  <= beats_remaining - 16'd1;
                        burst_beat_count <= burst_beat_count + 8'd1;

                        if (m_axi_wlast) begin
                            m_axi_bready <= 1'b1;
                            state        <= S_WAIT_RESP;
                        end
                    end
                end

                S_WAIT_RESP: begin
                    busy <= 1'b1;
                    if (m_axi_bvalid && m_axi_bready) begin
                        m_axi_bready <= 1'b0;
                        if (beats_remaining == 16'd0) begin
                            state <= S_DONE;
                        end else begin
                            state <= S_AW_ADDR;
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
