`timescale 1ns / 1ps
// =============================================================================
// Module: dma_sram_bridge
// Description: Streams 128-bit AXI bursts into and out of multi-bank SRAM.
//              Maps 128-bit beats directly to individual 128-bit banks:
//                bank_idx = (start_bank + beat_counter) % NUM_BANKS;
//                row_addr = start_row  + ((start_bank + beat_counter) / NUM_BANKS);
//              Features:
//                - Combinational SRAM Port A multiplexing for single-cycle writes.
//                - Direct transparent read out from SRAM to AXI Write Master.
//                - 100% backpressure tolerant without phase skew or data loss.
// =============================================================================

module dma_sram_bridge #(
    parameter NUM_BANKS  = 64,
    parameter WORD_SIZE  = 128,
    parameter ADDR_WIDTH = 8
)(
    input  wire                                aclk,
    input  wire                                aresetn,

    // Inbound Stream from AXI Read Master (DDR -> SRAM)
    input  wire [WORD_SIZE-1:0]                stream_rd_data,
    input  wire                                stream_rd_valid,
    output wire                                stream_rd_ready,
    input  wire                                rd_dma_start,
    input  wire [15:0]                         rd_sram_bank,
    input  wire [15:0]                         rd_sram_row,
    input  wire [15:0]                         rd_beats,

    // Outbound Stream to AXI Write Master (SRAM -> DDR)
    output wire [WORD_SIZE-1:0]                stream_wr_data,
    output wire                                stream_wr_valid,
    input  wire                                stream_wr_ready,
    input  wire                                wr_dma_start,
    input  wire [15:0]                         wr_sram_bank,
    input  wire [15:0]                         wr_sram_row,
    input  wire [15:0]                         wr_beats,

    // Interface to Multi-Bank Dual-Port Memory Port A
    output reg                                 ext_mem_override,
    output reg                                 ext_mem_en,
    output reg  [NUM_BANKS-1:0]                ext_mem_we,
    output reg  [ADDR_WIDTH-1:0]               ext_mem_addr,
    output reg  [NUM_BANKS*WORD_SIZE-1:0]      ext_mem_din,
    input  wire [NUM_BANKS*WORD_SIZE-1:0]      ext_mem_dout
);

    // =========================================================================
    // Sequencer Registers
    // =========================================================================
    reg [15:0]           rd_beat_count;
    reg [15:0]           rd_total_beats;
    reg [15:0]           rd_cur_bank;
    reg [ADDR_WIDTH-1:0] rd_cur_row;
    reg                  rd_active;

    reg [15:0]           wr_beat_count;
    reg [15:0]           wr_total_beats;
    reg [15:0]           wr_cur_bank;
    reg [ADDR_WIDTH-1:0] wr_cur_row;
    reg                  wr_active;
    reg                  row_read_pending;

    // Ready/Valid Handshakes
    assign stream_rd_ready = rd_active;
    assign stream_wr_valid = wr_active && !row_read_pending;

    // Outbound Stream Data Multiplexer (Static slices, 1-of-NUM_BANKS)
    reg [WORD_SIZE-1:0] wr_data_mux;
    integer b_idx;
    always @(*) begin
        wr_data_mux = ext_mem_dout[WORD_SIZE-1:0];
        for (b_idx = 0; b_idx < NUM_BANKS; b_idx = b_idx + 1) begin
            if (wr_cur_bank == b_idx) begin
                wr_data_mux = ext_mem_dout[(b_idx+1)*WORD_SIZE-1 -: WORD_SIZE];
            end
        end
    end
    assign stream_wr_data = wr_data_mux;

    // =========================================================================
    // Combinational Port A Drive (Zero latency into SRAM)
    // =========================================================================
    always @(*) begin
        ext_mem_override = 1'b0;
        ext_mem_en       = 1'b0;
        ext_mem_we       = {NUM_BANKS{1'b0}};
        ext_mem_addr     = {ADDR_WIDTH{1'b0}};
        ext_mem_din      = {(NUM_BANKS*WORD_SIZE){1'b0}};

        if (rd_active) begin
            ext_mem_override = 1'b1;
            ext_mem_addr     = rd_cur_row;
            if (stream_rd_valid) begin
                ext_mem_en   = 1'b1;
                ext_mem_we   = ({{(NUM_BANKS-1){1'b0}}, 1'b1} << rd_cur_bank);
                ext_mem_din  = {NUM_BANKS{stream_rd_data}};
            end
        end else if (wr_active) begin
            ext_mem_override = 1'b1;
            ext_mem_en       = 1'b1;
            ext_mem_we       = {NUM_BANKS{1'b0}}; // Read mode
            ext_mem_addr     = wr_cur_row;
        end
    end

    // =========================================================================
    // Sequential State & Address Progression
    // =========================================================================
    always @(posedge aclk or negedge aresetn) begin
        if (!aresetn) begin
            rd_active        <= 1'b0;
            rd_beat_count    <= 16'd0;
            rd_cur_bank      <= 16'd0;
            rd_cur_row       <= {ADDR_WIDTH{1'b0}};

            wr_active        <= 1'b0;
            wr_beat_count    <= 16'd0;
            wr_cur_bank      <= 16'd0;
            wr_cur_row       <= {ADDR_WIDTH{1'b0}};
            row_read_pending <= 1'b0;
        end else begin
            row_read_pending <= 1'b0;

            // 1. RD DMA Start Trigger
            if (rd_dma_start && (rd_beats > 0)) begin
                rd_active      <= 1'b1;
                rd_beat_count  <= 16'd0;
                rd_total_beats <= rd_beats;
                rd_cur_bank    <= rd_sram_bank;
                rd_cur_row     <= rd_sram_row[ADDR_WIDTH-1:0];
            end

            // 2. Inbound Stream Data Beat Handshake
            if (rd_active && stream_rd_valid && stream_rd_ready) begin
                rd_beat_count <= rd_beat_count + 16'd1;
                if (rd_cur_bank == (NUM_BANKS - 1)) begin
                    rd_cur_bank <= 16'd0;
                    rd_cur_row  <= rd_cur_row + 1'b1;
                end else begin
                    rd_cur_bank <= rd_cur_bank + 16'd1;
                end

                if (rd_beat_count + 16'd1 == rd_total_beats) begin
                    rd_active <= 1'b0;
                end
            end

            // 3. WR DMA Start Trigger
            if (wr_dma_start && (wr_beats > 0)) begin
                wr_active        <= 1'b1;
                wr_beat_count    <= 16'd0;
                wr_total_beats   <= wr_beats;
                wr_cur_bank      <= wr_sram_bank;
                wr_cur_row       <= wr_sram_row[ADDR_WIDTH-1:0];
                row_read_pending <= 1'b1;
            end

            // 4. Outbound Stream Data Beat Handshake
            if (wr_active && stream_wr_valid && stream_wr_ready) begin
                wr_beat_count <= wr_beat_count + 16'd1;
                if (wr_cur_bank == (NUM_BANKS - 1)) begin
                    wr_cur_bank <= 16'd0;
                    wr_cur_row  <= wr_cur_row + 1'b1;
                    if (wr_beat_count + 16'd1 < wr_total_beats) begin
                        row_read_pending <= 1'b1;
                    end
                end else begin
                    wr_cur_bank <= wr_cur_bank + 16'd1;
                end

                if (wr_beat_count + 16'd1 == wr_total_beats) begin
                    wr_active <= 1'b0;
                end
            end
        end
    end

endmodule
