`timescale 1ns / 1ps
// =============================================================================
// Module: axi_lite_slave
// Description: AXI4-Lite 32-bit Slave Register File for PS / Host Configuration.
//              Provides memory-mapped control, status, DMA transfer pointers,
//              microcode configuration registers, and interrupt management.
//              Optimized for Xilinx Zynq UltraScale+ (ZCU104) and generic AXI4-Lite.
// =============================================================================

module axi_lite_slave #(
    parameter ADDR_WIDTH = 8,
    parameter DATA_WIDTH = 32
)(
    input  wire                    aclk,
    input  wire                    aresetn,

    // AXI4-Lite Write Address Channel
    input  wire [ADDR_WIDTH-1:0]   s_axi_awaddr,
    input  wire [2:0]              s_axi_awprot,
    input  wire                    s_axi_awvalid,
    output reg                     s_axi_awready,

    // AXI4-Lite Write Data Channel
    input  wire [DATA_WIDTH-1:0]   s_axi_wdata,
    input  wire [(DATA_WIDTH/8)-1:0] s_axi_wstrb,
    input  wire                    s_axi_wvalid,
    output reg                     s_axi_wready,

    // AXI4-Lite Write Response Channel
    output reg  [1:0]              s_axi_bresp,
    output reg                     s_axi_bvalid,
    input  wire                    s_axi_bready,

    // AXI4-Lite Read Address Channel
    input  wire [ADDR_WIDTH-1:0]   s_axi_araddr,
    input  wire [2:0]              s_axi_arprot,
    input  wire                    s_axi_arvalid,
    output reg                     s_axi_arready,

    // AXI4-Lite Read Data Channel
    output reg  [DATA_WIDTH-1:0]   s_axi_rdata,
    output reg  [1:0]              s_axi_rresp,
    output reg                     s_axi_rvalid,
    input  wire                    s_axi_rready,

    // User / Internal Hardware Control Outputs
    output reg                     start_rd_dma,
    output reg                     start_wr_dma,
    output reg                     start_engine,
    output reg                     soft_reset,
    output reg  [31:0]             rd_ddr_addr,
    output reg  [15:0]             rd_sram_bank,
    output reg  [15:0]             rd_sram_row,
    output reg  [15:0]             rd_beats,
    output reg  [31:0]             wt_ddr_addr,
    output reg  [31:0]             wr_ddr_addr,
    output reg  [15:0]             wr_sram_bank,
    output reg  [15:0]             wr_sram_row,
    output reg  [15:0]             wr_beats,
    output reg  [15:0]             num_chunks,

    // Microcode Configuration Port (to config_mem)
    output reg                     cfg_wr_en,
    output reg  [9:0]              cfg_wr_addr,
    output reg  [127:0]            cfg_wr_data,
    output reg                     cfg_mode_select,

    // Hardware Status Inputs
    input  wire                    rd_dma_busy,
    input  wire                    rd_dma_done,
    input  wire                    wr_dma_busy,
    input  wire                    wr_dma_done,
    input  wire                    engine_busy,
    input  wire                    engine_done,
    input  wire [31:0]             cycle_count_lo,
    input  wire [31:0]             cycle_count_hi,
    input  wire [7:0]              active_chunk_idx,
    input  wire [9:0]              active_pc,
    input  wire [4:0]              active_fsm_state,

    // Hardware Post-Processing Ports
    output reg                     start_postproc,
    output reg  [15:0]             conf_thresh,
    output reg  [15:0]             iou_thresh,
    output reg  [3:0]              det_rd_idx,
    input  wire                    postproc_busy,
    input  wire                    postproc_done,
    input  wire [4:0]              det_count,
    input  wire [15:0]             det_x1_0,  input wire [15:0] det_y1_0,
    input  wire [15:0]             det_x2_0,  input wire [15:0] det_y2_0,
    input  wire [15:0]             det_score_0, input wire [15:0] det_class_0,
    input  wire [15:0]             det_x1_1,  input wire [15:0] det_y1_1,
    input  wire [15:0]             det_x2_1,  input wire [15:0] det_y2_1,
    input  wire [15:0]             det_score_1, input wire [15:0] det_class_1,
    input  wire [15:0]             det_x1_2,  input wire [15:0] det_y1_2,
    input  wire [15:0]             det_x2_2,  input wire [15:0] det_y2_2,
    input  wire [15:0]             det_score_2, input wire [15:0] det_class_2,
    input  wire [15:0]             det_x1_3,  input wire [15:0] det_y1_3,
    input  wire [15:0]             det_x2_3,  input wire [15:0] det_y2_3,
    input  wire [15:0]             det_score_3, input wire [15:0] det_class_3,
    input  wire [15:0]             det_cur_x1, input wire [15:0] det_cur_y1,
    input  wire [15:0]             det_cur_x2, input wire [15:0] det_cur_y2,
    input  wire [15:0]             det_cur_score, input wire [15:0] det_cur_class,

    // Candidate Streaming Ports to Hardware Post-Processor
    output reg                     cand_valid,
    output reg  [4:0]              cand_grid_x,
    output reg  [4:0]              cand_grid_y,
    output reg                     cand_head_id,
    output reg  [1:0]              cand_anchor_id,
    output reg  signed [7:0]       cand_tx,
    output reg  signed [7:0]       cand_ty,
    output reg  signed [7:0]       cand_tw,
    output reg  signed [7:0]       cand_th,
    output reg  signed [7:0]       cand_to,
    output reg  signed [7:0]       cand_cls_score,
    output reg  [6:0]              cand_cls_id,

    // Dedicated Interrupt Output to PS GIC (ZCU104 pl_ps_irq0)
    output wire                    interrupt
);

    // -------------------------------------------------------------------------
    // Register Address Map
    // -------------------------------------------------------------------------
    localparam ADDR_CTRL          = 8'h00; // [0] start_rd_dma, [1] start_wr_dma, [2] start_engine, [3] soft_reset
    localparam ADDR_STATUS        = 8'h04; // [0] rd_busy, [1] rd_done, [2] wr_busy, [3] wr_done, [4] eng_busy, [5] eng_done
    localparam ADDR_IRQ_EN        = 8'h08; // [0] rd_done_ie, [1] wr_done_ie, [2] eng_done_ie
    localparam ADDR_IRQ_STATUS    = 8'h0C; // [0] rd_done_irq, [1] wr_done_irq, [2] eng_done_irq (W1C: Write 1 to clear)
    localparam ADDR_RD_DDR        = 8'h10; // DDR Read Source Address
    localparam ADDR_RD_SRAM_LOC   = 8'h14; // [15:0] Row, [31:16] Bank
    localparam ADDR_RD_BEATS      = 8'h18; // Number of 128-bit beats
    localparam ADDR_WT_DDR        = 8'h1C; // Weights DDR Physical Address [31:0]
    localparam ADDR_WR_DDR        = 8'h20; // DDR Write Destination Address
    localparam ADDR_WR_SRAM_LOC   = 8'h24; // [15:0] Row, [31:16] Bank
    localparam ADDR_WR_BEATS      = 8'h28; // Number of 128-bit beats
    localparam ADDR_NUM_CHUNKS    = 8'h2C; // Number of spatial chunks (default: 196 for ResNet-50, 1 for MNIST)
    localparam ADDR_CFG_ADDR      = 8'h30; // Microcode target write address [9:0]
    localparam ADDR_CFG_DATA0     = 8'h34; // Microcode data [31:0]
    localparam ADDR_CFG_DATA1     = 8'h38; // Microcode data [63:32]
    localparam ADDR_CFG_DATA2     = 8'h3C; // Microcode data [95:64]
    localparam ADDR_CFG_DATA3     = 8'h40; // Microcode data [127:96], auto-strobes cfg_wr_en
    localparam ADDR_MODE_CTRL     = 8'h44; // [0] cfg_mode_select (1: microcode sequencing)
    localparam ADDR_CYCLES_LO     = 8'h48; // Performance cycle counter [31:0]
    localparam ADDR_CYCLES_HI     = 8'h4C; // Performance cycle counter [63:32]
    localparam ADDR_DEBUG_STATUS  = 8'h50; // [7:0] active_chunk_idx, [17:8] active_pc, [22:18] active_fsm_state

    // Post-Processing Hardware Accelerator Registers
    localparam ADDR_DET_CTRL      = 8'h80; // [0] start_postproc
    localparam ADDR_DET_STATUS    = 8'h84; // [0] postproc_busy, [1] postproc_done
    localparam ADDR_DET_COUNT     = 8'h88; // [4:0] valid detection count (0..16)
    localparam ADDR_DET_THRESH    = 8'h8C; // [15:0] conf_thresh (Q8.8), [31:16] iou_thresh (Q8.8)
    localparam ADDR_DET_INDEX     = 8'h90; // [3:0] detection index to read out
    localparam ADDR_DET_CUR_XY1   = 8'h94; // [31:16] x1, [15:0] y1 of selected detection
    localparam ADDR_DET_CUR_XY2   = 8'h98; // [31:16] x2, [15:0] y2 of selected detection
    localparam ADDR_DET_CUR_INFO  = 8'h9C; // [31:16] score (Q8.8), [15:0] class_id of selected detection
    localparam ADDR_DET0_XY1      = 8'hA0; // Detection 0 X1/Y1
    localparam ADDR_DET0_XY2      = 8'hA4; // Detection 0 X2/Y2
    localparam ADDR_DET0_INFO     = 8'hA8; // Detection 0 Score/Class
    localparam ADDR_DET1_XY1      = 8'hB0; // Detection 1 X1/Y1
    localparam ADDR_DET1_XY2      = 8'hB4; // Detection 1 X2/Y2
    localparam ADDR_DET1_INFO     = 8'hB8; // Detection 1 Score/Class
    localparam ADDR_DET2_XY1      = 8'hC0; // Detection 2 X1/Y1
    localparam ADDR_DET2_XY2      = 8'hC4; // Detection 2 X2/Y2
    localparam ADDR_DET2_INFO     = 8'hC8; // Detection 2 Score/Class
    localparam ADDR_DET3_XY1      = 8'hD0; // Detection 3 X1/Y1
    localparam ADDR_DET3_XY2      = 8'hD4; // Detection 3 X2/Y2
    localparam ADDR_DET3_INFO     = 8'hD8; // Detection 3 Score/Class
    localparam ADDR_CAND_DATA0    = 8'hE0; // [31:16] = {cand_head_id, cand_anchor_id[1:0], cand_grid_x[4:0], cand_grid_y[4:0]}, [15:0] = {cand_to[7:0], cand_cls_score[7:0]}
    localparam ADDR_CAND_DATA1    = 8'hE4; // [31:24] = cand_cls_id[6:0], [23:16] = cand_tx[7:0], [15:8] = cand_ty[7:0], [7:0] = cand_tw[7:0]
    localparam ADDR_CAND_DATA2    = 8'hE8; // [7:0] = cand_th[7:0], [31] = trigger cand_valid pulse

    // -------------------------------------------------------------------------
    // Internal Registers
    // -------------------------------------------------------------------------
    reg [2:0] irq_en;
    reg [2:0] irq_status;

    // Sticky Done Registers
    reg rd_done_sticky;
    reg wr_done_sticky;
    reg eng_done_sticky;
    reg postproc_done_sticky;

    // Pulse generation edge detectors
    reg rd_dma_done_d;
    reg wr_dma_done_d;
    reg engine_done_d;
    reg postproc_done_d;

    wire rd_done_pulse       = rd_dma_done       && !rd_dma_done_d;
    wire wr_done_pulse       = wr_dma_done       && !wr_dma_done_d;
    wire eng_done_pulse      = engine_done      && !engine_done_d;
    wire postproc_done_pulse = postproc_done     && !postproc_done_d;

    // Write Channel State & Handshake registers
    reg                  aw_done;
    reg                  w_done;
    reg [ADDR_WIDTH-1:0] wr_addr;
    reg [DATA_WIDTH-1:0] wr_data;

    always @(posedge aclk or negedge aresetn) begin
        if (!aresetn) begin
            rd_dma_done_d        <= 1'b0;
            wr_dma_done_d        <= 1'b0;
            engine_done_d        <= 1'b0;
            postproc_done_d      <= 1'b0;
            rd_done_sticky       <= 1'b0;
            wr_done_sticky       <= 1'b0;
            eng_done_sticky      <= 1'b0;
            postproc_done_sticky <= 1'b0;
            irq_status           <= 3'b000;
        end else begin
            rd_dma_done_d   <= rd_dma_done;
            wr_dma_done_d   <= wr_dma_done;
            engine_done_d   <= engine_done;
            postproc_done_d <= postproc_done;

            if (rd_done_pulse) begin
                rd_done_sticky <= 1'b1;
                irq_status[0]  <= 1'b1;
            end
            if (wr_done_pulse) begin
                wr_done_sticky <= 1'b1;
                irq_status[1]  <= 1'b1;
            end
            if (eng_done_pulse) begin
                eng_done_sticky <= 1'b1;
                irq_status[2]   <= 1'b1;
            end
            if (postproc_done_pulse || postproc_done) begin
                postproc_done_sticky <= 1'b1;
            end

            // Clear on user write to IRQ_STATUS (W1C)
            if (aw_done && w_done && !s_axi_bvalid && (wr_addr[7:0] == ADDR_IRQ_STATUS)) begin
                if (wr_data[0]) begin
                    irq_status[0]  <= 1'b0;
                    rd_done_sticky <= 1'b0;
                end
                if (wr_data[1]) begin
                    irq_status[1]  <= 1'b0;
                    wr_done_sticky <= 1'b0;
                end
                if (wr_data[2]) begin
                    irq_status[2]   <= 1'b0;
                    eng_done_sticky <= 1'b0;
                end
            end

            // Clear postproc_done_sticky on new start_postproc write or ADDR_DET_STATUS write
            if (aw_done && w_done && !s_axi_bvalid) begin
                if (wr_addr[7:0] == ADDR_DET_CTRL && wr_data[0]) begin
                    postproc_done_sticky <= 1'b0;
                end
                if (wr_addr[7:0] == ADDR_DET_STATUS && wr_data[1]) begin
                    postproc_done_sticky <= 1'b0;
                end
            end
        end
    end

    // Level-sensitive Interrupt Generation
    assign interrupt = |(irq_status & irq_en);

    // -------------------------------------------------------------------------
    // AXI-Lite Write Channel State & Handshake
    // -------------------------------------------------------------------------

    always @(posedge aclk or negedge aresetn) begin
        if (!aresetn) begin
            s_axi_awready   <= 1'b0;
            s_axi_wready    <= 1'b0;
            s_axi_bvalid    <= 1'b0;
            s_axi_bresp     <= 2'b00;
            aw_done         <= 1'b0;
            w_done          <= 1'b0;
            wr_addr         <= {ADDR_WIDTH{1'b0}};
            wr_data         <= {DATA_WIDTH{1'b0}};
            start_rd_dma    <= 1'b0;
            start_wr_dma    <= 1'b0;
            start_engine    <= 1'b0;
            soft_reset      <= 1'b0;
            rd_ddr_addr     <= 32'd0;
            rd_sram_bank    <= 16'd0;
            rd_sram_row     <= 16'd0;
            rd_beats        <= 16'd0;
            wt_ddr_addr     <= 32'd0;
            wr_ddr_addr     <= 32'd0;
            wr_sram_bank    <= 16'd0;
            wr_sram_row     <= 16'd0;
            wr_beats        <= 16'd0;
            num_chunks      <= 16'd196; // Default to 196 for ResNet-50 backwards compatibility
            cfg_wr_en       <= 1'b0;
            cfg_wr_addr     <= 10'd0;
            cfg_wr_data     <= 128'd0;
            cfg_mode_select <= 1'b1;
            irq_en          <= 3'b000;
            cand_valid      <= 1'b0;
            cand_grid_x     <= 5'd0;
            cand_grid_y     <= 5'd0;
            cand_head_id    <= 1'b0;
            cand_anchor_id  <= 2'd0;
            cand_tx         <= 8'sd0;
            cand_ty         <= 8'sd0;
            cand_tw         <= 8'sd0;
            cand_th         <= 8'sd0;
            cand_to         <= 8'sd0;
            cand_cls_score  <= 8'sd0;
            cand_cls_id     <= 7'd0;
        end else begin
            // Default single-cycle pulse clears
            start_rd_dma   <= 1'b0;
            start_wr_dma   <= 1'b0;
            start_engine   <= 1'b0;
            start_postproc <= 1'b0;
            cfg_wr_en      <= 1'b0;
            cand_valid     <= 1'b0;

            // 1. Independent Write Address Channel Handshake
            if (!aw_done && !s_axi_bvalid) begin
                if (s_axi_awvalid && !s_axi_awready) begin
                    s_axi_awready <= 1'b1;
                    wr_addr       <= s_axi_awaddr;
                    aw_done       <= 1'b1;
                end else begin
                    s_axi_awready <= 1'b0;
                end
            end else begin
                s_axi_awready <= 1'b0;
            end

            // 2. Independent Write Data Channel Handshake
            if (!w_done && !s_axi_bvalid) begin
                if (s_axi_wvalid && !s_axi_wready) begin
                    s_axi_wready <= 1'b1;
                    wr_data      <= s_axi_wdata;
                    w_done       <= 1'b1;
                end else begin
                    s_axi_wready <= 1'b0;
                end
            end else begin
                s_axi_wready <= 1'b0;
            end

            // 3. Write Execution & Response Generation
            if (aw_done && w_done && !s_axi_bvalid) begin
                s_axi_bvalid <= 1'b1;
                s_axi_bresp  <= 2'b00; // OKAY

                case (wr_addr[7:0])
                    ADDR_CTRL: begin
                        start_rd_dma <= wr_data[0];
                        start_wr_dma <= wr_data[1];
                        start_engine <= wr_data[2];
                        soft_reset   <= wr_data[3];
                    end
                    ADDR_IRQ_EN: begin
                        irq_en <= wr_data[2:0];
                    end
                    ADDR_RD_DDR: begin
                        rd_ddr_addr <= wr_data;
                    end
                    ADDR_RD_SRAM_LOC: begin
                        rd_sram_row  <= wr_data[15:0];
                        rd_sram_bank <= wr_data[31:16];
                    end
                    ADDR_RD_BEATS: begin
                        rd_beats <= wr_data[15:0];
                    end
                    ADDR_WT_DDR: begin
                        wt_ddr_addr <= wr_data;
                    end
                    ADDR_WR_DDR: begin
                        wr_ddr_addr <= wr_data;
                    end
                    ADDR_WR_SRAM_LOC: begin
                        wr_sram_row  <= wr_data[15:0];
                        wr_sram_bank <= wr_data[31:16];
                    end
                    ADDR_WR_BEATS: begin
                        wr_beats <= wr_data[15:0];
                    end
                    ADDR_NUM_CHUNKS: begin
                        num_chunks <= wr_data[15:0];
                    end
                    ADDR_CFG_ADDR: begin
                        cfg_wr_addr <= wr_data[9:0];
                    end
                    ADDR_CFG_DATA0: begin
                        cfg_wr_data[31:0] <= wr_data;
                    end
                    ADDR_CFG_DATA1: begin
                        cfg_wr_data[63:32] <= wr_data;
                    end
                    ADDR_CFG_DATA2: begin
                        cfg_wr_data[95:64] <= wr_data;
                    end
                    ADDR_CFG_DATA3: begin
                        cfg_wr_data[127:96] <= wr_data;
                        cfg_wr_en           <= 1'b1; // Auto-pulse strobe to config_mem on DATA3 write
                    end
                    ADDR_MODE_CTRL: begin
                        cfg_mode_select <= wr_data[0];
                    end
                    ADDR_DET_CTRL: begin
                        start_postproc <= wr_data[0];
                    end
                    ADDR_DET_THRESH: begin
                        conf_thresh <= wr_data[15:0];
                        iou_thresh  <= wr_data[31:16];
                    end
                    ADDR_DET_INDEX: begin
                        det_rd_idx <= wr_data[3:0];
                    end
                    ADDR_CAND_DATA0: begin
                        cand_head_id   <= wr_data[31];
                        cand_anchor_id <= wr_data[30:29];
                        cand_grid_x    <= wr_data[28:24];
                        cand_grid_y    <= wr_data[23:19];
                        cand_to        <= wr_data[15:8];
                        cand_cls_score <= wr_data[7:0];
                    end
                    ADDR_CAND_DATA1: begin
                        cand_cls_id    <= wr_data[30:24];
                        cand_tx        <= wr_data[23:16];
                        cand_ty        <= wr_data[15:8];
                        cand_tw        <= wr_data[7:0];
                    end
                    ADDR_CAND_DATA2: begin
                        cand_th        <= wr_data[7:0];
                        cand_valid     <= wr_data[31]; // bit 31=1 pulses cand_valid for 1 cycle
                    end
                    default: ;
                endcase
            end else if (s_axi_bvalid && s_axi_bready) begin
                s_axi_bvalid <= 1'b0;
                aw_done      <= 1'b0;
                w_done       <= 1'b0;
            end
        end
    end

    // -------------------------------------------------------------------------
    // AXI-Lite Read Channel State & Handshake
    // -------------------------------------------------------------------------
    reg                  ar_done;
    reg [ADDR_WIDTH-1:0] rd_addr;

    always @(posedge aclk or negedge aresetn) begin
        if (!aresetn) begin
            s_axi_arready <= 1'b0;
            s_axi_rvalid  <= 1'b0;
            s_axi_rresp   <= 2'b00;
            s_axi_rdata   <= 32'd0;
            ar_done       <= 1'b0;
            rd_addr       <= {ADDR_WIDTH{1'b0}};
        end else begin
            // 1. Read Address Channel Handshake
            if (!ar_done && !s_axi_rvalid) begin
                if (s_axi_arvalid && !s_axi_arready) begin
                    s_axi_arready <= 1'b1;
                    rd_addr       <= s_axi_araddr;
                    ar_done       <= 1'b1;
                end else begin
                    s_axi_arready <= 1'b0;
                end
            end else begin
                s_axi_arready <= 1'b0;
            end

            // 2. Read Data Channel Handshake & Multiplexing
            if (ar_done && !s_axi_rvalid) begin
                s_axi_rvalid <= 1'b1;
                s_axi_rresp  <= 2'b00; // OKAY

                case (rd_addr[7:0])
                    ADDR_CTRL:        s_axi_rdata <= {28'd0, soft_reset, 1'b0, 1'b0, 1'b0};
                    ADDR_STATUS:      s_axi_rdata <= {26'd0, eng_done_sticky, engine_busy, wr_done_sticky, wr_dma_busy, rd_done_sticky, rd_dma_busy};
                    ADDR_IRQ_EN:      s_axi_rdata <= {29'd0, irq_en};
                    ADDR_IRQ_STATUS:  s_axi_rdata <= {29'd0, irq_status};
                    ADDR_RD_DDR:      s_axi_rdata <= rd_ddr_addr;
                    ADDR_RD_SRAM_LOC: s_axi_rdata <= {rd_sram_bank, rd_sram_row};
                    ADDR_RD_BEATS:    s_axi_rdata <= {16'd0, rd_beats};
                    ADDR_WT_DDR:      s_axi_rdata <= wt_ddr_addr;
                    ADDR_WR_DDR:      s_axi_rdata <= wr_ddr_addr;
                    ADDR_WR_SRAM_LOC: s_axi_rdata <= {wr_sram_bank, wr_sram_row};
                    ADDR_WR_BEATS:    s_axi_rdata <= {16'd0, wr_beats};
                    ADDR_NUM_CHUNKS:  s_axi_rdata <= {16'd0, num_chunks};
                    ADDR_CFG_ADDR:    s_axi_rdata <= {22'd0, cfg_wr_addr};
                    ADDR_CFG_DATA0:   s_axi_rdata <= cfg_wr_data[31:0];
                    ADDR_CFG_DATA1:   s_axi_rdata <= cfg_wr_data[63:32];
                    ADDR_CFG_DATA2:   s_axi_rdata <= cfg_wr_data[95:64];
                    ADDR_CFG_DATA3:   s_axi_rdata <= cfg_wr_data[127:96];
                    ADDR_MODE_CTRL:   s_axi_rdata <= {31'd0, cfg_mode_select};
                    ADDR_CYCLES_LO:   s_axi_rdata <= cycle_count_lo;
                    ADDR_CYCLES_HI:   s_axi_rdata <= cycle_count_hi;
                    ADDR_DEBUG_STATUS:s_axi_rdata <= {9'd0, active_fsm_state, active_pc, active_chunk_idx};

                    // Post-Processing Registers
                    ADDR_DET_CTRL:     s_axi_rdata <= 32'd0;
                    ADDR_DET_STATUS:   s_axi_rdata <= {30'd0, (postproc_done | postproc_done_sticky), postproc_busy};
                    ADDR_DET_COUNT:    s_axi_rdata <= {27'd0, det_count};
                    ADDR_DET_THRESH:   s_axi_rdata <= {iou_thresh, conf_thresh};
                    ADDR_DET_INDEX:    s_axi_rdata <= {28'd0, det_rd_idx};
                    ADDR_DET_CUR_XY1:  s_axi_rdata <= {det_cur_x1, det_cur_y1};
                    ADDR_DET_CUR_XY2:  s_axi_rdata <= {det_cur_x2, det_cur_y2};
                    ADDR_DET_CUR_INFO: s_axi_rdata <= {det_cur_score, det_cur_class};

                    ADDR_DET0_XY1:     s_axi_rdata <= {det_x1_0, det_y1_0};
                    ADDR_DET0_XY2:     s_axi_rdata <= {det_x2_0, det_y2_0};
                    ADDR_DET0_INFO:    s_axi_rdata <= {det_score_0, det_class_0};
                    ADDR_DET1_XY1:     s_axi_rdata <= {det_x1_1, det_y1_1};
                    ADDR_DET1_XY2:     s_axi_rdata <= {det_x2_1, det_y2_1};
                    ADDR_DET1_INFO:    s_axi_rdata <= {det_score_1, det_class_1};
                    ADDR_DET2_XY1:     s_axi_rdata <= {det_x1_2, det_y1_2};
                    ADDR_DET2_XY2:     s_axi_rdata <= {det_x2_2, det_y2_2};
                    ADDR_DET2_INFO:    s_axi_rdata <= {det_score_2, det_class_2};
                    ADDR_DET3_XY1:     s_axi_rdata <= {det_x1_3, det_y1_3};
                    ADDR_DET3_XY2:     s_axi_rdata <= {det_x2_3, det_y2_3};
                    ADDR_DET3_INFO:    s_axi_rdata <= {det_score_3, det_class_3};
                    ADDR_CAND_DATA0:   s_axi_rdata <= {cand_head_id, cand_anchor_id, cand_grid_x, cand_grid_y, 3'b000, cand_to, cand_cls_score};
                    ADDR_CAND_DATA1:   s_axi_rdata <= {1'b0, cand_cls_id, cand_tx, cand_ty, cand_tw};
                    ADDR_CAND_DATA2:   s_axi_rdata <= {31'd0, cand_th};

                    default:           s_axi_rdata <= 32'hDEADBEEF;
                endcase
            end else if (s_axi_rvalid && s_axi_rready) begin
                s_axi_rvalid <= 1'b0;
                ar_done      <= 1'b0;
            end
        end
    end

endmodule
