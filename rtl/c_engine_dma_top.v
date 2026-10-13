`timescale 1ns / 1ps
`include "hw_config.vh"
// =============================================================================
// Module: c_engine_dma_top
// Description: Complete Top-Level System with Dual 128-bit AXI-4 Masters,
//              32-bit AXI4-Lite Slave Configuration Interface, and Dedicated
//              Level-Sensitive Interrupt Output for Zynq UltraScale+ ZCU104.
//              Contains:
//                1. axi_lite_slave: Host PS Control & Status Register File.
//                2. axi4_master_rd: 128-bit AXI4 Read Master (DDR -> SRAM).
//                3. axi4_master_wr: 128-bit AXI4 Write Master (SRAM -> DDR).
//                4. dma_sram_bridge: Multi-Bank SRAM Streaming Bridge.
//                5. c_engine_top: 2D Systolic Array + Multi-bank Dual-port SRAM.
// =============================================================================

module c_engine_dma_top #(
    parameter R                   = `CFG_R,
    parameter C                   = `CFG_C,
    parameter x_DATA_WIDTH        = `CFG_X_DATA_WIDTH,
    parameter w_DATA_WIDTH        = `CFG_W_DATA_WIDTH,
    parameter b_DATA_WIDTH        = `CFG_B_DATA_WIDTH,
    parameter y_DATA_WIDTH        = `CFG_Y_DATA_WIDTH,
    parameter NUM_BANKS           = `CFG_NUM_BANKS,
    parameter WORD_SIZE           = `CFG_WORD_SIZE,
    parameter MEM_DEPTH           = `CFG_MEM_DEPTH,
    parameter ADDR_WIDTH          = `CFG_ADDR_WIDTH,
    parameter MAX_WEIGHTS         = `CFG_MAX_WEIGHTS,
    parameter CFG_WIDTH           = `CFG_CFG_WIDTH,
    parameter CFG_DEPTH           = `CFG_CFG_DEPTH,
    parameter CFG_ADDR_WIDTH      = `CFG_CFG_ADDR_WIDTH,
    parameter NUM_CHUNKS          = `CFG_NUM_CHUNKS,
    parameter C_M_AXI_ADDR_WIDTH  = 32,
    parameter C_M_AXI_DATA_WIDTH  = 128,
    parameter integer CLIP_MAX    = `CFG_CLIP_MAX,
    parameter integer CLIP_MIN    = `CFG_CLIP_MIN
)(
    input  wire                                aclk,
    input  wire                                aresetn,

    // -------------------------------------------------------------------------
    // AXI4-Lite Slave Interface (from PS / Host CPU)
    // -------------------------------------------------------------------------
    input  wire [7:0]                          s_axi_lite_awaddr,
    input  wire [2:0]                          s_axi_lite_awprot,
    input  wire                                s_axi_lite_awvalid,
    output wire                                s_axi_lite_awready,
    input  wire [31:0]                         s_axi_lite_wdata,
    input  wire [3:0]                          s_axi_lite_wstrb,
    input  wire                                s_axi_lite_wvalid,
    output wire                                s_axi_lite_wready,
    output wire [1:0]                          s_axi_lite_bresp,
    output wire                                s_axi_lite_bvalid,
    input  wire                                s_axi_lite_bready,
    input  wire [7:0]                          s_axi_lite_araddr,
    input  wire [2:0]                          s_axi_lite_arprot,
    input  wire                                s_axi_lite_arvalid,
    output wire                                s_axi_lite_arready,
    output wire [31:0]                         s_axi_lite_rdata,
    output wire [1:0]                          s_axi_lite_rresp,
    output wire                                s_axi_lite_rvalid,
    input  wire                                s_axi_lite_rready,

    // -------------------------------------------------------------------------
    // AXI4 Master 0: Read Master (128-bit) - Fetch from DDR to SRAM
    // -------------------------------------------------------------------------
    output wire [3:0]                          m0_axi_arid,
    output wire [C_M_AXI_ADDR_WIDTH-1:0]       m0_axi_araddr,
    output wire [7:0]                          m0_axi_arlen,
    output wire [2:0]                          m0_axi_arsize,
    output wire [1:0]                          m0_axi_arburst,
    output wire                                m0_axi_arlock,
    output wire [3:0]                          m0_axi_arcache,
    output wire [2:0]                          m0_axi_arprot,
    output wire [3:0]                          m0_axi_arqos,
    output wire                                m0_axi_arvalid,
    input  wire                                m0_axi_arready,
    input  wire [3:0]                          m0_axi_rid,
    input  wire [C_M_AXI_DATA_WIDTH-1:0]       m0_axi_rdata,
    input  wire [1:0]                          m0_axi_rresp,
    input  wire                                m0_axi_rlast,
    input  wire                                m0_axi_rvalid,
    output wire                                m0_axi_rready,

    // -------------------------------------------------------------------------
    // AXI4 Master 1: Write Master (128-bit) - Send from SRAM to DDR
    // -------------------------------------------------------------------------
    output wire [3:0]                          m1_axi_awid,
    output wire [C_M_AXI_ADDR_WIDTH-1:0]       m1_axi_awaddr,
    output wire [7:0]                          m1_axi_awlen,
    output wire [2:0]                          m1_axi_awsize,
    output wire [1:0]                          m1_axi_awburst,
    output wire                                m1_axi_awlock,
    output wire [3:0]                          m1_axi_awcache,
    output wire [2:0]                          m1_axi_awprot,
    output wire [3:0]                          m1_axi_awqos,
    output wire                                m1_axi_awvalid,
    input  wire                                m1_axi_awready,
    output wire [C_M_AXI_DATA_WIDTH-1:0]       m1_axi_wdata,
    output wire [(C_M_AXI_DATA_WIDTH/8)-1:0]   m1_axi_wstrb,
    output wire                                m1_axi_wlast,
    output wire                                m1_axi_wvalid,
    input  wire                                m1_axi_wready,
    input  wire [3:0]                          m1_axi_bid,
    input  wire [1:0]                          m1_axi_bresp,
    input  wire                                m1_axi_bvalid,
    output wire                                m1_axi_bready,

    // Dedicated Interrupt Line to PS GIC (ZCU104 pl_ps_irq0)
    output wire                                interrupt
);

    // -------------------------------------------------------------------------
    // Interconnect Signals
    // -------------------------------------------------------------------------
    wire        start_rd_dma;
    wire        start_wr_dma;
    wire        start_engine;
    wire        soft_reset;
    wire [31:0] rd_ddr_addr;
    wire [15:0] rd_sram_bank;
    wire [15:0] rd_sram_row;
    wire [15:0] rd_beats;
    wire [31:0] wt_ddr_addr;
    wire [31:0] wr_ddr_addr;
    wire [15:0] wr_sram_bank;
    wire [15:0] wr_sram_row;
    wire [15:0] wr_beats;
    wire [15:0] num_chunks;

    wire                          cfg_wr_en;
    wire [9:0]                    slave_cfg_wr_addr;
    wire [CFG_ADDR_WIDTH-1:0]     cfg_wr_addr = slave_cfg_wr_addr[CFG_ADDR_WIDTH-1:0];
    wire [CFG_WIDTH-1:0]          cfg_wr_data;
    wire                          cfg_mode_select;

    wire        rd_dma_busy;
    wire        rd_dma_done;
    wire        wr_dma_busy;
    wire        wr_dma_done;
    wire        engine_busy;
    wire        engine_done;

    // -------------------------------------------------------------------------
    // Performance Cycle Counter (Active Inference Duration)
    // -------------------------------------------------------------------------
    reg [63:0] cycle_counter;
    reg        engine_active_r;
    always @(posedge aclk or negedge aresetn) begin
        if (!aresetn) begin
            cycle_counter   <= 64'd0;
            engine_active_r <= 1'b0;
        end else begin
            if (start_engine) begin
                cycle_counter   <= 64'd0;
                engine_active_r <= 1'b1;
            end else if (engine_active_r) begin
                cycle_counter <= cycle_counter + 64'd1;
                if (engine_done) begin
                    engine_active_r <= 1'b0;
                end
            end
        end
    end

    // Autonomous PL DMA Controls
    wire                      ctrl_start_rd_dma;
    wire                      ctrl_rd_is_weight;
    wire [31:0]               ctrl_rd_ddr_offset;
    wire [ADDR_WIDTH-1:0]     ctrl_rd_sram_row;
    wire [15:0]               ctrl_rd_beats;
    wire                      ctrl_start_wr_dma;
    wire [ADDR_WIDTH-1:0]     ctrl_wr_sram_row;
    wire [15:0]               ctrl_wr_beats;
    wire [7:0]                active_chunk_idx;
    wire [CFG_ADDR_WIDTH-1:0] active_pc;
    wire [4:0]                active_fsm_state;

    wire        eff_start_rd_dma = start_rd_dma | ctrl_start_rd_dma;
    wire [31:0] eff_rd_ddr_addr  = ctrl_start_rd_dma ? (ctrl_rd_is_weight ? (wt_ddr_addr + ctrl_rd_ddr_offset) : (rd_ddr_addr + ctrl_rd_ddr_offset)) : rd_ddr_addr;
    wire [15:0] eff_rd_sram_row  = ctrl_start_rd_dma ? { {(16-ADDR_WIDTH){1'b0}}, ctrl_rd_sram_row } : rd_sram_row;
    wire [15:0] eff_rd_sram_bank = ctrl_start_rd_dma ? 16'd0 : rd_sram_bank;
    wire [15:0] eff_rd_beats     = ctrl_start_rd_dma ? ctrl_rd_beats : rd_beats;

    wire        eff_start_wr_dma = start_wr_dma | ctrl_start_wr_dma;
    wire [15:0] eff_wr_sram_row  = ctrl_start_wr_dma ? { {(16-ADDR_WIDTH){1'b0}}, ctrl_wr_sram_row } : wr_sram_row;
    wire [15:0] eff_wr_sram_bank = ctrl_start_wr_dma ? 16'd0 : wr_sram_bank;
    wire [15:0] eff_wr_beats     = ctrl_start_wr_dma ? ctrl_wr_beats : wr_beats;

    // Post-Processing Core Interconnect
    wire        start_postproc;
    wire [15:0] conf_thresh;
    wire [15:0] iou_thresh;
    wire [3:0]  det_rd_idx;
    wire        postproc_busy;
    wire        postproc_done;
    wire [4:0]  det_count;

    wire [15:0] det_x1_0, det_y1_0, det_x2_0, det_y2_0, det_score_0, det_class_0;
    wire [15:0] det_x1_1, det_y1_1, det_x2_1, det_y2_1, det_score_1, det_class_1;
    wire [15:0] det_x1_2, det_y1_2, det_x2_2, det_y2_2, det_score_2, det_class_2;
    wire [15:0] det_x1_3, det_y1_3, det_x2_3, det_y2_3, det_score_3, det_class_3;
    wire [15:0] det_cur_x1, det_cur_y1, det_cur_x2, det_cur_y2, det_cur_score, det_cur_class;

    // Candidate Streaming Interconnect (Slave Register -> Post-Processor)
    wire               cand_valid_reg;
    wire [4:0]         cand_grid_x_reg;
    wire [4:0]         cand_grid_y_reg;
    wire               cand_head_id_reg;
    wire [1:0]         cand_anchor_id_reg;
    wire signed [7:0]  cand_tx_reg;
    wire signed [7:0]  cand_ty_reg;
    wire signed [7:0]  cand_tw_reg;
    wire signed [7:0]  cand_th_reg;
    wire signed [7:0]  cand_to_reg;
    wire signed [7:0]  cand_cls_score_reg;
    wire [6:0]         cand_cls_id_reg;

    // Simulation / testbench override hook (hierarchical access from testbench)
    wire               cand_valid_tb;
    wire [4:0]         cand_grid_x_tb;
    wire [4:0]         cand_grid_y_tb;
    wire               cand_head_id_tb;
    wire [1:0]         cand_anchor_id_tb;
    wire signed [7:0]  cand_tx_tb;
    wire signed [7:0]  cand_ty_tb;
    wire signed [7:0]  cand_tw_tb;
    wire signed [7:0]  cand_th_tb;
    wire signed [7:0]  cand_to_tb;
    wire signed [7:0]  cand_cls_score_tb;
    wire [6:0]         cand_cls_id_tb;

    wire               cand_tb_active = (cand_valid_tb === 1'b1);
    wire               cand_valid     = cand_valid_reg | cand_tb_active;
    wire [4:0]         cand_grid_x    = cand_tb_active ? cand_grid_x_tb : cand_grid_x_reg;
    wire [4:0]         cand_grid_y    = cand_tb_active ? cand_grid_y_tb : cand_grid_y_reg;
    wire               cand_head_id   = cand_tb_active ? cand_head_id_tb : cand_head_id_reg;
    wire [1:0]         cand_anchor_id = cand_tb_active ? cand_anchor_id_tb : cand_anchor_id_reg;
    wire signed [7:0]  cand_tx        = cand_tb_active ? cand_tx_tb : cand_tx_reg;
    wire signed [7:0]  cand_ty        = cand_tb_active ? cand_ty_tb : cand_ty_reg;
    wire signed [7:0]  cand_tw        = cand_tb_active ? cand_tw_tb : cand_tw_reg;
    wire signed [7:0]  cand_th        = cand_tb_active ? cand_th_tb : cand_th_reg;
    wire signed [7:0]  cand_to        = cand_tb_active ? cand_to_tb : cand_to_reg;
    wire signed [7:0]  cand_cls_score = cand_tb_active ? cand_cls_score_tb : cand_cls_score_reg;
    wire [6:0]         cand_cls_id    = cand_tb_active ? cand_cls_id_tb : cand_cls_id_reg;

    wire [WORD_SIZE-1:0] stream_rd_data;
    wire                 stream_rd_valid;
    wire                 stream_rd_ready;

    wire [WORD_SIZE-1:0] stream_wr_data;
    wire                 stream_wr_valid;
    wire                 stream_wr_ready;

    wire                                bridge_mem_override;
    wire                                bridge_mem_en;
    wire [NUM_BANKS-1:0]               bridge_mem_we;
    wire [ADDR_WIDTH-1:0]              bridge_mem_addr;
    wire [NUM_BANKS*WORD_SIZE-1:0]     bridge_mem_din;
    wire [NUM_BANKS*WORD_SIZE-1:0]     bridge_mem_dout;

    // Reset combination
    wire sys_reset = !aresetn || soft_reset;

    // -------------------------------------------------------------------------
    // 1. AXI-Lite Slave Register File
    // -------------------------------------------------------------------------
    axi_lite_slave #(
        .ADDR_WIDTH (8),
        .DATA_WIDTH (32)
    ) u_axi_lite (
        .aclk            (aclk),
        .aresetn         (aresetn),
        .s_axi_awaddr    (s_axi_lite_awaddr),
        .s_axi_awprot    (s_axi_lite_awprot),
        .s_axi_awvalid   (s_axi_lite_awvalid),
        .s_axi_awready   (s_axi_lite_awready),
        .s_axi_wdata     (s_axi_lite_wdata),
        .s_axi_wstrb     (s_axi_lite_wstrb),
        .s_axi_wvalid    (s_axi_lite_wvalid),
        .s_axi_wready    (s_axi_lite_wready),
        .s_axi_bresp     (s_axi_lite_bresp),
        .s_axi_bvalid    (s_axi_lite_bvalid),
        .s_axi_bready    (s_axi_lite_bready),
        .s_axi_araddr    (s_axi_lite_araddr),
        .s_axi_arprot    (s_axi_lite_arprot),
        .s_axi_arvalid   (s_axi_lite_arvalid),
        .s_axi_arready   (s_axi_lite_arready),
        .s_axi_rdata     (s_axi_lite_rdata),
        .s_axi_rresp     (s_axi_lite_rresp),
        .s_axi_rvalid    (s_axi_lite_rvalid),
        .s_axi_rready    (s_axi_lite_rready),
        .start_rd_dma    (start_rd_dma),
        .start_wr_dma    (start_wr_dma),
        .start_engine    (start_engine),
        .soft_reset      (soft_reset),
        .rd_ddr_addr     (rd_ddr_addr),
        .rd_sram_bank    (rd_sram_bank),
        .rd_sram_row     (rd_sram_row),
        .rd_beats        (rd_beats),
        .wt_ddr_addr     (wt_ddr_addr),
        .wr_ddr_addr     (wr_ddr_addr),
        .wr_sram_bank    (wr_sram_bank),
        .wr_sram_row     (wr_sram_row),
        .wr_beats        (wr_beats),
        .num_chunks      (num_chunks),
        .cfg_wr_en       (cfg_wr_en),
        .cfg_wr_addr     (slave_cfg_wr_addr),
        .cfg_wr_data     (cfg_wr_data),
        .cfg_mode_select (cfg_mode_select),
        .rd_dma_busy     (rd_dma_busy),
        .rd_dma_done     (rd_dma_done),
        .wr_dma_busy     (wr_dma_busy),
        .wr_dma_done     (wr_dma_done),
        .engine_busy     (engine_busy),
        .engine_done     (engine_done),
        .cycle_count_lo  (cycle_counter[31:0]),
        .cycle_count_hi  (cycle_counter[63:32]),
        .active_chunk_idx(active_chunk_idx),
        .active_pc       ({ {(10-CFG_ADDR_WIDTH){1'b0}}, active_pc }),
        .active_fsm_state(active_fsm_state),

        // Post-Processing Ports
        .start_postproc  (start_postproc),
        .conf_thresh     (conf_thresh),
        .iou_thresh      (iou_thresh),
        .det_rd_idx      (det_rd_idx),
        .postproc_busy   (postproc_busy),
        .postproc_done   (postproc_done),
        .det_count       (det_count),
        .det_x1_0        (det_x1_0), .det_y1_0(det_y1_0),
        .det_x2_0        (det_x2_0), .det_y2_0(det_y2_0),
        .det_score_0     (det_score_0), .det_class_0(det_class_0),
        .det_x1_1        (det_x1_1), .det_y1_1(det_y1_1),
        .det_x2_1        (det_x2_1), .det_y2_1(det_y2_1),
        .det_score_1     (det_score_1), .det_class_1(det_class_1),
        .det_x1_2        (det_x1_2), .det_y1_2(det_y1_2),
        .det_x2_2        (det_x2_2), .det_y2_2(det_y2_2),
        .det_score_2     (det_score_2), .det_class_2(det_class_2),
        .det_x1_3        (det_x1_3), .det_y1_3(det_y1_3),
        .det_x2_3        (det_x2_3), .det_y2_3(det_y2_3),
        .det_score_3     (det_score_3), .det_class_3(det_class_3),
        .det_cur_x1      (det_cur_x1),
        .det_cur_y1      (det_cur_y1),
        .det_cur_x2      (det_cur_x2),
        .det_cur_y2      (det_cur_y2),
        .det_cur_score   (det_cur_score),
        .det_cur_class   (det_cur_class),

        // Candidate Streaming Ports
        .cand_valid      (cand_valid_reg),
        .cand_grid_x     (cand_grid_x_reg),
        .cand_grid_y     (cand_grid_y_reg),
        .cand_head_id    (cand_head_id_reg),
        .cand_anchor_id  (cand_anchor_id_reg),
        .cand_tx         (cand_tx_reg),
        .cand_ty         (cand_ty_reg),
        .cand_tw         (cand_tw_reg),
        .cand_th         (cand_th_reg),
        .cand_to         (cand_to_reg),
        .cand_cls_score  (cand_cls_score_reg),
        .cand_cls_id     (cand_cls_id_reg),

        .interrupt       (interrupt)
    );

    // -------------------------------------------------------------------------
    // Dedicated Hardware Post-Processing Accelerator
    // -------------------------------------------------------------------------
    // Bypass unused YOLO post-processing engine to eliminate CLB/routing congestion for CNNs
    assign postproc_busy  = 1'b0;
    assign postproc_done  = 1'b0;
    assign det_count      = 5'd0;
    assign det_cur_x1     = 16'd0;
    assign det_cur_y1     = 16'd0;
    assign det_cur_x2     = 16'd0;
    assign det_cur_y2     = 16'd0;
    assign det_cur_score  = 16'd0;
    assign det_cur_class  = 16'd0;
    assign det_x1_0       = 16'd0;
    assign det_y1_0       = 16'd0;
    assign det_x2_0       = 16'd0;
    assign det_y2_0       = 16'd0;
    assign det_score_0    = 16'd0;
    assign det_class_0    = 16'd0;
    assign det_x1_1       = 16'd0;
    assign det_y1_1       = 16'd0;
    assign det_x2_1       = 16'd0;
    assign det_y2_1       = 16'd0;
    assign det_score_1    = 16'd0;
    assign det_class_1    = 16'd0;
    assign det_x1_2       = 16'd0;
    assign det_y1_2       = 16'd0;
    assign det_x2_2       = 16'd0;
    assign det_y2_2       = 16'd0;
    assign det_score_2    = 16'd0;
    assign det_class_2    = 16'd0;
    assign det_x1_3       = 16'd0;
    assign det_y1_3       = 16'd0;
    assign det_x2_3       = 16'd0;
    assign det_y2_3       = 16'd0;
    assign det_score_3    = 16'd0;
    assign det_class_3    = 16'd0;

    // -------------------------------------------------------------------------
    // 2. AXI4 Master 0: Read Master (128-bit)
    // -------------------------------------------------------------------------
    axi4_master_rd #(
        .C_M_AXI_ADDR_WIDTH (C_M_AXI_ADDR_WIDTH),
        .C_M_AXI_DATA_WIDTH (C_M_AXI_DATA_WIDTH)
    ) u_master_rd (
        .aclk          (aclk),
        .aresetn       (!sys_reset),
        .start         (eff_start_rd_dma),
        .base_addr     (eff_rd_ddr_addr),
        .total_beats   (eff_rd_beats),
        .busy          (rd_dma_busy),
        .done          (rd_dma_done),
        .stream_data   (stream_rd_data),
        .stream_valid  (stream_rd_valid),
        .stream_ready  (stream_rd_ready),
        .m_axi_arid    (m0_axi_arid),
        .m_axi_araddr  (m0_axi_araddr),
        .m_axi_arlen   (m0_axi_arlen),
        .m_axi_arsize  (m0_axi_arsize),
        .m_axi_arburst (m0_axi_arburst),
        .m_axi_arlock  (m0_axi_arlock),
        .m_axi_arcache (m0_axi_arcache),
        .m_axi_arprot  (m0_axi_arprot),
        .m_axi_arqos   (m0_axi_arqos),
        .m_axi_arvalid (m0_axi_arvalid),
        .m_axi_arready (m0_axi_arready),
        .m_axi_rid     (m0_axi_rid),
        .m_axi_rdata   (m0_axi_rdata),
        .m_axi_rresp   (m0_axi_rresp),
        .m_axi_rlast   (m0_axi_rlast),
        .m_axi_rvalid  (m0_axi_rvalid),
        .m_axi_rready  (m0_axi_rready)
    );

    // -------------------------------------------------------------------------
    // 3. AXI4 Master 1: Write Master (128-bit)
    // -------------------------------------------------------------------------
    axi4_master_wr #(
        .C_M_AXI_ADDR_WIDTH (C_M_AXI_ADDR_WIDTH),
        .C_M_AXI_DATA_WIDTH (C_M_AXI_DATA_WIDTH)
    ) u_master_wr (
        .aclk          (aclk),
        .aresetn       (!sys_reset),
        .start         (eff_start_wr_dma),
        .base_addr     (wr_ddr_addr),
        .total_beats   (eff_wr_beats),
        .busy          (wr_dma_busy),
        .done          (wr_dma_done),
        .stream_data   (stream_wr_data),
        .stream_valid  (stream_wr_valid),
        .stream_ready  (stream_wr_ready),
        .m_axi_awid    (m1_axi_awid),
        .m_axi_awaddr  (m1_axi_awaddr),
        .m_axi_awlen   (m1_axi_awlen),
        .m_axi_awsize  (m1_axi_awsize),
        .m_axi_awburst (m1_axi_awburst),
        .m_axi_awlock  (m1_axi_awlock),
        .m_axi_awcache (m1_axi_awcache),
        .m_axi_awprot  (m1_axi_awprot),
        .m_axi_awqos   (m1_axi_awqos),
        .m_axi_awvalid (m1_axi_awvalid),
        .m_axi_awready (m1_axi_awready),
        .m_axi_wdata   (m1_axi_wdata),
        .m_axi_wstrb   (m1_axi_wstrb),
        .m_axi_wlast   (m1_axi_wlast),
        .m_axi_wvalid  (m1_axi_wvalid),
        .m_axi_wready  (m1_axi_wready),
        .m_axi_bid     (m1_axi_bid),
        .m_axi_bresp   (m1_axi_bresp),
        .m_axi_bvalid  (m1_axi_bvalid),
        .m_axi_bready  (m1_axi_bready)
    );

    // -------------------------------------------------------------------------
    // 4. DMA-to-SRAM Bridge
    // -------------------------------------------------------------------------
    dma_sram_bridge #(
        .NUM_BANKS  (NUM_BANKS),
        .WORD_SIZE  (WORD_SIZE),
        .ADDR_WIDTH (ADDR_WIDTH)
    ) u_bridge (
        .aclk             (aclk),
        .aresetn          (!sys_reset),
        .stream_rd_data   (stream_rd_data),
        .stream_rd_valid  (stream_rd_valid),
        .stream_rd_ready  (stream_rd_ready),
        .rd_dma_start     (eff_start_rd_dma),
        .rd_sram_bank     (eff_rd_sram_bank),
        .rd_sram_row      (eff_rd_sram_row),
        .rd_beats         (eff_rd_beats),
        .stream_wr_data   (stream_wr_data),
        .stream_wr_valid  (stream_wr_valid),
        .stream_wr_ready  (stream_wr_ready),
        .wr_dma_start     (eff_start_wr_dma),
        .wr_sram_bank     (eff_wr_sram_bank),
        .wr_sram_row      (eff_wr_sram_row),
        .wr_beats         (eff_wr_beats),
        .ext_mem_override (bridge_mem_override),
        .ext_mem_en       (bridge_mem_en),
        .ext_mem_we       (bridge_mem_we),
        .ext_mem_addr     (bridge_mem_addr),
        .ext_mem_din      (bridge_mem_din),
        .ext_mem_dout     (bridge_mem_dout)
    );

    // -------------------------------------------------------------------------
    // 5. Accelerator Compute Engine Top
    // -------------------------------------------------------------------------
    c_engine_top #(
        .R                   (R),
        .C                   (C),
        .x_DATA_WIDTH        (x_DATA_WIDTH),
        .w_DATA_WIDTH        (w_DATA_WIDTH),
        .b_DATA_WIDTH        (b_DATA_WIDTH),
        .y_DATA_WIDTH        (y_DATA_WIDTH),
        .NUM_BANKS           (NUM_BANKS),
        .WORD_SIZE           (WORD_SIZE),
        .MEM_DEPTH           (MEM_DEPTH),
        .ADDR_WIDTH          (ADDR_WIDTH),
        .MAX_WEIGHTS         (MAX_WEIGHTS),
        .CFG_WIDTH           (CFG_WIDTH),
        .CFG_DEPTH           (CFG_DEPTH),
        .CFG_ADDR_WIDTH      (CFG_ADDR_WIDTH),
        .NUM_CHUNKS          (NUM_CHUNKS),
        .CLIP_MAX            (CLIP_MAX),
        .CLIP_MIN            (CLIP_MIN)
    ) u_engine (
        .clk                 (aclk),
        .reset               (sys_reset),
        .start               (start_engine),
        .cfg_mode_select     (cfg_mode_select),
        .cfg_wr_en           (cfg_wr_en),
        .cfg_wr_addr         (cfg_wr_addr[CFG_ADDR_WIDTH-1:0]),
        .cfg_wr_data         (cfg_wr_data[CFG_WIDTH-1:0]),
        .mode                (1'b0),
        .cfg_chin            (8'd0),
        .cfg_chout           (8'd0),
        .cfg_kw              (4'd3),
        .cfg_kh              (4'd3),
        .cfg_shift_bits      (5'd0),
        .act_type            (2'b00),
        .stride              (3'd1),
        .pad_type            (1'b1),
        .pool_type           (2'b00),
        .pool_size           (2'b00),
        .pool_stride         (3'd1),
        .cfg_residual_en     (1'b0),
        .cfg_residual_base_addr ({ADDR_WIDTH{1'b0}}),
        .cfg_input_base_addr ({ADDR_WIDTH{1'b0}}),
        .cfg_weight_base_addr({ADDR_WIDTH{1'b0}}),
        .cfg_output_base_addr({ADDR_WIDTH{1'b0}}),
        .busy                (engine_busy),
        .done                (engine_done),
        .wt_ddr_addr         (wt_ddr_addr),
        .reg_num_chunks      (num_chunks),
        .ctrl_start_rd_dma   (ctrl_start_rd_dma),
        .ctrl_rd_is_weight   (ctrl_rd_is_weight),
        .ctrl_rd_ddr_offset  (ctrl_rd_ddr_offset),
        .ctrl_rd_sram_row    (ctrl_rd_sram_row),
        .ctrl_rd_beats       (ctrl_rd_beats),
        .rd_dma_busy         (rd_dma_busy),
        .rd_dma_done         (rd_dma_done),
        .ctrl_start_wr_dma   (ctrl_start_wr_dma),
        .ctrl_wr_sram_row    (ctrl_wr_sram_row),
        .ctrl_wr_beats       (ctrl_wr_beats),
        .wr_dma_busy         (wr_dma_busy),
        .wr_dma_done         (wr_dma_done),
        .active_chunk_idx    (active_chunk_idx),
        .active_pc           (active_pc),
        .active_fsm_state    (active_fsm_state),
        .ext_mem_override    (bridge_mem_override),
        .ext_mem_en          (bridge_mem_en),
        .ext_mem_we          (bridge_mem_we),
        .ext_mem_addr        (bridge_mem_addr),
        .ext_mem_din         (bridge_mem_din),
        .ext_mem_dout        (bridge_mem_dout),
        .ext_gemv_x_val      ({x_DATA_WIDTH{1'b0}}),
        .engdata_out         ()
    );

endmodule
