`timescale 1ps/1ps
// The whole pi0 chunk (or a cut-down one) on a reduced node array, host-free: N_VN vector nodes, N_CH int8
// chain nodes and N_PV uint8 PV chain nodes share one NAP (colpar_nap_mux.sv) in front of tb_axi_gddr6_model.sv.
// The image and the node programs come from paper/sw/pi0_chunk_program.py; the host starts every node once at its
// program base (nodes.txt) and the nodes sequence themselves with GDDR6 flags (WAIT / POST).  Every beat of every
// intermediate region, of the flags and of the actions must match the bit-exact reference, and nothing else may
// be written.
module tb_pi0_chunk;
    parameter F_GELU  = "vu_tbl_gelu.mem";
    parameter F_SIGM  = "vu_tbl_sigm.mem";
    parameter F_EXP   = "vu_tbl_exp.mem";
    parameter F_RSQRT = "vu_tbl_rsqrt.mem";
    parameter F_QUANT = "vu_tbl_quant.mem";
    parameter integer N_VN = 2;
    parameter integer N_CH = 2;
    parameter integer N_PV = 1;
    parameter integer N_LANE  = 2;
    parameter integer N_LD    = 2;
    parameter integer SLOT_BITS = 12;
    parameter integer MAX_OUT = 8;
    parameter integer N_STAGE = 16;
    parameter integer WR_PIPE_EVERY = (N_STAGE > 16) ? 4 : 0;
    parameter integer VALID_COPIES = (N_STAGE > 16) ? 4 : 1;
    // mixed depths as pi0_chip_top builds them: the first N_DEEP int8 chains are 32-stage (WR_PIPE_EVERY 4,
    // VALID_COPIES 4), the other int8 chains and the PV chains N_STAGE deep
    parameter integer N_DEEP = 0;
    parameter integer VN_RD_FIFO_LOG2 = 7;   // vu_node_ml RD_FIFO_LOG2 (5 = the fan-out before 2026-09-18)
    localparam integer N_M = N_VN + N_CH + N_PV;

    reg i_clk = 1'b0, i_fclk = 1'b0, i_rstn = 1'b0;
    always #500  i_clk  = ~i_clk;      // 1 GHz array clock
    always #1500 i_fclk = ~i_fclk;     // 333 MHz fabric / vector clock

    logic [255:0] rmem [longint];
    logic [255:0] emem [longint];
    logic [255:0] gmem [longint];

    wire [N_M-1:0]      s_arvalid, s_arready, s_rvalid, s_rready, s_awvalid, s_awready, s_wvalid, s_wready;
    wire [N_M-1:0]      s_wlast, s_bvalid, s_bready;
    wire [N_M*42-1:0]   s_araddr, s_awaddr;
    wire [N_M*8-1:0]    s_arlen, s_awlen;
    wire [N_M*3-1:0]    s_arsize, s_awsize;
    wire [N_M*2-1:0]    s_arburst, s_awburst;
    wire [N_M*256-1:0]  s_wdata;
    wire [N_M*32-1:0]   s_wstrb;
    // response buses per node: one shared bus behind colpar_nap_mux (PER_NODE_MEM 0), or each node's own port
    wire [N_M*256-1:0]  s_rdata;
    wire [N_M*2-1:0]    s_rresp, s_bresp;
    wire [N_M-1:0]      s_rlast;
    wire [N_M-1:0]      halt, err;
    reg  [N_M-1:0]      start = '0;
    reg  [41:0]         pbase [0:N_M-1];

    genvar g;
    generate
        for (g = 0; g < N_M; g = g + 1) begin : g_node
            if (g < N_VN) begin : g_vn
                vu_node_ml #(
                    .F_GELU(F_GELU), .F_SIGM(F_SIGM), .F_EXP(F_EXP), .F_RSQRT(F_RSQRT), .F_QUANT(F_QUANT),
                    .N_LANE(N_LANE), .SLOT_BITS(SLOT_BITS), .N_LD(N_LD), .RD_MAX_OUTSTANDING(MAX_OUT), .WR_MAX_OUTSTANDING(MAX_OUT),
                    .WR_FIFO_LOG2(6), .RD_FIFO_LOG2(VN_RD_FIFO_LOG2)
                ) u_vn (
                    .i_clk(i_fclk), .i_rstn(i_rstn), .i_prog_start(start[g]), .i_prog_base(pbase[g]),
                    .o_arvalid(s_arvalid[g]), .i_arready(s_arready[g]), .o_araddr(s_araddr[g*42 +: 42]),
                    .o_arlen(s_arlen[g*8 +: 8]), .o_arsize(s_arsize[g*3 +: 3]), .o_arburst(s_arburst[g*2 +: 2]),
                    .o_rready(s_rready[g]), .i_rvalid(s_rvalid[g]), .i_rdata(s_rdata[g*256 +: 256]), .i_rresp(s_rresp[g*2 +: 2]), .i_rlast(s_rlast[g]),
                    .o_awvalid(s_awvalid[g]), .i_awready(s_awready[g]), .o_awaddr(s_awaddr[g*42 +: 42]),
                    .o_awlen(s_awlen[g*8 +: 8]), .o_awsize(s_awsize[g*3 +: 3]), .o_awburst(s_awburst[g*2 +: 2]),
                    .o_wvalid(s_wvalid[g]), .i_wready(s_wready[g]), .o_wdata(s_wdata[g*256 +: 256]),
                    .o_wstrb(s_wstrb[g*32 +: 32]), .o_wlast(s_wlast[g]), .i_bvalid(s_bvalid[g]), .o_bready(s_bready[g]),
                    .i_bresp(s_bresp[g*2 +: 2]), .o_halt(halt[g]), .o_error(err[g]));
            end else begin : g_ch
                localparam integer DEEP = (g - N_VN) < N_DEEP;
                colpar_chain_node #(
                    .N_STAGE(DEEP ? 32 : N_STAGE), .WR_PIPE_EVERY(DEEP ? 4 : WR_PIPE_EVERY),
                    .VALID_COPIES(DEEP ? 4 : VALID_COPIES), .PROG_FROM_GDDR6(1),
                    .WR_FIFO_LOG2(8), .WR_AFULL_MARGIN(8),
                    .RD_MAX_OUTSTANDING(MAX_OUT), .WR_MAX_OUTSTANDING(MAX_OUT),
                    .MULT_MODE((g >= N_VN + N_CH) ? 5'h13 : 5'h00)
                ) u_chain (
                    .i_clk_array(i_clk), .i_clk_fabric(i_fclk), .i_rstn(i_rstn),
                    .i_prog_start(start[g]), .i_prog_base(pbase[g]),
                    .i_cmd(128'd0), .i_cmd_valid(1'b0), .o_cmd_ready(),
                    .o_arvalid(s_arvalid[g]), .i_arready(s_arready[g]), .o_araddr(s_araddr[g*42 +: 42]),
                    .o_arlen(s_arlen[g*8 +: 8]), .o_arsize(s_arsize[g*3 +: 3]), .o_arburst(s_arburst[g*2 +: 2]),
                    .o_rready(s_rready[g]), .i_rvalid(s_rvalid[g]), .i_rdata(s_rdata[g*256 +: 256]), .i_rresp(s_rresp[g*2 +: 2]),
                    .i_rlast(s_rlast[g]),
                    .o_awvalid(s_awvalid[g]), .i_awready(s_awready[g]), .o_awaddr(s_awaddr[g*42 +: 42]),
                    .o_awlen(s_awlen[g*8 +: 8]), .o_awsize(s_awsize[g*3 +: 3]), .o_awburst(s_awburst[g*2 +: 2]),
                    .o_wvalid(s_wvalid[g]), .i_wready(s_wready[g]), .o_wdata(s_wdata[g*256 +: 256]),
                    .o_wstrb(s_wstrb[g*32 +: 32]), .o_wlast(s_wlast[g]), .i_bvalid(s_bvalid[g]),
                    .o_bready(s_bready[g]), .i_bresp(s_bresp[g*2 +: 2]), .o_halt(halt[g]), .o_error(err[g]));
            end
        end
    endgenerate

    // PER_NODE_MEM 0: every node behind one colpar_nap_mux (one burst at a time) in front of one memory model -- the
    // calibrated runs.  PER_NODE_MEM 1: each node has its own memory model port on the shared contents, as on the chip
    // (every node its own NAPs), so +rd_lat / +wr_lat measure what a node's own read-ahead hides.
    parameter integer PER_NODE_MEM = 0;
    parameter integer STRIPE = 0;            // PER_NODE_MEM 1 only: 12 = the chip's 4 KB channel striping
    function automatic [41:0] unstripe(input [41:0] a);   // NoC address -> logical (inverse of axi_stripe_pkg::stripe_addr)
        unstripe = (STRIPE == 0) ? a : ((42'(a[30:0]) >> STRIPE) << (STRIPE + 4)) | (42'(a[36:33]) << STRIPE) |
                                       (42'(a[30:0]) & ((42'd1 << STRIPE) - 42'd1));
    endfunction
    wire [31:0]   n_rb, n_wb, n_err;
    longint unsigned rd_beats = 0, wr_beats = 0;
    generate
    if (PER_NODE_MEM == 0) begin : g_shared
        wire [255:0] sh_rdata;
        wire [1:0]   sh_rresp, sh_bresp;
        wire         sh_rlast;
        assign s_rdata = {N_M{sh_rdata}};
        assign s_rresp = {N_M{sh_rresp}};
        assign s_bresp = {N_M{sh_bresp}};
        assign s_rlast = {N_M{sh_rlast}};
        wire          m_arvalid, m_arready, m_rvalid, m_rready, m_rlast;
        wire          m_awvalid, m_awready, m_wvalid, m_wready, m_wlast, m_bvalid, m_bready, wr_fire;
        wire [41:0]   m_araddr, m_awaddr, rd_addr, wr_addr;
        wire [7:0]    m_arlen, m_awlen;
        wire [2:0]    m_arsize, m_awsize;
        wire [1:0]    m_arburst, m_awburst;
        wire [255:0]  m_rdata, m_wdata, wr_data;
        reg  [255:0]  rd_beat;
        wire [31:0]   m_wstrb;

        colpar_nap_mux #(.N_M(N_M)) u_mux (
            .i_clk(i_fclk), .i_rstn(i_rstn),
            .s_arvalid(s_arvalid), .s_arready(s_arready), .s_araddr(s_araddr), .s_arlen(s_arlen),
            .s_arsize(s_arsize), .s_arburst(s_arburst), .s_rvalid(s_rvalid), .s_rready(s_rready),
            .s_rdata(sh_rdata), .s_rresp(sh_rresp), .s_rlast(sh_rlast),
            .s_awvalid(s_awvalid), .s_awready(s_awready), .s_awaddr(s_awaddr), .s_awlen(s_awlen),
            .s_awsize(s_awsize), .s_awburst(s_awburst), .s_wvalid(s_wvalid), .s_wready(s_wready),
            .s_wdata(s_wdata), .s_wstrb(s_wstrb), .s_wlast(s_wlast), .s_bvalid(s_bvalid),
            .s_bready(s_bready), .s_bresp(sh_bresp),
            .m_arvalid(m_arvalid), .m_arready(m_arready), .m_araddr(m_araddr), .m_arlen(m_arlen),
            .m_arsize(m_arsize), .m_arburst(m_arburst), .m_rvalid(m_rvalid), .m_rready(m_rready),
            .m_rdata(m_rdata), .m_rresp(2'b00), .m_rlast(m_rlast),
            .m_awvalid(m_awvalid), .m_awready(m_awready), .m_awaddr(m_awaddr), .m_awlen(m_awlen),
            .m_awsize(m_awsize), .m_awburst(m_awburst), .m_wvalid(m_wvalid), .m_wready(m_wready),
            .m_wdata(m_wdata), .m_wstrb(m_wstrb), .m_wlast(m_wlast), .m_bvalid(m_bvalid),
            .m_bready(m_bready), .m_bresp(2'b00));

        always @* rd_beat = rmem.exists(longint'(rd_addr >> 5)) ? rmem[longint'(rd_addr >> 5)] : 256'd0;

        tb_axi_gddr6_model #(.MAX_BEATS(16)) u_mem (
            .i_clk(i_fclk), .i_rstn(i_rstn),
            .i_arvalid(m_arvalid), .o_arready(m_arready), .i_araddr(m_araddr), .i_arlen(m_arlen),
            .i_arsize(m_arsize), .i_arburst(m_arburst), .o_rvalid(m_rvalid), .i_rready(m_rready),
            .o_rdata(m_rdata), .o_rlast(m_rlast), .o_rd_addr(rd_addr), .i_rd_beat(rd_beat),
            .i_awvalid(m_awvalid), .o_awready(m_awready), .i_awaddr(m_awaddr), .i_awlen(m_awlen),
            .i_awsize(m_awsize), .i_awburst(m_awburst), .i_wvalid(m_wvalid), .o_wready(m_wready),
            .i_wdata(m_wdata), .i_wstrb(m_wstrb), .i_wlast(m_wlast), .o_bvalid(m_bvalid), .i_bready(m_bready),
            .o_wr_fire(wr_fire), .o_wr_addr(wr_addr), .o_wr_data(wr_data),
            .o_n_rbursts(n_rb), .o_n_wbursts(n_wb), .o_n_err(n_err), .o_max_rq(), .o_max_wq());

            always @(posedge i_fclk) begin
                if (m_rvalid && m_rready) rd_beats <= rd_beats + 1;
                if (wr_fire) wr_beats <= wr_beats + 1;
            end
            always @(posedge i_fclk) begin
                if (wr_fire) begin
                    gmem[longint'(wr_addr >> 5)] = wr_data;
                    rmem[longint'(wr_addr >> 5)] = wr_data;
                end
            end


    end else begin : g_per_node
        wire [N_M-1:0]      p_wr_fire, p_rfire;
        wire [N_M*42-1:0]   p_wr_addr, p_rd_addr;
        wire [N_M*256-1:0]  p_wr_data;
        reg  [N_M*256-1:0]  p_rd_beat;
        wire [N_M*32-1:0]   p_nrb, p_nwb, p_nerr;
        for (g = 0; g < N_M; g = g + 1) begin : g_mem
            // STRIPE != 0: the chip's GDDR6 channel striping (axi_stripe.sv, pi0_chip_top STRIPE) between the node and
            // its memory port; the model sees NoC addresses, the image and the checks stay logical (unstripe below)
            wire         x_arvalid, x_arready, x_rvalid, x_rready, x_rlast, x_awvalid, x_awready, x_wvalid, x_wready, x_wlast;
            wire         x_bvalid, x_bready;
            wire [41:0]  x_araddr, x_awaddr, x_rd_addr, x_wr_addr;
            wire [7:0]   x_arlen, x_awlen;
            wire [2:0]   x_arsize, x_awsize;
            wire [1:0]   x_arburst, x_awburst;
            wire [255:0] x_rdata, x_wdata;
            wire [31:0]  x_wstrb;
            always @* p_rd_beat[g*256 +: 256] = rmem.exists(longint'(unstripe(x_rd_addr) >> 5)) ?
                                                 rmem[longint'(unstripe(x_rd_addr) >> 5)] : 256'd0;
            assign p_rd_addr[g*42 +: 42] = unstripe(x_rd_addr);
            assign p_wr_addr[g*42 +: 42] = unstripe(x_wr_addr);
            assign p_rfire[g] = s_rvalid[g] && s_rready[g];
            assign s_rresp[g*2 +: 2] = 2'b00;
            assign s_bresp[g*2 +: 2] = 2'b00;
            if (STRIPE != 0) begin : g_st
                // stripe -> axi_id_reorder (as pi0_chip_top) -> the in-order model; the model has no IDs, so the
                // RID / BID it would return are the ARID / AWID of the burst at the head of its queues
                wire         y_arvalid, y_arready, y_rvalid, y_rready, y_rlast, y_awvalid, y_awready, y_bvalid, y_bready;
                wire [41:0]  y_araddr, y_awaddr;
                wire [7:0]   y_arlen, y_awlen, x_arid, x_awid;
                wire [2:0]   y_arsize, y_awsize;
                wire [1:0]   y_arburst, y_awburst, y_rresp, y_bresp;
                wire [255:0] y_rdata;
                byte unsigned rid_q [$], bid_q [$];
                always @(posedge i_fclk) begin
                    if (x_arvalid && x_arready) rid_q.push_back(x_arid);
                    if (x_rvalid && x_rready && x_rlast) void'(rid_q.pop_front());
                    if (x_awvalid && x_awready) bid_q.push_back(x_awid);
                    if (x_bvalid && x_bready) void'(bid_q.pop_front());
                end
                wire [7:0] x_rid = (rid_q.size() > 0) ? rid_q[0] : 8'd0;
                wire [7:0] x_bid = (bid_q.size() > 0) ? bid_q[0] : 8'd0;
                axi_id_reorder #(.TAG_LOG2(4)) u_ro (.i_clk(i_fclk), .i_rstn(i_rstn),
                    .s_arvalid(y_arvalid), .s_arready(y_arready), .s_araddr(y_araddr), .s_arlen(y_arlen), .s_arsize(y_arsize),
                    .s_arburst(y_arburst), .s_rvalid(y_rvalid), .s_rready(y_rready), .s_rdata(y_rdata), .s_rresp(y_rresp),
                    .s_rlast(y_rlast),
                    .s_awvalid(y_awvalid), .s_awready(y_awready), .s_awaddr(y_awaddr), .s_awlen(y_awlen), .s_awsize(y_awsize),
                    .s_awburst(y_awburst), .s_bvalid(y_bvalid), .s_bready(y_bready), .s_bresp(y_bresp),
                    .m_arvalid(x_arvalid), .m_arready(x_arready), .m_araddr(x_araddr), .m_arlen(x_arlen), .m_arsize(x_arsize),
                    .m_arburst(x_arburst), .m_arid(x_arid), .m_rvalid(x_rvalid), .m_rready(x_rready), .m_rdata(x_rdata),
                    .m_rresp(2'b00), .m_rlast(x_rlast), .m_rid(x_rid),
                    .m_awvalid(x_awvalid), .m_awready(x_awready), .m_awaddr(x_awaddr), .m_awlen(x_awlen), .m_awsize(x_awsize),
                    .m_awburst(x_awburst), .m_awid(x_awid), .m_bvalid(x_bvalid), .m_bready(x_bready), .m_bresp(2'b00),
                    .m_bid(x_bid));
                axi_rd_stripe #(.STRIPE_LOG2(STRIPE)) u_rs (.i_clk(i_fclk), .i_rstn(i_rstn), .i_en(1'b1),
                    .s_arvalid(s_arvalid[g]), .s_arready(s_arready[g]), .s_araddr(s_araddr[g*42 +: 42]), .s_arlen(s_arlen[g*8 +: 8]),
                    .s_arsize(s_arsize[g*3 +: 3]), .s_arburst(s_arburst[g*2 +: 2]), .s_rvalid(s_rvalid[g]), .s_rready(s_rready[g]),
                    .s_rdata(s_rdata[g*256 +: 256]), .s_rresp(), .s_rlast(s_rlast[g]),
                    .m_arvalid(y_arvalid), .m_arready(y_arready), .m_araddr(y_araddr), .m_arlen(y_arlen), .m_arsize(y_arsize),
                    .m_arburst(y_arburst), .m_rvalid(y_rvalid), .m_rready(y_rready), .m_rdata(y_rdata), .m_rresp(y_rresp), .m_rlast(y_rlast));
                axi_wr_stripe #(.STRIPE_LOG2(STRIPE)) u_ws (.i_clk(i_fclk), .i_rstn(i_rstn), .i_en(1'b1),
                    .s_awvalid(s_awvalid[g]), .s_awready(s_awready[g]), .s_awaddr(s_awaddr[g*42 +: 42]), .s_awlen(s_awlen[g*8 +: 8]),
                    .s_awsize(s_awsize[g*3 +: 3]), .s_awburst(s_awburst[g*2 +: 2]), .s_wvalid(s_wvalid[g]), .s_wready(s_wready[g]),
                    .s_wdata(s_wdata[g*256 +: 256]), .s_wstrb(s_wstrb[g*32 +: 32]), .s_wlast(s_wlast[g]), .s_bvalid(s_bvalid[g]),
                    .s_bready(s_bready[g]), .s_bresp(),
                    .m_awvalid(y_awvalid), .m_awready(y_awready), .m_awaddr(y_awaddr), .m_awlen(y_awlen), .m_awsize(y_awsize),
                    .m_awburst(y_awburst), .m_wvalid(x_wvalid), .m_wready(x_wready), .m_wdata(x_wdata), .m_wstrb(x_wstrb),
                    .m_wlast(x_wlast), .m_bvalid(y_bvalid), .m_bready(y_bready), .m_bresp(y_bresp));
            end else begin : g_nost
                assign x_arvalid = s_arvalid[g]; assign s_arready[g] = x_arready; assign x_araddr = s_araddr[g*42 +: 42];
                assign x_arlen = s_arlen[g*8 +: 8]; assign x_arsize = s_arsize[g*3 +: 3]; assign x_arburst = s_arburst[g*2 +: 2];
                assign s_rvalid[g] = x_rvalid; assign x_rready = s_rready[g]; assign s_rdata[g*256 +: 256] = x_rdata;
                assign s_rlast[g] = x_rlast;
                assign x_awvalid = s_awvalid[g]; assign s_awready[g] = x_awready; assign x_awaddr = s_awaddr[g*42 +: 42];
                assign x_awlen = s_awlen[g*8 +: 8]; assign x_awsize = s_awsize[g*3 +: 3]; assign x_awburst = s_awburst[g*2 +: 2];
                assign x_wvalid = s_wvalid[g]; assign s_wready[g] = x_wready; assign x_wdata = s_wdata[g*256 +: 256];
                assign x_wstrb = s_wstrb[g*32 +: 32]; assign x_wlast = s_wlast[g];
                assign s_bvalid[g] = x_bvalid; assign x_bready = s_bready[g];
            end
            tb_axi_gddr6_model #(.MAX_BEATS(16)) u_mem (
                .i_clk(i_fclk), .i_rstn(i_rstn),
                .i_arvalid(x_arvalid), .o_arready(x_arready), .i_araddr(x_araddr), .i_arlen(x_arlen),
                .i_arsize(x_arsize), .i_arburst(x_arburst), .o_rvalid(x_rvalid), .i_rready(x_rready),
                .o_rdata(x_rdata), .o_rlast(x_rlast), .o_rd_addr(x_rd_addr),
                .i_rd_beat(p_rd_beat[g*256 +: 256]),
                .i_awvalid(x_awvalid), .o_awready(x_awready), .i_awaddr(x_awaddr), .i_awlen(x_awlen),
                .i_awsize(x_awsize), .i_awburst(x_awburst), .i_wvalid(x_wvalid), .o_wready(x_wready),
                .i_wdata(x_wdata), .i_wstrb(x_wstrb), .i_wlast(x_wlast), .o_bvalid(x_bvalid),
                .i_bready(x_bready),
                .o_wr_fire(p_wr_fire[g]), .o_wr_addr(x_wr_addr), .o_wr_data(p_wr_data[g*256 +: 256]),
                .o_n_rbursts(p_nrb[g*32 +: 32]), .o_n_wbursts(p_nwb[g*32 +: 32]), .o_n_err(p_nerr[g*32 +: 32]),
                .o_max_rq(), .o_max_wq());
        end
        reg [31:0] srb, swb, serr;
        integer    q;
        always @* begin
            srb = 0; swb = 0; serr = 0;
            for (q = 0; q < N_M; q = q + 1) begin
                srb = srb + p_nrb[q*32 +: 32]; swb = swb + p_nwb[q*32 +: 32]; serr = serr + p_nerr[q*32 +: 32];
            end
        end
        assign n_rb = srb; assign n_wb = swb; assign n_err = serr;
        always @(posedge i_fclk) begin
            for (q = 0; q < N_M; q = q + 1) begin
                if (p_rfire[q]) rd_beats = rd_beats + 1;
                if (p_wr_fire[q]) begin
                    wr_beats = wr_beats + 1;
                    gmem[longint'(p_wr_addr[q*42 +: 42] >> 5)] = p_wr_data[q*256 +: 256];
                    rmem[longint'(p_wr_addr[q*42 +: 42] >> 5)] = p_wr_data[q*256 +: 256];
                end
            end
        end
    end
    endgenerate

    // progress: the flags posted so far (one per stage part), once a second of simulated time
    integer n_flags_seen = 0;
    string  vec;
    integer fd, rc, n_exp = 0, n_wrong = 0, n_extra = 0, n_nodes = 0, i, kind_ok = 1;
    integer nd_idx;
    string  nd_kind;
    longint nd_base, idx;
    logic [255:0] beat;
    initial begin
        if (!$value$plusargs("vec=%s", vec)) $fatal(1, "+vec=<dir> required");
        fd = $fopen({vec, "/mem_in.hex"}, "r");
        if (fd == 0) $fatal(1, "no mem_in.hex");
        while (!$feof(fd)) begin
            rc = $fscanf(fd, "%h %h\n", idx, beat);
            if (rc == 2) rmem[idx] = beat;
        end
        $fclose(fd);
        fd = $fopen({vec, "/mem_exp.hex"}, "r");
        while (!$feof(fd)) begin
            rc = $fscanf(fd, "%h %h\n", idx, beat);
            if (rc == 2) begin emem[idx] = beat; n_exp = n_exp + 1; end
        end
        $fclose(fd);
        fd = $fopen({vec, "/nodes.txt"}, "r");
        if (fd == 0) $fatal(1, "no nodes.txt");
        while (!$feof(fd)) begin
            rc = $fscanf(fd, "%d %s %h\n", nd_idx, nd_kind, nd_base);
            if (rc == 3) begin
                if (nd_idx >= N_M) $fatal(1, "nodes.txt has node %0d but the testbench has %0d nodes", nd_idx, N_M);
                pbase[nd_idx] = 42'(nd_base);
                if ((nd_idx < N_VN && nd_kind != "vector") || (nd_idx >= N_VN && nd_idx < N_VN + N_CH && nd_kind != "int8")
                    || (nd_idx >= N_VN + N_CH && nd_kind != "uint8")) begin
                    $display("node %0d kind %s does not match the testbench's N_VN=%0d N_CH=%0d N_PV=%0d", nd_idx, nd_kind, N_VN, N_CH, N_PV);
                    kind_ok = 0;
                end
                n_nodes = n_nodes + 1;
            end
        end
        $fclose(fd);
        if (!kind_ok || n_nodes != N_M) $fatal(1, "node table mismatch (%0d nodes)", n_nodes);

        repeat (8) @(posedge i_fclk);
        @(negedge i_fclk);
        i_rstn = 1'b1;
        repeat (4) @(negedge i_fclk);
        start = '1;
        @(negedge i_fclk);
        start = '0;
        for (i = 0; i < N_M; i = i + 1) wait (halt[i] === 1'b1);
        repeat (16) @(negedge i_fclk);
        $display("ALL HALTED at %0d fabric cycles, node errors %b", $time / 3000, err);

        foreach (emem[j]) begin
            if (!gmem.exists(j)) begin
                if (n_wrong < 12) $display("MISSING beat %h", j);
                n_wrong = n_wrong + 1;
            end else if (gmem[j] !== emem[j]) begin
                if (n_wrong < 12) $display("MISMATCH beat %h: got %h expected %h", j, gmem[j], emem[j]);
                n_wrong = n_wrong + 1;
            end
        end
        foreach (gmem[j]) if (!emem.exists(j)) n_extra = n_extra + 1;
        fd = $fopen({vec, "/gmem.txt"}, "w");
        foreach (gmem[j]) $fwrite(fd, "%h %h\n", j, gmem[j]);
        $fclose(fd);
        $display("RESULT %s nodes=%0d expected_beats=%0d written_beats=%0d wrong=%0d extra=%0d rd_bursts=%0d wr_bursts=%0d rd_beats=%0d wr_beats=%0d fabric_cycles=%0d axi_err=%0d node_error=%b",
                 (n_wrong == 0 && n_extra == 0 && n_err == 0 && err == '0) ? "PASS" : "FAIL",
                 N_M, n_exp, gmem.num(), n_wrong, n_extra, n_rb, n_wb, rd_beats, wr_beats, $time / 3000, n_err, err);
        $finish;
    end

    // a progress line every 2 ms of simulated time: beats moved and which nodes have halted
    always begin
        #(64'd2_000_000_000);
        $display("PROGRESS t=%0d us rd_beats=%0d wr_beats=%0d halt=%b err=%b", $time / 1_000_000, rd_beats, wr_beats, halt, err);
        $fflush();
    end

    longint unsigned timeout_ps = 64'd4_000_000_000_000;
    initial begin
        void'($value$plusargs("timeout_ps=%d", timeout_ps));
        #(timeout_ps);
        $display("RESULT FAIL timeout (halt=%b err=%b rd_beats=%0d wr_beats=%0d)", halt, err, rd_beats, wr_beats);
        $fatal(1, "timeout");
    end
endmodule
