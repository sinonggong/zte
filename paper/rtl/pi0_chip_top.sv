// The whole node array as one chip-level design: N_CHAIN column-parallel INT8 chain nodes and
// N_VNODE vector nodes of N_LANE lanes, each on its own NAP, plus the host register window.
//
// Every node is an independent AXI initiator into the NoC (nap_axi_initiator, ACX_NAP_AXI_SLAVE), so
// it fetches its own program, weight images and activation rows out of GDDR6 and writes its results
// back without passing through anything shared in the fabric.  Between stages the nodes sequence each
// other with GDDR6 flags (the WAIT and POST commands, node_sync.sv), so within a chunk the host does
// nothing: it writes each node's program base and start bit once through pi0_chip_ctrl and waits for
// the halt vector.
//
// Three clock domains, as in every node's own synthesis:
//   i_clk_array   the MLP72 chain clock (725 MHz on silicon: docs/PI0_CLOCK_PLAN_20260917.md; 750 timed alone)
//   i_clk_fabric  the chain nodes' NAP / fabric clock and the host window (250 MHz)
//   i_clk_vec     the vector nodes' whole datapath and their NAPs (333 MHz measured for the 2-lane node)
// The chain node crosses between array and fabric internally (colpar_result_port.sv); nodes only meet in
// GDDR6, so nothing here crosses.
//
// PV attention needs uint8 activations against int8 weights, which is a static MLP72 multiplier mode
// (5'h13), so N_PV of the chain nodes are built that way and the compiler places the PV GEMMs on them.
// Everything else runs on the int8 chains.
module pi0_chip_top #(
    parameter F_GELU  = "vu_tbl_gelu.mem",
    parameter F_SIGM  = "vu_tbl_sigm.mem",
    parameter F_EXP   = "vu_tbl_exp.mem",
    parameter F_RSQRT = "vu_tbl_rsqrt.mem",
    parameter F_QUANT = "vu_tbl_quant.mem",
    parameter integer N_CHAIN  = 32,       // column-parallel INT8 chain nodes
    parameter integer N_PV     = 8,        // of those, built with uint8 x int8 stage multipliers
    parameter integer N_VNODE  = 4,        // vector nodes
    parameter integer N_LANE   = 4,        // lanes per vector node
    parameter integer N_STAGE  = 16,       // MLP72 stages per chain
    // A deep chain needs both fanout fixes to route and to hold 750 MHz (measured: 32 stages does not
    // route at all with the flat weight-load broadcast, and misses by 0.266 ns without the split bank).
    parameter integer WR_PIPE_EVERY = 0,
    parameter integer VALID_COPIES  = 1,
    // Mixed depths: the first N_DEEP int8 chains are DEEP_STAGE deep, built with both fanout fixes (WR_PIPE_EVERY 4,
    // VALID_COPIES 4); the other int8 chains and the PV chains are N_STAGE deep.  Deep chains halve the column tiles of
    // the wide GEMMs; shallow ones have more column groups to share the expert's small GEMMs out.
    parameter integer N_DEEP     = 0,
    parameter integer DEEP_STAGE = 32,
    // 1 or 2 NAPs per node.  With 2, reads go to one NAP and writes to the other: a node's AR/R and
    // AW/W/B are already separate ports, so this costs no RTL and no RLB -- and the NoC has 80 sites,
    // of which a 19-node array uses 23.  Per-transaction latency, not bandwidth, is what bounds the
    // chunk, so transactions in flight are worth more than bytes per second.
    parameter integer NAPS_PER_NODE = 1,
    parameter integer SLOT_BITS = 11,     // 12 for pi0 at full size (4,096-wide MLP rows)
    parameter integer N_LD     = 1,       // vector-node operand loaders (4: parallel loading, vu_rd_fanout.sv)
    parameter integer MAX_OUT  = 8,
    // NoC site of the host register window's NAP (4'hx: ACE chooses).  On the VP815 bitstreams the host's BAR0
    // maps NOC[3][4], so a real build pins it there.
    parameter         CTRL_NAP_COL = 4'hx,
    parameter         CTRL_NAP_ROW = 4'hx,
    parameter [23:0]  ID_WORD      = 24'h504930,
    parameter integer DBG          = 1,          // per-node debug snapshots readable by the host (a 128 x N_NODE mux)
    // host -> GDDR6 paging bridge (pi0_host_gddr_bridge.sv) behind its own target NAP, which BAR1 maps (256 MB);
    // the page base is the register window's beat N_NODE + 1.  0 = none.
    parameter integer HOST_BRIDGE  = 1,
    parameter         BRIDGE_NAP_COL = 4'hx,
    parameter         BRIDGE_NAP_ROW = 4'hx,
    // GDDR6 channel striping (paper/rtl/axi_stripe.sv): 0 = node and host addresses are NoC addresses as they are;
    // 12 = 4 KB stripes over the 16 channels (logical 32 GB), bursts cut at stripe edges, the host bridge translated too
    parameter integer STRIPE         = 0
) (
    input  wire i_clk_array,
    input  wire i_clk_fabric,
    input  wire i_clk_vec,
    input  wire i_rstn_array,
    input  wire i_rstn_fabric,
    input  wire i_rstn_vec,
    output wire o_all_halt,
    output wire o_any_error,
    output wire o_soft_rst           // host asked for a reset (one fabric cycle); the top stretches and applies it
);
    localparam integer N_NODE = N_CHAIN + N_VNODE;

    wire [N_NODE-1:0]    halt, node_err, nap_err;
    wire [8*N_NODE-1:0]  err_bits;
    wire [128*N_NODE-1:0] dbg;
    wire [42*N_NODE-1:0] prog_base;
    wire [N_NODE-1:0]    prog_start;

    // ---------------------------------------------------------------- host register window
    wire        c_arvalid, c_arready, c_rvalid, c_rready, c_rlast;
    wire        c_awvalid, c_awready, c_wvalid, c_wready, c_wlast, c_bvalid, c_bready;
    wire [27:0] c_araddr, c_awaddr;
    wire [7:0]  c_arlen, c_awlen;
    wire [2:0]  c_arsize, c_awsize;
    wire [1:0]  c_arburst, c_awburst, c_rresp, c_bresp;
    wire [255:0] c_rdata, c_wdata;
    wire [31:0] c_wstrb;
    wire        c_rstn, c_err_valid;
    wire [2:0]  c_err_info;

    nap_axi_target #(.COLUMN(CTRL_NAP_COL), .ROW(CTRL_NAP_ROW)) u_ctrl_nap (
        .i_clk(i_clk_fabric), .i_rstn(i_rstn_fabric), .o_output_rstn(c_rstn),
        .o_arvalid(c_arvalid), .i_arready(c_arready), .o_araddr(c_araddr), .o_arlen(c_arlen),
        .o_arsize(c_arsize), .o_arburst(c_arburst), .i_rvalid(c_rvalid), .o_rready(c_rready),
        .i_rdata(c_rdata), .i_rresp(c_rresp), .i_rlast(c_rlast),
        .o_awvalid(c_awvalid), .i_awready(c_awready), .o_awaddr(c_awaddr), .o_awlen(c_awlen),
        .o_awsize(c_awsize), .o_awburst(c_awburst), .o_wvalid(c_wvalid), .i_wready(c_wready),
        .o_wdata(c_wdata), .o_wstrb(c_wstrb), .o_wlast(c_wlast), .i_bvalid(c_bvalid),
        .o_bready(c_bready), .i_bresp(c_bresp),
        .o_error_valid(c_err_valid), .o_error_info(c_err_info));

    wire [41:0] page_base;          // host bridge page (declared before its first use: an implicit net is 1 bit)
    wire        stripe_en;          // channel striping on (window beat N_NODE + 1 bit 64), fabric clock
    pi0_chip_ctrl #(.N_NODE(N_NODE), .AXI_ADDR_WIDTH(28), .ID_WORD(ID_WORD)) u_ctrl (
        .i_clk(i_clk_fabric), .i_rstn(i_rstn_fabric & c_rstn),
        .i_arvalid(c_arvalid), .o_arready(c_arready), .i_araddr(c_araddr), .i_arlen(c_arlen),
        .o_rvalid(c_rvalid), .i_rready(c_rready), .o_rdata(c_rdata), .o_rresp(c_rresp), .o_rlast(c_rlast),
        .i_awvalid(c_awvalid), .o_awready(c_awready), .i_awaddr(c_awaddr),
        .i_wvalid(c_wvalid), .o_wready(c_wready), .i_wdata(c_wdata), .i_wstrb(c_wstrb), .i_wlast(c_wlast),
        .o_bvalid(c_bvalid), .i_bready(c_bready), .o_bresp(c_bresp),
        .o_prog_base(prog_base), .o_prog_start(prog_start), .o_soft_rst(o_soft_rst), .o_page_base(page_base), .o_stripe_en(stripe_en),
        .i_halt(halt), .i_error(node_err), .i_error_bits(err_bits), .i_dbg(DBG ? dbg : '0));

    // ---------------------------------------------------------------- host -> GDDR6 paging bridge
    wire        br_err;
    generate if (HOST_BRIDGE != 0) begin : g_bridge
        wire         t_arvalid, t_arready, t_rvalid, t_rready, t_rlast, t_awvalid, t_awready, t_wvalid, t_wready, t_wlast,
                     t_bvalid, t_bready;
        wire [27:0]  t_araddr, t_awaddr;
        wire [7:0]   t_arlen, t_awlen;
        wire [2:0]   t_arsize, t_awsize, t_err_info, m_err_info;
        wire [1:0]   t_arburst, t_awburst, t_rresp, t_bresp;
        wire [255:0] t_rdata, t_wdata;
        wire [31:0]  t_wstrb;
        wire         m_arvalid, m_arready, m_rvalid, m_rready, m_rlast, m_awvalid, m_awready, m_wvalid, m_wready, m_wlast,
                     m_bvalid, m_bready, t_rstn, m_rstn, t_err, m_err;
        wire [41:0]  m_araddr, m_awaddr;
        wire [7:0]   m_arlen, m_awlen;
        wire [2:0]   m_arsize, m_awsize;
        wire [1:0]   m_arburst, m_awburst, m_rresp, m_bresp;
        wire [255:0] m_rdata, m_wdata;
        wire [31:0]  m_wstrb;
        nap_axi_target #(.COLUMN(BRIDGE_NAP_COL), .ROW(BRIDGE_NAP_ROW)) u_br_tgt (
            .i_clk(i_clk_fabric), .i_rstn(i_rstn_fabric), .o_output_rstn(t_rstn),
            .o_arvalid(t_arvalid), .i_arready(t_arready), .o_araddr(t_araddr), .o_arlen(t_arlen),
            .o_arsize(t_arsize), .o_arburst(t_arburst), .i_rvalid(t_rvalid), .o_rready(t_rready),
            .i_rdata(t_rdata), .i_rresp(t_rresp), .i_rlast(t_rlast),
            .o_awvalid(t_awvalid), .i_awready(t_awready), .o_awaddr(t_awaddr), .o_awlen(t_awlen),
            .o_awsize(t_awsize), .o_awburst(t_awburst), .o_wvalid(t_wvalid), .i_wready(t_wready),
            .o_wdata(t_wdata), .o_wstrb(t_wstrb), .o_wlast(t_wlast), .i_bvalid(t_bvalid),
            .o_bready(t_bready), .i_bresp(t_bresp), .o_error_valid(t_err), .o_error_info(t_err_info));
        pi0_host_gddr_bridge u_bridge (
            .i_clk(i_clk_fabric), .i_rstn(i_rstn_fabric & t_rstn & m_rstn), .i_page_base(page_base),
            .t_arvalid(t_arvalid), .t_arready(t_arready), .t_araddr(t_araddr), .t_arlen(t_arlen), .t_arsize(t_arsize),
            .t_arburst(t_arburst), .t_rvalid(t_rvalid), .t_rready(t_rready), .t_rdata(t_rdata), .t_rresp(t_rresp),
            .t_rlast(t_rlast), .t_awvalid(t_awvalid), .t_awready(t_awready), .t_awaddr(t_awaddr), .t_awlen(t_awlen),
            .t_awsize(t_awsize), .t_awburst(t_awburst), .t_wvalid(t_wvalid), .t_wready(t_wready), .t_wdata(t_wdata),
            .t_wstrb(t_wstrb), .t_wlast(t_wlast), .t_bvalid(t_bvalid), .t_bready(t_bready), .t_bresp(t_bresp),
            .m_arvalid(m_arvalid), .m_arready(m_arready), .m_araddr(m_araddr), .m_arlen(m_arlen), .m_arsize(m_arsize),
            .m_arburst(m_arburst), .m_rvalid(m_rvalid), .m_rready(m_rready), .m_rdata(m_rdata), .m_rresp(m_rresp),
            .m_rlast(m_rlast), .m_awvalid(m_awvalid), .m_awready(m_awready), .m_awaddr(m_awaddr), .m_awlen(m_awlen),
            .m_awsize(m_awsize), .m_awburst(m_awburst), .m_wvalid(m_wvalid), .m_wready(m_wready), .m_wdata(m_wdata),
            .m_wstrb(m_wstrb), .m_wlast(m_wlast), .m_bvalid(m_bvalid), .m_bready(m_bready), .m_bresp(m_bresp),
            .o_n_reads(), .o_n_writes());
        // host accesses are PCIe TLPs, which never cross 4 KB: the logical -> NoC translation is all they need
        wire [41:0]  m_araddr_n = (STRIPE != 0 && stripe_en) ? axi_stripe_pkg::stripe_addr(m_araddr, STRIPE) : m_araddr;
        wire [41:0]  m_awaddr_n = (STRIPE != 0 && stripe_en) ? axi_stripe_pkg::stripe_addr(m_awaddr, STRIPE) : m_awaddr;
        nap_axi_initiator u_br_ini (
            .i_clk(i_clk_fabric), .i_rstn(i_rstn_fabric), .o_output_rstn(m_rstn),
            .i_arvalid(m_arvalid), .o_arready(m_arready), .i_araddr(m_araddr_n), .i_arlen(m_arlen),
            .i_arsize(m_arsize), .i_arburst(m_arburst), .i_arid(8'd0), .o_rvalid(m_rvalid), .i_rready(m_rready),
            .o_rdata(m_rdata), .o_rresp(m_rresp), .o_rlast(m_rlast), .o_rid(),
            .i_awvalid(m_awvalid), .o_awready(m_awready), .i_awaddr(m_awaddr_n), .i_awlen(m_awlen),
            .i_awsize(m_awsize), .i_awburst(m_awburst), .i_awid(8'd0), .i_wvalid(m_wvalid), .o_wready(m_wready),
            .i_wdata(m_wdata), .i_wstrb(m_wstrb), .i_wlast(m_wlast), .o_bvalid(m_bvalid),
            .i_bready(m_bready), .o_bresp(m_bresp), .o_bid(), .o_error_valid(m_err), .o_error_info(m_err_info));
        assign br_err = t_err | m_err;
    end else begin : g_no_bridge
        assign br_err = 1'b0;
    end endgenerate

    // ---------------------------------------------------------------- the array
    genvar g;
    generate
        for (g = 0; g < N_NODE; g = g + 1) begin : g_node
            wire         arvalid, arready, rvalid, rready, rlast;
            wire         awvalid, awready, wvalid, wready, wlast, bvalid, bready;
            wire [41:0]  araddr, awaddr;
            wire [7:0]   arlen, awlen;
            wire [2:0]   arsize, awsize;
            wire [1:0]   arburst, awburst, rresp, bresp;
            wire [255:0] rdata, wdata;
            wire [31:0]  wstrb;
            wire         nap_rstn;
            wire [2:0]   err_info;
            wire [41:0]  base = prog_base[42*g +: 42];
            // a chain node's NAPs run on the fabric clock, a vector node's on the vector clock
            wire         nclk  = (g < N_CHAIN) ? i_clk_fabric  : i_clk_vec;
            wire         nrstn = (g < N_CHAIN) ? i_rstn_fabric : i_rstn_vec;
            wire         n_halt, n_err;
            wire [7:0]   n_err_bits;
            wire [127:0] v_dbg;

            // the NAP side of the node's AXI master: straight, or through the channel-striping cutters
            wire         p_arvalid, p_arready, p_rvalid, p_rready, p_rlast, p_awvalid, p_awready, p_wvalid, p_wready,
                         p_wlast, p_bvalid, p_bready;
            wire [41:0]  p_araddr, p_awaddr;
            wire [7:0]   p_arlen, p_awlen;
            wire [2:0]   p_arsize, p_awsize;
            wire [1:0]   p_arburst, p_awburst, p_rresp, p_bresp;
            wire [255:0] p_rdata, p_wdata;
            wire [31:0]  p_wstrb;
            wire [7:0]   p_arid, p_awid, p_rid, p_bid;
            if (STRIPE != 0) begin : g_stripe
                // the host's striping switch, static between runs, into this node's clock
                (* syn_preserve = 1 *) reg [1:0] st_en_s;
                always @(posedge nclk) st_en_s <= nrstn ? {st_en_s[0], stripe_en} : 2'b00;
                // striping sends one master's bursts to different GDDR6 channels, and the NoC answers those out of
                // order (s1n, 2026-09-18): every burst gets its own ID and axi_id_reorder restores the order
                wire         z_arvalid, z_arready, z_rvalid, z_rready, z_rlast, z_awvalid, z_awready, z_bvalid, z_bready;
                wire [41:0]  z_araddr, z_awaddr;
                wire [7:0]   z_arlen, z_awlen;
                wire [2:0]   z_arsize, z_awsize;
                wire [1:0]   z_arburst, z_awburst, z_rresp, z_bresp;
                wire [255:0] z_rdata;
                axi_rd_stripe #(.STRIPE_LOG2(STRIPE)) u_rs (
                    .i_clk(nclk), .i_rstn(nrstn), .i_en(st_en_s[1]),
                    .s_arvalid(arvalid), .s_arready(arready), .s_araddr(araddr), .s_arlen(arlen), .s_arsize(arsize),
                    .s_arburst(arburst), .s_rvalid(rvalid), .s_rready(rready), .s_rdata(rdata), .s_rresp(rresp),
                    .s_rlast(rlast),
                    .m_arvalid(z_arvalid), .m_arready(z_arready), .m_araddr(z_araddr), .m_arlen(z_arlen),
                    .m_arsize(z_arsize), .m_arburst(z_arburst), .m_rvalid(z_rvalid), .m_rready(z_rready),
                    .m_rdata(z_rdata), .m_rresp(z_rresp), .m_rlast(z_rlast));
                axi_wr_stripe #(.STRIPE_LOG2(STRIPE)) u_ws (
                    .i_clk(nclk), .i_rstn(nrstn), .i_en(st_en_s[1]),
                    .s_awvalid(awvalid), .s_awready(awready), .s_awaddr(awaddr), .s_awlen(awlen), .s_awsize(awsize),
                    .s_awburst(awburst), .s_wvalid(wvalid), .s_wready(wready), .s_wdata(wdata), .s_wstrb(wstrb),
                    .s_wlast(wlast), .s_bvalid(bvalid), .s_bready(bready), .s_bresp(bresp),
                    .m_awvalid(z_awvalid), .m_awready(z_awready), .m_awaddr(z_awaddr), .m_awlen(z_awlen),
                    .m_awsize(z_awsize), .m_awburst(z_awburst), .m_wvalid(p_wvalid), .m_wready(p_wready),
                    .m_wdata(p_wdata), .m_wstrb(p_wstrb), .m_wlast(p_wlast), .m_bvalid(z_bvalid), .m_bready(z_bready),
                    .m_bresp(z_bresp));
                axi_id_reorder #(.TAG_LOG2(4)) u_ro (
                    .i_clk(nclk), .i_rstn(nrstn),
                    .s_arvalid(z_arvalid), .s_arready(z_arready), .s_araddr(z_araddr), .s_arlen(z_arlen), .s_arsize(z_arsize),
                    .s_arburst(z_arburst), .s_rvalid(z_rvalid), .s_rready(z_rready), .s_rdata(z_rdata), .s_rresp(z_rresp),
                    .s_rlast(z_rlast),
                    .s_awvalid(z_awvalid), .s_awready(z_awready), .s_awaddr(z_awaddr), .s_awlen(z_awlen), .s_awsize(z_awsize),
                    .s_awburst(z_awburst), .s_bvalid(z_bvalid), .s_bready(z_bready), .s_bresp(z_bresp),
                    .m_arvalid(p_arvalid), .m_arready(p_arready), .m_araddr(p_araddr), .m_arlen(p_arlen), .m_arsize(p_arsize),
                    .m_arburst(p_arburst), .m_arid(p_arid), .m_rvalid(p_rvalid), .m_rready(p_rready), .m_rdata(p_rdata),
                    .m_rresp(p_rresp), .m_rlast(p_rlast), .m_rid(p_rid),
                    .m_awvalid(p_awvalid), .m_awready(p_awready), .m_awaddr(p_awaddr), .m_awlen(p_awlen), .m_awsize(p_awsize),
                    .m_awburst(p_awburst), .m_awid(p_awid), .m_bvalid(p_bvalid), .m_bready(p_bready), .m_bresp(p_bresp),
                    .m_bid(p_bid));
            end else begin : g_straight
                assign p_arid = 8'd0;
                assign p_awid = 8'd0;
                assign p_arvalid = arvalid;  assign arready = p_arready;  assign p_araddr = araddr;
                assign p_arlen   = arlen;    assign p_arsize = arsize;    assign p_arburst = arburst;
                assign rvalid    = p_rvalid; assign p_rready = rready;    assign rdata = p_rdata;
                assign rresp     = p_rresp;  assign rlast = p_rlast;
                assign p_awvalid = awvalid;  assign awready = p_awready;  assign p_awaddr = awaddr;
                assign p_awlen   = awlen;    assign p_awsize = awsize;    assign p_awburst = awburst;
                assign p_wvalid  = wvalid;   assign wready = p_wready;    assign p_wdata = wdata;
                assign p_wstrb   = wstrb;    assign p_wlast = wlast;
                assign bvalid    = p_bvalid; assign p_bready = bready;    assign bresp = p_bresp;
            end
            wire nap_err_rd, nap_err_wr, nap_rstn_wr;
            if (NAPS_PER_NODE < 2) begin : g_one_nap
                nap_axi_initiator u_nap (
                    .i_clk(nclk), .i_rstn(nrstn), .o_output_rstn(nap_rstn),
                    .i_arvalid(p_arvalid), .o_arready(p_arready), .i_araddr(p_araddr), .i_arlen(p_arlen),
                    .i_arsize(p_arsize), .i_arburst(p_arburst), .i_arid(p_arid), .o_rvalid(p_rvalid), .i_rready(p_rready),
                    .o_rdata(p_rdata), .o_rresp(p_rresp), .o_rlast(p_rlast), .o_rid(p_rid),
                    .i_awvalid(p_awvalid), .o_awready(p_awready), .i_awaddr(p_awaddr), .i_awlen(p_awlen),
                    .i_awsize(p_awsize), .i_awburst(p_awburst), .i_awid(p_awid), .i_wvalid(p_wvalid), .o_wready(p_wready),
                    .i_wdata(p_wdata), .i_wstrb(p_wstrb), .i_wlast(p_wlast), .o_bvalid(p_bvalid),
                    .i_bready(p_bready), .o_bresp(p_bresp), .o_bid(p_bid),
                    .o_error_valid(nap_err_rd), .o_error_info(err_info));
                assign nap_err_wr  = 1'b0;
                assign nap_rstn_wr = 1'b1;
            end else begin : g_two_naps
                // reads (program fetch, weight images, activation rows) on one NAP
                nap_axi_initiator u_nap_rd (
                    .i_clk(nclk), .i_rstn(nrstn), .o_output_rstn(nap_rstn),
                    .i_arvalid(p_arvalid), .o_arready(p_arready), .i_araddr(p_araddr), .i_arlen(p_arlen),
                    .i_arsize(p_arsize), .i_arburst(p_arburst), .i_arid(p_arid), .o_rvalid(p_rvalid), .i_rready(p_rready),
                    .o_rdata(p_rdata), .o_rresp(p_rresp), .o_rlast(p_rlast), .o_rid(p_rid),
                    .i_awvalid(1'b0), .o_awready(), .i_awaddr(42'd0), .i_awlen(8'd0),
                    .i_awsize(3'd5), .i_awburst(2'b01), .i_awid(8'd0), .i_wvalid(1'b0), .o_wready(),
                    .i_wdata(256'd0), .i_wstrb(32'd0), .i_wlast(1'b0), .o_bvalid(),
                    .i_bready(1'b1), .o_bresp(), .o_bid(),
                    .o_error_valid(nap_err_rd), .o_error_info(err_info));
                // result writes and sync posts on the other
                nap_axi_initiator u_nap_wr (
                    .i_clk(nclk), .i_rstn(nrstn), .o_output_rstn(nap_rstn_wr),
                    .i_arvalid(1'b0), .o_arready(), .i_araddr(42'd0), .i_arlen(8'd0),
                    .i_arsize(3'd5), .i_arburst(2'b01), .i_arid(8'd0), .o_rvalid(), .i_rready(1'b1),
                    .o_rdata(), .o_rresp(), .o_rlast(), .o_rid(),
                    .i_awvalid(p_awvalid), .o_awready(p_awready), .i_awaddr(p_awaddr), .i_awlen(p_awlen),
                    .i_awsize(p_awsize), .i_awburst(p_awburst), .i_awid(p_awid), .i_wvalid(p_wvalid), .o_wready(p_wready),
                    .i_wdata(p_wdata), .i_wstrb(p_wstrb), .i_wlast(p_wlast), .o_bvalid(p_bvalid),
                    .i_bready(p_bready), .o_bresp(p_bresp), .o_bid(p_bid),
                    .o_error_valid(nap_err_wr), .o_error_info());
            end
            assign nap_err[g] = nap_err_rd | nap_err_wr;

            // The host window lives in the fabric domain.  A chain node's control side is that domain too; a
            // vector node runs entirely on i_clk_vec, so its start pulse is carried across as a toggle, its
            // program base is sampled in its own domain when that toggle arrives (the host wrote it long
            // before), and its halt / error come back through two flops.  Without this the first silicon
            // (2026-09-17) sometimes double-started a vector node and its first op's output came out shifted
            // by one element.
            wire        v_start;
            wire [41:0] v_base;
            if (g < N_CHAIN) begin : g_same_clock
                assign v_start = prog_start[g];
                assign v_base  = base;
                assign halt[g]     = n_halt;
                assign node_err[g] = n_err;
                assign err_bits[8*g +: 8] = n_err_bits;
            end else begin : g_cross_clock
                reg        start_tgl;
                (* syn_preserve = 1 *) reg [2:0] tgl_sync;
                reg        start_q;
                reg [41:0] base_q;
                (* syn_preserve = 1 *) reg [1:0] halt_sync, err_sync;
                reg [7:0]  err_bits_q;
                always @(posedge i_clk_fabric)
                    if (!i_rstn_fabric) start_tgl <= 1'b0;
                    else if (prog_start[g]) start_tgl <= ~start_tgl;
                always @(posedge i_clk_vec) begin
                    if (!i_rstn_vec) begin
                        tgl_sync <= 3'b000;
                        start_q  <= 1'b0;
                    end else begin
                        tgl_sync <= {tgl_sync[1:0], start_tgl};
                        start_q  <= tgl_sync[2] ^ tgl_sync[1];
                        if (tgl_sync[2] ^ tgl_sync[1]) base_q <= base;
                    end
                end
                always @(posedge i_clk_fabric) begin
                    halt_sync  <= i_rstn_fabric ? {halt_sync[0], n_halt} : 2'b00;
                    err_sync   <= i_rstn_fabric ? {err_sync[0], n_err}   : 2'b00;
                    err_bits_q <= n_err_bits;                       // sticky flags: safe to sample late
                end
                assign v_start = start_q;
                assign v_base  = base_q;
                assign halt[g]     = halt_sync[1];
                assign node_err[g] = err_sync[1];
                assign err_bits[8*g +: 8] = err_bits_q;
            end

            if (g < N_CHAIN) begin : g_chain
                localparam integer DEEP = (g < N_DEEP) && (g < N_CHAIN - N_PV);
                colpar_chain_node #(
                    .N_STAGE(DEEP ? DEEP_STAGE : N_STAGE), .ADDR_BITS(9), .PROG_FROM_GDDR6(1),
                    .RD_MAX_OUTSTANDING(MAX_OUT), .WR_MAX_OUTSTANDING(MAX_OUT),
                    .WR_FIFO_LOG2(8), .WR_AFULL_MARGIN(8),
                    .WR_PIPE_EVERY(DEEP ? 4 : WR_PIPE_EVERY), .VALID_COPIES(DEEP ? 4 : VALID_COPIES),
                    .MULT_MODE((g >= N_CHAIN - N_PV) ? 5'h13 : 5'h00)
                ) u_node (
                    .i_clk_array(i_clk_array), .i_clk_fabric(i_clk_fabric),
                    .i_rstn(i_rstn_fabric & i_rstn_array & nap_rstn & nap_rstn_wr),
                    .i_prog_start(prog_start[g]), .i_prog_base(base),
                    .i_cmd(128'd0), .i_cmd_valid(1'b0), .o_cmd_ready(),
                    .o_arvalid(arvalid), .i_arready(arready), .o_araddr(araddr), .o_arlen(arlen),
                    .o_arsize(arsize), .o_arburst(arburst), .o_rready(rready), .i_rvalid(rvalid),
                    .i_rdata(rdata), .i_rresp(rresp), .i_rlast(rlast),
                    .o_awvalid(awvalid), .i_awready(awready), .o_awaddr(awaddr), .o_awlen(awlen),
                    .o_awsize(awsize), .o_awburst(awburst), .o_wvalid(wvalid), .i_wready(wready),
                    .o_wdata(wdata), .o_wstrb(wstrb), .o_wlast(wlast), .i_bvalid(bvalid),
                    .o_bready(bready), .i_bresp(bresp),
                    .o_halt(n_halt), .o_error(n_err), .o_error_bits(n_err_bits[4:0]));
                assign n_err_bits[7:5] = 3'b000;
                assign dbg[128*g +: 128] = '0;
            end else begin : g_vector
                vu_node_ml #(
                    .F_GELU(F_GELU), .F_SIGM(F_SIGM), .F_EXP(F_EXP), .F_RSQRT(F_RSQRT), .F_QUANT(F_QUANT),
                    .N_LANE(N_LANE), .SLOT_BITS(SLOT_BITS), .N_LD(N_LD),
                    .RD_MAX_OUTSTANDING(MAX_OUT), .WR_MAX_OUTSTANDING(MAX_OUT), .WR_FIFO_LOG2(6)
                ) u_node (
                    .i_clk(i_clk_vec), .i_rstn(i_rstn_vec & nap_rstn & nap_rstn_wr),
                    .i_prog_start(v_start), .i_prog_base(v_base),
                    .o_arvalid(arvalid), .i_arready(arready), .o_araddr(araddr), .o_arlen(arlen),
                    .o_arsize(arsize), .o_arburst(arburst), .o_rready(rready), .i_rvalid(rvalid),
                    .i_rdata(rdata), .i_rresp(rresp), .i_rlast(rlast),
                    .o_awvalid(awvalid), .i_awready(awready), .o_awaddr(awaddr), .o_awlen(awlen),
                    .o_awsize(awsize), .o_awburst(awburst), .o_wvalid(wvalid), .i_wready(wready),
                    .o_wdata(wdata), .o_wstrb(wstrb), .o_wlast(wlast), .i_bvalid(bvalid),
                    .o_bready(bready), .i_bresp(bresp),
                    .o_halt(n_halt), .o_error(n_err), .o_error_bits(n_err_bits[6:0]), .o_dbg(v_dbg));
                assign n_err_bits[7] = 1'b0;
                assign dbg[128*g +: 128] = v_dbg;        // static between ops: sampled in the fabric domain as is
            end
        end
    endgenerate

    assign o_all_halt  = &halt;
    assign o_any_error = (|node_err) | (|nap_err) | c_err_valid | br_err;
endmodule
