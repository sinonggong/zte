// The two NAPs the node array needs, with plain wire ports.
//
// Achronix names a NAP after the role the NAP itself plays on the AXI link, which is the opposite of
// the fabric's role:
//
//   ACX_NAP_AXI_SLAVE   the FABRIC is the initiator: it drives arvalid / a 42-bit araddr and the NAP
//                       returns the data.  This is what every chain and vector node hangs on to read
//                       its program, weights and activations out of GDDR6 and to write results back.
//                       `src/rtl/nap_slave_wrapper.sv` is the same primitive behind the vendor's
//                       `t_AXI4` interface, and is what each deployed TC already uses.
//   ACX_NAP_AXI_MASTER  the NoC is the initiator and the FABRIC responds, over a 28-bit address.  This
//                       is the host register path: the host writes a node's program base and start bit
//                       and reads its halt and error, the same role `src/rtl/reg_control_block.sv` has.
//
// A generate loop over 36 nodes is clearer with wires than with an interface array, so these wrappers
// exist beside the vendor ones rather than instead of them.

module nap_axi_initiator #(          // fabric -> NoC (ACX_NAP_AXI_SLAVE)
    parameter        COLUMN        = 4'hx,
    parameter        ROW           = 4'hx,
    parameter [31:0] E2W_ARB_SCHED = 32'hffffffff,
    parameter [31:0] W2E_ARB_SCHED = 32'hffffffff
) (
    input  wire         i_clk,
    input  wire         i_rstn,
    output wire         o_output_rstn,

    input  wire         i_arvalid,
    output wire         o_arready,
    input  wire [41:0]  i_araddr,
    input  wire [7:0]   i_arlen,
    input  wire [2:0]   i_arsize,
    input  wire [1:0]   i_arburst,
    input  wire [7:0]   i_arid,              // 0 unless the master reorders by ID (axi_id_reorder.sv)
    output wire         o_rvalid,
    input  wire         i_rready,
    output wire [255:0] o_rdata,
    output wire [1:0]   o_rresp,
    output wire         o_rlast,
    output wire [7:0]   o_rid,

    input  wire         i_awvalid,
    output wire         o_awready,
    input  wire [41:0]  i_awaddr,
    input  wire [7:0]   i_awlen,
    input  wire [2:0]   i_awsize,
    input  wire [1:0]   i_awburst,
    input  wire [7:0]   i_awid,
    input  wire         i_wvalid,
    output wire         o_wready,
    input  wire [255:0] i_wdata,
    input  wire [31:0]  i_wstrb,
    input  wire         i_wlast,
    output wire         o_bvalid,
    input  wire         i_bready,
    output wire [1:0]   o_bresp,
    output wire [7:0]   o_bid,

    output wire         o_error_valid,
    output wire [2:0]   o_error_info
);
    ACX_NAP_AXI_SLAVE #(
        .column(COLUMN), .row(ROW), .must_keep(1), .ew_nap_ott_enable(0),
        .e2w_arbitration_schedule(E2W_ARB_SCHED), .w2e_arbitration_schedule(W2E_ARB_SCHED)
    ) i_nap (
        .clk(i_clk), .rstn(i_rstn), .output_rstn(o_output_rstn),
        .arready(o_arready), .arvalid(i_arvalid), .arqos(4'd0), .arburst(i_arburst),
        .arlock(1'b0), .arsize(i_arsize), .arlen(i_arlen), .arid(i_arid), .araddr(i_araddr),
        .awready(o_awready), .awvalid(i_awvalid), .awqos(4'd0), .awburst(i_awburst),
        .awlock(1'b0), .awsize(i_awsize), .awlen(i_awlen), .awid(i_awid), .awaddr(i_awaddr),
        .wready(o_wready), .wvalid(i_wvalid), .wdata(i_wdata), .wstrb(i_wstrb), .wlast(i_wlast),
        .rready(i_rready), .rvalid(o_rvalid), .rresp(o_rresp), .rid(o_rid), .rdata(o_rdata), .rlast(o_rlast),
        .bready(i_bready), .bvalid(o_bvalid), .bid(o_bid), .bresp(o_bresp),
        .error_valid(o_error_valid), .error_info(o_error_info)) /* synthesis syn_noprune=1 */;
endmodule


module nap_axi_target #(             // NoC -> fabric (ACX_NAP_AXI_MASTER), the host register path
    parameter        COLUMN        = 4'hx,
    parameter        ROW           = 4'hx,
    parameter [31:0] N2S_ARB_SCHED = 32'hffffffff,
    parameter [31:0] S2N_ARB_SCHED = 32'hffffffff
) (
    input  wire         i_clk,
    input  wire         i_rstn,
    output wire         o_output_rstn,

    output wire         o_arvalid,
    input  wire         i_arready,
    output wire [27:0]  o_araddr,
    output wire [7:0]   o_arlen,
    output wire [2:0]   o_arsize,
    output wire [1:0]   o_arburst,
    input  wire         i_rvalid,
    output wire         o_rready,
    input  wire [255:0] i_rdata,
    input  wire [1:0]   i_rresp,
    input  wire         i_rlast,

    output wire         o_awvalid,
    input  wire         i_awready,
    output wire [27:0]  o_awaddr,
    output wire [7:0]   o_awlen,
    output wire [2:0]   o_awsize,
    output wire [1:0]   o_awburst,
    output wire         o_wvalid,
    input  wire         i_wready,
    output wire [255:0] o_wdata,
    output wire [31:0]  o_wstrb,
    output wire         o_wlast,
    input  wire         i_bvalid,
    output wire         o_bready,
    input  wire [1:0]   i_bresp,

    output wire         o_error_valid,
    output wire [2:0]   o_error_info
);
    // The NoC initiator (the PCIe bridge) matches responses to requests by AXI ID, so the ID of each accepted
    // request is echoed on its response (as src/rtl/reg_control_block.sv does).  With rid / bid tied to 0 the
    // first S0 bitstream answered every BAR0 read with 0xffffffff (2026-09-17).  The window behind this NAP
    // holds one read and one write transaction at a time, so one register per channel is enough.
    wire [7:0] arid, awid;
    reg  [7:0] rid_q, bid_q;
    always @(posedge i_clk) begin
        if (o_arvalid && i_arready) rid_q <= arid;
        if (o_awvalid && i_awready) bid_q <= awid;
    end
    ACX_NAP_AXI_MASTER #(
        .column(COLUMN), .row(ROW), .must_keep(1),
        .n2s_arbitration_schedule(N2S_ARB_SCHED), .s2n_arbitration_schedule(S2N_ARB_SCHED)
    ) i_nap (
        .clk(i_clk), .rstn(i_rstn), .output_rstn(o_output_rstn),
        .arready(i_arready), .arvalid(o_arvalid), .arqos(), .arburst(o_arburst),
        .arlock(), .arsize(o_arsize), .arlen(o_arlen), .arid(arid), .araddr(o_araddr),
        .awready(i_awready), .awvalid(o_awvalid), .awqos(), .awburst(o_awburst),
        .awlock(), .awsize(o_awsize), .awlen(o_awlen), .awid(awid), .awaddr(o_awaddr),
        .wready(i_wready), .wvalid(o_wvalid), .wlast(o_wlast), .wstrb(o_wstrb), .wdata(o_wdata),
        .rready(o_rready), .rvalid(i_rvalid), .rresp(i_rresp), .rid(rid_q), .rlast(i_rlast), .rdata(i_rdata),
        .bready(o_bready), .bvalid(i_bvalid), .bid(bid_q), .bresp(i_bresp),
        .error_valid(o_error_valid), .error_info(o_error_info)) /* synthesis syn_noprune=1 */;
endmodule
