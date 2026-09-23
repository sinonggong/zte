// Burst-granular AXI4 sharing of one NAP by N_M chain nodes (colpar_chain_node.sv).
//
// Read and write sides are arbitrated independently, round-robin, one burst at a time:
//   read:  grant a master's AR, then route R to it until the RLAST handshake;
//   write: grant a master's AW, then route W to it until WLAST, then B to it.
// Bursts never interleave, so one AXI ID is enough and the NAP sees an ordinary master.
// The nodes send W only after their own AW handshake and keep ARVALID/AWVALID asserted
// until granted (AXI rule), which is all this mux relies on.  Single clock: the NAP clock.
// Payload buses toward the masters (RDATA, RRESP, RLAST, BRESP) are shared; only the
// valid/ready handshakes are steered.
module colpar_nap_mux #(
    parameter integer N_M            = 4,
    parameter integer AXI_ADDR_WIDTH = 42
) (
    input  wire                           i_clk,
    input  wire                           i_rstn,

    // masters, packed: master i at slice i
    input  wire [N_M-1:0]                 s_arvalid,
    output wire [N_M-1:0]                 s_arready,
    input  wire [N_M*AXI_ADDR_WIDTH-1:0]  s_araddr,
    input  wire [N_M*8-1:0]               s_arlen,
    input  wire [N_M*3-1:0]               s_arsize,
    input  wire [N_M*2-1:0]               s_arburst,
    output wire [N_M-1:0]                 s_rvalid,
    input  wire [N_M-1:0]                 s_rready,
    output wire [255:0]                   s_rdata,
    output wire [1:0]                     s_rresp,
    output wire                           s_rlast,

    input  wire [N_M-1:0]                 s_awvalid,
    output wire [N_M-1:0]                 s_awready,
    input  wire [N_M*AXI_ADDR_WIDTH-1:0]  s_awaddr,
    input  wire [N_M*8-1:0]               s_awlen,
    input  wire [N_M*3-1:0]               s_awsize,
    input  wire [N_M*2-1:0]               s_awburst,
    input  wire [N_M-1:0]                 s_wvalid,
    output wire [N_M-1:0]                 s_wready,
    input  wire [N_M*256-1:0]             s_wdata,
    input  wire [N_M*32-1:0]              s_wstrb,
    input  wire [N_M-1:0]                 s_wlast,
    output wire [N_M-1:0]                 s_bvalid,
    input  wire [N_M-1:0]                 s_bready,
    output wire [1:0]                     s_bresp,

    // the NAP (AXI4 master side)
    output wire                           m_arvalid,
    input  wire                           m_arready,
    output wire [AXI_ADDR_WIDTH-1:0]      m_araddr,
    output wire [7:0]                     m_arlen,
    output wire [2:0]                     m_arsize,
    output wire [1:0]                     m_arburst,
    input  wire                           m_rvalid,
    output wire                           m_rready,
    input  wire [255:0]                   m_rdata,
    input  wire [1:0]                     m_rresp,
    input  wire                           m_rlast,

    output wire                           m_awvalid,
    input  wire                           m_awready,
    output wire [AXI_ADDR_WIDTH-1:0]      m_awaddr,
    output wire [7:0]                     m_awlen,
    output wire [2:0]                     m_awsize,
    output wire [1:0]                     m_awburst,
    output wire                           m_wvalid,
    input  wire                           m_wready,
    output wire [255:0]                   m_wdata,
    output wire [31:0]                    m_wstrb,
    output wire                           m_wlast,
    input  wire                           m_bvalid,
    output wire                           m_bready,
    input  wire [1:0]                     m_bresp
);
    localparam integer IW = (N_M > 1) ? $clog2(N_M) : 1;

    // round-robin: the first requesting master after `last`; bit IW = found
    function automatic [IW:0] pick(input [N_M-1:0] req, input [IW-1:0] last);
        integer k, idx;
        pick = '0;
        for (k = 1; k <= N_M; k = k + 1) begin
            idx = (int'(last) + k) % N_M;
            if (req[idx] && !pick[IW]) pick = {1'b1, IW'(idx)};
        end
    endfunction

    // ---- read side ----
    reg          r_busy, r_ar_done;
    reg [IW-1:0] r_own, r_last;
    wire [IW:0]  r_pick = pick(s_arvalid, r_last);

    always @(posedge i_clk) begin
        if (!i_rstn) begin
            r_busy    <= 1'b0;
            r_ar_done <= 1'b0;
            r_own     <= '0;
            r_last    <= IW'(N_M - 1);
        end else if (!r_busy) begin
            if (r_pick[IW]) begin
                r_busy    <= 1'b1;
                r_ar_done <= 1'b0;
                r_own     <= r_pick[IW-1:0];
            end
        end else if (!r_ar_done) begin
            if (m_arvalid && m_arready) r_ar_done <= 1'b1;
        end else if (m_rvalid && m_rready && m_rlast) begin
            r_busy <= 1'b0;
            r_last <= r_own;
        end
    end

    assign m_arvalid = r_busy && !r_ar_done && s_arvalid[r_own];
    assign m_araddr  = s_araddr [r_own*AXI_ADDR_WIDTH +: AXI_ADDR_WIDTH];
    assign m_arlen   = s_arlen  [r_own*8 +: 8];
    assign m_arsize  = s_arsize [r_own*3 +: 3];
    assign m_arburst = s_arburst[r_own*2 +: 2];
    assign m_rready  = r_busy && r_ar_done && s_rready[r_own];
    assign s_rdata   = m_rdata;
    assign s_rresp   = m_rresp;
    assign s_rlast   = m_rlast;

    // ---- write side ----
    reg          w_busy, w_aw_done, w_w_done;
    reg [IW-1:0] w_own, w_last;
    wire [IW:0]  w_pick = pick(s_awvalid, w_last);

    always @(posedge i_clk) begin
        if (!i_rstn) begin
            w_busy    <= 1'b0;
            w_aw_done <= 1'b0;
            w_w_done  <= 1'b0;
            w_own     <= '0;
            w_last    <= IW'(N_M - 1);
        end else if (!w_busy) begin
            if (w_pick[IW]) begin
                w_busy    <= 1'b1;
                w_aw_done <= 1'b0;
                w_w_done  <= 1'b0;
                w_own     <= w_pick[IW-1:0];
            end
        end else if (!w_aw_done) begin
            if (m_awvalid && m_awready) w_aw_done <= 1'b1;
        end else if (!w_w_done) begin
            if (m_wvalid && m_wready && m_wlast) w_w_done <= 1'b1;
        end else if (m_bvalid && m_bready) begin
            w_busy <= 1'b0;
            w_last <= w_own;
        end
    end

    assign m_awvalid = w_busy && !w_aw_done && s_awvalid[w_own];
    assign m_awaddr  = s_awaddr [w_own*AXI_ADDR_WIDTH +: AXI_ADDR_WIDTH];
    assign m_awlen   = s_awlen  [w_own*8 +: 8];
    assign m_awsize  = s_awsize [w_own*3 +: 3];
    assign m_awburst = s_awburst[w_own*2 +: 2];
    assign m_wvalid  = w_busy && w_aw_done && !w_w_done && s_wvalid[w_own];
    assign m_wdata   = s_wdata[w_own*256 +: 256];
    assign m_wstrb   = s_wstrb[w_own*32 +: 32];
    assign m_wlast   = s_wlast[w_own];
    assign m_bready  = w_busy && w_w_done && s_bready[w_own];
    assign s_bresp   = m_bresp;

    genvar i;
    generate
        for (i = 0; i < N_M; i = i + 1) begin : g_m
            assign s_arready[i] = r_busy && !r_ar_done && (r_own == IW'(i)) && m_arready;
            assign s_rvalid[i]  = r_busy && r_ar_done && (r_own == IW'(i)) && m_rvalid;
            assign s_awready[i] = w_busy && !w_aw_done && (w_own == IW'(i)) && m_awready;
            assign s_wready[i]  = w_busy && w_aw_done && !w_w_done && (w_own == IW'(i)) && m_wready;
            assign s_bvalid[i]  = w_busy && w_w_done && (w_own == IW'(i)) && m_bvalid;
        end
    endgenerate
endmodule
