// AXI4 write merge for the vector node's two beat writers (element and summary) onto one write port, with bursts
// of both writers in flight at once.
//
// Why: colpar_nap_mux.sv grants one burst at a time and holds the grant until its B response, so the node had ONE
// write burst open whatever the writers' MAX_OUTSTANDING.  With a GDDR6 round trip on the write response that is
// the node's bottleneck: the smoke chunk took 1.23x its ideal-memory cycles with a 128-cycle B round trip, and 8,
// 16 or 32 open writes per writer made no difference (tb_pi0_chunk PER_NODE_MEM=1, 2026-09-18).
//
//   AW  round-robin over the writers with AWVALID; the accepted AW pushes its writer's index into the W queue and
//       the B queue (one AXI ID: W data follow AW order, B responses come back in AW order).
//   W   from the writer at the head of the W queue; WLAST pops it.
//   B   to the writer at the head of the B queue; the B handshake pops it.
// A writer sends W only after its own AW handshake and only for beats it already holds (vu_beat_writer.sv), so the
// head writer always has its data: no deadlock.  Q_LOG2 must cover the writers' outstanding bursts together.
// Single clock.
module vu_wr_merge #(
    parameter integer N_M            = 2,
    parameter integer AXI_ADDR_WIDTH = 42,
    parameter integer Q_LOG2         = 4
) (
    input  wire                           i_clk,
    input  wire                           i_rstn,
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
    output wire                           m_bready
);
    localparam integer IB = (N_M > 1) ? $clog2(N_M) : 1;
    localparam integer QD = 1 << Q_LOG2;

    // ---------------------------------------------------------------- queues of burst owners, in AW order
    reg  [IB-1:0]   wq [0:QD-1];
    reg  [IB-1:0]   bq [0:QD-1];
    reg  [Q_LOG2:0] ww, wr, bw, br;
    wire            q_full = ((ww - wr) == (Q_LOG2+1)'(QD)) || ((bw - br) == (Q_LOG2+1)'(QD));
    wire [IB-1:0]   w_own  = wq[wr[Q_LOG2-1:0]];
    wire [IB-1:0]   b_own  = bq[br[Q_LOG2-1:0]];
    wire            w_have = (ww != wr);
    wire            b_have = (bw != br);

    // ---------------------------------------------------------------- AW: round-robin, one presented at a time
    reg  [IB-1:0]   aw_sel;
    reg             aw_hold;
    function automatic [IB-1:0] next_req(input [IB-1:0] from, input [N_M-1:0] req);
        integer k;
        next_req = from;
        for (k = N_M; k >= 1; k = k - 1)
            if (req[(32'(from) + k) % N_M]) next_req = IB'((32'(from) + k) % N_M);
    endfunction
    assign m_awvalid = aw_hold;
    assign m_awaddr  = s_awaddr[aw_sel * AXI_ADDR_WIDTH +: AXI_ADDR_WIDTH];
    assign m_awlen   = s_awlen[aw_sel * 8 +: 8];
    assign m_awsize  = s_awsize[aw_sel * 3 +: 3];
    assign m_awburst = s_awburst[aw_sel * 2 +: 2];
    wire            aw_fire = m_awvalid && m_awready;
    genvar gm;
    generate
        for (gm = 0; gm < N_M; gm = gm + 1) begin : g_m
            assign s_awready[gm] = aw_fire && (aw_sel == IB'(gm));
            assign s_wready[gm]  = w_have && (w_own == IB'(gm)) && m_wready;
            assign s_bvalid[gm]  = b_have && (b_own == IB'(gm)) && m_bvalid;
        end
    endgenerate

    // ---------------------------------------------------------------- W and B routed by the queue heads
    assign m_wvalid = w_have && s_wvalid[w_own];
    assign m_wdata  = s_wdata[w_own * 256 +: 256];
    assign m_wstrb  = s_wstrb[w_own * 32 +: 32];
    assign m_wlast  = s_wlast[w_own];
    assign m_bready = b_have && s_bready[b_own];
    wire            w_end  = m_wvalid && m_wready && m_wlast;
    wire            b_fire = m_bvalid && m_bready;

    always @(posedge i_clk) begin
        if (!i_rstn) begin
            aw_sel  <= '0;
            aw_hold <= 1'b0;
            ww <= '0; wr <= '0; bw <= '0; br <= '0;
        end else begin
            if (aw_fire) begin
                wq[ww[Q_LOG2-1:0]] <= aw_sel;
                bq[bw[Q_LOG2-1:0]] <= aw_sel;
                ww      <= ww + 1'b1;
                bw      <= bw + 1'b1;
                aw_hold <= 1'b0;                          // re-arbitrate: AWVALID of the next pick next cycle
            end else if (!aw_hold && !q_full && (s_awvalid != '0)) begin
                aw_sel  <= next_req(aw_sel, s_awvalid);
                aw_hold <= 1'b1;
            end
            if (w_end)  wr <= wr + 1'b1;
            if (b_fire) br <= br + 1'b1;
        end
    end
endmodule
