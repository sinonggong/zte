// GDDR6 channel striping for the node array's AXI initiators.
//
// Why: the VP815's GDDR6 is 16 channels of 2 GB at NoC address k << 33 (measured 2026-09-18), and a node addresses one
// contiguous region per operand.  With a region in ONE channel, the 12 chain nodes that stream the same activation
// rows all queue on that channel (~30 GB/s): the half array ran its whole chunk 63 % slower than the calibrated model,
// and the same smoke chunk took 0.081 s with every region in channel 0 against 0.059 s with regions round-robin.
// Striping every address at 4 KB over the 16 channels lets any region use all of them.
//
// Logical address L (what nodes, programs and the host use; 32 GB flat, L < 2^35):
//     channel = L[S+3:S]  (S = STRIPE_LOG2, 12: 4 KB stripes)
//     offset  = {L[34:S+4], L[S-1:0]}          (31 bits: 2 GB per channel)
//     NoC     = channel << 33 | offset
// A burst must not cross a stripe edge (its two halves live in different channels), and the masters do not keep to
// 4 KB boundaries, so the read and write modules below cut such a burst in two:
//   axi_rd_stripe  AR -> one or two ARs; R passes straight through with the first part's RLAST hidden
//   axi_wr_stripe  AW -> one or two AWs; W beats pass through with WLAST re-made per part; the parts' B responses are
//                  merged into one (resp = OR)
// Both keep AXI order: all parts use the master's single ID and the NoC returns R / B in order per ID (the array has
// relied on that since the first silicon: every node already mixes channels within its outstanding window).
// The host bridge's accesses are PCIe TLPs (never across 4 KB) and only need stripe_addr.
package axi_stripe_pkg;
    function automatic [41:0] stripe_addr(input [41:0] l, input integer s);
        reg [3:0]  ch;
        reg [30:0] off;
        begin
            ch  = 4'((l >> s) & 42'hF);
            off = 31'((((l >> (s + 4)) << s) | (l & ((42'd1 << s) - 42'd1))));
            stripe_addr = {5'd0, ch, 2'b00, off};          // ch at bits [36:33]
        end
    endfunction
endpackage

module axi_rd_stripe #(
    parameter integer STRIPE_LOG2 = 12,
    parameter integer QDEPTH_LOG2 = 5               // parts in flight (>= the master's outstanding reads x 2)
) (
    input  wire         i_clk,
    input  wire         i_rstn,
    input  wire         i_en,              // striping on (static between runs); off: addresses and bursts pass as they are
    // from the master (logical)
    input  wire         s_arvalid,
    output wire         s_arready,
    input  wire [41:0]  s_araddr,
    input  wire [7:0]   s_arlen,
    input  wire [2:0]   s_arsize,
    input  wire [1:0]   s_arburst,
    output wire         s_rvalid,
    input  wire         s_rready,
    output wire [255:0] s_rdata,
    output wire [1:0]   s_rresp,
    output wire         s_rlast,
    // to the NAP (NoC addresses)
    output wire         m_arvalid,
    input  wire         m_arready,
    output wire [41:0]  m_araddr,
    output wire [7:0]   m_arlen,
    output wire [2:0]   m_arsize,
    output wire [1:0]   m_arburst,
    input  wire         m_rvalid,
    output wire         m_rready,
    input  wire [255:0] m_rdata,
    input  wire [1:0]   m_rresp,
    input  wire         m_rlast
);
    import axi_stripe_pkg::*;
    localparam integer SB = STRIPE_LOG2 - 5;         // beats per stripe = 2^SB (32-byte beats)
    // current master burst, cut at the stripe edge
    wire [SB-1:0] beat_in_stripe = s_araddr[STRIPE_LOG2-1:5];
    wire [SB:0]   room           = (SB+1)'(1 << SB) - (SB+1)'(beat_in_stripe);   // beats to the edge
    wire [8:0]    beats          = 9'(s_arlen) + 9'd1;
`ifdef AXI_STRIPE_NEG_NOSPLIT                                  // negative control: never cut (must FAIL tb_axi_stripe)
    wire          split          = 1'b0;
`else
    wire          split          = i_en && (9'(room) < beats);
`endif
    reg           second;                                     // part 2 of a split burst is next
    wire [8:0]    len1           = split ? 9'(room) : beats;
    wire [41:0]   a2             = s_araddr + {28'd0, len1, 5'd0};
    // the part on the bus
    wire [41:0]   part_addr      = second ? a2 : s_araddr;
    wire [8:0]    part_beats     = second ? (beats - len1) : len1;
    wire          part_hide      = !second && split;          // hide this part's RLAST
    // queue of RLAST masks, one per issued part
    reg  [(1<<QDEPTH_LOG2)-1:0] hideq;
    reg  [QDEPTH_LOG2:0]        qw, qr;
    wire          q_full   = (qw - qr) == (QDEPTH_LOG2+1)'(1 << QDEPTH_LOG2);
    assign m_arvalid = s_arvalid && !q_full;
    assign m_araddr  = i_en ? stripe_addr(part_addr, STRIPE_LOG2) : part_addr;
    assign m_arlen   = 8'(part_beats - 9'd1);
    assign m_arsize  = s_arsize;
    assign m_arburst = s_arburst;
    wire          ar_fire  = m_arvalid && m_arready;
    assign s_arready = ar_fire && !part_hide;                     // the master's AR completes with its last part
    // R: straight through, the first part's RLAST hidden
    wire          hide_now = hideq[qr[QDEPTH_LOG2-1:0]];
    assign s_rvalid = m_rvalid;
    assign m_rready = s_rready;
    assign s_rdata  = m_rdata;
    assign s_rresp  = m_rresp;
    assign s_rlast  = m_rlast && !hide_now;
    always @(posedge i_clk) begin
        if (!i_rstn) begin
            second <= 1'b0;
            qw     <= '0;
            qr     <= '0;
        end else begin
            if (ar_fire) begin
                hideq[qw[QDEPTH_LOG2-1:0]] <= part_hide;
                qw     <= qw + 1'b1;
                second <= part_hide;                              // after part 1 comes part 2, then a new burst
            end
            if (m_rvalid && s_rready && m_rlast) qr <= qr + 1'b1;
        end
    end
endmodule

module axi_wr_stripe #(
    parameter integer STRIPE_LOG2 = 12,
    parameter integer QDEPTH_LOG2 = 5
) (
    input  wire         i_clk,
    input  wire         i_rstn,
    input  wire         i_en,
    input  wire         s_awvalid,
    output wire         s_awready,
    input  wire [41:0]  s_awaddr,
    input  wire [7:0]   s_awlen,
    input  wire [2:0]   s_awsize,
    input  wire [1:0]   s_awburst,
    input  wire         s_wvalid,
    output wire         s_wready,
    input  wire [255:0] s_wdata,
    input  wire [31:0]  s_wstrb,
    input  wire         s_wlast,
    output wire         s_bvalid,
    input  wire         s_bready,
    output wire [1:0]   s_bresp,
    output wire         m_awvalid,
    input  wire         m_awready,
    output wire [41:0]  m_awaddr,
    output wire [7:0]   m_awlen,
    output wire [2:0]   m_awsize,
    output wire [1:0]   m_awburst,
    output wire         m_wvalid,
    input  wire         m_wready,
    output wire [255:0] m_wdata,
    output wire [31:0]  m_wstrb,
    output wire         m_wlast,
    input  wire         m_bvalid,
    output wire         m_bready,
    input  wire [1:0]   m_bresp
);
    import axi_stripe_pkg::*;
    localparam integer SB = STRIPE_LOG2 - 5;
    localparam integer QD = 1 << QDEPTH_LOG2;
    wire [SB-1:0] beat_in_stripe = s_awaddr[STRIPE_LOG2-1:5];
    wire [SB:0]   room           = (SB+1)'(1 << SB) - (SB+1)'(beat_in_stripe);
    wire [8:0]    beats          = 9'(s_awlen) + 9'd1;
`ifdef AXI_STRIPE_NEG_NOSPLIT                                  // negative control: never cut (must FAIL tb_axi_stripe)
    wire          split          = 1'b0;
`else
    wire          split          = i_en && (9'(room) < beats);
`endif
    reg           second;
    wire [8:0]    len1           = split ? 9'(room) : beats;
    wire [41:0]   a2             = s_awaddr + {28'd0, len1, 5'd0};
    wire [41:0]   part_addr      = second ? a2 : s_awaddr;
    wire [8:0]    part_beats     = second ? (beats - len1) : len1;
    wire          part_first     = !second && split;          // part 1 of a split burst (its B is absorbed)
    // W queue: the beat count of every issued part, in order; B queue: 1 = absorb this B (part 1 of a split)
    reg  [8:0]              wq [0:QD-1];
    reg  [QD-1:0]           bq;
    reg  [QDEPTH_LOG2:0]    ww, wr_, bw, br;
    wire          wq_full = (ww - wr_) == (QDEPTH_LOG2+1)'(QD);
    wire          bq_full = (bw - br) == (QDEPTH_LOG2+1)'(QD);
    assign m_awvalid = s_awvalid && !wq_full && !bq_full;
    assign m_awaddr  = i_en ? stripe_addr(part_addr, STRIPE_LOG2) : part_addr;
    assign m_awlen   = 8'(part_beats - 9'd1);
    assign m_awsize  = s_awsize;
    assign m_awburst = s_awburst;
    wire          aw_fire = m_awvalid && m_awready;
    assign s_awready = aw_fire && !part_first;
    // W: beats pass through once their part is known; WLAST at the end of every part
    wire          w_have  = (ww != wr_);
    reg  [8:0]    w_cnt;                                      // beats sent of the head part
    wire [8:0]    w_len   = wq[wr_[QDEPTH_LOG2-1:0]];
    wire          w_last  = (w_cnt + 9'd1 == w_len);
    assign m_wvalid = s_wvalid && w_have;
    assign s_wready = m_wready && w_have;
    assign m_wdata  = s_wdata;
    assign m_wstrb  = s_wstrb;
    assign m_wlast  = w_last;
    wire          w_fire  = m_wvalid && m_wready;
    // B: an absorbed part-1 response is dropped (its resp kept for the merge), the next one goes up
    reg  [1:0]    bresp_acc;
    wire          b_absorb = bq[br[QDEPTH_LOG2-1:0]];
    wire          b_have   = (bw != br);
    assign s_bvalid = m_bvalid && b_have && !b_absorb;
    assign s_bresp  = m_bresp | bresp_acc;
    assign m_bready = b_have && (b_absorb || s_bready);
    wire          b_fire   = m_bvalid && m_bready;
    always @(posedge i_clk) begin
        if (!i_rstn) begin
            second    <= 1'b0;
            ww        <= '0; wr_ <= '0; bw <= '0; br <= '0;
            w_cnt     <= 9'd0;
            bresp_acc <= 2'b00;
        end else begin
            if (aw_fire) begin
                wq[ww[QDEPTH_LOG2-1:0]] <= part_beats;
                bq[bw[QDEPTH_LOG2-1:0]] <= part_first;
                ww     <= ww + 1'b1;
                bw     <= bw + 1'b1;
                second <= part_first;
            end
            if (w_fire) begin
                if (w_last) begin
                    w_cnt <= 9'd0;
                    wr_   <= wr_ + 1'b1;
                end else begin
                    w_cnt <= w_cnt + 9'd1;
                end
            end
            if (b_fire) begin
                br        <= br + 1'b1;
                bresp_acc <= b_absorb ? (bresp_acc | m_bresp) : 2'b00;
            end
        end
    end
endmodule
