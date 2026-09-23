// Restores request order for a master whose bursts the NoC may answer out of order.
//
// Why: the VP815's NoC returns the responses of one AXI ID out of order when they come from different GDDR6 channels
// (bitstream s1n, 2026-09-18: with 4 KB channel striping the vector nodes' loaders saw RLAST in the wrong place and the
// 32-stage chains' writers failed, while the same chunk against an in-order memory model was bit-exact).  The node
// masters and axi_stripe.sv rely on in-order responses.  This block gives every burst in flight its own ID (a tag)
// and hands the responses back in request order:
//
//   AR  tag = issue count mod 2^TAG_LOG2, driven as ARID; at most 2^TAG_LOG2 read bursts open.
//   R   each beat is written into the reorder RAM at (RID, its beat index); the head tag's beats leave in order as
//       soon as they are there (a burst that arrives in order streams through with two cycles of latency).
//   AW  tag as AWID, at most 2^TAG_LOG2 write bursts open; W passes straight through (AXI4 W carries no ID and the
//       NoC takes W in AW order).
//   B   a B marks its tag done; B responses leave in AW order.
// Bursts are at most 16 beats (the NoC limit).  RAM: 2^TAG_LOG2 x 16 x (256 + 3) bits (TAG_LOG2 4: two BRAM72K).
// Single clock.
module axi_id_reorder #(
    parameter integer TAG_LOG2 = 4
) (
    input  wire         i_clk,
    input  wire         i_rstn,
    // master side (in-order AXI, no IDs)
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
    input  wire         s_awvalid,
    output wire         s_awready,
    input  wire [41:0]  s_awaddr,
    input  wire [7:0]   s_awlen,
    input  wire [2:0]   s_awsize,
    input  wire [1:0]   s_awburst,
    output wire         s_bvalid,
    input  wire         s_bready,
    output wire [1:0]   s_bresp,
    // NAP side (IDs)
    output wire         m_arvalid,
    input  wire         m_arready,
    output wire [41:0]  m_araddr,
    output wire [7:0]   m_arlen,
    output wire [2:0]   m_arsize,
    output wire [1:0]   m_arburst,
    output wire [7:0]   m_arid,
    input  wire         m_rvalid,
    output wire         m_rready,
    input  wire [255:0] m_rdata,
    input  wire [1:0]   m_rresp,
    input  wire         m_rlast,
    input  wire [7:0]   m_rid,
    output wire         m_awvalid,
    input  wire         m_awready,
    output wire [41:0]  m_awaddr,
    output wire [7:0]   m_awlen,
    output wire [2:0]   m_awsize,
    output wire [1:0]   m_awburst,
    output wire [7:0]   m_awid,
    input  wire         m_bvalid,
    output wire         m_bready,
    input  wire [1:0]   m_bresp,
    input  wire [7:0]   m_bid
);
    localparam integer NT = 1 << TAG_LOG2;
    localparam integer TL = TAG_LOG2;

    // ================================================================ reads
    reg  [TL:0]   r_iss, r_ret;                      // tags issued / retired (head = r_ret)
    wire          r_full  = (r_iss - r_ret) == (TL+1)'(NT);
    assign m_arvalid = s_arvalid && !r_full;
    assign s_arready = m_arready && !r_full;
    assign m_araddr  = s_araddr;
    assign m_arlen   = s_arlen;
    assign m_arsize  = s_arsize;
    assign m_arburst = s_arburst;
    assign m_arid    = 8'(r_iss[TL-1:0]);
    wire          ar_fire = m_arvalid && m_arready;

    // incoming beats: counted per tag, written at (tag, beat index); RLAST marks the burst complete
    reg  [4:0]    r_cnt  [0:NT-1];
    reg  [NT-1:0] r_last;
    reg  [258:0]  rmem   [0:NT*16-1] /* synthesis syn_ramstyle = "block_ram" */;
    assign m_rready = 1'b1;                          // a tag's 16 slots are reserved at issue: never full
    wire [TL-1:0] w_tag = m_rid[TL-1:0];

    // outgoing: the head tag's beats in order, through a registered-read RAM and a two-entry output queue
    wire [TL-1:0] h_tag  = r_ret[TL-1:0];
    reg  [3:0]    h_pos;
    wire          h_open = (r_iss != r_ret);
    reg           rd_v;
    reg  [258:0]  rd_q;
    reg  [258:0]  oq [0:1];
    reg  [1:0]    oq_n;
    wire          o_pop  = (oq_n != 2'd0) && s_rready;
    wire [2:0]    room   = 3'd2 - 3'(oq_n) - 3'(rd_v) + 3'(o_pop);
    wire          do_rd  = h_open && (5'(h_pos) < r_cnt[h_tag]) && (room != 3'd0);
    wire          h_end  = do_rd && r_last[h_tag] && (5'(h_pos) + 5'd1 == r_cnt[h_tag]);   // the burst's last beat
    assign s_rvalid = (oq_n != 2'd0);
    assign s_rdata  = oq[0][255:0];
    assign s_rresp  = oq[0][257:256];
    assign s_rlast  = oq[0][258];
    always @(posedge i_clk) begin
        if (m_rvalid) rmem[{w_tag, r_cnt[w_tag][3:0]}] <= {m_rlast, m_rresp, m_rdata};
        rd_q <= rmem[{h_tag, h_pos}];
    end
    integer k;
    always @(posedge i_clk) begin
        if (!i_rstn) begin
            r_iss <= '0; r_ret <= '0; h_pos <= 4'd0; rd_v <= 1'b0; oq_n <= 2'd0; r_last <= '0;
            for (k = 0; k < NT; k = k + 1) r_cnt[k] <= 5'd0;
        end else begin
            if (ar_fire) r_iss <= r_iss + 1'b1;
            if (m_rvalid) begin
                r_cnt[w_tag] <= r_cnt[w_tag] + 5'd1;
                if (m_rlast) r_last[w_tag] <= 1'b1;
            end
            rd_v <= do_rd;
            if (h_end) begin                         // retire the head tag (no beat of it can arrive any more)
                r_ret         <= r_ret + 1'b1;
                h_pos         <= 4'd0;
                r_cnt[h_tag]  <= 5'd0;
                r_last[h_tag] <= 1'b0;
            end else if (do_rd) begin
                h_pos <= h_pos + 4'd1;
            end
            // output queue: pop the front, append the RAM word read last cycle
            case ({o_pop, rd_v})
            2'b10: begin oq[0] <= oq[1]; oq_n <= oq_n - 2'd1; end
            2'b01: begin if (oq_n == 2'd0) oq[0] <= rd_q; else oq[1] <= rd_q; oq_n <= oq_n + 2'd1; end
            2'b11: begin if (oq_n == 2'd1) oq[0] <= rd_q; else begin oq[0] <= oq[1]; oq[1] <= rd_q; end end
            default: ;
            endcase
        end
    end

    // ================================================================ writes
    reg  [TL:0]   w_iss, w_ret;
    wire          w_full = (w_iss - w_ret) == (TL+1)'(NT);
    assign m_awvalid = s_awvalid && !w_full;
    assign s_awready = m_awready && !w_full;
    assign m_awaddr  = s_awaddr;
    assign m_awlen   = s_awlen;
    assign m_awsize  = s_awsize;
    assign m_awburst = s_awburst;
    assign m_awid    = 8'(w_iss[TL-1:0]);
    reg  [NT-1:0] b_done;
    reg  [1:0]    b_resp [0:NT-1];
    assign m_bready  = 1'b1;
    wire [TL-1:0] b_head = w_ret[TL-1:0];
    assign s_bvalid  = (w_iss != w_ret) && b_done[b_head];
    assign s_bresp   = b_resp[b_head];
    always @(posedge i_clk) begin
        if (!i_rstn) begin
            w_iss <= '0; w_ret <= '0; b_done <= '0;
        end else begin
            if (m_awvalid && m_awready) w_iss <= w_iss + 1'b1;
            if (m_bvalid) begin
                b_done[m_bid[TL-1:0]] <= 1'b1;
                b_resp[m_bid[TL-1:0]] <= m_bresp;
            end
            if (s_bvalid && s_bready) begin
                w_ret <= w_ret + 1'b1;
                b_done[b_head] <= 1'b0;           // its tag is not re-issued before this retires
            end
        end
    end
endmodule
