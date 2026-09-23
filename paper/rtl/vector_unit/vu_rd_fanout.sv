// AXI4 read fan-out for the vector node's operand loaders (vu_node_ml.sv, N_LD > 1): N_M read masters share
// one read port without serialising their bursts.
//
//   AR   round-robin over the masters with ARVALID, issued while fewer than MAX_OUTSTANDING bursts are open;
//        each accepted AR pushes its master's index into an owner queue (one AXI ID, so R bursts come back in
//        AR order).
//   R    beats go to the owner at the head of the queue, into that master's beat FIFO; RLAST pops the queue.
//        The port's RREADY is the owner FIFO's space, so a master that consumes slowly only stalls the port
//        when its own FIFO is full.
//   out  each master reads its FIFO as an ordinary R channel (first-word fall-through).
//
// A master that keeps at most FIFO_BEATS / 16 bursts outstanding (its own MAX_OUTSTANDING) never fills its
// FIFO, so no master ever waits for another one's consumption.  Single clock.
module vu_rd_fanout #(
    parameter integer N_M             = 2,
    parameter integer AXI_ADDR_WIDTH  = 42,
    parameter integer MAX_OUTSTANDING = 8,
    parameter integer FIFO_LOG2       = 5          // beats per master FIFO = 2^FIFO_LOG2
) (
    input  wire                           i_clk,
    input  wire                           i_rstn,

    input  wire [N_M-1:0]                 s_arvalid,
    output wire [N_M-1:0]                 s_arready,
    input  wire [N_M*AXI_ADDR_WIDTH-1:0]  s_araddr,
    input  wire [N_M*8-1:0]               s_arlen,
    input  wire [N_M*3-1:0]               s_arsize,
    input  wire [N_M*2-1:0]               s_arburst,
    output wire [N_M-1:0]                 s_rvalid,
    input  wire [N_M-1:0]                 s_rready,
    output wire [N_M*256-1:0]             s_rdata,
    output wire [N_M*2-1:0]               s_rresp,
    output wire [N_M-1:0]                 s_rlast,

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
    input  wire                           m_rlast
);
    localparam integer IB = (N_M > 1) ? $clog2(N_M) : 1;
    localparam integer OB = $clog2(MAX_OUTSTANDING + 1);
    localparam integer QL = (MAX_OUTSTANDING > 1) ? $clog2(MAX_OUTSTANDING) : 1;   // owner queue: 2^QL entries
    localparam integer FD = 1 << FIFO_LOG2;
    initial if (MAX_OUTSTANDING > 1 && (1 << QL) != MAX_OUTSTANDING)
        $fatal(1, "vu_rd_fanout: MAX_OUTSTANDING must be 1 or a power of two");

    // ---------------------------------------------------------------- AR: round-robin, bounded outstanding
    reg  [IB-1:0] ar_sel;          // the master whose AR is presented
    reg           ar_hold;         // ar_sel is presenting an AR (held until the handshake)
    reg  [OB-1:0] open_n;          // bursts accepted and not yet finished (RLAST)
    wire          r_end;
    wire          ar_fire = m_arvalid && m_arready;

    function automatic [IB-1:0] next_req(input [IB-1:0] from, input [N_M-1:0] req);
        integer k;
        next_req = from;
        for (k = N_M; k >= 1; k = k - 1)
            if (req[(32'(from) + k) % N_M]) next_req = IB'((32'(from) + k) % N_M);
    endfunction

    assign m_arvalid = ar_hold;
    assign m_araddr  = s_araddr[ar_sel * AXI_ADDR_WIDTH +: AXI_ADDR_WIDTH];
    assign m_arlen   = s_arlen[ar_sel * 8 +: 8];
    assign m_arsize  = s_arsize[ar_sel * 3 +: 3];
    assign m_arburst = s_arburst[ar_sel * 2 +: 2];
    genvar gm;
    generate
        for (gm = 0; gm < N_M; gm = gm + 1) begin : g_arr
            assign s_arready[gm] = ar_hold && (ar_sel == IB'(gm)) && m_arready;
        end
    endgenerate

    // owner queue
    reg  [IB-1:0] owner [0:(1 << QL)-1];
    reg  [QL:0]   q_wr, q_rd;
    // the head owner, the queue's emptiness and each FIFO's fullness are registered from their next-cycle values
    // (exact): m_rready and the R push into the owner's block RAM were wp -> (wp - rp == FD) -> owner mux ->
    // rready -> push in one cycle, the vector node's longest path at 320 MHz (2026-09-22)
    reg  [IB-1:0] head;
    reg           q_empty;
    wire [QL:0]   q_wr_n = q_wr + (QL+1)'(ar_fire), q_rd_n = q_rd + (QL+1)'(r_end);

    always @(posedge i_clk) begin
        if (!i_rstn) begin
            ar_sel  <= '0;
            ar_hold <= 1'b0;
            open_n  <= '0;
            q_wr    <= '0;
            q_rd    <= '0;
            q_empty <= 1'b1;
            head    <= '0;
        end else begin
            open_n  <= open_n + OB'(ar_fire) - OB'(r_end);
            q_wr    <= q_wr_n;
            q_rd    <= q_rd_n;
            q_empty <= (q_wr_n == q_rd_n);
            head    <= (ar_fire && (q_wr[QL-1:0] == q_rd_n[QL-1:0])) ? ar_sel : owner[q_rd_n[QL-1:0]];
            if (ar_fire) begin
                owner[q_wr[QL-1:0]] <= ar_sel;
                ar_hold <= 1'b0;
                ar_sel  <= next_req(ar_sel, s_arvalid & ~(N_M'(1) << ar_sel));
            end else if (!ar_hold && 32'(open_n) + 32'(ar_fire) < MAX_OUTSTANDING) begin
                if (s_arvalid[ar_sel]) ar_hold <= 1'b1;
                else ar_sel <= next_req(ar_sel, s_arvalid);
            end
        end
    end

    // ---------------------------------------------------------------- R: to the head owner's FIFO
    wire [N_M-1:0] f_full;
    assign m_rready = !q_empty && !f_full[head];
    wire   r_fire   = m_rvalid && m_rready;
    assign r_end    = r_fire && m_rlast;

    generate
        for (gm = 0; gm < N_M; gm = gm + 1) begin : g_fifo
            // block RAM with a registered read, shown ahead: q_r holds entry rp (the read address is advanced
            // in the cycle of a pop), and an entry becomes visible one cycle after its write (wp_d)
            reg  [258:0] mem [0:FD-1] /* synthesis syn_ramstyle = "block_ram" */;
            reg  [258:0] q_r;
            reg  [FIFO_LOG2:0] wp, rp, wp_d;
            wire push = r_fire && (head == IB'(gm));
            wire pop  = s_rvalid[gm] && s_rready[gm];
            wire [FIFO_LOG2:0] rp_n = rp + (FIFO_LOG2+1)'(pop);
            wire [FIFO_LOG2:0] wp_n = wp + (FIFO_LOG2+1)'(push);
            reg                f_full_q;                     // one slot of margin: the flag is a cycle old
            always @(posedge i_clk) f_full_q <= i_rstn && ((wp_n - rp_n) >= (FIFO_LOG2+1)'(FD - 1));
            assign f_full[gm] = f_full_q;
            always @(posedge i_clk) begin
                if (push) mem[wp[FIFO_LOG2-1:0]] <= {m_rlast, m_rresp, m_rdata};
                q_r <= mem[rp_n[FIFO_LOG2-1:0]];
                if (!i_rstn) begin
                    wp   <= '0;
                    rp   <= '0;
                    wp_d <= '0;
                end else begin
                    if (push) wp <= wp + 1'b1;
                    rp   <= rp_n;
                    wp_d <= wp;
                end
            end
            assign s_rvalid[gm]              = (wp_d != rp);
            assign s_rdata[gm * 256 +: 256]  = q_r[255:0];
            assign s_rresp[gm * 2 +: 2]      = q_r[257:256];
            assign s_rlast[gm]               = q_r[258];
        end
    endgenerate
endmodule
