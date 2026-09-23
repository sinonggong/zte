// Result writer: fabric-clock row-pass results -> AXI4 write bursts into GDDR6 (NAP master).
//
// A record is one row pass of one chain: its N_STAGE column sums truncated to SUM_BITS (32 by
// default).  Truncation is exact while the compiler keeps K <= 8,192 bytes per slice
// (|acc| <= 127^2 * 8192 < 2^27; uint8-P attention <= 255 * 127 * T_k).  Records are
// little-endian (stage 1 in bits [SUM_BITS-1:0]) and zero-padded to BPR = ceil(N_STAGE *
// SUM_BITS / 256) beats: 2 beats (64 bytes) for N_STAGE = 16.
//
// Storage, RECORD_FIFO = 0 (the original): records enter a 2-record skid and are serialised one beat
// per cycle into a beat FIFO (256 bits x 2^FIFO_LOG2, registered read).  A record-wide FIFO (768 bits
// x 128) mapped to eleven BRAM72K_SDP; every BRAM72K site it takes is a site a chain stage cannot use.
// RECORD_FIFO = 1: the record is written into a record-wide block-RAM FIFO in the cycle i_sums_valid is
// high, straight from i_sums (colpar_result_port with FABRIC_COPY = 0 hands over its array-clock bank,
// stable for that cycle), and beats are muxed out of the RAM's read register.  No fabric copy of the
// sums, no skid, no serialiser: 4 x N_STAGE x SUM_BITS fewer flops (2,048 at 16 stages), for 2^(FIFO_LOG2
// - log2 BPR) records of BRAM (the same number of beats).  A record takes BPR + 1 cycles to leave, far
// less than the >= 14 array cycles between records.
//
// AXI: bursts of at most BURST_BEATS beats (the NoC carries at most 16, UG086), up to
// MAX_OUTSTANDING of them issued ahead of their B responses (one AXI ID).  An AW is raised for
// beats already queued and not yet assigned to a burst, when BURST_BEATS of them wait, when the
// FIFO is almost full, or while i_flush is high; W beats of a burst are sent only after its AW
// handshake, WLAST positions come from a queue of issued burst lengths.  Consecutive bursts
// write consecutive addresses from i_out_base (re-armed by i_restart while idle).
//
// Row striding (i_blk_beats != 0): the node writes a COLUMN BLOCK of a wider matrix, so after every
// i_blk_beats beats the address jumps by i_gap_bytes, and no burst crosses a block boundary.  That is what
// lets one node cover 16 P columns of a row while other nodes (or later weight tiles on the same node) fill
// the rest, and the rows still come out row-major in GDDR6.  i_blk_beats = 0 writes one contiguous run.
//
// o_afull: fewer than AFULL_MARGIN records (plus the skid and the record being serialised) fit;
// wire it to colpar_result_port.i_fifo_afull.  A record arriving with the skid full is dropped
// and o_error set (sticky), as is BRESP != OKAY.  o_idle: nothing queued, serialising, waiting
// in the skid, raised or outstanding.
module colpar_result_writer #(
    parameter integer N_STAGE         = 16,
    parameter integer SUM_BITS        = 32,
    parameter integer AXI_ADDR_WIDTH  = 42,
    parameter integer FIFO_LOG2       = 8,           // beats
    parameter integer BURST_BEATS     = 16,          // 1..256
    parameter integer MAX_OUTSTANDING = 8,           // 1..255
    parameter integer AFULL_MARGIN    = 8,           // records
    parameter integer RECORD_FIFO     = 0            // 1: record-wide FIFO written from i_sums (see above)
) (
    input  wire                       i_clk,
    input  wire                       i_rstn,
    input  wire [AXI_ADDR_WIDTH-1:0]  i_out_base,
    input  wire [9:0]                 i_blk_beats,   // beats per column block (0 = contiguous)
    input  wire [23:0]                i_gap_bytes,   // added to the address at a block boundary
    input  wire                       i_restart,
    input  wire                       i_flush,
    input  wire [48*N_STAGE-1:0]      i_sums,
    input  wire                       i_sums_valid,
    output wire                       o_afull,

    output reg                        o_awvalid,
    input  wire                       i_awready,
    output reg  [AXI_ADDR_WIDTH-1:0]  o_awaddr,
    output reg  [7:0]                 o_awlen,
    output wire [2:0]                 o_awsize,
    output wire [1:0]                 o_awburst,
    output wire                       o_wvalid,
    input  wire                       i_wready,
    output wire [255:0]               o_wdata,
    output wire [31:0]                o_wstrb,
    output wire                       o_wlast,
    input  wire                       i_bvalid,
    output wire                       o_bready,
    input  wire [1:0]                 i_bresp,

    output wire                       o_idle,
    output wire                       o_drained,     // every beat sent (B responses may still be open)
    output reg                        o_error
);
    localparam integer REC_BITS = N_STAGE * SUM_BITS;
    localparam integer BPR      = (REC_BITS + 255) / 256;
    localparam integer RW       = BPR * 256;
    localparam integer DEP      = 1 << FIFO_LOG2;
    localparam integer PW       = FIFO_LOG2 + 1;
    localparam integer LQ       = (MAX_OUTSTANDING > 1) ? $clog2(MAX_OUTSTANDING) : 1;
    // RECORD_FIFO: 2^RL records of RW bits, as many beats as the beat FIFO holds (at least 2 records)
    localparam integer RL       = (FIFO_LOG2 > $clog2(BPR) + 1) ? FIFO_LOG2 - $clog2(BPR) : 1;
    localparam integer RDEP     = 1 << RL;

    assign o_awsize  = 3'd5;
    assign o_awburst = 2'b01;
    assign o_wstrb   = 32'hFFFF_FFFF;

    // ---- sums -> record ----
    wire [RW-1:0] rec_in;
    genvar gs;
    generate
        for (gs = 0; gs < N_STAGE; gs = gs + 1) begin : g_trunc
            assign rec_in[gs*SUM_BITS +: SUM_BITS] = i_sums[gs*48 +: SUM_BITS];
        end
        if (RW > REC_BITS) begin : g_pad
            assign rec_in[RW-1:REC_BITS] = '0;
        end
    endgenerate

    // ---- storage (declared per RECORD_FIFO below) ----
    wire [PW:0]   queued;       // beats stored and not yet sent
    wire          st_valid;     // a beat is presented on o_wdata
    wire          st_idle;

    // ---- AXI bookkeeping ----
    reg [PW:0]               assigned;     // queued beats already covered by a raised or handshaked AW
    reg [PW:0]               w_credit;     // beats of handshaked AWs not yet sent
    reg [7:0]                outstanding;  // AW handshakes minus B handshakes
    reg [8:0]                aw_len_q;
    reg [8:0]                lenq [0:(1<<LQ)-1] /* synthesis syn_ramstyle = "registers" */;  // power-of-two depth: pointers wrap at 2^LQ
    reg [LQ-1:0]             lq_h, lq_t;
    reg [8:0]                w_pos;
    reg [AXI_ADDR_WIDTH-1:0] next_addr;
    reg [9:0]                blk_left;      // beats until the next column-block jump

    wire        w_fire = o_wvalid && i_wready;
    wire        st_error;     // a record arrived with no room: dropped
    wire        aw_fire = o_awvalid && i_awready;
    wire        b_fire  = i_bvalid && o_bready;
    // avail = queued - assigned, registered one cycle old (as vu_beat_writer.sv): it is never stale across a
    // decision (the next one waits for aw_pend and the AW handshake, >= 3 cycles) and beats pushed meanwhile only
    // make the true value larger.  An AW then takes two cycles: length decided (beats reserved) in the first, the
    // 42-bit address stepped and AWVALID raised in the second -- the fabric domain's longest path at 317 MHz was
    // nq -> avail -> len -> next_addr in one cycle (2026-09-22).
    reg  [PW:0] avail;
    reg         aw_pend, aw_jump;
    wire [8:0]  len_a  = (int'(avail) > BURST_BEATS) ? 9'(BURST_BEATS) : 9'(avail);
    // a burst never crosses a column-block boundary
    // blk_left = 0 with striding on means the program's block size does not divide the record stream:
    // fall back to a contiguous burst and raise o_error rather than issuing a zero-length one
    wire        blk_bad = (i_blk_beats != 10'd0) && (blk_left == 10'd0);
    wire [8:0]  len    = (i_blk_beats != 10'd0 && !blk_bad && len_a > 9'(blk_left)) ? 9'(blk_left) : len_a;

    assign o_wvalid = st_valid && (w_credit != '0);
    assign o_wlast  = (w_pos == lenq[lq_h] - 9'd1);
    assign o_bready = (outstanding != 8'd0);
    assign o_drained = (queued == '0) && st_idle && !o_awvalid && !aw_pend;
    wire        raise = !o_awvalid && !aw_pend && outstanding < 8'(MAX_OUTSTANDING) && avail != '0 &&
                        !(i_restart && o_drained) && (int'(avail) >= BURST_BEATS || o_afull || i_flush);
    assign o_idle   = o_drained && (outstanding == 8'd0);

    generate
    if (RECORD_FIFO == 0) begin : g_beat_fifo
        // ---- beat FIFO (inferred block RAM, registered read) ----
        reg [255:0]   mem [0:DEP-1] /* synthesis syn_ramstyle = "block_ram" */;   // not LRAM2K: each blocks an MLP72 site
        reg [PW-1:0]  wp, rp;
        wire [PW-1:0] mem_count = wp - rp;
        reg [255:0]   out_beat;
        reg           out_valid;

        // ---- skid + serialiser ----
        reg [RW-1:0]  skid [0:1];
        reg [1:0]     skid_n;
        reg [RW-1:0]  ser;
        reg           ser_valid;
        reg [7:0]     ser_bi;
        wire          ser_last   = (ser_bi == 8'(BPR - 1));
        wire          push_beat  = ser_valid && (mem_count != PW'(DEP));
        wire          ser_free   = !ser_valid || (push_beat && ser_last);
        wire          take_skid  = ser_free && (skid_n != 2'd0);
        wire          take_in    = ser_free && (skid_n == 2'd0) && i_sums_valid;
        wire          in_to_skid = i_sums_valid && !take_in;
        wire          fetch      = (!out_valid || w_fire) && (mem_count != '0);
        reg           err_q;

        always @(posedge i_clk) begin
            if (push_beat) mem[wp[FIFO_LOG2-1:0]] <= ser[ser_bi*256 +: 256];
            if (fetch)     out_beat <= mem[rp[FIFO_LOG2-1:0]];
        end
        assign queued   = {1'b0, mem_count} + (PW+1)'(out_valid);
        assign st_valid = out_valid;
        assign o_wdata  = out_beat;
        assign st_idle  = !ser_valid && (skid_n == 2'd0);
        assign st_error = err_q;
        // integer compares: sized casts into PW+1 bits would truncate 256 / BURST_BEATS
        assign o_afull  = (DEP - int'(mem_count)) < (AFULL_MARGIN + 3) * BPR;

        always @(posedge i_clk) begin
            err_q <= 1'b0;
            if (!i_rstn) begin
                wp        <= '0;
                rp        <= '0;
                out_valid <= 1'b0;
                ser_valid <= 1'b0;
                ser_bi    <= 8'd0;
                skid_n    <= 2'd0;
            end else begin
                // serialiser
                if (push_beat) wp <= wp + 1'b1;
                if (take_skid) begin
                    ser       <= skid[0];
                    ser_valid <= 1'b1;
                    ser_bi    <= 8'd0;
                end else if (take_in) begin
                    ser       <= rec_in;
                    ser_valid <= 1'b1;
                    ser_bi    <= 8'd0;
                end else if (push_beat) begin
                    if (ser_last) ser_valid <= 1'b0;
                    else          ser_bi    <= ser_bi + 8'd1;
                end
                case ({take_skid, in_to_skid})
                2'b10: begin
                    skid[0] <= skid[1];
                    skid_n  <= skid_n - 2'd1;
                end
                2'b01: begin
                    if (skid_n == 2'd2) err_q <= 1'b1;
                    else begin
                        skid[skid_n[0]] <= rec_in;
                        skid_n          <= skid_n + 2'd1;
                    end
                end
                2'b11: begin
                    skid[0] <= (skid_n == 2'd2) ? skid[1] : rec_in;
                    skid[1] <= rec_in;
                end
                default: ;
                endcase

                // beat FIFO output register
                if (fetch)       begin rp <= rp + 1'b1; out_valid <= 1'b1; end
                else if (w_fire) out_valid <= 1'b0;
            end
        end
    end else begin : g_record_fifo
        reg [RW-1:0]  rmem [0:RDEP-1] /* synthesis syn_ramstyle = "block_ram" */;   // not LRAM2K (MLP72 sites)
        reg [RL:0]    rwp, rrp;             // records written / loaded into the read register
        wire [RL:0]   rcount = rwp - rrp;
        reg [RW-1:0]  head;                 // the RAM's read register: the record being sent
        reg           head_v;
        reg [7:0]     hbi;                  // its next beat
        reg [PW:0]    nq;
        reg           err_q;
        wire          push = i_sums_valid && (rcount != (RL+1)'(RDEP));
        wire          load = !head_v && (rcount != '0);
        always @(posedge i_clk) begin
            if (push) rmem[rwp[RL-1:0]] <= rec_in;
            if (load) head <= rmem[rrp[RL-1:0]];
        end
        assign queued   = nq;
        assign st_valid = head_v;
`ifdef COLPAR_WR_NEG_BEAT_ORDER    // negative control: a record's beats leave in reverse order
        assign o_wdata  = head[(BPR - 1 - hbi)*256 +: 256];
`else
        assign o_wdata  = head[hbi*256 +: 256];
`endif
        assign st_idle  = 1'b1;
        assign st_error = err_q;
        assign o_afull  = (RDEP - int'(rcount)) < (AFULL_MARGIN + 3);
        always @(posedge i_clk) begin
            err_q <= i_rstn && i_sums_valid && !push;
            if (!i_rstn) begin
                rwp    <= '0;
                rrp    <= '0;
                head_v <= 1'b0;
                hbi    <= 8'd0;
                nq     <= '0;
            end else begin
                if (push) rwp <= rwp + 1'b1;
                if (load) begin
                    rrp    <= rrp + 1'b1;
                    head_v <= 1'b1;
                    hbi    <= 8'd0;
                end else if (w_fire) begin
                    if (hbi == 8'(BPR - 1)) head_v <= 1'b0;
                    else                    hbi    <= hbi + 8'd1;
                end
                nq <= nq + (push ? (PW+1)'(BPR) : '0) - (PW+1)'(w_fire);
            end
        end
    end
    endgenerate

    always @(posedge i_clk) begin
        if (!i_rstn) begin
            o_awvalid   <= 1'b0;
            o_error     <= 1'b0;
            assigned    <= '0;
            blk_left    <= '0;
            w_credit    <= '0;
            outstanding <= 8'd0;
            lq_h        <= '0;
            lq_t        <= '0;
            w_pos       <= 9'd0;
            next_addr   <= i_out_base;
            blk_left    <= i_blk_beats;
            avail       <= '0;
            aw_pend     <= 1'b0;
        end else begin
            avail <= queued - assigned;
            // AW
            if (i_restart && o_drained) begin
                next_addr <= i_out_base;
                blk_left  <= i_blk_beats;
            end else if (raise) begin
                aw_pend   <= 1'b1;
                o_awlen   <= 8'(len - 9'd1);
                aw_len_q  <= len;
                aw_jump   <= (i_blk_beats != 10'd0) && (10'(len) == blk_left);
                if (i_blk_beats != 10'd0 && 10'(len) == blk_left)
                    blk_left <= i_blk_beats;
                else if (i_blk_beats != 10'd0)
                    blk_left <= blk_left - 10'(len);
            end
            if (aw_pend) begin
                aw_pend   <= 1'b0;
                o_awvalid <= 1'b1;
                o_awaddr  <= next_addr;
                next_addr <= next_addr + AXI_ADDR_WIDTH'({aw_len_q, 5'd0}) + (aw_jump ? AXI_ADDR_WIDTH'(i_gap_bytes) : '0);
            end
            if (aw_fire) begin
                o_awvalid  <= 1'b0;
                lenq[lq_t] <= aw_len_q;
                lq_t       <= lq_t + 1'b1;
            end
            assigned <= assigned + (raise ? (PW+1)'(len) : '0) - (PW+1)'(w_fire);
            w_credit <= w_credit + (aw_fire ? (PW+1)'(aw_len_q) : '0) - (PW+1)'(w_fire);

            // W burst position
            if (w_fire) begin
                if (o_wlast) begin
                    w_pos <= 9'd0;
                    lq_h  <= lq_h + 1'b1;
                end else begin
                    w_pos <= w_pos + 9'd1;
                end
            end

            // B
            outstanding <= outstanding + {7'd0, aw_fire} - {7'd0, b_fire};
            if (b_fire && i_bresp != 2'b00) o_error <= 1'b1;
            if (blk_bad && avail != '0) o_error <= 1'b1;
            if (st_error) o_error <= 1'b1;
        end
    end
endmodule
