// Beat writer for the vector node: a stream of 256-bit beats -> AXI4 write bursts into GDDR6.
// Beats go into a FIFO (256 bits x 2^FIFO_LOG2, registered read); bursts of at most BURST_BEATS
// beats (NoC limit 16), up to MAX_OUTSTANDING issued ahead of their B responses (one AXI ID).
// An AW is raised for queued beats not yet assigned to a burst, when BURST_BEATS of them wait,
// when the FIFO is almost full, or while i_flush is high.  W beats of a burst are sent only after
// its AW handshake.  Consecutive bursts write consecutive addresses from i_base (re-armed by
// i_restart once drained: every beat sent; B responses of the previous run may still be open, they only count down).
// Row striding (i_blk_beats != 0), as in colpar_result_writer.sv: after every i_blk_beats beats the address jumps by
// i_gap_bytes and no burst crosses a block, so an op on head-major rows can write them as blocks of token-major
// rows.  i_blk_beats and i_gap_bytes are sampled with i_restart.  o_afull leaves AFULL_MARGIN beats; a beat pushed into a full FIFO is
// dropped and o_error set (sticky), as is BRESP != OKAY.  (Same AXI logic as
// paper/rtl/colpar_result_writer.sv, without its record serialiser.)
// An AW takes two cycles: the burst length is decided (and its beats reserved) in the first, the address is
// stepped and AWVALID raised in the second, so the FIFO-count -> length -> 42-bit address add is not one path
// (it was the node's critical path at 300 and 333 MHz).  The decision itself uses the unassigned-beat count registered
// one cycle earlier: it is never stale across a decision (the next one waits for the AW handshake, >= 2 cycles), and
// beats pushed meanwhile only make it smaller.
module vu_beat_writer #(
    parameter integer AXI_ADDR_WIDTH  = 42,
    parameter integer FIFO_LOG2       = 6,
    parameter integer BURST_BEATS     = 16,
    parameter integer MAX_OUTSTANDING = 8,
    parameter integer AFULL_MARGIN    = 4
) (
    input  wire                       i_clk,
    input  wire                       i_rstn,
    input  wire [AXI_ADDR_WIDTH-1:0]  i_base,
    input  wire [9:0]                 i_blk_beats,     // beats per block (0 = one contiguous run)
    input  wire [23:0]                i_gap_bytes,     // added to the address at a block boundary
    input  wire                       i_restart,
    input  wire                       i_flush,
    input  wire [255:0]               i_beat,
    input  wire                       i_beat_valid,
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
    localparam integer DEP = 1 << FIFO_LOG2;
    localparam integer PW  = FIFO_LOG2 + 1;
    localparam integer LQ  = (MAX_OUTSTANDING > 1) ? $clog2(MAX_OUTSTANDING) : 1;

    assign o_awsize  = 3'd5;
    assign o_awburst = 2'b01;
    assign o_wstrb   = 32'hFFFF_FFFF;

    reg [255:0]   mem [0:DEP-1] /* synthesis syn_ramstyle = "block_ram" */;   // not LRAM2K: each blocks an MLP72 site
    reg [PW-1:0]  wp, rp;
    wire [PW-1:0] mem_count = wp - rp;
    reg [255:0]   out_beat;
    reg           out_valid;

    reg [PW:0]               assigned, w_credit;
    reg [7:0]                outstanding;
    reg [8:0]                aw_len_q;
    reg [8:0]                lenq [0:(1<<LQ)-1] /* synthesis syn_ramstyle = "registers" */;
    reg [LQ-1:0]             lq_h, lq_t;
    reg [8:0]                w_pos;
    reg [AXI_ADDR_WIDTH-1:0] next_addr;
    reg [9:0]                blk_beats, blk_left;
    reg [23:0]               gap_bytes;
    reg                      aw_pend;       // a burst decided last cycle; its AW is raised this cycle
    reg                      aw_jump;       // ... and it ends a block: the address also jumps the gap

    wire        push    = i_beat_valid && (mem_count != PW'(DEP));
    wire        w_fire  = o_wvalid && i_wready;
    wire        aw_fire = o_awvalid && i_awready;
    wire        b_fire  = i_bvalid && o_bready;
    wire        fetch   = (!out_valid || w_fire) && (mem_count != '0);
    wire [PW:0] queued  = {1'b0, mem_count} + (PW+1)'(out_valid);
    reg  [PW:0] avail;                      // queued - assigned, one cycle old (see the header)
    wire [8:0]  len_a   = (int'(avail) > BURST_BEATS) ? 9'(BURST_BEATS) : 9'(avail);
    // a burst never crosses a block boundary; blk_left = 0 with striding on means the stream does not divide
    // into blocks: fall back to a contiguous burst and raise o_error
    wire        blk_bad = (blk_beats != 10'd0) && (blk_left == 10'd0);
    wire [8:0]  len     = (blk_beats != 10'd0 && !blk_bad && len_a > 9'(blk_left)) ? 9'(blk_left) : len_a;
    wire        raise   = !o_awvalid && !aw_pend && outstanding < 8'(MAX_OUTSTANDING) && avail != '0 &&
                          !(i_restart && o_drained) && (int'(avail) >= BURST_BEATS || o_afull || i_flush);

    always @(posedge i_clk) begin
        if (push)  mem[wp[FIFO_LOG2-1:0]] <= i_beat;
        if (fetch) out_beat <= mem[rp[FIFO_LOG2-1:0]];
    end

    assign o_afull  = (DEP - int'(mem_count)) <= AFULL_MARGIN;
    always @(posedge i_clk) avail <= i_rstn ? queued - assigned : '0;
    assign o_wvalid = out_valid && (w_credit != '0);
    assign o_wdata  = out_beat;
    assign o_wlast  = (w_pos == lenq[lq_h] - 9'd1);
    assign o_bready = (outstanding != 8'd0);
    assign o_drained = (queued == '0) && !o_awvalid && !aw_pend;
    assign o_idle   = o_drained && (outstanding == 8'd0);

    always @(posedge i_clk) begin
        if (!i_rstn) begin
            wp          <= '0;
            rp          <= '0;
            out_valid   <= 1'b0;
            o_awvalid   <= 1'b0;
            aw_pend     <= 1'b0;
            o_error     <= 1'b0;
            assigned    <= '0;
            w_credit    <= '0;
            outstanding <= 8'd0;
            lq_h        <= '0;
            lq_t        <= '0;
            w_pos       <= 9'd0;
            next_addr   <= i_base;
            blk_beats   <= '0;
            blk_left    <= '0;
            gap_bytes   <= '0;
        end else begin
            if (i_beat_valid) begin
                if (push) wp <= wp + 1'b1;
                else      o_error <= 1'b1;
            end
            if (fetch)       begin rp <= rp + 1'b1; out_valid <= 1'b1; end
            else if (w_fire) out_valid <= 1'b0;

            if (i_restart && o_drained) begin
                next_addr <= i_base;
                blk_beats <= i_blk_beats;
                blk_left  <= i_blk_beats;
                gap_bytes <= i_gap_bytes;
            end else if (raise) begin
                aw_pend  <= 1'b1;
                o_awlen  <= 8'(len - 9'd1);
                aw_len_q <= len;
                aw_jump  <= (blk_beats != 10'd0) && (10'(len) == blk_left);
                if (blk_beats != 10'd0 && 10'(len) == blk_left)
                    blk_left <= blk_beats;
                else if (blk_beats != 10'd0)
                    blk_left <= blk_left - 10'(len);
            end
            if (aw_pend) begin
                aw_pend   <= 1'b0;
                o_awvalid <= 1'b1;
                o_awaddr  <= next_addr;
                next_addr <= next_addr + AXI_ADDR_WIDTH'({aw_len_q, 5'd0}) + (aw_jump ? AXI_ADDR_WIDTH'(gap_bytes) : '0);
            end
            if (aw_fire) begin
                o_awvalid  <= 1'b0;
                lenq[lq_t] <= aw_len_q;
                lq_t       <= lq_t + 1'b1;
            end
            assigned <= assigned + (raise ? (PW+1)'(len) : '0) - (PW+1)'(w_fire);
            w_credit <= w_credit + (aw_fire ? (PW+1)'(aw_len_q) : '0) - (PW+1)'(w_fire);
            if (w_fire) begin
                if (o_wlast) begin
                    w_pos <= 9'd0;
                    lq_h  <= lq_h + 1'b1;
                end else begin
                    w_pos <= w_pos + 9'd1;
                end
            end
            outstanding <= outstanding + {7'd0, aw_fire} - {7'd0, b_fire};
            if (b_fire && i_bresp != 2'b00) o_error <= 1'b1;
            if (blk_bad && avail != '0) o_error <= 1'b1;
        end
    end
endmodule
