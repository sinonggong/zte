// Tile loader for one column-parallel chain: AXI4 read bursts from GDDR6 (through a
// NAP) straight into the chain's BRAM72K write port, on the NAP clock (= the chain's
// i_wclk).
//
// GDDR6 layout: raw int8 bytes, 16 per BRAM word, word w at byte offset 16 w.  A 256-bit
// beat is exactly two words.  Each beat is unpacked into the UG086 Table 107 word format
// {8'h0, b15..b8, 8'h0, b7..b0} and written as two BRAM words on consecutive cycles
// (RREADY is low on the second cycle).
//
// One command fills n_regions BRAMs (targets first_target + r: 0 = feeder, s = stage s).
// Each target's image is n_segs segments of n_words words, written from BRAM address 0 in
// segment order (from i_wbase instead of 0 when set).  Segment i of target r starts at GDDR6 word
//     base_word + r * tgt_step + i * seg_step
// (every segment start must be an even word, i.e. on a beat boundary; a segment of odd
// n_words ends in a half-used beat).  Two uses:
//   contiguous images (LOAD):  n_segs = 1, tgt_step = n_words rounded up to even
//   interleaved columns:       stage s = columns c = N_STAGE p + s of a column-contiguous
//                              matrix (W words per column): n_words = W, n_segs = P,
//                              seg_step = N_STAGE W, tgt_step = W (first_target 1)
//   feeder row slices:         rows r at stride S from an offset: n_regions = 1, target 0,
//                              n_words = W, n_segs = M, seg_step = S
//
// Bursts: at most BEATS_PER_BURST beats (the NoC carries at most 16, UG086), never across a
// segment, and up to MAX_OUTSTANDING AR requests are issued ahead of their R data (one AXI
// ID, so R bursts return in request order).  The AR issuer and the R consumer walk the same
// burst sequence independently; the consumer checks RLAST against its own count.
// MAX_OUTSTANDING = 1 gives strictly serial bursts.
//
// o_error (sticky) flags RRESP != OKAY or RLAST on the wrong beat.
module colpar_tile_loader #(
    parameter integer N_STAGE         = 16,
    parameter integer ADDR_BITS       = 9,
    parameter integer AXI_ADDR_WIDTH  = 42,
    parameter integer BEATS_PER_BURST = 16,          // 1..256
    parameter integer MAX_OUTSTANDING = 8            // 1..255
) (
    input  wire                       i_clk,
    input  wire                       i_rstn,

    // command: one-cycle i_arm, fields held until o_done
    input  wire                       i_arm,
    input  wire [AXI_ADDR_WIDTH-1:0]  i_base,
    input  wire [4:0]                 i_first_target,
    input  wire [4:0]                 i_n_regions,     // >= 1
    input  wire [ADDR_BITS:0]         i_n_words,       // words per segment, >= 1
    input  wire [ADDR_BITS:0]         i_n_segs,        // >= 1; n_segs x n_words <= 2^ADDR_BITS
    input  wire [13:0]                i_seg_step,      // words between segment starts of one target
    input  wire [13:0]                i_tgt_step,      // words between the first segments of targets r and r+1
    input  wire [ADDR_BITS-1:0]       i_wbase,         // BRAM word address of each target's first word (0; half 256 for
                                                       // the double-buffered GEMM feeder, colpar_node_ctrl.sv)

    // AXI4 read, NAP master side
    output reg                        o_arvalid,
    input  wire                       i_arready,
    output reg  [AXI_ADDR_WIDTH-1:0]  o_araddr,
    output reg  [7:0]                 o_arlen,
    output wire [2:0]                 o_arsize,
    output wire [1:0]                 o_arburst,
    output wire                       o_rready,
    input  wire                       i_rvalid,
    input  wire [255:0]               i_rdata,
    input  wire [1:0]                 i_rresp,
    input  wire                       i_rlast,

    // chain BRAM72K write port
    output reg  [143:0]               o_wdata,
    output reg  [ADDR_BITS-1:0]       o_waddr,
    output reg  [N_STAGE:0]           o_wen,

    output reg                        o_busy,
    output reg                        o_done,          // one-cycle pulse
    output reg                        o_error
);
    localparam integer AW = AXI_ADDR_WIDTH;

    assign o_arsize  = 3'd5;                          // 32 bytes per beat
    assign o_arburst = 2'b01;                         // INCR

    function automatic [143:0] word_of(input [127:0] b);
        word_of = {8'h00, b[127:64], 8'h00, b[63:0]};
    endfunction

    function automatic [8:0] burst_of(input [ADDR_BITS:0] beats_left);
        burst_of = (beats_left > BEATS_PER_BURST) ? 9'(BEATS_PER_BURST) : 9'(beats_left);
    endfunction

    reg [ADDR_BITS:0]   n_words_q, n_segs_q, beats_seg;
    reg [AW-1:0]        seg_step_b, tgt_step_b;       // in bytes
    reg [7:0]           outstanding;

    // ---- AR issuer ----
    reg                 ar_active;
    reg [4:0]           ar_tgt_left;
    reg [ADDR_BITS:0]   ar_seg_left, ar_beats_left;
    reg [AW-1:0]        ar_seg_addr, ar_tgt_addr;
    reg [8:0]           ar_len_q;
    wire                ar_fire   = o_arvalid && i_arready;

    // ---- R consumer ----
    reg                 rc_active;
`ifdef COLPAR_LD_NEG_TARGET5      // negative control: the 5-bit target of the 16-stage node (stage 32 wraps to the feeder)
    reg [4:0]           rc_target;
`else
    reg [5:0]           rc_target;                    // 0 feeder, 1..N_STAGE stages (32 needs 6 bits)
`endif
    reg [4:0]           rc_tgt_left;
    reg [ADDR_BITS:0]   rc_seg_left, rc_beats_left, rc_word_idx, rc_seg_word;
    reg [ADDR_BITS-1:0] wbase_q;
    reg [8:0]           rc_burst_left;
    reg                 pending;
    reg [143:0]         word1;
    wire [N_STAGE:0]    target_onehot = {{N_STAGE{1'b0}}, 1'b1} << rc_target;
    wire                r_fire = o_rready && i_rvalid;
    wire                r_end  = r_fire && i_rlast;

    assign o_rready = rc_active && !pending;

    wire [AW-1:0]       next_seg_addr = ar_seg_addr +
`ifdef COLPAR_LD_NEG_SEG_STEP    // negative control: segments advance by the target step
                                    tgt_step_b;
`else
                                    seg_step_b;
`endif
    wire [AW-1:0]       next_tgt_addr = ar_tgt_addr + tgt_step_b;

    always @(posedge i_clk) begin
        o_done <= 1'b0;
        o_wen  <= '0;
        if (!i_rstn) begin
            o_arvalid   <= 1'b0;
            o_busy      <= 1'b0;
            o_error     <= 1'b0;
            ar_active   <= 1'b0;
            rc_active   <= 1'b0;
            pending     <= 1'b0;
            outstanding <= 8'd0;
        end else begin
            outstanding <= outstanding + {7'd0, ar_fire} - {7'd0, r_end};

            if (i_arm && !o_busy) begin
                n_words_q     <= i_n_words;
                n_segs_q      <= i_n_segs;
                beats_seg     <= (i_n_words + 1'b1) >> 1;
                seg_step_b    <= AW'(i_seg_step) << 4;
                tgt_step_b    <= AW'(i_tgt_step) << 4;
                o_araddr      <= i_base;
                ar_seg_addr   <= i_base;
                ar_tgt_addr   <= i_base;
                ar_active     <= 1'b1;
                ar_tgt_left   <= i_n_regions;
                ar_seg_left   <= i_n_segs;
                ar_beats_left <= (i_n_words + 1'b1) >> 1;
                rc_active     <= 1'b1;
                rc_target     <= i_first_target;
                rc_tgt_left   <= i_n_regions;
                rc_seg_left   <= i_n_segs;
                rc_beats_left <= (i_n_words + 1'b1) >> 1;
                rc_word_idx   <= (ADDR_BITS+1)'(i_wbase);
                wbase_q       <= i_wbase;
                rc_seg_word   <= '0;
                rc_burst_left <= burst_of((i_n_words + 1'b1) >> 1);
                o_busy        <= 1'b1;
            end

            // ---- issuer ----
            if (ar_active && !o_arvalid && outstanding < 8'(MAX_OUTSTANDING)) begin
                o_arvalid <= 1'b1;
                o_arlen   <= 8'(burst_of(ar_beats_left) - 9'd1);
                ar_len_q  <= burst_of(ar_beats_left);
            end
            if (ar_fire) begin
                o_arvalid <= 1'b0;
                o_araddr  <= o_araddr + {ar_len_q, 5'd0};
                if (ar_beats_left == {1'b0, ar_len_q}) begin              // segment done
                    ar_beats_left <= beats_seg;
                    if (ar_seg_left == (ADDR_BITS+1)'(1)) begin           // target done
                        if (ar_tgt_left == 5'd1) begin
                            ar_active <= 1'b0;
                        end else begin
                            ar_tgt_left <= ar_tgt_left - 1'b1;
                            ar_seg_left <= n_segs_q;
                            ar_tgt_addr <= next_tgt_addr;
                            ar_seg_addr <= next_tgt_addr;
                            o_araddr    <= next_tgt_addr;
                        end
                    end else begin
                        ar_seg_left <= ar_seg_left - 1'b1;
                        ar_seg_addr <= next_seg_addr;
                        o_araddr    <= next_seg_addr;
                    end
                end else begin
                    ar_beats_left <= ar_beats_left - ar_len_q;
                end
            end

            // ---- consumer ----
            if (rc_active && !pending && i_rvalid) begin
                o_wdata       <= word_of(i_rdata[127:0]);
                o_waddr       <= rc_word_idx[ADDR_BITS-1:0];
                o_wen         <= target_onehot;
                rc_word_idx   <= rc_word_idx + 1'b1;
                rc_seg_word   <= rc_seg_word + 1'b1;
                word1         <= word_of(i_rdata[255:128]);
                pending       <= 1'b1;
                rc_beats_left <= rc_beats_left - 1'b1;
                rc_burst_left <= rc_burst_left - 1'b1;
                if (i_rresp != 2'b00 || i_rlast != (rc_burst_left == 9'd1))
                    o_error <= 1'b1;
            end else if (pending) begin
                pending <= 1'b0;
                if (rc_seg_word < n_words_q) begin
                    o_wdata     <= word1;
                    o_waddr     <= rc_word_idx[ADDR_BITS-1:0];
                    o_wen       <= target_onehot;
                    rc_word_idx <= rc_word_idx + 1'b1;
                    rc_seg_word <= rc_seg_word + 1'b1;
                end
                if (rc_burst_left == 9'd0) begin
                    if (rc_beats_left != '0) begin
                        rc_burst_left <= burst_of(rc_beats_left);
                    end else begin                                       // segment done
                        rc_beats_left <= beats_seg;
                        rc_burst_left <= burst_of(beats_seg);
                        rc_seg_word   <= '0;
                        if (rc_seg_left == (ADDR_BITS+1)'(1)) begin      // target done
                            if (rc_tgt_left == 5'd1) begin
                                rc_active <= 1'b0;
                                o_busy    <= 1'b0;
                                o_done    <= 1'b1;
                            end else begin
                                rc_tgt_left <= rc_tgt_left - 1'b1;
                                rc_target   <= rc_target + 1'b1;
                                rc_seg_left <= n_segs_q;
                                rc_word_idx <= (ADDR_BITS+1)'(wbase_q);
                            end
                        end else begin
                            rc_seg_left <= rc_seg_left - 1'b1;
                        end
                    end
                end
            end
        end
    end
endmodule
