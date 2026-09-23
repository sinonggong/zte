// Command executor for one column-parallel chain node, on the fabric clock.
//
// A static program (compiled off line) is a stream of 128-bit commands, [127:124] = opcode:
//   0 END    stop and raise o_halt
//   1 LOAD   [41:0] base  [46:42] first_target  [51:47] n_regions  [61:52] n_words
//            fill BRAM72K images from GDDR6 with colpar_tile_loader; waits for it
//   2 TILE   [8:0] act_base  [17:9] wt_base  [27:18] W  [37:28] M  [47:38] P  [52:48] G
//            run one tile on colpar_row_sequencer; waits until the sequencer is done AND all
//            M x P records have reached the result writer
//   3 OUT    [41:0] out_base  [51:42] block beats  [75:52] gap bytes: waits for the writer to be idle, then
//            restarts its write address.  With block beats != 0 the writer jumps by the gap after every
//            block, so this node's records land as a COLUMN BLOCK of a wider row-major matrix.
//   4 FLUSH  forces the writer to write everything it holds; waits until it is idle
//   5 LOADX  [41:0] base  [46:42] first_target  [51:47] n_regions  [61:52] n_words (per segment)
//            [71:62] n_segs  [85:72] seg_step  [99:86] tgt_step (words): segmented BRAM72K images,
//            segment i of target r at GDDR6 word base + r tgt_step + i seg_step (colpar_tile_loader.sv);
//            e.g. stage s = columns 16 p + s of a column-contiguous matrix (K codes, V channel rows)
//   7 WAIT   [41:0] addr  [95:64] value: poll the 32-bit word at addr until it is >= value (node_sync.sv)
//   8 POST   [41:0] addr  [95:64] value: write that word, after the result writer has gone idle
//   6 GEMM column group: [41:0] act_base (GDDR6: T rows x W words, contiguous from word 0)
//            [51:42] T  [60:52] wt_base  [70:61] W  [80:71] M  [90:81] P  [95:91] G
//            the node itself loads the feeder with min(M, rows left) rows at a time and runs
//            that tile against P columns per stage, until all T rows are done; records come out
//            row group outer, then row r, column p.  M x W must be even and <= 512 (a feeder
//            load starts on a beat boundary).  One command replaces 2 x ceil(T/M) LOAD/TILE pairs.
//            [105:96] S: feeder row stride in words (0 = W, rows contiguous).  [106] double buffer: the feeder's two
//            256-word halves alternate, the next group loading while one runs (M x W <= 256).  With S != W row r is the
//            W words at act_base + r S (a slice of a wider row, e.g. the prefix or suffix keys of a
//            P row); S and the act_base word must be even.
// Waiting for the records, not only for the sequencer's done, makes a LOAD right after a TILE
// safe: the last row pass has left the chain before any BRAM72K is rewritten.
//
// The sequencer fields are quasi-static toward the array clock: they are registered here
// before o_seq_start rises and held until the next TILE; the sequencer synchronises the start.
module colpar_node_ctrl #(
    parameter integer AXI_ADDR_WIDTH = 42,
    parameter integer ADDR_BITS      = 9
) (
    input  wire                       i_clk,
    input  wire                       i_rstn,
    input  wire                       i_restart,      // leave END/halt and fetch again (program start)

    input  wire [127:0]               i_cmd,
    input  wire                       i_cmd_valid,
    output wire                       o_cmd_ready,

    output reg                        o_ld_arm,
    output reg  [AXI_ADDR_WIDTH-1:0]  o_ld_base,
    output reg  [4:0]                 o_ld_first,
    output reg  [4:0]                 o_ld_nreg,
    output reg  [ADDR_BITS:0]         o_ld_nwords,
    output reg  [ADDR_BITS:0]         o_ld_nsegs,
    output reg  [13:0]                o_ld_seg_step,
    output reg  [13:0]                o_ld_tgt_step,
    output reg  [ADDR_BITS-1:0]       o_ld_wbase,
    input  wire                       i_ld_done,

    output reg                        o_seq_start,
    output reg  [ADDR_BITS-1:0]       o_act_base,
    output reg  [ADDR_BITS-1:0]       o_wt_base,
    output reg  [ADDR_BITS:0]         o_words,
    output reg  [ADDR_BITS:0]         o_rows,
    output reg  [ADDR_BITS:0]         o_passes,
    output reg  [4:0]                 o_gap,
    input  wire                       i_seq_busy_async,
    input  wire                       i_seq_done_async,

    input  wire                       i_rec_valid,
    output reg  [AXI_ADDR_WIDTH-1:0]  o_out_base,
    output reg  [9:0]                 o_out_blk,
    output reg  [23:0]                o_out_gap,
    output reg                        o_wr_restart,
    output reg                        o_wr_flush,
    input  wire                       i_wr_idle,
    input  wire                       i_wr_drained,   // every beat sent (B may be open)

    output reg                        o_sync_arm,
    output reg                        o_sync_post,
    output reg  [AXI_ADDR_WIDTH-1:0]  o_sync_addr,
    output reg  [31:0]                o_sync_value,
    input  wire                       i_sync_done,

    output reg                        o_halt,
    output wire                       o_fetching      // waiting for a command (tile loader idle)
);
    localparam [3:0] S_FETCH = 4'd0, S_LOAD = 4'd1, S_TSTART = 4'd2, S_TRUN = 4'd3,
                     S_OUT = 4'd4, S_FLUSH = 4'd5, S_HALT = 4'd6, S_CG = 4'd7,
                     S_SYNC = 4'd8, S_POSTW = 4'd9, S_DBL = 4'd10, S_DBGO = 4'd11, S_DBTS = 4'd12, S_DBTR = 4'd13,
                     S_DBP = 4'd14;

    reg [3:0]   st;
    (* syn_preserve = 1 *) reg [1:0] busy_s, done_s;
    reg [19:0]  rec_cnt, rec_expected;
    reg [1:0]   settle;
    // The sequencer fields (act / wt base, words, rows, passes, gap) cross to the array clock unconstrained: the
    // sequencer samples them when its synchronised start rises.  Raising o_seq_start in the same cycle as a field
    // change left the fields ~2 array cycles to arrive, and on a non-timing-driven route they sometimes did not: the
    // double-buffered GEMM, whose act base and row count change at every group, failed on silicon (s1q, 2026-09-18:
    // prefix / expert GEMMs wrong, not deterministic; the same RTL bit-exact in simulation).  seq_arm raises the start
    // three fabric cycles after the fields.
    reg [1:0]   seq_arm;

    // column-group loop
    reg                      in_cg;
    reg [AXI_ADDR_WIDTH-1:0] cg_base;
    reg [ADDR_BITS:0]        cg_rows_left, cg_m, cg_W, cg_M, cg_P, cg_S;
    // the byte step between row groups, M x S x 16, registered from cg_M / cg_S: the multiply was in series with
    // the 42-bit add into cg_base (the fabric domain's longest path, 3.2 ns).  Only a group that is followed by
    // another one advances cg_base, and such a group has M rows, so the step never needs the last group's cg_m.
    reg [AXI_ADDR_WIDTH-1:0] cg_stepM;
    reg [ADDR_BITS-1:0]      cg_wt;
    reg [4:0]                cg_G;
    wire [ADDR_BITS:0]       cg_m_next = (cg_rows_left > cg_M) ? cg_M : cg_rows_left;
    // double-buffered GEMM (opcode 6 bit 106): the feeder is two halves of 256 words; while a row group runs on one
    // half, the next group loads into the other, so the feeder load and its GDDR6 round trip leave the critical
    // path (they were serial with every group's row passes).  The command's M x W must be <= 256.
    reg                      cg_db, db_ld_ok, db_pf;
    reg                      db_h;                         // half being loaded (0: words 0..255, 1: 256..511)
    reg [ADDR_BITS:0]        db_m_ld;                      // rows of the group being / last loaded

    assign o_cmd_ready = (st == S_FETCH);
    assign o_fetching  = (st == S_FETCH);

    always @(posedge i_clk) begin
        busy_s       <= {busy_s[0], i_seq_busy_async};
        done_s       <= {done_s[0], i_seq_done_async};
        o_ld_arm     <= 1'b0;
        if (i_ld_done) db_ld_ok <= 1'b1;
        o_wr_restart <= 1'b0;
        o_sync_arm   <= 1'b0;
        if (i_rec_valid) rec_cnt <= rec_cnt + 1'b1;
        cg_stepM <= AXI_ADDR_WIDTH'(20'(cg_M) * 20'(cg_S) * 20'd16);

        if (seq_arm != 2'd0) begin
            seq_arm <= seq_arm - 2'd1;
            if (seq_arm == 2'd1) o_seq_start <= 1'b1;
        end
        if (!i_rstn) begin
            st          <= S_FETCH;
            o_seq_start <= 1'b0;
            seq_arm     <= 2'd0;
            o_wr_flush  <= 1'b0;
            o_halt      <= 1'b0;
            rec_cnt     <= '0;
            in_cg       <= 1'b0;
            db_ld_ok    <= 1'b0;
            db_pf       <= 1'b0;
            o_ld_wbase  <= '0;
        end else if (i_restart) begin
            st     <= S_FETCH;
            o_halt <= 1'b0;
            in_cg  <= 1'b0;
        end else begin
            case (st)
            S_FETCH: if (i_cmd_valid) begin
                case (i_cmd[127:124])
                4'd1: begin
                    o_ld_base     <= i_cmd[AXI_ADDR_WIDTH-1:0];
                    o_ld_first    <= i_cmd[46:42];
                    o_ld_nreg     <= i_cmd[51:47];
                    o_ld_nwords   <= i_cmd[52 +: ADDR_BITS+1];
                    o_ld_nsegs    <= (ADDR_BITS+1)'(1);
                    o_ld_seg_step <= 14'd0;
                    o_ld_tgt_step <= 14'(i_cmd[52 +: ADDR_BITS+1]) + 14'(i_cmd[52]);   // whole beats
                    o_ld_wbase    <= '0;
                    o_ld_arm      <= 1'b1;
                    st            <= S_LOAD;
                end
                4'd5: begin
                    o_ld_base     <= i_cmd[AXI_ADDR_WIDTH-1:0];
                    o_ld_first    <= i_cmd[46:42];
                    o_ld_nreg     <= i_cmd[51:47];
                    o_ld_nwords   <= i_cmd[52 +: ADDR_BITS+1];
                    o_ld_nsegs    <= i_cmd[62 +: ADDR_BITS+1];
                    o_ld_seg_step <= i_cmd[85:72];
                    o_ld_tgt_step <= i_cmd[99:86];
                    o_ld_wbase    <= '0;
                    o_ld_arm      <= 1'b1;
                    st            <= S_LOAD;
                end
                4'd2: begin
                    o_act_base   <= i_cmd[ADDR_BITS-1:0];
                    o_wt_base    <= i_cmd[9 +: ADDR_BITS];
                    o_words      <= i_cmd[18 +: ADDR_BITS+1];
                    o_rows       <= i_cmd[28 +: ADDR_BITS+1];
                    o_passes     <= i_cmd[38 +: ADDR_BITS+1];
                    o_gap        <= i_cmd[52:48];
                    rec_expected <= 20'(i_cmd[28 +: ADDR_BITS+1]) * 20'(i_cmd[38 +: ADDR_BITS+1]);
                    rec_cnt      <= '0;
                    seq_arm      <= 2'd3;                 // fields now, start 3 cycles later (see seq_arm)
                    st           <= S_TSTART;
                end
                4'd3: begin
                    o_out_base <= i_cmd[AXI_ADDR_WIDTH-1:0];
                    o_out_blk  <= i_cmd[51:42];
                    o_out_gap  <= i_cmd[75:52];
                    st         <= S_OUT;
                end
                4'd6: begin
                    cg_base      <= i_cmd[AXI_ADDR_WIDTH-1:0];
                    cg_rows_left <= i_cmd[42 +: ADDR_BITS+1];
                    cg_wt        <= i_cmd[52 +: ADDR_BITS];
                    cg_W         <= i_cmd[61 +: ADDR_BITS+1];
                    cg_M         <= i_cmd[71 +: ADDR_BITS+1];
                    cg_P         <= i_cmd[81 +: ADDR_BITS+1];
                    cg_G         <= i_cmd[95:91];
                    cg_S         <= (i_cmd[96 +: ADDR_BITS+1] == '0) ? i_cmd[61 +: ADDR_BITS+1] : i_cmd[96 +: ADDR_BITS+1];
                    in_cg        <= 1'b1;
                    cg_db        <= i_cmd[106];
                    db_h         <= 1'b0;
                    st           <= i_cmd[106] ? S_DBP : S_CG;    // S_DBP: one cycle for cg_stepM
                end
                4'd7, 4'd8: begin
                    o_sync_addr  <= i_cmd[AXI_ADDR_WIDTH-1:0];
                    o_sync_value <= i_cmd[95:64];
                    o_sync_post  <= (i_cmd[127:124] == 4'd8);   // opcode 8 = POST, 7 = WAIT
                    if (i_cmd[127:124] == 4'd8) begin
                        st <= S_POSTW;                   // a POST waits for the writer to drain first
                    end else begin
                        o_sync_arm <= 1'b1;
                        st         <= S_SYNC;
                    end
                end
                4'd4: begin
                    o_wr_flush <= 1'b1;
                    settle     <= 2'd3;
                    st         <= S_FLUSH;
                end
                default: begin
                    o_halt <= 1'b1;
                    st     <= S_HALT;
                end
                endcase
            end
            S_LOAD: if (i_ld_done) begin
                if (in_cg) begin
                    o_act_base   <= '0;
                    o_wt_base    <= cg_wt;
                    o_words      <= cg_W;
                    o_rows       <= cg_m;
                    o_passes     <= cg_P;
                    o_gap        <= cg_G;
                    rec_expected <= 20'(cg_m) * 20'(cg_P);
                    rec_cnt      <= '0;
                    seq_arm      <= 2'd3;                 // fields now, start 3 cycles later (see seq_arm)
                    st           <= S_TSTART;
                end else begin
                    st <= S_FETCH;
                end
            end
            S_CG: begin
                if (cg_rows_left == '0) begin
                    in_cg <= 1'b0;
                    st    <= S_FETCH;
                end else begin
                    cg_m          <= cg_m_next;
                    o_ld_base     <= cg_base;
                    o_ld_first    <= 5'd0;
                    o_ld_nreg     <= 5'd1;
                    if (cg_S == cg_W) begin                      // contiguous rows: one image
                        o_ld_nwords   <= (ADDR_BITS+1)'(cg_m_next * cg_W);
                        o_ld_nsegs    <= (ADDR_BITS+1)'(1);
                    end else begin                               // row slices: one segment per row
                        o_ld_nwords   <= cg_W;
                        o_ld_nsegs    <= cg_m_next;
                    end
                    o_ld_seg_step <= 14'(cg_S);
                    o_ld_tgt_step <= 14'd0;
                    o_ld_wbase    <= '0;
                    o_ld_arm      <= 1'b1;
                    st            <= S_LOAD;
                end
            end
            S_TSTART: if (busy_s[1]) begin
                o_seq_start <= 1'b0;
                st          <= S_TRUN;
            end
            S_TRUN: if (done_s[1] && !busy_s[1] && rec_cnt == rec_expected) begin
                if (in_cg) begin
                    cg_base      <= cg_base + cg_stepM;
                    cg_rows_left <= cg_rows_left - cg_m;
                    st           <= S_CG;
                end else begin
                    st <= S_FETCH;
                end
            end
            // ---- double-buffered GEMM ----
            S_DBP: st <= S_DBL;
            S_DBL: begin                                   // the first group into half 0
                o_ld_base     <= cg_base;
                o_ld_first    <= 5'd0;
                o_ld_nreg     <= 5'd1;
                if (cg_S == cg_W) begin
                    o_ld_nwords <= (ADDR_BITS+1)'(cg_m_next * cg_W);
                    o_ld_nsegs  <= (ADDR_BITS+1)'(1);
                end else begin
                    o_ld_nwords <= cg_W;
                    o_ld_nsegs  <= cg_m_next;
                end
                o_ld_seg_step <= 14'(cg_S);
                o_ld_tgt_step <= 14'd0;
                o_ld_wbase    <= '0;
                o_ld_arm      <= 1'b1;
                db_ld_ok      <= 1'b0;
                db_pf         <= 1'b1;
                db_h          <= 1'b0;
                db_m_ld       <= cg_m_next;
                cg_base       <= cg_base + cg_stepM;
                cg_rows_left  <= cg_rows_left - cg_m_next;
                st            <= S_DBGO;
            end
            S_DBGO: if (db_ld_ok || i_ld_done) begin      // the loaded group runs; the next one loads meanwhile
                o_act_base   <= db_h ? ADDR_BITS'(1 << (ADDR_BITS - 1)) : '0;
                o_wt_base    <= cg_wt;
                o_words      <= cg_W;
                o_rows       <= db_m_ld;
                o_passes     <= cg_P;
                o_gap        <= cg_G;
                rec_expected <= 20'(db_m_ld) * 20'(cg_P);
                rec_cnt      <= '0;
                seq_arm      <= 2'd3;                 // fields now, start 3 cycles later (see seq_arm)
                db_ld_ok     <= 1'b0;
                if (cg_rows_left != '0) begin
                    o_ld_base     <= cg_base;
                    if (cg_S == cg_W) begin
                        o_ld_nwords <= (ADDR_BITS+1)'(cg_m_next * cg_W);
                        o_ld_nsegs  <= (ADDR_BITS+1)'(1);
                    end else begin
                        o_ld_nwords <= cg_W;
                        o_ld_nsegs  <= cg_m_next;
                    end
                    o_ld_wbase    <= db_h ? '0 : ADDR_BITS'(1 << (ADDR_BITS - 1));
                    o_ld_arm      <= 1'b1;
                    db_h          <= ~db_h;
                    db_m_ld       <= cg_m_next;
                    db_pf         <= 1'b1;
                    cg_base       <= cg_base + cg_stepM;
                    cg_rows_left  <= cg_rows_left - cg_m_next;
                end else begin
                    db_pf         <= 1'b0;
                end
                st           <= S_DBTS;
            end
            S_DBTS: if (busy_s[1]) begin
                o_seq_start <= 1'b0;
                st          <= S_DBTR;
            end
            S_DBTR: if (done_s[1] && !busy_s[1] && rec_cnt == rec_expected) begin
                if (db_pf) begin
                    st <= S_DBGO;
                end else begin
                    in_cg <= 1'b0;
                    st    <= S_FETCH;
                end
            end
            S_POSTW: if (i_wr_idle) begin
                o_sync_arm <= 1'b1;
                st         <= S_SYNC;
            end
            S_SYNC: if (i_sync_done) st <= S_FETCH;
            S_OUT: begin
                o_wr_flush <= 1'b1;                 // drain whatever is still queued, then re-arm the address
`ifdef NODE_DRAIN_ONLY
                if (i_wr_drained) begin
`else
                if (i_wr_idle) begin
`endif
                    o_wr_flush   <= 1'b0;
                    o_wr_restart <= 1'b1;
                    st           <= S_FETCH;
                end
            end
            S_FLUSH: begin
                if (settle != 2'd0)  settle <= settle - 1'b1;
`ifdef NODE_DRAIN_ONLY
                else if (i_wr_drained) begin
`else
                else if (i_wr_idle) begin
`endif
                    o_wr_flush <= 1'b0;
                    st         <= S_FETCH;
                end
            end
            default: ;
            endcase
        end
    end
endmodule
