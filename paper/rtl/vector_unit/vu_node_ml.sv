// Multi-lane vector node: N_LANE vector-unit lanes (vu_lane.sv) sharing one program fetch, one operand
// loader and one pair of output writers, on the fabric clock.  Same program, operand descriptors and
// output streams as the one-lane vu_node.sv (the verified reference): the beats written are identical
// for every N_LANE.  Design: docs/PI0_FULL_MODEL_ON_CHIP_DESIGN_20260916.md section 3.3.
//
// Program: 6-word op records and END, exactly as vu_node.sv (see its header for the field layout).
//
// Chunking.  rows_per_lane = min(512, 2^SLOT_BITS >> ceil(log2 L')), L' = L when L is a multiple of 4,
// else L + 3 (so a lane's elements plus the word alignment of its first element fit 2^SLOT_BITS / 4
// words; L = 2^SLOT_BITS - 2 or - 1 is rejected with o_error).  A chunk is N_LANE * rows_per_lane rows;
// lane l takes chunk rows [l * rpl, (l+1) * rpl), trailing lanes of the last chunk may be empty.
//
// Operand slots.  Each lane has its own slot memories (x b c d e rs mask), 2^SLOT_BITS / 4 words of 4
// elements (block RAM, 128 bits x 512 for 32-bit slots).  Words are aligned to the operand tensor's
// element grid (word m = elements 4m .. 4m+3), which is also the beat grid, so vu_word_loader.sv takes
// them straight out of the beats, 4 elements per cycle.  One load per operand per chunk (E: the chunk's
// R x L elements; R: its rows; C: L elements once per op) is steered word by word to every lane whose
// element range touches the word, at the word index relative to the lane's first word.  A lane reads
// its elements with (word, position) counters started at its first element's position; the 4:1 element
// select follows the registered block RAM read (look-ahead read addresses, as in vu_node.sv).
//
// Outputs.  Lanes run without back-pressure (i_ready = 1).  Each lane packs its elements into beats
// itself, starting at the bit position its first element has in the op's element region (known before
// the run: the element count of every earlier lane mod 256), and pads its last partial beat with zeros.
// The beats go into the lane's beat buffer (256 bits x 2^(SLOT_BITS-3) + 1 beats, block RAM: a full lane
// of 32-bit elements plus the partial beat its first element may start in); summary records into its
// record buffer (64 x 512).  After every lane has finished, the buffers are drained in lane order: element beats
// one per cycle, a lane's first beat OR-ed with the partial beat carried from before it (their bit ranges
// are disjoint), and summary records one per cycle into the shared summary packer.  The carried partial
// beat crosses chunks and is written at the end of the op.  Writers as in vu_node.sv, merged by vu_wr_merge.sv.
//
//   w0[7] x_rot: the x slot (shape E) is read rotate-half within the row -- element j reads x[j + L/2]
//   for j < L/2 and x[j - L/2] otherwise, the partner RoPE pass B needs (L must be even).
//
// Parallel loading (N_LD > 1).  The lanes are split into N_LD equal groups, and each group has its own word loader,
// which loads only the elements (E), rows (R) or column operand (C) its lanes hold.  The loaders share the read
// port through vu_rd_fanout.sv (up to RD_MAX_OUTSTANDING open bursts across loaders, R routed in AR order into a
// 32-beat FIFO per loader, each loader keeping at most 2 bursts open), so they fill disjoint slot memories at
// the same time.  N_LD = 1 is the single loader, cycle for cycle.
//
// Negative control: VU_NODE_ML_NEG_MERGE_ORDER drains the lanes in reverse order (must fail, N_LANE >= 2).
module vu_node_ml #(
    parameter F_GELU  = "vu_tbl_gelu.mem",
    parameter F_SIGM  = "vu_tbl_sigm.mem",
    parameter F_EXP   = "vu_tbl_exp.mem",
    parameter F_RSQRT = "vu_tbl_rsqrt.mem",
    parameter F_QUANT = "vu_tbl_quant.mem",
    parameter integer N_LANE             = 4,
    parameter integer AXI_ADDR_WIDTH     = 42,
    parameter integer SLOT_BITS          = 11,
    parameter integer RD_BURST_BEATS     = 16,
    parameter integer RD_MAX_OUTSTANDING = 8,
    parameter integer WR_BURST_BEATS     = 16,
    parameter integer WR_MAX_OUTSTANDING = 8,
    parameter integer WR_FIFO_LOG2       = 6,
    // beats per loader in the read fan-out (N_LD > 1); a loader keeps 2^(RD_FIFO_LOG2 - 4) bursts open (capped by
    // RD_MAX_OUTSTANDING).  5 (2 bursts per loader) left the vector node latency-bound on silicon: with a 64-cycle
    // GDDR6 round trip the smoke chunk took 2.34x its ideal-memory cycles.  The FIFO is two BRAM72K at any depth <= 512.
    parameter integer RD_FIFO_LOG2       = 7,
    // pre-dequant in front of each lane (vu_pdq.sv): a record with w0[119] feeds its int32 x and d through
    // DEQUANT(x; c, e) and DEQUANT(d; c, b) first (fusion 2).  0: the flag is ignored and nothing is built.
    parameter integer PDQ                = 1,
    parameter integer N_LD               = 1,     // operand loaders: 1, or a divisor of N_LANE (lane groups)
    parameter integer QTAIL              = 1      // fused QUANT tail in every lane (record flag w0[118])
) (
    input  wire                       i_clk,
    input  wire                       i_rstn,
    input  wire                       i_prog_start,
    input  wire [AXI_ADDR_WIDTH-1:0]  i_prog_base,

    output wire                       o_arvalid,
    input  wire                       i_arready,
    output wire [AXI_ADDR_WIDTH-1:0]  o_araddr,
    output wire [7:0]                 o_arlen,
    output wire [2:0]                 o_arsize,
    output wire [1:0]                 o_arburst,
    output wire                       o_rready,
    input  wire                       i_rvalid,
    input  wire [255:0]               i_rdata,
    input  wire [1:0]                 i_rresp,
    input  wire                       i_rlast,

    output wire                       o_awvalid,
    input  wire                       i_awready,
    output wire [AXI_ADDR_WIDTH-1:0]  o_awaddr,
    output wire [7:0]                 o_awlen,
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

    output reg                        o_halt,
    output wire                       o_error,
    output wire [6:0]                 o_error_bits,     // {lane overflow, sync, length, summary writer, element writer, loader, prog fetch}
    // silicon debug: a snapshot of the element packer's end-of-op state, taken on every entry into C_FLUSH
    // (after an op's last chunk).  {op, pad_w, ow_log2, L_len[7:0], epos, off_end, obits[0], obits[1],
    //  nb[0], nb[1], part[1:0], rows_l[0], rows_l[1], ecarry != 0, n_flush[7:0]} -- see the assign below
    output reg  [127:0]               o_dbg
);
    localparam integer AW    = AXI_ADDR_WIDTH;
    localparam integer NL    = N_LANE;
    localparam integer LG    = $clog2(N_LANE);           // 0 for one lane
    localparam integer LB    = (LG > 0) ? LG : 1;
    localparam integer WB    = SLOT_BITS - 2;            // slot word address bits
    localparam integer CB    = SLOT_BITS + LG;           // elements per chunk <= 2^CB
    localparam integer RB    = 10 + LG;                  // rows per chunk < 2^RB
    // element beat buffer: 2^SLOT_BITS 32-bit elements are 2^(SLOT_BITS-3) beats, and a lane's first element
    // may start inside a beat (its bit offset in the op's element region), which can need one beat more
    localparam integer EDEPTH = (1 << (SLOT_BITS - 3)) + 1;
    localparam integer EAB    = $clog2(EDEPTH);          // beat buffer address bits
    localparam integer EKB    = $clog2(EDEPTH + 1);      // beat count bits (0 .. EDEPTH)
    localparam [3:0] OP_RMS_STAT = 4'd1, OP_LN_STAT = 4'd3, OP_ROPE_A = 4'd5, OP_SMAX_SUM = 4'd10,
                     OP_SMAX_Q8 = 4'd12, OP_QUANT = 4'd13, OP_DEQUANT = 4'd14, OP_EULER = 4'd15;
    localparam [1:0] SH_E = 2'd0, SH_R = 2'd1, SH_C = 2'd2, SH_K = 2'd3;
    localparam integer NSLOT = 7;   // x b c d e rs mask

    // ------------------------------------------------------------------ control state
    localparam [4:0] C_FETCH = 5'd0, C_DECODE = 5'd1, C_DECODE2 = 5'd2, C_DECODE3 = 5'd3, C_LOADC = 5'd4,
                     C_LOADC_W = 5'd5, C_SETTLE = 5'd6, C_CHUNK = 5'd7, C_SETUP1 = 5'd8, C_SETUP2 = 5'd9,
                     C_SETUP3 = 5'd10, C_LOADE = 5'd11, C_LOADE_W = 5'd12, C_RUN = 5'd13, C_DRAIN = 5'd14,
                     C_FLUSH = 5'd15, C_FLUSH_W = 5'd16, C_HALT = 5'd17, C_SYNC = 5'd18, C_POSTW = 5'd19;
    reg [4:0] st;

    // ------------------------------------------------------------------ program fetch
    wire [127:0] cmd;
    wire         cmd_valid, pf_error;
    wire         fetching = (st == C_FETCH);
    wire         f_arvalid, f_rready, l_arvalid, l_rready;
    wire [AW-1:0] f_araddr, l_araddr;
    wire [7:0]   f_arlen, l_arlen;
    wire [2:0]   f_arsize, l_arsize;
    wire [1:0]   f_arburst, l_arburst;

    colpar_prog_fetch #(.AXI_ADDR_WIDTH(AW), .RECORD_WORDS(6)) u_fetch (
        .i_clk(i_clk), .i_rstn(i_rstn), .i_start(i_prog_start), .i_base(i_prog_base), .i_grant(fetching),
        .o_arvalid(f_arvalid), .i_arready(i_arready && fetching), .o_araddr(f_araddr), .o_arlen(f_arlen),
        .o_arsize(f_arsize), .o_arburst(f_arburst), .i_rvalid(i_rvalid && fetching), .o_rready(f_rready),
        .i_rdata(i_rdata), .i_rresp(i_rresp), .i_rlast(i_rlast),
        .o_cmd(cmd), .o_cmd_valid(cmd_valid), .i_cmd_ready(fetching), .o_error(pf_error));

    assign o_arvalid = sy_busy ? sy_arvalid : fetching ? f_arvalid : l_arvalid;
    assign o_araddr  = sy_busy ? sy_araddr  : fetching ? f_araddr  : l_araddr;
    assign o_arlen   = sy_busy ? sy_arlen   : fetching ? f_arlen   : l_arlen;
    assign o_arsize  = sy_busy ? sy_arsize  : fetching ? f_arsize  : l_arsize;
    assign o_arburst = sy_busy ? sy_arburst : fetching ? f_arburst : l_arburst;
    assign o_rready  = sy_busy ? sy_rready  : fetching ? f_rready  : l_rready;

    // ------------------------------------------------------------------ op record
    reg [127:0] w [0:5];
    reg [2:0]   wn;
    reg [3:0]   op;
    reg         b_fp32, bias_en, out_fp32;
    reg [31:0]  k_const;
    reg [19:0]  R_rows;
    reg [15:0]  L_len;
    reg         x_rot;            // w0[7]: rotate-half read of the x slot (RoPE pass B)
    reg         q_tail;           // w0[118]: fused QUANT -- the op's bf16 rows leave the lane as int8 codes + s_row
    reg         pdq_en;           // w0[119]: pre-dequant x and d (vu_pdq.sv)
    reg [AW-1:0] out_base, sum_base;
    reg [9:0]    e_blk;                           // w5[9:0]: element beats per row block (0 = contiguous)
    reg [23:0]   e_gap;                           // w5[33:10]: bytes skipped after each block
    reg [47:0]  desc [0:NSLOT-1];
    wire [1:0]  shape [0:NSLOT-1];
    genvar gs, gl;
    generate
        for (gs = 0; gs < NSLOT; gs = gs + 1) begin : g_shape
            assign shape[gs] = desc[gs][43:42];
        end
    endgenerate

    // output element width (elem_w: element beats, pad_w: width of every element of the op's element region)
    reg [5:0]   elem_w, pad_w;
    reg [2:0]   ow_log2;
    reg         len_err;

    // ------------------------------------------------------------------ chunking
    reg [19:0]  rows_left;
    reg [4:0]   lc;                              // ceil(log2 L')
    reg [3:0]   rpl_log2;                        // log2(rows per lane)
    reg [RB-1:0] rows_c;
    reg [CB:0]  n_elem;
    reg [SLOT_BITS:0] epl;                       // elements per full lane
    reg [31:0]  e_off, r_off;
    reg [2:0]   slot_i;
    reg [1:0]   settle;
    reg         ret_e;
    wire [RB:0] chunk_rows = (RB+1)'(NL) << rpl_log2;

    reg [9:0]    rows_l  [0:NL-1];
    reg [CB-2:0] wloE    [0:NL-1];
    reg [CB-2:0] whiE    [0:NL-1];
    reg [CB-2:0] wloR    [0:NL-1];
    reg [CB-2:0] whiR    [0:NL-1];
    reg [1:0]    pos0E   [0:NL-1];
    reg [1:0]    pos0R   [0:NL-1];
    reg [7:0]    obits   [0:NL-1];
    // Each lane's element bits mod 256 (where its last beat ends).  Only (rows x L) mod 2^(8 - ow_log2) matters and
    // ow_log2 >= 3, so a 5 x 5 product per lane is enough.  This used to be a 32 x 32 multiply on a blocking
    // temporary inside the C_SETUP2 lane loop, and on silicon (Synplify W-2025.03X, S0 builds 2026-09-17) lane 1
    // got lane 0's value: with uneven lanes the op's end offset came out wrong and its partial last element beat was
    // never flushed (SMAX_OUT R 8 x L 67 lost 16 bytes).  One wire per lane, LUT multipliers.
    wire [7:0]   obits_w [0:NL-1];
    genvar gob;
    generate
        for (gob = 0; gob < NL; gob = gob + 1) begin : g_obits
            (* syn_multstyle = "logic" *) wire [9:0] prod5 = rows_l[gob][4:0] * L_len[4:0];
            wire [9:0] nb = (pad_w == 6'd0) ? 10'd0 : (op == OP_LN_STAT) ? 10'(rows_l[gob][4:0]) : prod5;
            assign obits_w[gob] = 8'({nb, 5'd0} >> (3'd5 - ow_log2));   // nb << ow_log2, mod 256
        end
    endgenerate
    reg [7:0]    dbg_n = 8'd0;                   // C_FLUSH entries (ops finished) since configuration
    reg [7:0]    off_q   [0:NL-1];
    reg [NL-1:0] en_q;
    reg [7:0]    off_end, epos;

    function automatic [4:0] clog2_17(input [16:0] v);
        integer i;
        clog2_17 = 5'd0;
        for (i = 16; i >= 0; i = i - 1)
            if (clog2_17 == 5'd0 && v > (17'd1 << i)) clog2_17 = 5'(i + 1);
    endfunction

    // ------------------------------------------------------------------ operand loaders + lane steering
    localparam integer ND  = N_LD;
    localparam integer LPD = NL / ND;                    // lanes per loader
    initial if (ND < 1 || NL % ND != 0) $fatal(1, "vu_node_ml: N_LD must divide N_LANE");
    reg          ld_arm;                                 // arm the loaders flagged in ld_go
    reg [ND-1:0] ld_go;
    reg [AW-1:0] ld_base;
    reg [1:0]    ld_wsel, ld_field;
    reg [2:0]    ld_slot;
    reg [31:0]   ld_first [0:ND-1];
    reg [CB:0]   ld_count [0:ND-1];
    reg [CB-2:0] ld_woff  [0:ND-1];                      // loader d's first word, relative to the load's first word
    reg [ND-1:0] ld_pend;
    wire [ND-1:0] ldv_we, ldv_done, ldv_error;
    wire [CB-2:0] ldv_widx  [0:ND-1];
    wire [127:0]  ldv_wword [0:ND-1];
    wire         ld_error = |ldv_error;
    wire         ld_all   = ((ld_pend & ~ldv_done) == '0);   // every armed loader done (this cycle at the latest)
    reg [NL-1:0] st_en;
    reg [CB-2:0] st_wlo [0:NL-1];
    reg [CB-2:0] st_whi [0:NL-1];

    wire [ND-1:0]       lf_arvalid, lf_arready, lf_rvalid, lf_rready, lf_rlast;
    wire [ND*AW-1:0]    lf_araddr;
    wire [ND*8-1:0]     lf_arlen;
    wire [ND*3-1:0]     lf_arsize;
    wire [ND*2-1:0]     lf_arburst, lf_rresp;
    wire [ND*256-1:0]   lf_rdata;

    genvar gd;
    generate
        for (gd = 0; gd < ND; gd = gd + 1) begin : g_ld
            vu_word_loader #(
                .AXI_ADDR_WIDTH(AW), .CNT_BITS(CB), .BEATS_PER_BURST(RD_BURST_BEATS),
                .MAX_OUTSTANDING((ND == 1) ? RD_MAX_OUTSTANDING : ((1 << (RD_FIFO_LOG2 - 4)) < RD_MAX_OUTSTANDING ? (1 << (RD_FIFO_LOG2 - 4)) : RD_MAX_OUTSTANDING))
            ) u_ld (
                .i_clk(i_clk), .i_rstn(i_rstn), .i_arm(ld_arm && ld_go[gd]), .i_base(ld_base),
                .i_first(ld_first[gd]), .i_count(ld_count[gd]), .i_wsel(ld_wsel), .i_field(ld_field),
                .o_arvalid(lf_arvalid[gd]), .i_arready(lf_arready[gd]), .o_araddr(lf_araddr[gd*AW +: AW]),
                .o_arlen(lf_arlen[gd*8 +: 8]), .o_arsize(lf_arsize[gd*3 +: 3]), .o_arburst(lf_arburst[gd*2 +: 2]),
                .o_rready(lf_rready[gd]), .i_rvalid(lf_rvalid[gd]), .i_rdata(lf_rdata[gd*256 +: 256]),
                .i_rresp(lf_rresp[gd*2 +: 2]), .i_rlast(lf_rlast[gd]),
                .o_we(ldv_we[gd]), .o_widx(ldv_widx[gd]), .o_wword(ldv_wword[gd]),
                .o_busy(), .o_done(ldv_done[gd]), .o_error(ldv_error[gd]));
        end
        if (ND == 1) begin : g_rd1
            assign l_arvalid     = lf_arvalid[0];
            assign l_araddr      = lf_araddr;
            assign l_arlen       = lf_arlen;
            assign l_arsize      = lf_arsize;
            assign l_arburst     = lf_arburst;
            assign l_rready      = lf_rready[0];
            assign lf_arready[0] = i_arready && !fetching;
            assign lf_rvalid[0]  = i_rvalid && !fetching;
            assign lf_rdata      = i_rdata;
            assign lf_rresp      = i_rresp;
            assign lf_rlast[0]   = i_rlast;
        end else begin : g_rdn
            vu_rd_fanout #(.N_M(ND), .AXI_ADDR_WIDTH(AW), .MAX_OUTSTANDING(RD_MAX_OUTSTANDING), .FIFO_LOG2(RD_FIFO_LOG2)) u_fan (
                .i_clk(i_clk), .i_rstn(i_rstn),
                .s_arvalid(lf_arvalid), .s_arready(lf_arready), .s_araddr(lf_araddr), .s_arlen(lf_arlen),
                .s_arsize(lf_arsize), .s_arburst(lf_arburst), .s_rvalid(lf_rvalid), .s_rready(lf_rready),
                .s_rdata(lf_rdata), .s_rresp(lf_rresp), .s_rlast(lf_rlast),
                .m_arvalid(l_arvalid), .m_arready(i_arready && !fetching), .m_araddr(l_araddr), .m_arlen(l_arlen),
                .m_arsize(l_arsize), .m_arburst(l_arburst), .m_rvalid(i_rvalid && !fetching), .m_rready(l_rready),
                .m_rdata(i_rdata), .m_rresp(i_rresp), .m_rlast(i_rlast));
        end
    endgenerate

    // steered slot writes (one register stage): lane l takes the words of its group's loader
    reg [NL-1:0] sw_we;
    reg [WB-1:0] sw_addr [0:NL-1];
    reg [127:0]  sw_word [0:NL-1];
    reg [2:0]    sw_slot;
    always @(posedge i_clk) begin
        sw_slot <= ld_slot;
        for (int l = 0; l < NL; l++) begin : steer
            reg [CB-2:0] aw;
            aw          = ldv_widx[l / LPD] + ld_woff[l / LPD];
            sw_word[l] <= ldv_wword[l / LPD];
            sw_we[l]   <= i_rstn && ldv_we[l / LPD] && st_en[l] && (aw >= st_wlo[l]) && (aw <= st_whi[l]);
            sw_addr[l] <= WB'(aw - st_wlo[l]);
        end
    end

    // ------------------------------------------------------------------ lanes
    reg          desc_we;
    reg          primed;
    wire         go_run = (st == C_LOADE) && (slot_i == 3'(NSLOT));
    wire [NL-1:0] lane_done, lane_part, lane_ovf;
    wire [EKB-1:0] nb [0:NL-1];
    wire [255:0] eq   [0:NL-1];
    wire [63:0]  sq   [0:NL-1];

    // drain read addresses (shared by the lanes' buffers)
    reg [EKB-1:0] ed_k;
    reg [9:0]    sd_k;

    generate
        for (gl = 0; gl < NL; gl = gl + 1) begin : g_lane
            // The read pointers are kept one element AHEAD (cur / nxt): a fire only selects a register, so the slot
            // memories' read address is one mux after the fire decision.  Before: st -> in_valid -> fire -> +1 ->
            // shape mux -> BRAM address, 6 levels, the vector node's limit at ~350 MHz (2026-09-22).
            reg  [WB+1:0]      eI, eI1;         // element pointer {word, phase}: current element / next
            reg  [WB+1:0]      rI, rI1;         // row pointer {word, phase}
            reg  [WB+1:0]      xI, xI1;         // x-slot pointer: element +- L/2 with the rotate-half flag, else eI
            reg  [SLOT_BITS:0] jc, jc1;         // element index within the row: current / next
            reg                last_q, last1;   // jc == L - 1: current / next
            reg  [9:0]         rc;
            reg                run_done;
            reg                in_valid;        // registered, exact (see its always block)
            wire               lane_ready, lane_idle, lane_ovalid;
            wire [1:0]         lane_okind;
            wire [63:0]        lane_odata;
            wire               fire     = in_valid && lane_ready;
            wire               last_j   = last_q;
            wire               radv     = fire && last_j;
            wire [SLOT_BITS:0] l2    = {1'b0, L_len[SLOT_BITS:1]};
            wire [SLOT_BITS:0] jc2   = last1 ? '0 : jc1 + 1'b1;                    // the element after next
            wire               last2 = (32'(jc2) + 32'd1 == 32'(L_len));
            wire               lo2   = (32'(jc2) < 32'(l2));                       // jc2 < L / 2
            wire [WB+1:0]      eI2   = eI1 + 1'b1;
`ifdef VU_NODE_NEG_NO_ROT     // negative control: ignore the rotate-half flag
            wire [WB+1:0]      xI2   = eI2;
`else
            wire [WB+1:0]      xI2   = !x_rot ? eI2 : lo2 ? eI2 + (WB+2)'(l2) : eI2 - (WB+2)'(l2);
`endif
            wire [WB+1:0]      cI    = (WB+2)'(jc), cI1 = (WB+2)'(jc1);
            // record start: element 0 and element jc1 (1, or 0 when L = 1)
            wire [WB+1:0]      eI0   = (WB+2)'(pos0E[gl]);
            wire [WB+1:0]      eI0n  = eI0 + 1'b1;
            wire [SLOT_BITS:0] jc1_0 = (L_len == 16'd1) ? '0 : (SLOT_BITS+1)'(1);
            wire               lo0   = (l2 != '0), lo1_0 = (32'(jc1_0) < 32'(l2));
`ifdef VU_NODE_NEG_NO_ROT
            wire [WB+1:0]      xI0 = eI0, xI1_0 = eI0n;
`else
            wire [WB+1:0]      xI0   = !x_rot ? eI0  : lo0   ? eI0  + (WB+2)'(l2) : eI0  - (WB+2)'(l2);
            wire [WB+1:0]      xI1_0 = !x_rot ? eI0n : lo1_0 ? eI0n + (WB+2)'(l2) : eI0n - (WB+2)'(l2);
`endif
            // in_valid one cycle ahead, exact: it was (st == C_RUN) && primed && !run_done && en_q, and primed is
            // exactly "C_RUN last cycle"; so: C_RUN now and staying (no drain, reset or restart), rows left after
            // this cycle's fire, lane enabled
            wire               run_done_n = run_done || (radv && (rc + 10'd1 == rows_l[gl]));
            always @(posedge i_clk)
                in_valid <= i_rstn && !i_prog_start && (st == C_RUN) && !(&lane_done) && !run_done_n && en_q[gl];

            // ---- slot memories ----
            wire [31:0] q [0:NSLOT-1];
            for (gs = 0; gs < NSLOT; gs = gs + 1) begin : g_slot
                localparam integer WID = (gs == 5) ? 16 : (gs == 6) ? 1 : 32;
                reg  [4*WID-1:0] mem [0:(1<<WB)-1] /* synthesis syn_ramstyle = "block_ram" */;
                reg  [4*WID-1:0] qr;
                reg  [1:0]       pq;
                wire [4*WID-1:0] wv;
                wire [WB+1:0]    a_cur = (shape[gs] == SH_R) ? rI  : (shape[gs] == SH_C) ? cI  : (gs == 0) ? xI  : eI;
                wire [WB+1:0]    a_nxt = (shape[gs] == SH_R) ? rI1 : (shape[gs] == SH_C) ? cI1 : (gs == 0) ? xI1 : eI1;
                wire [WB+1:0]    a_rd  = fire ? a_nxt : a_cur;
                wire [WB-1:0]    ra    = a_rd[WB+1:2];
                if (WID == 32) begin : g_w32
                    assign wv = sw_word[gl];
                end else if (WID == 16) begin : g_w16
                    assign wv = {sw_word[gl][111:96], sw_word[gl][79:64], sw_word[gl][47:32], sw_word[gl][15:0]};
                end else begin : g_w1
                    assign wv = {sw_word[gl][96], sw_word[gl][64], sw_word[gl][32], sw_word[gl][0]};
                end
                always @(posedge i_clk) begin
                    if (sw_we[gl] && sw_slot == 3'(gs)) mem[sw_addr[gl]] <= wv;
                    qr <= mem[ra];
                    pq <= a_rd[1:0];
                end
                assign q[gs] = 32'(qr[32'(pq) * WID +: WID]);
            end

            wire [31:0] x32  = (shape[0] == SH_K) ? desc[0][31:0] : q[0];
            wire [47:0] x48  = (desc[0][45:44] == 2'd2) ? {{16{x32[31]}}, x32} : {16'd0, x32};
            wire [31:0] b32  = (shape[1] == SH_K) ? desc[1][31:0] : q[1];
            wire [31:0] c32  = (shape[2] == SH_K) ? desc[2][31:0] : q[2];
            wire [31:0] d32  = (shape[3] == SH_K) ? desc[3][31:0] : q[3];
            wire [31:0] e32  = (shape[4] == SH_K) ? desc[4][31:0] : q[4];
            wire [31:0] rs32 = (shape[5] == SH_K) ? desc[5][31:0] : q[5];
            wire        mk1  = (shape[6] == SH_K) ? desc[6][0] : q[6][0];

            // pre-dequant (fusion 2): a wire unless the record has w0[119]
            wire               p_valid, p_ready, p_last, p_mask, p_empty, l_ready, l_idle;
            wire [47:0]        p_x;
            wire [15:0]        p_rs;
            wire [31:0]        p_b, p_c, p_d, p_e;
            if (PDQ != 0) begin : g_pdq
                vu_pdq u_pdq (
                    .i_clk(i_clk), .i_rstn(i_rstn), .i_en(pdq_en),
                    .i_valid(in_valid), .o_ready(lane_ready), .i_last(last_j), .i_mask(mk1), .i_x(x48), .i_rs(rs32[15:0]),
                    .i_b(b32), .i_c(c32), .i_d(d32), .i_e(e32),
                    .o_valid(p_valid), .i_ready(l_ready), .o_last(p_last), .o_mask(p_mask), .o_x(p_x), .o_rs(p_rs),
                    .o_b(p_b), .o_c(p_c), .o_d(p_d), .o_e(p_e), .o_empty(p_empty));
            end else begin : g_nopdq
                assign p_valid = in_valid; assign lane_ready = l_ready; assign p_last = last_j; assign p_mask = mk1;
                assign p_x = x48; assign p_rs = rs32[15:0]; assign p_b = b32; assign p_c = c32; assign p_d = d32;
                assign p_e = e32; assign p_empty = 1'b1;
            end
            assign lane_idle = l_idle && p_empty;
            vu_lane #(
                .F_GELU(F_GELU), .F_SIGM(F_SIGM), .F_EXP(F_EXP), .F_RSQRT(F_RSQRT), .F_QUANT(F_QUANT),
                .QTAIL(QTAIL), .ROW_AW(SLOT_BITS)
            ) u_lane (
                .i_clk(i_clk), .i_rstn(i_rstn),
                .i_desc_we(desc_we), .i_op(op), .i_b_fp32(b_fp32), .i_bias_en(bias_en), .i_out_fp32(out_fp32), .i_qtail(q_tail),
                .i_k(k_const), .o_idle(l_idle),
                .i_valid(p_valid), .o_ready(l_ready), .i_last(p_last),
                .i_mask(p_mask), .i_x(p_x), .i_rs(p_rs), .i_b(p_b), .i_c(p_c), .i_d(p_d), .i_e(p_e),
                .o_valid(lane_ovalid), .i_ready(1'b1), .o_kind(lane_okind), .o_data(lane_odata));

            // ---- per-lane output: beat packer + beat buffer, record buffer ----
            reg          ov_q;
            reg  [1:0]   ok_q;
            reg  [63:0]  od_q;
            reg  [255:0] pack;
            reg  [8:0]   pos;
            reg  [EKB-1:0] wp;
            reg  [9:0]   sc;
            reg          part_q, done_q, ovf_q;
            wire         e_in  = ov_q && (ok_q != 2'd1);
            wire         s_in  = ov_q && (ok_q == 2'd1);
            // fin registered: the lane's idle came through the PDQ's valid mux (pdq_en) and then tail / pack / the
            // beat buffer in one cycle (the node's longest path once the pointers were retimed, 2026-09-22).  A cycle
            // late is exact: once the rows are done and the lane idle, nothing can arrive any more.  Cleared with
            // go_run: the previous record's 1 must not be seen in the new record's first C_RUN cycle.
            reg          fin;
            always @(posedge i_clk)
                fin <= !go_run && (!en_q[gl] || (run_done && lane_idle && !lane_ovalid && !ov_q && sc == rows_l[gl]));
            wire         tail  = (st == C_RUN) && fin && (pos != 9'd0) && !done_q;
            wire         shift = e_in || tail;
            wire [31:0]  din   = e_in ? od_q[31:0] : 32'd0;
            wire [255:0] pack_n = (pad_w == 6'd8)  ? {din[7:0],  pack[255:8]}  :
                                  (pad_w == 6'd16) ? {din[15:0], pack[255:16]} :
                                                     {din,       pack[255:32]};
            wire         full  = shift && (pos + 9'(pad_w) == 9'd256);

            reg  [255:0] ebuf [0:EDEPTH-1] /* synthesis syn_ramstyle = "block_ram" */;
            reg  [63:0]  sbuf [0:511] /* synthesis syn_ramstyle = "block_ram" */;
            reg  [255:0] eq_r;
            reg  [63:0]  sq_r;
            always @(posedge i_clk) begin
                if (full) ebuf[EAB'(wp)] <= pack_n;
                if (s_in) sbuf[sc[8:0]] <= od_q;
                eq_r <= ebuf[EAB'(ed_k)];
                sq_r <= sbuf[sd_k[8:0]];
            end

            always @(posedge i_clk) begin
                ov_q <= i_rstn && lane_ovalid;
                ok_q <= lane_okind;
                od_q <= lane_odata;
                if (!i_rstn) begin
                    run_done <= 1'b1;
                    done_q   <= 1'b0;
                    ovf_q    <= 1'b0;
                    pos      <= 9'd0;
                end else if (go_run) begin
                    eI       <= eI0;
                    eI1      <= eI0n;
                    rI       <= (WB+2)'(pos0R[gl]);
                    rI1      <= (WB+2)'(pos0R[gl]) + (WB+2)'(L_len == 16'd1);
                    jc       <= '0;
                    jc1      <= jc1_0;
                    last_q   <= (L_len == 16'd1);
                    last1    <= (32'(jc1_0) + 32'd1 == 32'(L_len));
                    xI       <= xI0;
                    xI1      <= xI1_0;
                    rc       <= 10'd0;
                    run_done <= 1'b0;
                    pack     <= '0;
                    pos      <= en_q[gl] ? {1'b0, off_q[gl]} : 9'd0;
                    wp       <= '0;
                    sc       <= 10'd0;
                    part_q   <= 1'b0;
                    done_q   <= 1'b0;
                end else begin
                    if (fire) begin
                        eI     <= eI1;   eI1   <= eI2;
                        rI     <= rI1;   rI1   <= last1 ? rI1 + 1'b1 : rI1;
                        jc     <= jc1;   jc1   <= jc2;
                        last_q <= last1; last1 <= last2;
                        xI     <= xI1;   xI1   <= xI2;
                    end
                    if (radv) begin
                        rc <= rc + 10'd1;
                        if (rc + 10'd1 == rows_l[gl]) run_done <= 1'b1;
                    end
                    if (shift) begin
                        pack <= pack_n;
                        if (full) begin
                            if (32'(wp) == EDEPTH) ovf_q <= 1'b1;
                            wp  <= wp + 1'b1;
                            pos <= 9'd0;
                            if (tail) part_q <= 1'b1;
                        end else begin
                            pos <= pos + 9'(pad_w);
                        end
                    end
                    if (s_in) sc <= sc + 10'd1;
                    if ((st == C_RUN) && fin && pos == 9'd0 && !done_q) done_q <= 1'b1;
                end
            end

            assign lane_done[gl] = done_q;
            assign lane_part[gl] = part_q;
            assign lane_ovf[gl]  = ovf_q;
            assign nb[gl]        = wp;
            assign eq[gl]        = eq_r;
            assign sq[gl]        = sq_r;
        end
    endgenerate

    // ------------------------------------------------------------------ shared packers + writers
    reg  [255:0] spack, ecarry;
    reg  [8:0]   spos;
    reg          e_push, s_push;
    reg  [255:0] e_beat, s_beat;
    reg          e_restart, s_restart, wr_flush;
    wire         e_afull, s_afull, e_idle, s_idle, e_err, s_err, e_drained, s_drained;

    // drains
    reg          ed_busy, sd_busy, ev1, sv1, efirst1, epart1;
    reg [LB-1:0] ed_i, sd_i, el1, sl1;
`ifdef VU_NODE_ML_NEG_MERGE_ORDER    // negative control: lanes drained in reverse order
    wire [LB-1:0] ed_lane = LB'(NL - 1) - ed_i;
    wire [LB-1:0] sd_lane = LB'(NL - 1) - sd_i;
`else
    wire [LB-1:0] ed_lane = ed_i;
    wire [LB-1:0] sd_lane = sd_i;
`endif

    wire [1:0]         wm_awvalid, wm_awready, wm_wvalid, wm_wready, wm_wlast, wm_bvalid, wm_bready;
    wire [2*AW-1:0]    wm_awaddr;
    wire [15:0]        wm_awlen;
    wire [5:0]         wm_awsize;
    wire [3:0]         wm_awburst;
    wire [511:0]       wm_wdata;
    wire [63:0]        wm_wstrb;

    vu_beat_writer #(.AXI_ADDR_WIDTH(AW), .FIFO_LOG2(WR_FIFO_LOG2), .BURST_BEATS(WR_BURST_BEATS),
                     .MAX_OUTSTANDING(WR_MAX_OUTSTANDING)) u_ew (
        .i_clk(i_clk), .i_rstn(i_rstn), .i_base(out_base), .i_blk_beats(e_blk), .i_gap_bytes(e_gap),
        .i_restart(e_restart), .i_flush(wr_flush),
        .i_beat(e_beat), .i_beat_valid(e_push), .o_afull(e_afull),
        .o_awvalid(wm_awvalid[0]), .i_awready(wm_awready[0]), .o_awaddr(wm_awaddr[0 +: AW]),
        .o_awlen(wm_awlen[0 +: 8]), .o_awsize(wm_awsize[0 +: 3]), .o_awburst(wm_awburst[0 +: 2]),
        .o_wvalid(wm_wvalid[0]), .i_wready(wm_wready[0]), .o_wdata(wm_wdata[0 +: 256]),
        .o_wstrb(wm_wstrb[0 +: 32]), .o_wlast(wm_wlast[0]), .i_bvalid(wm_bvalid[0]),
        .o_bready(wm_bready[0]), .i_bresp(i_bresp), .o_idle(e_idle), .o_drained(e_drained), .o_error(e_err));

    vu_beat_writer #(.AXI_ADDR_WIDTH(AW), .FIFO_LOG2(WR_FIFO_LOG2), .BURST_BEATS(WR_BURST_BEATS),
                     .MAX_OUTSTANDING(WR_MAX_OUTSTANDING)) u_sw (
        .i_clk(i_clk), .i_rstn(i_rstn), .i_base(sum_base), .i_blk_beats(10'd0), .i_gap_bytes(24'd0),
        .i_restart(s_restart), .i_flush(wr_flush),
        .i_beat(s_beat), .i_beat_valid(s_push), .o_afull(s_afull),
        .o_awvalid(wm_awvalid[1]), .i_awready(wm_awready[1]), .o_awaddr(wm_awaddr[AW +: AW]),
        .o_awlen(wm_awlen[8 +: 8]), .o_awsize(wm_awsize[3 +: 3]), .o_awburst(wm_awburst[2 +: 2]),
        .o_wvalid(wm_wvalid[1]), .i_wready(wm_wready[1]), .o_wdata(wm_wdata[256 +: 256]),
        .o_wstrb(wm_wstrb[32 +: 32]), .o_wlast(wm_wlast[1]), .i_bvalid(wm_bvalid[1]),
        .o_bready(wm_bready[1]), .i_bresp(i_bresp), .o_idle(s_idle), .o_drained(s_drained), .o_error(s_err));

    // both writers' bursts in flight at once (colpar_nap_mux held one burst until its B: one write open per node)
    vu_wr_merge #(.N_M(2), .AXI_ADDR_WIDTH(AW), .Q_LOG2(WR_MAX_OUTSTANDING > 8 ? $clog2(2 * WR_MAX_OUTSTANDING) : 4)) u_wmux (
        .i_clk(i_clk), .i_rstn(i_rstn),
        .s_awvalid(wm_awvalid), .s_awready(wm_awready), .s_awaddr(wm_awaddr), .s_awlen(wm_awlen),
        .s_awsize(wm_awsize), .s_awburst(wm_awburst), .s_wvalid(wm_wvalid), .s_wready(wm_wready),
        .s_wdata(wm_wdata), .s_wstrb(wm_wstrb), .s_wlast(wm_wlast), .s_bvalid(wm_bvalid),
        .s_bready(wm_bready),
        .m_awvalid(wx_awvalid), .m_awready(i_awready && !sy_busy), .m_awaddr(wx_awaddr), .m_awlen(wx_awlen),
        .m_awsize(wx_awsize), .m_awburst(wx_awburst), .m_wvalid(wx_wvalid), .m_wready(i_wready && !sy_busy),
        .m_wdata(wx_wdata), .m_wstrb(wx_wstrb), .m_wlast(wx_wlast), .m_bvalid(i_bvalid && !sy_busy),
        .m_bready(wx_bready));

    wire                      wx_awvalid, wx_wvalid, wx_wlast, wx_bready;
    wire [AW-1:0]             wx_awaddr;
    wire [7:0]                wx_awlen;
    wire [2:0]                wx_awsize;
    wire [1:0]                wx_awburst;
    wire [255:0]              wx_wdata;
    wire [31:0]               wx_wstrb;
    assign o_awvalid = sy_busy ? sy_awvalid : wx_awvalid;
    assign o_awaddr  = sy_busy ? sy_awaddr  : wx_awaddr;
    assign o_awlen   = sy_busy ? sy_awlen   : wx_awlen;
    assign o_awsize  = sy_busy ? sy_awsize  : wx_awsize;
    assign o_awburst = sy_busy ? sy_awburst : wx_awburst;
    assign o_wvalid  = sy_busy ? sy_wvalid  : wx_wvalid;
    assign o_wdata   = sy_busy ? sy_wdata   : wx_wdata;
    assign o_wstrb   = sy_busy ? sy_wstrb   : wx_wstrb;
    assign o_wlast   = sy_busy ? sy_wlast   : wx_wlast;
    assign o_bready  = sy_busy ? sy_bready  : wx_bready;

    // ---- node-to-node synchronisation (WAIT / POST records, node_sync.sv) ----
    reg                       sy_arm;
    reg                       sy_post_q;
    reg  [AW-1:0]             sy_addr_q;
    reg  [31:0]               sy_value_q;
    wire                      sy_busy, sy_done, sy_error;
    wire                      sy_arvalid, sy_rready, sy_awvalid, sy_wvalid, sy_wlast, sy_bready;
    wire [AW-1:0]             sy_araddr, sy_awaddr;
    wire [7:0]                sy_arlen, sy_awlen;
    wire [2:0]                sy_arsize, sy_awsize;
    wire [1:0]                sy_arburst, sy_awburst;
    wire [255:0]              sy_wdata;
    wire [31:0]               sy_wstrb;

    node_sync #(.AXI_ADDR_WIDTH(AW)) u_sync (
        .i_clk(i_clk), .i_rstn(i_rstn), .i_arm(sy_arm), .i_post(sy_post_q), .i_addr(sy_addr_q),
        .i_value(sy_value_q), .o_busy(sy_busy), .o_done(sy_done), .o_error(sy_error),
        .o_arvalid(sy_arvalid), .i_arready(i_arready && sy_busy), .o_araddr(sy_araddr), .o_arlen(sy_arlen),
        .o_arsize(sy_arsize), .o_arburst(sy_arburst), .o_rready(sy_rready), .i_rvalid(i_rvalid && sy_busy),
        .i_rdata(i_rdata), .i_rresp(i_rresp),
        .o_awvalid(sy_awvalid), .i_awready(i_awready && sy_busy), .o_awaddr(sy_awaddr), .o_awlen(sy_awlen),
        .o_awsize(sy_awsize), .o_awburst(sy_awburst), .o_wvalid(sy_wvalid), .i_wready(i_wready && sy_busy),
        .o_wdata(sy_wdata), .o_wstrb(sy_wstrb), .o_wlast(sy_wlast), .i_bvalid(i_bvalid && sy_busy),
        .o_bready(sy_bready), .i_bresp(i_bresp));

    assign o_error = pf_error | ld_error | e_err | s_err | len_err | sy_error | (|lane_ovf);
    assign o_error_bits = {|lane_ovf, sy_error, len_err, s_err, e_err, ld_error, pf_error};

    // ------------------------------------------------------------------ control
    always @(posedge i_clk) begin
        ld_arm    <= 1'b0;
        ld_pend   <= ld_pend & ~ldv_done;            // a loader's done retires it (states below may override)
        sy_arm    <= 1'b0;
        desc_we   <= 1'b0;
        e_push    <= 1'b0;
        s_push    <= 1'b0;
        e_restart <= 1'b0;
        s_restart <= 1'b0;
        ev1       <= 1'b0;
        sv1       <= 1'b0;

        // drain stage 1 (the buffers' registered read of the address issued by stage 0)
        if (ev1) begin : est1
            reg [255:0] v;
            v = eq[el1] | (efirst1 ? ecarry : 256'd0);
            if (epart1) begin
                ecarry <= v;
            end else begin
                e_beat <= v;
                e_push <= 1'b1;
                ecarry <= '0;
            end
        end
        if (sv1) begin
            spack <= {sq[sl1], spack[255:64]};
            if (spos == 9'd192) begin
                s_beat <= {sq[sl1], spack[255:64]};
                s_push <= 1'b1;
                spos   <= 9'd0;
            end else begin
                spos <= spos + 9'd64;
            end
        end

        if (!i_rstn) begin
            st       <= C_FETCH;
            wn       <= 3'd0;
            o_halt   <= 1'b0;
            primed   <= 1'b0;
            wr_flush <= 1'b0;
            len_err  <= 1'b0;
            spos     <= 9'd0;
            spack    <= '0;
            ecarry   <= '0;
            epos     <= 8'd0;
            ed_busy  <= 1'b0;
            sd_busy  <= 1'b0;
            en_q     <= '0;
            st_en    <= '0;
            ld_pend  <= '0;
        end else if (i_prog_start) begin   // host restart: leave HALT and fetch the program at i_prog_base
            st       <= C_FETCH;
            wn       <= 3'd0;
            o_halt   <= 1'b0;
            primed   <= 1'b0;
            wr_flush <= 1'b0;
            len_err  <= 1'b0;
            spos     <= 9'd0;
            spack    <= '0;
            ecarry   <= '0;
            epos     <= 8'd0;
            ed_busy  <= 1'b0;
            sd_busy  <= 1'b0;
            en_q     <= '0;
            st_en    <= '0;
            ld_pend  <= '0;
        end else begin
            case (st)
            C_FETCH: if (cmd_valid) begin
                w[wn] <= cmd;
                if (wn == 3'd0 && cmd[127:124] == 4'd0) begin
                    o_halt <= 1'b1;
                    st     <= C_HALT;
                end else if (wn == 3'd5) begin
                    wn <= 3'd0;
                    st <= C_DECODE;
                end else begin
                    wn <= wn + 3'd1;
                end
            end
            C_DECODE: if (w[0][127:124] == 4'hB || w[0][127:124] == 4'hC) begin
                sy_addr_q  <= w[1][AW-1:0];
                sy_value_q <= w[0][39:8];
                sy_post_q  <= (w[0][127:124] == 4'hC);   // 0xC = POST, 0xB = WAIT
                st         <= (w[0][127:124] == 4'hC) ? C_POSTW : C_SYNC;
                if (w[0][127:124] != 4'hC) sy_arm <= 1'b1;
            end else begin
                op        <= w[0][3:0];
                b_fp32    <= w[0][4];
                bias_en   <= w[0][5];
                out_fp32  <= w[0][6];
                x_rot     <= w[0][7];
                q_tail    <= w[0][118] && (QTAIL != 0);
                pdq_en    <= w[0][119] && (PDQ != 0);
                k_const   <= w[0][39:8];
                R_rows    <= w[0][59:40];
                L_len     <= w[0][75:60];
                out_base  <= w[0][117:76];
                e_blk     <= w[5][9:0];
                e_gap     <= w[5][33:10];
                sum_base  <= w[1][41:0];
                desc[0]   <= w[1][89:42];
                desc[1]   <= w[2][47:0];
                desc[2]   <= w[2][95:48];
                desc[3]   <= w[3][47:0];
                desc[4]   <= w[3][95:48];
                desc[5]   <= w[4][47:0];
                desc[6]   <= w[4][95:48];
                case (w[0][3:0])
                OP_RMS_STAT, OP_SMAX_SUM:             elem_w <= 6'd0;
                OP_LN_STAT:                           elem_w <= 6'd0;
                OP_QUANT, OP_SMAX_Q8:                 elem_w <= 6'd8;
                OP_ROPE_A, OP_EULER:                  elem_w <= 6'd32;
                OP_DEQUANT:                           elem_w <= w[0][6] ? 6'd32 : 6'd16;
                default:                              elem_w <= 6'd16;
                endcase
                case (w[0][3:0])
                OP_RMS_STAT, OP_SMAX_SUM:             pad_w <= 6'd0;
                OP_QUANT, OP_SMAX_Q8:                 pad_w <= 6'd8;
                OP_LN_STAT, OP_ROPE_A, OP_EULER:      pad_w <= 6'd32;
                OP_DEQUANT:                           pad_w <= w[0][6] ? 6'd32 : 6'd16;
                default:                              pad_w <= 6'd16;
                endcase
                // fused QUANT (w0[118], bf16-output element ops): the lane emits int8 codes
                if (w[0][118] && (QTAIL != 0) && (w[0][3:0] != OP_RMS_STAT) && (w[0][3:0] != OP_SMAX_SUM) &&
                    (w[0][3:0] != OP_LN_STAT) && (w[0][3:0] != OP_QUANT) && (w[0][3:0] != OP_SMAX_Q8) &&
                    (w[0][3:0] != OP_ROPE_A) && (w[0][3:0] != OP_EULER) && !((w[0][3:0] == OP_DEQUANT) && w[0][6])) begin
                    elem_w <= 6'd8;
                    pad_w  <= 6'd8;
                end
                rows_left <= w[0][59:40];
                e_off     <= '0;
                r_off     <= '0;
                desc_we   <= 1'b1;
                e_restart <= 1'b1;
                s_restart <= 1'b1;
                st        <= C_DECODE2;
            end
            C_DECODE2: begin : d2
                reg [16:0] le;
                le      = (L_len[1:0] == 2'd0) ? {1'b0, L_len} : {1'b0, L_len} + 17'd3;
                lc      <= clog2_17(le);
                ow_log2 <= (pad_w == 6'd32) ? 3'd5 : (pad_w == 6'd16) ? 3'd4 : 3'd3;
                st      <= C_DECODE3;
            end
            C_DECODE3: begin
                if (R_rows != '0 && (L_len == '0 || lc > 5'(SLOT_BITS) || (x_rot && L_len[0]))) begin
                    len_err <= 1'b1;
                    o_halt  <= 1'b1;
                    st      <= C_HALT;
                end else begin
                    rpl_log2 <= (5'(SLOT_BITS) - lc > 5'd9) ? 4'd9 : 4'(5'(SLOT_BITS) - lc);
                    slot_i   <= 3'd0;
                    st       <= C_LOADC;
                end
            end
            C_LOADC: begin
                if (slot_i == 3'(NSLOT)) begin
                    st <= C_CHUNK;
                end else if (shape[slot_i] == SH_C) begin
                    ld_slot  <= slot_i;
                    ld_base  <= desc[slot_i][41:0];
                    for (int d = 0; d < ND; d++) begin
                        ld_first[d] <= '0;
                        ld_count[d] <= (CB+1)'(L_len);
                        ld_woff[d]  <= '0;
                    end
                    ld_wsel  <= (desc[slot_i][45:44] == 2'd0) ? 2'd0 : (desc[slot_i][45:44] == 2'd3) ? 2'd2 : 2'd1;
                    ld_field <= desc[slot_i][47:46];
                    ld_arm   <= 1'b1;
                    ld_go    <= '1;
                    ld_pend  <= '1;
                    st_en    <= '1;
                    for (int l = 0; l < NL; l++) begin
                        st_wlo[l] <= '0;
                        st_whi[l] <= (CB-1)'((32'(L_len) - 32'd1) >> 2);
                    end
                    st       <= C_LOADC_W;
                end else begin
                    slot_i <= slot_i + 3'd1;
                end
            end
            C_LOADC_W: if (ld_all) begin
                ld_pend <= '0;
                settle <= 2'd2;
                ret_e  <= 1'b0;
                st     <= C_SETTLE;
            end
            C_SETTLE: begin
                // the last steered word reaches its slot memory two cycles after o_done
                if (settle == 2'd0) begin
                    st_en  <= '0;
                    slot_i <= slot_i + 3'd1;
                    st     <= ret_e ? C_LOADE : C_LOADC;
                end else begin
                    settle <= settle - 2'd1;
                end
            end
            C_CHUNK: begin
                if (rows_left == '0) begin
                    st <= C_FLUSH;
                    dbg_n <= dbg_n + 8'd1;
                    o_dbg <= {op, pad_w, ow_log2, L_len[7:0], epos, off_end, obits[0], obits[NL > 1 ? 1 : 0],
                              16'(nb[0]), 16'(nb[NL > 1 ? 1 : 0]), 2'(lane_part), rows_l[0], rows_l[NL > 1 ? 1 : 0],
                              1'b0, |ecarry, dbg_n + 8'd1};
                end else begin
                    rows_c <= (rows_left > 20'(chunk_rows)) ? RB'(chunk_rows) : RB'(rows_left);
                    st     <= C_SETUP1;
                end
            end
            C_SETUP1: begin
                n_elem <= (CB+1)'(32'(rows_c) * 32'(L_len));
                epl    <= (SLOT_BITS+1)'(32'(L_len) << rpl_log2);
                for (int l = 0; l < NL; l++) begin : rl
                    reg [31:0] lo, hi;
                    lo = 32'(l) << rpl_log2;
                    hi = 32'(l + 1) << rpl_log2;
                    rows_l[l] <= (32'(rows_c) >= hi) ? 10'(32'd1 << rpl_log2) :
                                 (32'(rows_c) > lo)  ? 10'(32'(rows_c) - lo) : 10'd0;
                end
                st <= C_SETUP2;
            end
            C_SETUP2: begin
                for (int l = 0; l < NL; l++) begin : rng
                    reg [31:0] sE, eE, sR, eR;
                    sE = 32'(l) * 32'(epl);
                    eE = (32'(l + 1) * 32'(epl) < 32'(n_elem)) ? 32'(l + 1) * 32'(epl) : 32'(n_elem);
                    sR = 32'(l) << rpl_log2;
                    eR = sR + 32'(rows_l[l]);
                    wloE[l]  <= (CB-1)'((32'(e_off[1:0]) + sE) >> 2);
                    whiE[l]  <= (CB-1)'((32'(e_off[1:0]) + eE - 32'd1) >> 2);
                    pos0E[l] <= 2'(32'(e_off[1:0]) + sE);
                    wloR[l]  <= (CB-1)'((32'(r_off[1:0]) + sR) >> 2);
                    whiR[l]  <= (CB-1)'((32'(r_off[1:0]) + eR - 32'd1) >> 2);
                    pos0R[l] <= 2'(32'(r_off[1:0]) + sR);
                    en_q[l]  <= (rows_l[l] != 10'd0);
                    obits[l] <= obits_w[l];
                end
                st <= C_SETUP3;
            end
            C_SETUP3: begin : s3
                reg [7:0] acc;
                acc = epos;
                for (int l = 0; l < NL; l++) begin
                    off_q[l] <= acc;
                    acc = acc + obits[l];
                end
                off_end <= acc;
                slot_i  <= 3'd0;
                st      <= C_LOADE;
            end
            C_LOADE: begin
                if (slot_i == 3'(NSLOT)) begin
                    primed <= 1'b0;
                    st     <= C_RUN;
                end else if (shape[slot_i] == SH_E || shape[slot_i] == SH_R) begin
                    ld_slot  <= slot_i;
                    ld_base  <= desc[slot_i][41:0];
                    for (int d = 0; d < ND; d++) begin : split
                        reg [31:0] per, tot, off, lo, hi;
                        per = (shape[slot_i] == SH_E) ? 32'(epl) * 32'(LPD) : (32'd1 << rpl_log2) * 32'(LPD);
                        tot = (shape[slot_i] == SH_E) ? 32'(n_elem) : 32'(rows_c);
                        off = (shape[slot_i] == SH_E) ? e_off : r_off;
                        lo  = 32'(d) * per;
                        hi  = (32'(d + 1) * per < tot) ? 32'(d + 1) * per : tot;
                        ld_first[d] <= off + lo;
                        ld_count[d] <= (CB+1)'(hi - lo);
                        ld_woff[d]  <= (CB-1)'(((off + lo) >> 2) - (off >> 2));
                        ld_go[d]    <= (lo < tot);
                        ld_pend[d]  <= (lo < tot);
                    end
                    ld_wsel  <= (desc[slot_i][45:44] == 2'd0) ? 2'd0 : (desc[slot_i][45:44] == 2'd3) ? 2'd2 : 2'd1;
                    ld_field <= desc[slot_i][47:46];
                    ld_arm   <= 1'b1;
                    st_en    <= en_q;
                    for (int l = 0; l < NL; l++) begin
                        st_wlo[l] <= (shape[slot_i] == SH_E) ? wloE[l] : wloR[l];
                        st_whi[l] <= (shape[slot_i] == SH_E) ? whiE[l] : whiR[l];
                    end
                    st       <= C_LOADE_W;
                end else begin
                    slot_i <= slot_i + 3'd1;
                end
            end
            C_LOADE_W: if (ld_all) begin
                ld_pend <= '0;
                settle <= 2'd2;
                ret_e  <= 1'b1;
                st     <= C_SETTLE;
            end
            C_RUN: begin
                primed <= 1'b1;                      // read addresses settled one cycle after entry
                if (&lane_done) begin
                    ed_busy <= 1'b1;
                    ed_i    <= '0;
                    ed_k    <= '0;
                    sd_busy <= 1'b1;
                    sd_i    <= '0;
                    sd_k    <= 10'd0;
                    st      <= C_DRAIN;
                end
            end
            C_DRAIN: begin
                // element beats, stage 0: issue the read of beat ed_k of lane ed_lane
                if (ed_busy && !e_afull) begin
                    if (ed_k == nb[ed_lane]) begin
                        if (32'(ed_i) == NL - 1) ed_busy <= 1'b0;
                        else begin ed_i <= ed_i + 1'b1; ed_k <= '0; end
                    end else begin
                        ev1     <= 1'b1;
                        el1     <= ed_lane;
                        efirst1 <= (ed_k == '0);
                        epart1  <= lane_part[ed_lane] && (EKB'(ed_k + 1'b1) == nb[ed_lane]);
                        ed_k    <= ed_k + 1'b1;
                    end
                end
                // summary records, stage 0
                if (sd_busy && !s_afull) begin
                    if (sd_k == rows_l[sd_lane]) begin
                        if (32'(sd_i) == NL - 1) sd_busy <= 1'b0;
                        else begin sd_i <= sd_i + 1'b1; sd_k <= 10'd0; end
                    end else begin
                        sv1  <= 1'b1;
                        sl1  <= sd_lane;
                        sd_k <= sd_k + 10'd1;
                    end
                end
                if (!ed_busy && !sd_busy && !ev1 && !sv1 && !e_push && !s_push) begin
                    e_off     <= e_off + 32'(n_elem);
                    r_off     <= r_off + 32'(rows_c);
                    rows_left <= rows_left - 20'(rows_c);
                    epos      <= off_end;
                    st        <= C_CHUNK;
                end
            end
            C_POSTW: if (e_idle && s_idle) begin   // a POST waits for both writers to drain
                sy_arm <= 1'b1;
                st     <= C_SYNC;
            end
            C_SYNC: if (sy_done) st <= C_FETCH;
            C_FLUSH: begin
                // the carried partial element beat (already zero padded); pad the summary beat with
                // zero records, one per cycle (at most 3)
                if (epos != 8'd0 && !e_push && !e_afull) begin
                    e_beat <= ecarry;
                    e_push <= 1'b1;
                    ecarry <= '0;
                    epos   <= 8'd0;
                end
                if (spos != 9'd0 && !s_push && !s_afull) begin
                    spack <= {64'd0, spack[255:64]};
                    if (spos == 9'd192) begin
                        s_beat <= {64'd0, spack[255:64]};
                        s_push <= 1'b1;
                        spos   <= 9'd0;
                    end else begin
                        spos <= spos + 9'd64;
                    end
                end
                if (epos == 8'd0 && spos == 9'd0 && !e_push && !s_push) begin
                    wr_flush <= 1'b1;
                    st       <= C_FLUSH_W;
                end
            end
`ifdef NODE_DRAIN_ONLY                             // experiment: next op once every beat is sent, B still open
            C_FLUSH_W: if (e_drained && s_drained && !e_push && !s_push) begin
`else
            C_FLUSH_W: if (e_idle && s_idle && !e_push && !s_push) begin
`endif
                wr_flush <= 1'b0;
                st       <= C_FETCH;
            end
            default: ;
            endcase
        end
    end
endmodule
