// Vector node: one vector-unit lane (vu_lane.sv) with its NoC data path, on the fabric clock.
// Design: docs/PI0_FULL_MODEL_ON_CHIP_DESIGN_20260916.md section 3.4.
//
// Program (GDDR6, fetched by colpar_prog_fetch.sv, RECORD_WORDS = 6): one 6-word op record per
// op (128-bit words), then an END word ([127:124] = 0):
//   w0  [127:124] 4'hA  [3:0] op  [4] b_fp32  [5] bias_en  [6] out_fp32
//   w0[7] x_rot: the x slot (shape E) is read rotate-half within the row -- element j reads x[j + L/2]
//   for j < L/2 and x[j - L/2] otherwise, the partner RoPE pass B needs (L must be even).
//       [39:8] k (fp32)  [59:40] R rows  [75:60] L elements per row  [117:76] element out base
//   w1  [41:0] summary out base   [89:42] x operand
//   w2  [47:0] b operand          [95:48] c operand
//   w3  [47:0] d operand          [95:48] e operand
//   w4  [47:0] rs operand         [95:48] mask operand
//   w5  reserved
// Operand (48 bits): [41:0] base (or, for shape K, the value in [31:0]), [43:42] shape,
// [45:44] format, [47:46] field.
//   shape  0 E  R x L values, contiguous    1 R  one value per row
//          2 C  L values, the same every row 3 K  constant
//   format 0 bf16 (16 bits)  1 fp32 (32)  2 int32 (32; sign-extended for x, the DEQUANT sum)
//          3 summary record (64 bits {amax, max, rs0}); field 0 rs0, 1 max, 2 amax
//
// Execution per op: C operands are loaded once into their slot memories; then rows are taken in
// chunks of at most 2^SLOT_BITS elements and 512 rows (rows per chunk = min(512, 2^SLOT_BITS >>
// ceil(log2 L))).  Per chunk the E and R operands are loaded, then the lane runs over the chunk,
// reading every slot in lockstep (look-ahead read addresses, one element per cycle when the lane
// is ready).  L <= 2^SLOT_BITS; longer rows need splitting by the compiler (not in this version).
//
// Outputs: element beats (bf16 16 per beat, fp32 8, int8 32; LN_STAT's r as fp32) packed from
// the element out base, and summary records (64 bits, 4 per beat) from the summary out base; both
// contiguous over the op, the last beat zero-padded.  They leave through two vu_beat_writer
// instances sharing the write channel (colpar_nap_mux).  The lane is back-pressured when either
// writer is almost full.
module vu_node #(
    parameter F_GELU  = "vu_tbl_gelu.mem",
    parameter F_SIGM  = "vu_tbl_sigm.mem",
    parameter F_EXP   = "vu_tbl_exp.mem",
    parameter F_RSQRT = "vu_tbl_rsqrt.mem",
    parameter F_QUANT = "vu_tbl_quant.mem",
    parameter integer AXI_ADDR_WIDTH     = 42,
    parameter integer SLOT_BITS          = 11,
    parameter integer RD_BURST_BEATS     = 16,
    parameter integer RD_MAX_OUTSTANDING = 8,
    parameter integer WR_BURST_BEATS     = 16,
    parameter integer WR_MAX_OUTSTANDING = 8,
    parameter integer WR_FIFO_LOG2       = 6
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
    output wire                       o_error
);
    localparam integer AW    = AXI_ADDR_WIDTH;
    localparam integer DEPTH = 1 << SLOT_BITS;
    localparam [3:0] OP_RMS_STAT = 4'd1, OP_LN_STAT = 4'd3, OP_ROPE_A = 4'd5, OP_SMAX_SUM = 4'd10,
                     OP_SMAX_Q8 = 4'd12, OP_QUANT = 4'd13, OP_DEQUANT = 4'd14, OP_EULER = 4'd15;
    localparam [1:0] SH_E = 2'd0, SH_R = 2'd1, SH_C = 2'd2, SH_K = 2'd3;
    localparam integer NSLOT = 7;   // x b c d e rs mask

    // ------------------------------------------------------------------ control state
    localparam [3:0] C_FETCH = 4'd0, C_DECODE = 4'd1, C_LOADC = 4'd2, C_LOADC_W = 4'd3, C_CHUNK = 4'd4,
                     C_LOADE = 4'd5, C_LOADE_W = 4'd6, C_RUN = 4'd7, C_DRAIN = 4'd8, C_FLUSH = 4'd9,
                     C_FLUSH_W = 4'd10, C_HALT = 4'd11, C_SYNC = 4'd12, C_POSTW = 4'd13;
    reg [3:0] st;

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
    reg [AW-1:0] out_base, sum_base;
    reg [47:0]  desc [0:NSLOT-1];
    wire [1:0]  shape [0:NSLOT-1];
    genvar gs;
    generate
        for (gs = 0; gs < NSLOT; gs = gs + 1) begin : g_shape
            assign shape[gs] = desc[gs][43:42];
        end
    endgenerate

    // output element width: 0 none, 8, 16, 32; pad_w: width of the element region (LN_STAT: its fp32 r)
    reg [5:0]   elem_w, pad_w;

    // ------------------------------------------------------------------ chunking
    reg [19:0]  rows_left;
    reg [10:0]  rows_per_chunk, rows_c;
    reg [SLOT_BITS:0] n_elem;
    reg [31:0]  e_off, r_off;
    reg [2:0]   slot_i;

    function automatic [4:0] clog2_16(input [15:0] v);
        integer i;
        clog2_16 = 5'd0;
        for (i = 15; i >= 0; i = i - 1)
            if (clog2_16 == 5'd0 && v > (16'd1 << i)) clog2_16 = 5'(i + 1);
    endfunction

    // ------------------------------------------------------------------ slot loader
    reg          ld_arm;
    reg [AW-1:0] ld_base;
    reg [31:0]   ld_first;
    reg [SLOT_BITS:0] ld_count;
    reg [1:0]    ld_wsel;
    reg [2:0]    ld_slot;
    wire         ld_we, ld_busy, ld_done, ld_error;
    wire [SLOT_BITS-1:0] ld_waddr;
    wire [63:0]  ld_wdata;

    vu_slot_loader #(
        .AXI_ADDR_WIDTH(AW), .ADDR_BITS(SLOT_BITS), .BEATS_PER_BURST(RD_BURST_BEATS),
        .MAX_OUTSTANDING(RD_MAX_OUTSTANDING)
    ) u_ld (
        .i_clk(i_clk), .i_rstn(i_rstn), .i_arm(ld_arm), .i_base(ld_base), .i_first(ld_first),
        .i_count(ld_count), .i_wsel(ld_wsel),
        .o_arvalid(l_arvalid), .i_arready(i_arready && !fetching), .o_araddr(l_araddr), .o_arlen(l_arlen),
        .o_arsize(l_arsize), .o_arburst(l_arburst), .o_rready(l_rready), .i_rvalid(i_rvalid && !fetching),
        .i_rdata(i_rdata), .i_rresp(i_rresp), .i_rlast(i_rlast),
        .o_we(ld_we), .o_waddr(ld_waddr), .o_wdata(ld_wdata),
        .o_busy(ld_busy), .o_done(ld_done), .o_error(ld_error));

    // value stored in a slot: summary records keep one field
    wire [1:0]  ld_fmt   = desc[ld_slot][45:44];
    wire [1:0]  ld_field = desc[ld_slot][47:46];
    wire [31:0] ld_val   = (ld_fmt != 2'd3) ? ld_wdata[31:0] :
`ifdef VU_NODE_NEG_FIELD_SWAP    // negative control: max and amax fields swapped
                           (ld_field == 2'd2) ? {16'd0, ld_wdata[47:32]} :
                           (ld_field == 2'd1) ? {16'd0, ld_wdata[63:48]} : ld_wdata[31:0];
`else
                           (ld_field == 2'd1) ? {16'd0, ld_wdata[47:32]} :
                           (ld_field == 2'd2) ? {16'd0, ld_wdata[63:48]} : ld_wdata[31:0];
`endif

    // ------------------------------------------------------------------ slot memories
    reg [SLOT_BITS-1:0] e_idx, r_idx, j_idx;
    wire                lane_fire;
    wire [SLOT_BITS-1:0] e_n, r_n, j_n, x_n;
    wire [31:0]         q [0:NSLOT-1];

    generate
        for (gs = 0; gs < NSLOT; gs = gs + 1) begin : g_slot
            localparam integer WID = (gs == 5) ? 16 : (gs == 6) ? 1 : 32;
            reg [WID-1:0] mem [0:DEPTH-1] /* synthesis syn_ramstyle = "block_ram" */;
            reg [WID-1:0] qr;
            wire [SLOT_BITS-1:0] ra = (shape[gs] == SH_R) ? r_n : (shape[gs] == SH_C) ? j_n :
                                      (gs == 0) ? x_n : e_n;
            always @(posedge i_clk) begin
                if (ld_we && ld_slot == 3'(gs)) mem[ld_waddr] <= ld_val[WID-1:0];
                qr <= mem[ra];
            end
            assign q[gs] = 32'(qr);
        end
    endgenerate

    function automatic [31:0] opnd(input integer s);
        opnd = (shape[s] == SH_K) ? desc[s][31:0] : q[s];
    endfunction

    // ------------------------------------------------------------------ lane
    reg          desc_we;
    reg          primed, run_done;
    reg  [10:0]  sum_cnt;          // summaries received for the current chunk
    wire         lane_idle, lane_ready, lane_ovalid;
    wire [1:0]   lane_okind;
    wire [63:0]  lane_odata;
    wire         in_valid = (st == C_RUN) && primed && !run_done;
    wire         out_ready;
    wire         last_j   = (32'(j_idx) + 32'd1 == 32'(L_len));
    wire [31:0]  x32      = opnd(0);
    wire [47:0]  x48      = (desc[0][45:44] == 2'd2) ? {{16{x32[31]}}, x32} : {16'd0, x32};
    wire [31:0]  rs32     = opnd(5);
    wire [31:0]  b32      = opnd(1);
    wire [31:0]  c32      = opnd(2);
    wire [31:0]  d32      = opnd(3);
    wire [31:0]  e32      = opnd(4);

    assign lane_fire = in_valid && lane_ready;
    assign e_n = lane_fire ? e_idx + 1'b1 : e_idx;
    assign j_n = lane_fire ? (last_j ? '0 : j_idx + 1'b1) : j_idx;
    assign r_n = lane_fire && last_j ? r_idx + 1'b1 : r_idx;
    // rotate-half read of the x slot (w0[7]); L must be even
    wire [SLOT_BITS:0] l2 = {1'b0, L_len[SLOT_BITS:1]};
`ifdef VU_NODE_NEG_NO_ROT     // negative control: ignore the rotate-half flag
    assign x_n = e_n;
`else
    assign x_n = !x_rot ? e_n : (32'(j_n) < 32'(l2)) ? SLOT_BITS'({1'b0, e_n} + l2)
                                                    : SLOT_BITS'({1'b0, e_n} - l2);
`endif

    vu_lane #(
        .QTAIL(0),
        .F_GELU(F_GELU), .F_SIGM(F_SIGM), .F_EXP(F_EXP), .F_RSQRT(F_RSQRT), .F_QUANT(F_QUANT)
    ) u_lane (
        .i_clk(i_clk), .i_rstn(i_rstn),
        .i_desc_we(desc_we), .i_op(op), .i_b_fp32(b_fp32), .i_bias_en(bias_en), .i_out_fp32(out_fp32), .i_qtail(1'b0),
        .i_k(k_const), .o_idle(lane_idle),
        .i_valid(in_valid), .o_ready(lane_ready), .i_last(last_j),
        .i_mask((shape[6] == SH_K) ? desc[6][0] : q[6][0]),
        .i_x(x48), .i_rs(rs32[15:0]), .i_b(b32), .i_c(c32), .i_d(d32), .i_e(e32),
        .o_valid(lane_ovalid), .i_ready(out_ready), .o_kind(lane_okind), .o_data(lane_odata));

    // ------------------------------------------------------------------ output packers + writers
    reg  [255:0] epack, spack;
    reg  [8:0]   epos, spos;
    reg          e_push, s_push;
    reg  [255:0] e_beat, s_beat;
    reg          e_restart, s_restart, wr_flush;
    wire         e_afull, s_afull, e_idle, s_idle, e_err, s_err;
    assign out_ready = !e_afull && !s_afull && !e_push && !s_push;

    wire         out_fire = lane_ovalid && out_ready;
    wire         is_elem  = out_fire && (lane_okind == 2'd0 || lane_okind == 2'd2);
    wire         is_sum   = out_fire && (lane_okind == 2'd1);
    wire [5:0]   ew       = (lane_okind == 2'd2) ? 6'd32 : elem_w;
    // shift-register packers: a new element enters at the top, so element i ends at bit i*w
    wire [255:0] epack_n  = (ew == 6'd8)  ? {lane_odata[7:0],  epack[255:8]}  :
                            (ew == 6'd16) ? {lane_odata[15:0], epack[255:16]} :
                                            {lane_odata[31:0], epack[255:32]};
    wire [255:0] epad_n   = (pad_w == 6'd8)  ? {8'd0,  epack[255:8]}  :
                            (pad_w == 6'd16) ? {16'd0, epack[255:16]} :
                                               {32'd0, epack[255:32]};
    wire [255:0] spack_n  = {lane_odata, spack[255:64]};

    wire [1:0]         wm_awvalid, wm_awready, wm_wvalid, wm_wready, wm_wlast, wm_bvalid, wm_bready;
    wire [2*AW-1:0]    wm_awaddr;
    wire [15:0]        wm_awlen;
    wire [5:0]         wm_awsize;
    wire [3:0]         wm_awburst;
    wire [511:0]       wm_wdata;
    wire [63:0]        wm_wstrb;

    vu_beat_writer #(.AXI_ADDR_WIDTH(AW), .FIFO_LOG2(WR_FIFO_LOG2), .BURST_BEATS(WR_BURST_BEATS),
                     .MAX_OUTSTANDING(WR_MAX_OUTSTANDING)) u_ew (
        .i_clk(i_clk), .i_rstn(i_rstn), .i_base(out_base), .i_blk_beats(10'd0), .i_gap_bytes(24'd0),
        .i_restart(e_restart), .i_flush(wr_flush),
        .i_beat(e_beat), .i_beat_valid(e_push), .o_afull(e_afull),
        .o_awvalid(wm_awvalid[0]), .i_awready(wm_awready[0]), .o_awaddr(wm_awaddr[0 +: AW]),
        .o_awlen(wm_awlen[0 +: 8]), .o_awsize(wm_awsize[0 +: 3]), .o_awburst(wm_awburst[0 +: 2]),
        .o_wvalid(wm_wvalid[0]), .i_wready(wm_wready[0]), .o_wdata(wm_wdata[0 +: 256]),
        .o_wstrb(wm_wstrb[0 +: 32]), .o_wlast(wm_wlast[0]), .i_bvalid(wm_bvalid[0]),
        .o_bready(wm_bready[0]), .i_bresp(i_bresp), .o_idle(e_idle), .o_error(e_err));

    vu_beat_writer #(.AXI_ADDR_WIDTH(AW), .FIFO_LOG2(WR_FIFO_LOG2), .BURST_BEATS(WR_BURST_BEATS),
                     .MAX_OUTSTANDING(WR_MAX_OUTSTANDING)) u_sw (
        .i_clk(i_clk), .i_rstn(i_rstn), .i_base(sum_base), .i_blk_beats(10'd0), .i_gap_bytes(24'd0),
        .i_restart(s_restart), .i_flush(wr_flush),
        .i_beat(s_beat), .i_beat_valid(s_push), .o_afull(s_afull),
        .o_awvalid(wm_awvalid[1]), .i_awready(wm_awready[1]), .o_awaddr(wm_awaddr[AW +: AW]),
        .o_awlen(wm_awlen[8 +: 8]), .o_awsize(wm_awsize[3 +: 3]), .o_awburst(wm_awburst[2 +: 2]),
        .o_wvalid(wm_wvalid[1]), .i_wready(wm_wready[1]), .o_wdata(wm_wdata[256 +: 256]),
        .o_wstrb(wm_wstrb[32 +: 32]), .o_wlast(wm_wlast[1]), .i_bvalid(wm_bvalid[1]),
        .o_bready(wm_bready[1]), .i_bresp(i_bresp), .o_idle(s_idle), .o_error(s_err));

    colpar_nap_mux #(.N_M(2), .AXI_ADDR_WIDTH(AW)) u_wmux (
        .i_clk(i_clk), .i_rstn(i_rstn),
        .s_arvalid(2'b00), .s_arready(), .s_araddr('0), .s_arlen('0), .s_arsize('0), .s_arburst('0),
        .s_rvalid(), .s_rready(2'b00), .s_rdata(), .s_rresp(), .s_rlast(),
        .s_awvalid(wm_awvalid), .s_awready(wm_awready), .s_awaddr(wm_awaddr), .s_awlen(wm_awlen),
        .s_awsize(wm_awsize), .s_awburst(wm_awburst), .s_wvalid(wm_wvalid), .s_wready(wm_wready),
        .s_wdata(wm_wdata), .s_wstrb(wm_wstrb), .s_wlast(wm_wlast), .s_bvalid(wm_bvalid),
        .s_bready(wm_bready), .s_bresp(),
        .m_arvalid(), .m_arready(1'b0), .m_araddr(), .m_arlen(), .m_arsize(), .m_arburst(),
        .m_rvalid(1'b0), .m_rready(), .m_rdata(256'd0), .m_rresp(2'd0), .m_rlast(1'b0),
        .m_awvalid(wx_awvalid), .m_awready(i_awready && !sy_busy), .m_awaddr(wx_awaddr), .m_awlen(wx_awlen),
        .m_awsize(wx_awsize), .m_awburst(wx_awburst), .m_wvalid(wx_wvalid), .m_wready(i_wready && !sy_busy),
        .m_wdata(wx_wdata), .m_wstrb(wx_wstrb), .m_wlast(wx_wlast), .m_bvalid(i_bvalid && !sy_busy),
        .m_bready(wx_bready), .m_bresp(i_bresp));

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

    assign o_error = pf_error | ld_error | e_err | s_err | sy_error;

    // ------------------------------------------------------------------ control
    always @(posedge i_clk) begin
        ld_arm    <= 1'b0;
        sy_arm    <= 1'b0;
        desc_we   <= 1'b0;
        e_push    <= 1'b0;
        s_push    <= 1'b0;
        e_restart <= 1'b0;
        s_restart <= 1'b0;

        // packers (any state: the lane may still emit while draining)
        if (is_elem) begin
            epack <= epack_n;
            if (epos + 9'(ew) == 9'd256) begin
                e_beat <= epack_n;
                e_push <= 1'b1;
                epos   <= 9'd0;
            end else begin
                epos <= epos + 9'(ew);
            end
        end
        if (is_sum) begin
            sum_cnt <= sum_cnt + 11'd1;
            spack   <= spack_n;
            if (spos == 9'd192) begin
                s_beat <= spack_n;
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
            run_done <= 1'b0;
            wr_flush <= 1'b0;
            epos     <= 9'd0;
            spos     <= 9'd0;
            epack    <= '0;
            spack    <= '0;
        end else if (i_prog_start) begin   // host restart: leave HALT and fetch the program at i_prog_base
            st       <= C_FETCH;
            wn       <= 3'd0;
            o_halt   <= 1'b0;
            primed   <= 1'b0;
            run_done <= 1'b0;
            wr_flush <= 1'b0;
            epos     <= 9'd0;
            spos     <= 9'd0;
            epack    <= '0;
            spack    <= '0;
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
                k_const   <= w[0][39:8];
                R_rows    <= w[0][59:40];
                L_len     <= w[0][75:60];
                out_base  <= w[0][117:76];
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
                begin : chunk_rows
                    reg [15:0] rp;
                    rp = 16'(DEPTH) >> clog2_16(w[0][75:60]);
                    rows_per_chunk <= (rp > 16'd512) ? 11'd512 : 11'(rp);
                end
                rows_left <= w[0][59:40];
                e_off     <= '0;
                r_off     <= '0;
                desc_we   <= 1'b1;
                e_restart <= 1'b1;
                s_restart <= 1'b1;
                slot_i    <= 3'd0;
                st        <= C_LOADC;
            end
            C_LOADC: begin
                if (slot_i == 3'(NSLOT)) begin
                    st <= C_CHUNK;
                end else if (shape[slot_i] == SH_C) begin
                    ld_slot  <= slot_i;
                    ld_base  <= desc[slot_i][41:0];
                    ld_first <= '0;
                    ld_count <= (SLOT_BITS+1)'(L_len);
                    ld_wsel  <= (desc[slot_i][45:44] == 2'd0) ? 2'd0 : (desc[slot_i][45:44] == 2'd3) ? 2'd2 : 2'd1;
                    ld_arm   <= 1'b1;
                    st       <= C_LOADC_W;
                end else begin
                    slot_i <= slot_i + 3'd1;
                end
            end
            C_LOADC_W: if (ld_done) begin
                slot_i <= slot_i + 3'd1;
                st     <= C_LOADC;
            end
            C_CHUNK: begin
                if (rows_left == '0) begin
                    st <= C_FLUSH;
                end else begin : next_chunk
                    reg [10:0] rc;
                    rc = (rows_left > 20'(rows_per_chunk)) ? rows_per_chunk : 11'(rows_left);
                    rows_c <= rc;
                    n_elem <= (SLOT_BITS+1)'(32'(rc) * 32'(L_len));
                    slot_i <= 3'd0;
                    st     <= C_LOADE;
                end
            end
            C_LOADE: begin
                if (slot_i == 3'(NSLOT)) begin
                    e_idx    <= '0;
                    r_idx    <= '0;
                    j_idx    <= '0;
                    primed   <= 1'b0;
                    run_done <= 1'b0;
                    sum_cnt  <= 11'd0;
                    st       <= C_RUN;
                end else if (shape[slot_i] == SH_E || shape[slot_i] == SH_R) begin
                    ld_slot  <= slot_i;
                    ld_base  <= desc[slot_i][41:0];
                    ld_first <= (shape[slot_i] == SH_E) ? e_off : r_off;
                    ld_count <= (shape[slot_i] == SH_E) ? n_elem : (SLOT_BITS+1)'(rows_c);
                    ld_wsel  <= (desc[slot_i][45:44] == 2'd0) ? 2'd0 : (desc[slot_i][45:44] == 2'd3) ? 2'd2 : 2'd1;
                    ld_arm   <= 1'b1;
                    st       <= C_LOADE_W;
                end else begin
                    slot_i <= slot_i + 3'd1;
                end
            end
            C_LOADE_W: if (ld_done) begin
                slot_i <= slot_i + 3'd1;
                st     <= C_LOADE;
            end
            C_RUN: begin
                primed <= 1'b1;                      // read addresses settled one cycle after entry
                if (lane_fire) begin
                    e_idx <= e_n;
                    j_idx <= j_n;
                    r_idx <= r_n;
                    if (32'(e_idx) + 32'd1 == 32'(n_elem)) begin
                        run_done <= 1'b1;
                        st       <= C_DRAIN;
                    end
                end
            end
            C_DRAIN: begin
                // row reductions emit their summary after the microcode, possibly after o_idle: wait
                // for every row's summary, not only for an idle lane
`ifdef VU_NODE_NEG_EARLY_DRAIN   // negative control: the old drain condition (lane idle only)
                if (lane_idle && !lane_ovalid && !e_push && !s_push) begin
`else
                if (lane_idle && !lane_ovalid && !e_push && !s_push && sum_cnt == rows_c) begin
`endif
                    e_off     <= e_off + 32'(n_elem);
                    r_off     <= r_off + 32'(rows_c);
                    rows_left <= rows_left - 20'(rows_c);
                    st        <= C_CHUNK;
                end
            end
            C_POSTW: if (e_idle && s_idle) begin   // a POST waits for both writers to drain
                sy_arm <= 1'b1;
                st     <= C_SYNC;
            end
            C_SYNC: if (sy_done) st <= C_FETCH;
            C_FLUSH: begin
                // pad a partial beat with zero elements / records, one per cycle (at most 31)
                if (epos != 9'd0 && !e_push && !e_afull) begin
                    epack <= epad_n;
                    if (epos + 9'(pad_w) == 9'd256) begin
                        e_beat <= epad_n;
                        e_push <= 1'b1;
                        epos   <= 9'd0;
                    end else begin
                        epos <= epos + 9'(pad_w);
                    end
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
                if (epos == 9'd0 && spos == 9'd0 && !e_push && !s_push) begin
                    wr_flush <= 1'b1;
                    st       <= C_FLUSH_W;
                end
            end
            C_FLUSH_W: if (e_idle && s_idle && !e_push && !s_push) begin
                wr_flush <= 1'b0;
                st       <= C_FETCH;
            end
            default: ;
            endcase
        end
    end
endmodule
