// ------------------------------------------------------------------
// pi0 vector-unit lane: executes one op descriptor over a row stream.
// Bit-exact spec: paper/sw/vector_unit_ref.py; op table: OPS.md.
//
// Descriptor (quasi-static, load with i_desc_we while o_idle):
//   i_op        OP_* of vu_pkg
//   i_b_fp32    b operand is fp32 (ADD); for OP_SMAX_Q8 it means 255 levels instead of 127:
//               uint8 probability codes and s_row = (1/S)/255, for the PV chain's unsigned operand
//   i_b_fp32    ADD: operand b is fp32 (position table) instead of bf16
//   i_bias_en   DEQUANT: add operand e (fp32 bias)
//   i_out_fp32  DEQUANT: emit fp32 instead of bf16
//   i_qtail     fused QUANT (QTAIL = 1, bf16-output element ops only): the op's bf16 row is not emitted; it is
//               held in a row FIFO until the row ends, then quantised with the R / s_row of its own amax --
//               the lane emits exactly what a following OP_QUANT record over that output would: int8 element
//               beats and one summary {0, 0, s_row} per row (the summary may precede its row's elements)
//   i_k         fp32 constant: 1/N for RMS_STAT / LN_STAT, dt for EULER
//
// Element stream in (AXI-stream style valid/ready, i_last ends a row):
//   i_x   bf16 [15:0] | fp32 [31:0] (EULER v) | int48 [47:0] (DEQUANT sum)
//   i_rs  row statistic, constant over the row: amax code (RMS_STAT,
//         LN_STAT, QUANT) or signed max code over valid keys (SMAX_*)
//   i_mask SMAX_*: key valid; others: 1 (used by the output max tracker)
//   i_b   ADD: b (bf16 or fp32) | LN_APPLY: mu (fp32)
//   i_c   RMS/LN_APPLY: r | ROPE_A: cos | ROPE_B: +-sin | SMAX_OUT: 1/S |
//         DEQUANT: s_row   (fp32)
//   i_d   RMS_APPLY: 1+w | LN_APPLY: gamma | DEQUANT: s_col (fp32) |
//         GEGLU: up (bf16)
//   i_e   LN_APPLY: beta | ROPE_B: pass-A word | DEQUANT: bias |
//         EULER: x_t  (fp32)
//
// Stream out: per row, the element beats (o_kind 0, o_data[31:0] = bf16
// zero-extended | fp32 | int8 sign-extended), then for LN_STAT one
// o_kind 2 beat (o_data[31:0] = r), then one o_kind 1 summary beat
// o_data = {amax[15:0], max[15:0], rs0[31:0]}: amax/max of the bf16 output
// row (0 for non-bf16 outputs), rs0 = r (RMS_STAT) | mu (LN_STAT) |
// 1/S (SMAX_SUM) | s_row (SMAX_Q8, QUANT) | 0.
//
// Datapath (free running; flow control by credits into the output FIFO):
//   S0 -> U(3: unpack | n64) -> ADD1(6) ------------------+-> MUL1(4) ->
//                            \-> PREP(5) -> TBL(6) -> POST(1) /
//   MUL2(4) -> ADD2(6) -> ROUND(3) -> FIFO
// Unused units get identity operands (add 0, mul 1: exact in the ref).
//
// Row handling, two classes:
//   element ops (ADD, *_APPLY, ROPE_*, GELU, GEGLU, SILU, SMAX_OUT, QUANT,
//     DEQUANT, EULER): rows stream back to back with a 1-cycle accept gap;
//     the summary beat is written the cycle after the row's last element.
//     QUANT's row constants R = 127/amax and s_row come from a 256-entry
//     table (vu_tbl_quant.mem, derived from the ref's quant_setup).
//   row reductions (RMS_STAT, LN_STAT, SMAX_SUM, SMAX_Q8): fixed-point
//     accumulators, then row-end microcode tokens through the datapath
//     (2 / 5 / 1 / 1 steps); the next row is accepted after the summary.
// ------------------------------------------------------------------
`include "vu_pkg.sv"

module vu_lane
#(
    parameter F_GELU  = "vu_tbl_gelu.mem",
    parameter F_SIGM  = "vu_tbl_sigm.mem",
    parameter F_EXP   = "vu_tbl_exp.mem",
    parameter F_RSQRT = "vu_tbl_rsqrt.mem",
    parameter F_QUANT = "vu_tbl_quant.mem",
    parameter integer FIFO_AW = 6,
    // 1: the c / d / e operands ride a block-RAM ring (vu_delay_ram) for their first T_M1 - 1 cycles instead of
    //    96 bits of shift register per stage; bit-identical, ~1,250 fewer flops per lane.  0: all flops.
    parameter integer DELAY_RAM = 1,
    // fused QUANT tail (i_qtail): a row FIFO of 2^ROW_AW bf16 codes (>= the elements a lane gets per chunk), a
    // 512-row statistic FIFO, one multiplier and one rounder.  0: the input is ignored and nothing is built.
    parameter integer QTAIL  = 1,
    parameter integer ROW_AW = 12
)
(
    input  wire        i_clk,
    input  wire        i_rstn,

    input  wire        i_desc_we,
    input  wire [3:0]  i_op,
    input  wire        i_b_fp32,
    input  wire        i_bias_en,
    input  wire        i_out_fp32,
    input  wire        i_qtail,
    input  wire [31:0] i_k,
    output wire        o_idle,

    input  wire        i_valid,
    output wire        o_ready,
    input  wire        i_last,
    input  wire        i_mask,
    input  wire [47:0] i_x,
    input  wire [15:0] i_rs,
    input  wire [31:0] i_b,
    input  wire [31:0] i_c,
    input  wire [31:0] i_d,
    input  wire [31:0] i_e,

    output wire        o_valid,
    input  wire        i_ready,
    output wire [1:0]  o_kind,
    output wire [63:0] o_data
);
    import vu_pkg::*;

    // ---------------------------------------------------------------
    // stage offsets (cycles after S0)
    // ---------------------------------------------------------------
    localparam integer L_U = 3, L_ADD = 6, L_PREP = 5, L_TBL = 6, L_MUL = 4, L_RND = 3;
    localparam integer T_U   = L_U;               //  3 XU, BU
    localparam integer T_X1  = T_U + L_ADD;       //  9 ADD1 out
    localparam integer T_TA  = T_U + L_PREP;      //  8 table inputs
    localparam integer T_RAW = T_TA + 4;          // 12 table raw (softmax accumulator)
    localparam integer T_TZ  = T_TA + L_TBL;      // 14 table xf
    localparam integer T_M1  = T_TZ + 1;          // 15 MUL1 operands
    localparam integer T_M2  = T_M1 + L_MUL;      // 19 MUL2 operands
    localparam integer T_A2  = T_M2 + L_MUL;      // 23 ADD2 operands
    localparam integer T_Y3  = T_A2 + L_ADD;      // 29 ADD2 out (microcode capture)
    localparam integer T_OUT = T_Y3 + L_RND;      // 32 stream code
    localparam integer T_QR  = T_M1 - 3;          // 12 QUANT table read
    localparam integer DEPTH = 1 << FIFO_AW;

    // ---------------------------------------------------------------
    // descriptor and its static decodes
    // ---------------------------------------------------------------
    reg [3:0]  op;
    reg        b_fp32, bias_en, out_fp32;
    reg [36:0] kx;
    reg        d_out, d_bf16out, d_acc12, d_accexp, d_quant, d_hasrs, d_ptab, d_fast, d_qtail;
    reg [1:0]  d_fmt, d_fn, d_xsrc;
    reg [4:0]  d_ftop;
    reg [2:0]  d_last_step;

    // ---------------------------------------------------------------
    // S0 and the control pipeline
    // ---------------------------------------------------------------
    localparam integer CW = 7;   // {mc, step[2:0], last, first, mask}
    reg [T_OUT:0] vld;
    reg [CW-1:0]  ctl [0:T_OUT];
    reg [53:0]    s0_x54;
    reg [1:0]     s0_xsrc;
    reg [36:0]    s0_xf;
    reg [5:0]     s0_fb;
    reg [11:0]    s0_eoff;
    reg [15:0]    rs_d [0:T_OUT];
    reg [31:0]    s0_b;
    reg [31:0]    c_d [0:T_M1-1];
    reg [31:0]    d_d [0:T_M2-1];
    reg [31:0]    e_d [0:T_A2-1];

    wire          accept;
    reg           ml_en;
    reg [2:0]     ml_step;
    reg [1:0]     ml_xsrc;
    reg [36:0]    ml_xf;
    reg [53:0]    ml_x54;
    reg [5:0]     ml_fb;
    reg [11:0]    ml_eoff;
    reg           first_pend;

    function automatic ctl_mc(input [CW-1:0] c);         ctl_mc    = c[6];   endfunction
    function automatic [2:0] ctl_step(input [CW-1:0] c); ctl_step  = c[5:3]; endfunction
    function automatic ctl_last(input [CW-1:0] c);       ctl_last  = c[2];   endfunction
    function automatic ctl_first(input [CW-1:0] c);      ctl_first = c[1];   endfunction
    function automatic ctl_mask(input [CW-1:0] c);       ctl_mask  = c[0];   endfunction

    integer k;
    always @(posedge i_clk) begin
        if (!i_rstn)
            vld <= '0;
        else
            vld <= {vld[T_OUT-1:0], accept | ml_en};
        for (k = 1; k <= T_OUT; k = k + 1)
            ctl[k] <= ctl[k-1];
        for (k = 1; k <= T_OUT; k = k + 1)
            rs_d[k] <= rs_d[k-1];

        rs_d[0] <= i_rs;
        s0_b    <= i_b;
        if (accept) begin
            ctl[0]  <= {1'b0, 3'd0, i_last, first_pend, i_mask};
            s0_x54  <= {{6{i_x[47]}}, i_x};
            s0_xsrc <= d_xsrc;
            s0_xf   <= XF_ZERO;
            s0_fb   <= 6'd0;
            s0_eoff <= 12'd0;
        end else begin
            ctl[0]  <= {1'b1, ml_step, 3'b000};
            s0_x54  <= ml_x54;
            s0_xsrc <= ml_xsrc;
            s0_xf   <= ml_xf;
            s0_fb   <= ml_fb;
            s0_eoff <= ml_eoff;
        end
    end

    // operand delays: c to MUL1 (T_M1), d to MUL2 (T_M2), e to ADD2 (T_A2); stage k holds the operand of k + 1
    // cycles ago.  With DELAY_RAM the first T_RAM stages are a RAM ring and only the stages from T_RAM on are flops.
    localparam integer T_RAM = T_M1 - 1;
    generate
        if (DELAY_RAM != 0) begin : g_dly_ram
            wire [95:0] dq;
            integer j;
`ifdef VU_LANE_NEG_DELAY_RAM    // negative control: the ring one cycle short
            vu_delay_ram #(.W(96), .D(T_RAM - 2)) u_dly (.i_clk(i_clk), .i_d({i_e, i_d, i_c}), .o_q(dq));
`else
            vu_delay_ram #(.W(96), .D(T_RAM - 1)) u_dly (.i_clk(i_clk), .i_d({i_e, i_d, i_c}), .o_q(dq));
`endif
            always @(posedge i_clk) begin
                c_d[T_RAM] <= dq[31:0];
                d_d[T_RAM] <= dq[63:32];
                e_d[T_RAM] <= dq[95:64];
                for (j = T_RAM + 1; j < T_M2; j = j + 1)
                    d_d[j] <= d_d[j-1];
                for (j = T_RAM + 1; j < T_A2; j = j + 1)
                    e_d[j] <= e_d[j-1];
            end
        end else begin : g_dly_ff
            integer j;
            always @(posedge i_clk) begin
                c_d[0] <= i_c;
                d_d[0] <= i_d;
                e_d[0] <= i_e;
                for (j = 1; j < T_M1; j = j + 1)
                    c_d[j] <= c_d[j-1];
                for (j = 1; j < T_M2; j = j + 1)
                    d_d[j] <= d_d[j-1];
                for (j = 1; j < T_A2; j = j + 1)
                    e_d[j] <= e_d[j-1];
            end
        end
    endgenerate

    // ---------------------------------------------------------------
    // U: unpack | int -> xf (n64); operand B of ADD1
    // ---------------------------------------------------------------
    wire [36:0] u_in = (s0_xsrc == 2'd0) ? bf16_to_xf(s0_x54[15:0]) :
                       (s0_xsrc == 2'd1) ? fp32_to_xf(s0_x54[31:0]) : s0_xf;
    wire [36:0] b_in = ctl_mc(ctl[0]) ? (((op == OP_LN_STAT) && (ctl_step(ctl[0]) == 3'd3)) ? XF_EPS : XF_ZERO) :
                       (op == OP_ADD)      ? (b_fp32 ? fp32_to_xf(s0_b) : bf16_to_xf(s0_b[15:0])) :
                       (op == OP_LN_APPLY) ? xf_neg(fp32_to_xf(s0_b)) : XF_ZERO;

    reg [53:0] na1, na2;
    reg        ns1, ns2, nz2, nsrc1, nsrc2;
    reg [11:0] ne1, ne2;
    reg [5:0]  nlz2;
    reg [36:0] u1, u2, b1, b2, xu, bu;
    wire [6:0]  nlz_w = lzc64({na1, 10'h3FF});
    wire [53:0] nn3   = na2 << nlz2;

    always @(posedge i_clk) begin
        na1   <= s0_x54[53] ? (54'd0 - s0_x54) : s0_x54;
        ns1   <= s0_x54[53];
        ne1   <= s0_eoff - {6'd0, s0_fb} + 12'd53;
        nsrc1 <= (s0_xsrc == 2'd2);
        u1    <= u_in;
        b1    <= b_in;

        na2   <= na1;
        ns2   <= ns1;
        nz2   <= (na1 == 54'd0);
        ne2   <= ne1;
        nlz2  <= nlz_w[5:0];
        nsrc2 <= nsrc1;
        u2    <= u1;
        b2    <= b1;

        xu    <= nsrc2 ? (nz2 ? XF_ZERO : {ns2, ne2 - {6'd0, nlz2}, nn3[53:30]}) : u2;
        bu    <= b2;
    end

    // ---------------------------------------------------------------
    // ADD1 and the main operand's wait for the table
    // ---------------------------------------------------------------
    wire [36:0] x1;
    vu_add u_add1 (.i_clk(i_clk), .i_x(xu), .i_y(bu), .o_z(x1));

    reg [36:0] x1d [T_X1+1:T_M1];
    always @(posedge i_clk) begin
        x1d[T_X1+1] <= x1;
        for (k = T_X1 + 2; k <= T_M1; k = k + 1)
            x1d[k] <= x1d[k-1];
    end

    // ---------------------------------------------------------------
    // PREP (5): table address / fraction / exponent offset
    //   gate:  u = 2^22 +- (|a| in Q.19 (GELU) or Q.18 (sigmoid))
    //   exp:   t = fix(max) - fix(a) in Q.23, k = t >> 23, f = t[22:0]
    //   rsqrt: {e[0], m[22:13]}, frac m[12:1], eoff -((e - e[0]) / 2)
    // ---------------------------------------------------------------
    wire signed [12:0] xe   = $signed({xu[35], xu[35:24]});
    wire signed [12:0] nxe  = -xe;
    wire [36:0]        mxx  = bf16_to_xf(rs_d[T_U]);
    wire signed [12:0] me   = $signed({mxx[35], mxx[35:24]});
    wire signed [12:0] nme  = -me;
    wire signed [12:0] grs  = $signed({8'd0, d_ftop}) - xe;
    wire [CW-1:0]      c3   = ctl[T_U];

    // P1 (+4)
    reg        g_sat1, g_neg1;
    reg [4:0]  g_rs1;
    reg [23:0] g_m1;
    reg [23:0] fa_m1, fm_m1;
    reg        fa_s1, fa_z1, fa_sat1, fa_left1, fm_s1, fm_z1, fm_sat1, fm_left1;
    reg [4:0]  fa_sh1, fm_sh1;
    reg [36:0] r_op1;
    always @(posedge i_clk) begin
        g_sat1   <= xe >= ($signed({8'd0, d_ftop}) - 13'sd1);
        g_neg1   <= xu[36];
        g_rs1    <= (grs < 13'sd0) ? 5'd0 : (grs > 13'sd24) ? 5'd24 : grs[4:0];
        g_m1     <= xu[23:0];

        fa_m1    <= xu[23:0];
        fa_s1    <= xu[36];
        fa_z1    <= (xu[23:0] == 24'd0);
        fa_sat1  <= xe > 13'sd17;
        fa_left1 <= xe >= 13'sd0;
        fa_sh1   <= (xe >= 13'sd0) ? ((xe > 13'sd17) ? 5'd17 : xe[4:0]) : ((xe < -13'sd24) ? 5'd24 : nxe[4:0]);

        fm_m1    <= mxx[23:0];
        fm_s1    <= mxx[36];
        fm_z1    <= (mxx[23:0] == 24'd0);
        fm_sat1  <= me > 13'sd17;
        fm_left1 <= me >= 13'sd0;
        fm_sh1   <= (me >= 13'sd0) ? ((me > 13'sd17) ? 5'd17 : me[4:0]) : ((me < -13'sd24) ? 5'd24 : nme[4:0]);

        r_op1    <= (ctl_mc(c3) && (op == OP_LN_STAT) && (ctl_step(c3) == 3'd4) &&
                     (xu[36] || (xu[23:0] == 24'd0))) ? XF_EPS : xu;
    end

    // P2 (+5)
    wire signed [12:0] r_es = $signed({r_op1[35], r_op1[35:24]}) - $signed({12'd0, r_op1[24]});
    wire signed [12:0] r_eh = r_es >>> 1;
    reg [23:0] g_zm2;
    reg        g_sat2, g_neg2, fa_s2, fm_s2;
    reg [41:0] fa_mag2, fm_mag2;
    reg [10:0] r_addr2;
    reg [11:0] r_frac2, r_eoff2;
    always @(posedge i_clk) begin
        g_zm2   <= g_m1 >> g_rs1;
        g_sat2  <= g_sat1;
        g_neg2  <= g_neg1;
        fa_mag2 <= fa_z1 ? 42'd0 : fa_left1 ? (fa_sat1 ? 42'h1FFFFFFFFFF : ({18'd0, fa_m1} << fa_sh1))
                                            : ({18'd0, fa_m1} >> fa_sh1);
        fm_mag2 <= fm_z1 ? 42'd0 : fm_left1 ? (fm_sat1 ? 42'h1FFFFFFFFFF : ({18'd0, fm_m1} << fm_sh1))
                                            : ({18'd0, fm_m1} >> fm_sh1);
        fa_s2   <= fa_s1;
        fm_s2   <= fm_s1;
        r_addr2 <= {r_op1[24], r_op1[22:13]};
        r_frac2 <= r_op1[12:1];
        r_eoff2 <= 12'd0 - r_eh[11:0];
    end

    // P3 (+6)
    reg [22:0]        g_u3;
    reg               g_sat3, g_neg3;
    reg signed [42:0] fa_fix3, fm_fix3;
    reg [10:0]        r_addr3;
    reg [11:0]        r_frac3, r_eoff3;
    always @(posedge i_clk) begin
        g_u3    <= g_neg2 ? (23'h400000 - g_zm2[22:0]) : (23'h400000 + g_zm2[22:0]);
        g_sat3  <= g_sat2;
        g_neg3  <= g_neg2;
        fa_fix3 <= fa_s2 ? -$signed({1'b0, fa_mag2}) : $signed({1'b0, fa_mag2});
        fm_fix3 <= fm_s2 ? -$signed({1'b0, fm_mag2}) : $signed({1'b0, fm_mag2});
        r_addr3 <= r_addr2;
        r_frac3 <= r_frac2;
        r_eoff3 <= r_eoff2;
    end

    // P4 (+7)
    reg [22:0]        g_u4;
    reg               g_sat4, g_neg4;
    reg signed [42:0] x_t4;
    reg [10:0]        r_addr4;
    reg [11:0]        r_frac4, r_eoff4;
    always @(posedge i_clk) begin
        g_u4    <= g_u3;
        g_sat4  <= g_sat3;
        g_neg4  <= g_neg3;
        x_t4    <= fm_fix3 - fa_fix3;
        r_addr4 <= r_addr3;
        r_frac4 <= r_frac3;
        r_eoff4 <= r_eoff3;
    end

    // P5 (+8): select by function
    wire [CW-1:0] c7   = ctl[T_TA-1];
    wire [1:0]    fn7  = ctl_mc(c7) ? FN_RSQRT : d_fn;
    wire [42:0]   tc   = x_t4[42] ? 43'd0 : x_t4;
    wire          kval = (tc[42:29] == 14'd0);
    wire [5:0]    kk   = kval ? tc[28:23] : 6'd63;
    reg [1:0]     ta_fn;
    reg [10:0]    ta_addr;
    reg [11:0]    ta_frac, ta_eoff;
    reg           ta_zero, ta_isgate, ta_sat, ta_neg;
    reg [5:0]     ta_k;
    always @(posedge i_clk) begin
        ta_fn <= fn7;
        case (fn7)
            FN_GELU, FN_SIGM: begin
                ta_addr <= g_u4[22:12];
                ta_frac <= g_u4[11:0];
                ta_eoff <= 12'd0;
                ta_zero <= 1'b0;
            end
            FN_EXP: begin
                ta_addr <= tc[22:12];
                ta_frac <= tc[11:0];
                ta_eoff <= 12'd0 - {6'd0, kk};
                ta_zero <= !(kval && ctl_mask(c7));
            end
            default: begin
                ta_addr <= r_addr4;
                ta_frac <= r_frac4;
                ta_eoff <= r_eoff4;
                ta_zero <= 1'b0;
            end
        endcase
        ta_isgate <= (fn7 == FN_GELU) || (fn7 == FN_SIGM);
        ta_sat    <= g_sat4;
        ta_neg    <= g_neg4;
        ta_k      <= kk;
    end

    // ---------------------------------------------------------------
    // TBL (6) + POST (1)
    // ---------------------------------------------------------------
    wire [33:0] t_raw;
    wire [36:0] t_z;
    vu_tbl #(.F_GELU(F_GELU), .F_SIGM(F_SIGM), .F_EXP(F_EXP), .F_RSQRT(F_RSQRT)) u_tbl (
        .i_clk(i_clk), .i_fn(ta_fn), .i_addr(ta_addr), .i_frac(ta_frac), .i_eoff(ta_eoff),
        .i_zero(ta_zero), .o_raw(t_raw), .o_z(t_z));

    reg [T_TZ:T_TA+1] isg_d, sat_d, neg_d;
    reg [5:0]         k_d [T_TA+1:T_RAW];
    always @(posedge i_clk) begin
        isg_d <= {isg_d[T_TZ-1:T_TA+1], ta_isgate};
        sat_d <= {sat_d[T_TZ-1:T_TA+1], ta_sat};
        neg_d <= {neg_d[T_TZ-1:T_TA+1], ta_neg};
        k_d[T_TA+1] <= ta_k;
        for (k = T_TA + 2; k <= T_RAW; k = k + 1)
            k_d[k] <= k_d[k-1];
    end

    reg [36:0] tpost;   // +15
    always @(posedge i_clk)
        tpost <= (isg_d[T_TZ] && sat_d[T_TZ]) ? (neg_d[T_TZ] ? XF_ZERO : XF_ONE) : t_z;

    // ---------------------------------------------------------------
    // QUANT row constants from the table (two registered reads of one ROM):
    //   R     = {0, gR - (e - odd), MR}   at +14, for MUL1 of the elements
    //   s_row = {0, gS + (e - odd), MS}   for the row's summary beat
    // ---------------------------------------------------------------
    wire [15:0] rsq = rs_d[T_QR];
    wire [15:0] rso = rs_d[T_OUT-2];       // one stage earlier than the summary needs: the fp32 conversion is split in two
    wire [63:0] qw13, qs_w;
    wire [7:0]  t_qaddr;                  // the tail's table address (port A is free while the op is not QUANT)
    vu_qtab #(.F_QUANT(F_QUANT)) u_qtab (.i_clk(i_clk), .i_addr_a(d_qtail ? t_qaddr : {~rsq[7], rsq[6:0]}),
                                         .i_addr_b({~rso[7], rso[6:0]}), .o_a(qw13), .o_b(qs_w));
    reg         qz13;
    reg  [11:0] qes13;
    reg  [36:0] r_q14;
    always @(posedge i_clk) begin
        qz13  <= (rsq[14:7] == 8'd0);
        qes13 <= {4'd0, rsq[14:7]} - 12'd127 - {11'd0, ~rsq[7]};
        r_q14 <= qz13 ? XF_ZERO : {1'b0, {{4{qw13[63]}}, qw13[63:56]} - qes13, qw13[55:32]};
    end

    // ---------------------------------------------------------------
    // row accumulators (fixed point relative to the row amax exponent E)
    //   acc2 = sum m8^2 << (2(e-E)+24)   (Q.38 of 2^2E)     -> ACC_A
    //   acc1 = sum +-m8 << ((e-E)+25)    (Q.32 of 2^E)      -> ACC_B
    //   S    = sum y << (8-k)            (Q.40)             -> ACC_A
    // ---------------------------------------------------------------
    reg signed [53:0] acc_a, acc_b;
    reg               acc_done;

    wire [7:0]         xe8   = s0_x54[14:7];
    wire [7:0]         re8   = rs_d[0][14:7];
    wire signed [9:0]  dxe   = $signed({2'b0, xe8}) - $signed({2'b0, re8});
    wire signed [10:0] sh2w  = (re8 == 8'd0) ? 11'sd24 : (dxe > 10'sd0) ? 11'sd24 : ($signed({dxe, 1'b0}) + 11'sd24);
    wire signed [10:0] sh1w  = (re8 == 8'd0) ? 11'sd25 : (dxe > 10'sd0) ? 11'sd25 : ($signed({dxe[9], dxe}) + 11'sd25);
    wire [7:0]         am8   = {1'b1, s0_x54[6:0]};
    wire signed [10:0] nsh2w = -sh2w;
    wire signed [10:0] nsh1w = -sh1w;

    // +1
`ifdef VU_MUL_LOGIC
    reg [15:0] a_sq1 /* synthesis syn_multstyle = "logic" */;
`else
    reg [15:0] a_sq1;
`endif
    reg [7:0]  a_m8_1;
    reg        a_z2_1, a_z1_1, a_s1, a_l2_1, a_l1_1;
    reg [4:0]  a_sh2_1, a_sh1_1;
    always @(posedge i_clk) begin
        a_sq1   <= am8 * am8;
        a_m8_1  <= am8;
        a_s1    <= s0_x54[15];
        a_z2_1  <= (xe8 == 8'd0) || (sh2w <= -11'sd17);
        a_z1_1  <= (xe8 == 8'd0) || (sh1w <= -11'sd9);
        a_l2_1  <= sh2w >= 11'sd0;
        a_l1_1  <= sh1w >= 11'sd0;
        a_sh2_1 <= (sh2w >= 11'sd0) ? sh2w[4:0] : nsh2w[4:0];
        a_sh1_1 <= (sh1w >= 11'sd0) ? sh1w[4:0] : nsh1w[4:0];
    end

    // +2
    reg [39:0]        t2_2;
    reg signed [34:0] t1_2;
    wire [32:0]       t1m = a_l1_1 ? ({25'd0, a_m8_1} << a_sh1_1) : ({25'd0, a_m8_1} >> a_sh1_1);
    always @(posedge i_clk) begin
        t2_2 <= a_z2_1 ? 40'd0 : a_l2_1 ? ({24'd0, a_sq1} << a_sh2_1) : ({24'd0, a_sq1} >> a_sh2_1);
        t1_2 <= a_z1_1 ? 35'sd0 : a_s1 ? -$signed({2'b0, t1m}) : $signed({2'b0, t1m});
    end

    // softmax term at T_RAW + 1, accumulated at T_RAW + 2
    reg [41:0] te;
    always @(posedge i_clk)
        te <= (k_d[T_RAW] >= 6'd41) ? 42'd0 :
              (k_d[T_RAW] <= 6'd8) ? ({8'd0, t_raw} << (6'd8 - k_d[T_RAW])) : ({8'd0, t_raw} >> (k_d[T_RAW] - 6'd8));

    wire [CW-1:0] c2  = ctl[2];
    wire [CW-1:0] c13 = ctl[T_RAW+1];
    always @(posedge i_clk) begin
        if (!i_rstn) begin
            acc_done <= 1'b0;
        end else begin
            acc_done <= 1'b0;
            if (d_acc12 && vld[2] && !ctl_mc(c2)) begin
                acc_a    <= (ctl_first(c2) ? 54'sd0 : acc_a) + $signed({14'd0, t2_2});
                acc_b    <= (ctl_first(c2) ? 54'sd0 : acc_b) + {{19{t1_2[34]}}, t1_2};
                acc_done <= ctl_last(c2);
            end
            if (d_accexp && vld[T_RAW+1] && !ctl_mc(c13)) begin
                acc_a    <= (ctl_first(c13) ? 54'sd0 : acc_a) + $signed({12'd0, te});
                acc_done <= ctl_last(c13);
            end
        end
    end

    // ---------------------------------------------------------------
    // microcode registers
    // ---------------------------------------------------------------
    reg [36:0] mr_t, mr_mu, mr_mu2, mr_var, mr_ve;
    reg [31:0] rs0, rs1;
    reg        mcap;
    reg [2:0]  mcap_step;

    // ---------------------------------------------------------------
    // MUL1: P = x1d | t ; Q = c | t | const | row register
    // ---------------------------------------------------------------
    wire [CW-1:0] c14  = ctl[T_M1-1];
    wire          mc14 = ctl_mc(c14);
    wire [2:0]    st14 = ctl_step(c14);
    wire [36:0]   cx14 = fp32_to_xf(c_d[T_M1-1]);
    reg           p1t, q1t;
    reg [36:0]    q1pre;
    always @(posedge i_clk) begin
        if (mc14) begin
            p1t   <= ((op == OP_RMS_STAT) && (st14 == 3'd1)) || ((op == OP_LN_STAT) && (st14 == 3'd4)) ||
                     (op == OP_SMAX_SUM) || (op == OP_SMAX_Q8);
            q1t   <= (op == OP_SMAX_SUM) || (op == OP_SMAX_Q8);
            case (op)
                OP_RMS_STAT: q1pre <= (st14 == 3'd0) ? kx : XF_ONE;
                OP_LN_STAT:  q1pre <= ((st14 == 3'd0) || (st14 == 3'd2)) ? kx : (st14 == 3'd1) ? mr_mu : XF_ONE;
                default:     q1pre <= XF_ONE;
            endcase
        end else begin
            p1t <= d_ptab;
            q1t <= (op == OP_GELU) || (op == OP_GEGLU) || (op == OP_SILU);
            case (op)
                OP_RMS_APPLY, OP_LN_APPLY, OP_ROPE_A, OP_ROPE_B, OP_SMAX_OUT, OP_DEQUANT: q1pre <= cx14;
                OP_SMAX_Q8:  q1pre <= b_fp32 ? XF_C255 : XF_C127;
                OP_QUANT:    q1pre <= r_q14;
                OP_EULER:    q1pre <= kx;
                default:     q1pre <= XF_ONE;
            endcase
        end
    end

    wire [36:0] y1;
    vu_mul u_mul1 (.i_clk(i_clk), .i_x(p1t ? tpost : x1d[T_M1]), .i_y(q1t ? tpost : q1pre), .o_z(y1));

    // ---------------------------------------------------------------
    // MUL2: Q = d | const
    // ---------------------------------------------------------------
    wire [CW-1:0] c18  = ctl[T_M2-1];
    reg  [36:0]   q2r;
    always @(posedge i_clk) begin
        if (ctl_mc(c18))
            q2r <= ((op == OP_SMAX_Q8) && (ctl_step(c18) == 3'd0)) ? (b_fp32 ? XF_INV255 : XF_INV127) : XF_ONE;
        else
            case (op)
                OP_RMS_APPLY, OP_LN_APPLY, OP_DEQUANT: q2r <= fp32_to_xf(d_d[T_M2-1]);
                OP_GEGLU:                               q2r <= bf16_to_xf(d_d[T_M2-1][15:0]);
                default:                                q2r <= XF_ONE;
            endcase
    end

    wire [36:0] y2;
    vu_mul u_mul2 (.i_clk(i_clk), .i_x(y1), .i_y(q2r), .o_z(y2));

    // ---------------------------------------------------------------
    // ADD2: second operand = e | const | -mu^2
    // ---------------------------------------------------------------
    wire [CW-1:0] c22 = ctl[T_A2-1];
    reg  [36:0]   e2r;
    always @(posedge i_clk) begin
        if (ctl_mc(c22))
            e2r <= ((op == OP_RMS_STAT) && (ctl_step(c22) == 3'd0)) ? XF_EPS :
                   ((op == OP_LN_STAT) && (ctl_step(c22) == 3'd2)) ? xf_neg(mr_mu2) : XF_ZERO;
        else
            case (op)
                OP_LN_APPLY, OP_ROPE_B, OP_EULER: e2r <= fp32_to_xf(e_d[T_A2-1]);
                OP_DEQUANT:                       e2r <= bias_en ? fp32_to_xf(e_d[T_A2-1]) : XF_ZERO;
                default:                          e2r <= XF_ZERO;
            endcase
    end

    wire [36:0] y3;
    vu_add u_add2 (.i_clk(i_clk), .i_x(y2), .i_y(e2r), .o_z(y3));

    // ---------------------------------------------------------------
    // microcode capture (+30)
    // ---------------------------------------------------------------
    wire [CW-1:0] c29 = ctl[T_Y3];
    always @(posedge i_clk) begin
        if (!i_rstn) begin
            mcap <= 1'b0;
        end else begin
            mcap <= vld[T_Y3] && ctl_mc(c29);
            if (vld[T_Y3] && ctl_mc(c29)) begin
                mcap_step <= ctl_step(c29);
                case (op)
                    OP_RMS_STAT:
                        if (ctl_step(c29) == 3'd0) mr_t <= y3;
                        else                       rs0  <= xf_to_fp32(y3);
                    OP_LN_STAT:
                        case (ctl_step(c29))
                            3'd0: begin mr_mu <= y3; rs0 <= xf_to_fp32(y3); end
                            3'd1: mr_mu2 <= y3;
                            3'd2: mr_var <= y3;
                            3'd3: mr_ve  <= y3;
                            default: rs1 <= xf_to_fp32(y3);
                        endcase
                    OP_SMAX_SUM, OP_SMAX_Q8:
                        rs0 <= xf_to_fp32(y3);
                    default: ;
                endcase
            end
        end
    end

    // ---------------------------------------------------------------
    // ROUND (3) and the output side
    // ---------------------------------------------------------------
    wire [31:0] code;
    vu_round u_round (.i_clk(i_clk), .i_fmt(d_fmt), .i_x(y3), .o_code(code));

    wire [CW-1:0] c32   = ctl[T_OUT];
    wire          el_we = vld[T_OUT] && !ctl_mc(c32) && d_out;
    reg [14:0]    tr_amax;
    reg [15:0]    tr_max;
    always @(posedge i_clk) begin
        if (el_we) begin
            if (ctl_first(c32)) begin
                tr_amax <= code[14:0];
                tr_max  <= ctl_mask(c32) ? code[15:0] : 16'hFF80;
            end else begin
                if (code[14:0] > tr_amax)
                    tr_amax <= code[14:0];
                if (ctl_mask(c32) && bf16_gt(code[15:0], tr_max))
                    tr_max <= code[15:0];
            end
        end
    end

    // QUANT s_row for the summary beat: the row statistic travels with its elements (rs_d); the table is read
    // while the row's last element is two stages before the output, and converted to fp32 over the next two
    // cycles, i.e. registered before the cycle that writes the summary.
    reg         qs_z, qs_z2;
    reg  [11:0] qs_es, qs_sum;
    reg  [23:0] qs_lo;
    reg  [31:0] srow_fp;
    reg         sum_pend;
    always @(posedge i_clk) begin
        qs_z    <= (rso[14:7] == 8'd0);
        qs_es   <= {4'd0, rso[14:7]} - 12'd127 - {11'd0, ~rso[7]};
        // table output -> exponent add -> normalise to fp32 was one cycle (the lane's longest path at 337 MHz,
        // 2026-09-22): the add is registered, the table is read a stage earlier, the result lands as before
        qs_z2   <= qs_z;
        qs_sum  <= {{4{qs_w[31]}}, qs_w[31:24]} + qs_es;
        qs_lo   <= qs_w[23:0];
        srow_fp <= qs_z2 ? 32'd0 : xf_to_fp32({1'b0, qs_sum, qs_lo});
        if (!i_rstn)
            sum_pend <= 1'b0;
        else
            sum_pend <= el_we && ctl_last(c32) && d_fast;
    end

    // ---------------------------------------------------------------
    // output FIFO (first word fall through, registered head)
    // ---------------------------------------------------------------
    reg [65:0]         fmem [0:DEPTH-1] /* synthesis syn_ramstyle = "block_ram" */;
    reg [FIFO_AW:0]    wptr, rptr;
    reg                ov;
    reg [65:0]         od;
    reg                sum_we;
    reg [65:0]         sum_wd;
    wire [65:0]        fast_wd = {2'd1, d_bf16out ? {1'b0, tr_amax} : 16'd0, d_bf16out ? tr_max : 16'd0,
                                  d_quant ? srow_fp : 32'd0};
    wire               t_el_we, t_sum_wr, t_idle;
    wire [31:0]        t_code, t_srow;
    wire               f_we   = d_qtail ? (t_el_we | t_sum_wr) : (el_we | sum_we | sum_pend);
    wire [65:0]        f_wd   = d_qtail ? (t_el_we ? {2'd0, 32'd0, t_code} : {2'd1, 32'd0, t_srow}) :
                                el_we ? {2'd0, 32'd0, code} : sum_pend ? fast_wd : sum_wd;
    wire [FIFO_AW:0]   stored = wptr - rptr;
    wire [FIFO_AW+1:0] fifo_cnt = {1'b0, stored} + {{(FIFO_AW+1){1'b0}}, ov};

    always @(posedge i_clk)
        if (f_we)
            fmem[wptr[FIFO_AW-1:0]] <= f_wd;

    always @(posedge i_clk) begin
        if (!i_rstn) begin
            wptr <= '0;
            rptr <= '0;
            ov   <= 1'b0;
        end else begin
            if (f_we)
                wptr <= wptr + 1'b1;
            if (!ov || i_ready) begin
                if (stored != '0) begin
                    od   <= fmem[rptr[FIFO_AW-1:0]];
                    rptr <= rptr + 1'b1;
                    ov   <= 1'b1;
                end else begin
                    ov   <= 1'b0;
                end
            end
        end
    end

    assign o_valid = ov;
    assign o_kind  = od[65:64];
    assign o_data  = od[63:0];

    // ---------------------------------------------------------------
    // fused QUANT tail.  Push side: every bf16 output code of the op goes into the row FIFO instead of the
    // output FIFO; the cycle after a row's last element (sum_pend: tr_amax is final, the accept gap keeps the
    // next row out) {row length, amax} goes into the statistic FIFO.  Fetch: statistic -> table (port A) ->
    // R, s_row registered as the next row.  Run: one bubble, in which the row's summary becomes pending, then
    // the row's codes one per cycle through bf16 -> xf -> x R (MUL) -> ROUND int8, exactly the element path of
    // OP_QUANT (whose ADD1 + 0, MUL2 x 1 and ADD2 + 0 are identities).  A pending summary takes the first
    // output cycle without an element, which the bubble guarantees once per row, so the tail keeps the main
    // path's rate (L + 1 cycles per row).
    // ---------------------------------------------------------------
    generate if (QTAIL != 0) begin : g_qtail
        localparam integer SAW = 9;                          // rows per lane per chunk <= 512
        localparam integer TL  = 1 + L_MUL + L_RND;          // read register + MUL + ROUND
        reg [15:0]          rmem [0:(1 << ROW_AW)-1] /* synthesis syn_ramstyle = "block_ram" */;
        reg [ROW_AW+15:0]   smem [0:(1 << SAW)-1]    /* synthesis syn_ramstyle = "block_ram" */;
        reg [ROW_AW:0]      r_wp, r_rp;
        reg [SAW:0]         s_wp, s_rp;
        reg [ROW_AW:0]      row_cnt, row_len;
        reg [15:0]          r_q;
        reg [ROW_AW+15:0]   s_q;
        wire                r_push = el_we && d_qtail;
        wire                s_push = sum_pend && d_qtail;

        always @(posedge i_clk) begin
            if (r_push) rmem[r_wp[ROW_AW-1:0]] <= code[15:0];
            if (s_push) smem[s_wp[SAW-1:0]]    <= {row_len, tr_amax};
            r_q <= rmem[r_rp[ROW_AW-1:0]];
            s_q <= smem[s_rp[SAW-1:0]];
        end

        // fetch: f_st 0 idle, 1 statistic read, 2 table read, 3 next-row registers
        reg [1:0]           f_st;
        reg                 nxt_v, f_qz;
        reg [11:0]          f_qes;
        reg [ROW_AW:0]      f_len, nxt_len;
        reg [36:0]          nxt_R, cur_R;
        reg [31:0]          nxt_s, sum_d;
        assign t_qaddr = {~s_q[7], s_q[6:0]};
        wire [36:0] f_sx = {1'b0, {{4{qw13[31]}}, qw13[31:24]} + f_qes, qw13[23:0]};

        // run
        reg                 run, sum_p;
        reg [ROW_AW:0]      left;
        reg [TL-1:0]        tv;
        wire                start = !run && nxt_v && !sum_p;
        wire                rd    = run;
        wire [36:0]         t_y;
        vu_mul   u_tmul (.i_clk(i_clk), .i_x(bf16_to_xf(r_q)), .i_y(cur_R), .o_z(t_y));
        vu_round u_trnd (.i_clk(i_clk), .i_fmt(FMT_INT8), .i_x(t_y), .o_code(t_code));
        assign t_el_we  = tv[TL-1];
        assign t_sum_wr = sum_p && !tv[TL-1];
        assign t_srow   = sum_d;
        // the tail is idle only when its last element has also left the output FIFO's memory: a row's summary is
        // written BEFORE its elements here, so the node's "all summaries seen" check no longer covers the cycle
        // between the last element's FIFO write and o_valid (found by ml_q: SILU L = 1, the chunk's last element
        // arrived after the node had closed the lane and was OR-ed into the next chunk's first beat)
        assign t_idle   = (r_wp == r_rp) && (s_wp == s_rp) && (f_st == 2'd0) && !nxt_v && !run && !sum_p && (tv == '0) &&
                          (wptr == rptr);

        always @(posedge i_clk) begin
            if (!i_rstn) begin
                r_wp <= '0; r_rp <= '0; s_wp <= '0; s_rp <= '0; row_cnt <= '0;
                f_st <= 2'd0; nxt_v <= 1'b0; run <= 1'b0; sum_p <= 1'b0; tv <= '0;
            end else begin
                if (r_push) begin
                    r_wp    <= r_wp + 1'b1;
                    row_cnt <= ctl_last(c32) ? '0 : row_cnt + 1'b1;
                    if (ctl_last(c32)) row_len <= row_cnt + 1'b1;
                end
                if (s_push) s_wp <= s_wp + 1'b1;

                case (f_st)
                    2'd0: if (!nxt_v && (s_wp != s_rp)) f_st <= 2'd1;          // s_q holds smem[s_rp] (registered)
                    2'd1: begin                                                  // table address = s_q this cycle
                        f_qz  <= (s_q[14:7] == 8'd0);
                        f_qes <= {4'd0, s_q[14:7]} - 12'd127 - {11'd0, ~s_q[7]};
                        f_len <= s_q[ROW_AW+15:15];
                        s_rp  <= s_rp + 1'b1;
                        f_st  <= 2'd2;
                    end
                    2'd2: begin                                                  // qw13 = the row's table word
                        nxt_R   <= f_qz ? XF_ZERO : {1'b0, {{4{qw13[63]}}, qw13[63:56]} - f_qes, qw13[55:32]};
                        nxt_s   <= f_qz ? 32'd0 : xf_to_fp32(f_sx);
                        nxt_len <= f_len;
                        nxt_v   <= 1'b1;
                        f_st    <= 2'd0;
                    end
                    default: f_st <= 2'd0;
                endcase

                if (start) begin                       // the bubble: no read this cycle
                    cur_R <= nxt_R;
                    left  <= nxt_len;
                    sum_d <= nxt_s;
                    sum_p <= 1'b1;
                    nxt_v <= 1'b0;
                    run   <= 1'b1;
                end else if (run) begin
                    r_rp <= r_rp + 1'b1;
                    left <= left - 1'b1;
                    if (left == (ROW_AW+1)'(1)) run <= 1'b0;
                end
                if (t_sum_wr) sum_p <= 1'b0;
                tv <= {tv[TL-2:0], rd};
            end
        end
    end else begin : g_no_qtail
        assign t_qaddr = 8'd0;
        assign t_el_we = 1'b0;
        assign t_sum_wr = 1'b0;
        assign t_idle = 1'b1;
        assign t_code = 32'd0;
        assign t_srow = 32'd0;
    end endgenerate

    // ---------------------------------------------------------------
    // credits: element tokens in flight + pending fast summaries + FIFO
    // ---------------------------------------------------------------
    reg [7:0] inflight;
    reg       credit_ok;
    wire      credit_ok_n = ({1'b0, inflight} + {1'b0, fifo_cnt}) < 9'(DEPTH - 4);
    wire      el_exit = vld[T_OUT] && !ctl_mc(c32);
    always @(posedge i_clk) begin
        if (!i_rstn) begin
            inflight  <= 8'd0;
            credit_ok <= 1'b0;
        end else begin
            inflight  <= inflight + {7'd0, accept} + {7'd0, accept && i_last && d_fast}
                         - {7'd0, el_exit} - {7'd0, sum_pend};
            credit_ok <= credit_ok_n;
        end
    end

    // ---------------------------------------------------------------
    // control FSM
    // ---------------------------------------------------------------
    localparam [3:0] S_ROW = 4'd0, S_RUN = 4'd3, S_END = 4'd4, S_MW = 4'd5, S_ML = 4'd6,
                     S_SUMW = 4'd7, S_SUM1 = 4'd8, S_SUM = 4'd9;
    reg [3:0]  st;
    reg [2:0]  cstep;
    reg        accd_f, outd_f, gap;
    reg [15:0] rs_row;

    wire [11:0] e_row = (rs_row[14:7] == 8'd0) ? EZ : ({4'd0, rs_row[14:7]} - 12'd127);
    wire        room  = fifo_cnt < (FIFO_AW+2)'(DEPTH);

    // o_ready registered, one cycle ahead and exact: S_RUN next cycle (S_ROW -> S_RUN, or S_RUN staying, no
    // descriptor load, no reset), credits next cycle (credit_ok's own D), no gap next cycle.  It was
    // (st == S_RUN) && credit_ok && !gap combinationally, at the head of the node's fire -> slot address path.
    reg       ready_q;
    wire      desc_ld  = i_desc_we && o_idle;
    wire      run_next = i_rstn && !desc_ld && ((st == S_ROW) || ((st == S_RUN) && !(accept && i_last && !d_fast)));
    wire      gap_next = (st == S_RUN) && accept && i_last && d_fast;
    always @(posedge i_clk) ready_q <= run_next && credit_ok_n && !gap_next;
    assign o_ready = ready_q;
    assign accept  = i_valid && o_ready;
    assign o_idle  = ((st == S_ROW) || ((st == S_RUN) && first_pend && !gap)) && (inflight == 8'd0) && !i_valid && t_idle;

    // microcode token sources (combinational from the step to launch)
    always @(*) begin
        ml_en   = 1'b0;
        ml_step = 3'd0;
        case (st)
            S_END:  if ((d_acc12 || d_accexp) && accd_f) ml_en = 1'b1;
            S_ML:   begin ml_en = 1'b1; ml_step = cstep + 3'd1; end
            default: ;
        endcase
        ml_xsrc = 2'd3;
        ml_xf   = XF_ZERO;
        ml_x54  = 54'd0;
        ml_fb   = 6'd0;
        ml_eoff = 12'd0;
        case (op)
            OP_RMS_STAT:
                if (ml_step == 3'd0) begin
                    ml_xsrc = 2'd2; ml_x54 = acc_a; ml_fb = 6'd38; ml_eoff = {e_row[10:0], 1'b0};
                end else
                    ml_xf = mr_t;
            OP_LN_STAT:
                case (ml_step)
                    3'd0: begin ml_xsrc = 2'd2; ml_x54 = acc_b; ml_fb = 6'd32; ml_eoff = e_row; end
                    3'd1: ml_xf = mr_mu;
                    3'd2: begin ml_xsrc = 2'd2; ml_x54 = acc_a; ml_fb = 6'd38; ml_eoff = {e_row[10:0], 1'b0}; end
                    3'd3: ml_xf = mr_var;
                    default: ml_xf = mr_ve;
                endcase
            OP_SMAX_SUM, OP_SMAX_Q8: begin
                ml_xsrc = 2'd2; ml_x54 = acc_a; ml_fb = 6'd40; ml_eoff = 12'd0;
            end
            default: ;
        endcase
    end

    always @(posedge i_clk) begin
        if (!i_rstn) begin
            st         <= S_ROW;
            first_pend <= 1'b1;
            accd_f     <= 1'b0;
            outd_f     <= 1'b0;
            sum_we     <= 1'b0;
            gap        <= 1'b0;
            op         <= OP_ADD;
            d_qtail    <= 1'b0;
        end else begin
            sum_we <= 1'b0;
            gap    <= 1'b0;
            if (acc_done)
                accd_f <= 1'b1;
            if (el_we && ctl_last(c32))
                outd_f <= 1'b1;
            if (accept)
                first_pend <= i_last;

            if (i_desc_we && o_idle) begin
                op          <= i_op;
                b_fp32      <= i_b_fp32;
                bias_en     <= i_bias_en;
                out_fp32    <= i_out_fp32;
                kx          <= fp32_to_xf(i_k);
                d_out       <= !((i_op == OP_RMS_STAT) || (i_op == OP_LN_STAT) || (i_op == OP_SMAX_SUM));
                d_fast      <= !((i_op == OP_RMS_STAT) || (i_op == OP_LN_STAT) || (i_op == OP_SMAX_SUM) ||
                                 (i_op == OP_SMAX_Q8));
                d_bf16out   <= (i_op == OP_ADD) || (i_op == OP_RMS_APPLY) || (i_op == OP_LN_APPLY) ||
                               (i_op == OP_ROPE_B) || (i_op == OP_GELU) || (i_op == OP_GEGLU) ||
                               (i_op == OP_SILU) || (i_op == OP_SMAX_OUT) || ((i_op == OP_DEQUANT) && !i_out_fp32);
                d_acc12     <= (i_op == OP_RMS_STAT) || (i_op == OP_LN_STAT);
                d_accexp    <= (i_op == OP_SMAX_SUM) || (i_op == OP_SMAX_Q8);
                d_quant     <= (i_op == OP_QUANT);
                d_qtail     <= (QTAIL != 0) && i_qtail &&
                               ((i_op == OP_ADD) || (i_op == OP_RMS_APPLY) || (i_op == OP_LN_APPLY) ||
                                (i_op == OP_ROPE_B) || (i_op == OP_GELU) || (i_op == OP_GEGLU) ||
                                (i_op == OP_SILU) || (i_op == OP_SMAX_OUT) || ((i_op == OP_DEQUANT) && !i_out_fp32));
                d_hasrs     <= (i_op == OP_RMS_STAT) || (i_op == OP_LN_STAT) || (i_op == OP_SMAX_SUM) ||
                               (i_op == OP_SMAX_Q8);
                d_ptab      <= (i_op == OP_SMAX_SUM) || (i_op == OP_SMAX_OUT) || (i_op == OP_SMAX_Q8);
                d_fmt       <= (i_op == OP_QUANT) ? FMT_INT8 :
                               (i_op == OP_SMAX_Q8) ? (i_b_fp32 ? FMT_UINT8 : FMT_INT8) :
                               ((i_op == OP_ROPE_A) || (i_op == OP_EULER) || ((i_op == OP_DEQUANT) && i_out_fp32)) ?
                               FMT_FP32 : FMT_BF16;
                d_fn        <= (i_op == OP_SILU) ? FN_SIGM :
                               ((i_op == OP_SMAX_SUM) || (i_op == OP_SMAX_OUT) || (i_op == OP_SMAX_Q8)) ? FN_EXP : FN_GELU;
                d_xsrc      <= (i_op == OP_DEQUANT) ? 2'd2 : (i_op == OP_EULER) ? 2'd1 : 2'd0;
                d_ftop      <= (i_op == OP_SILU) ? 5'd5 : 5'd4;
                d_last_step <= (i_op == OP_LN_STAT) ? 3'd4 : (i_op == OP_RMS_STAT) ? 3'd1 : 3'd0;
                st          <= S_ROW;
                first_pend  <= 1'b1;
            end else begin
                case (st)
                    S_ROW:
                        st <= S_RUN;
                    S_RUN:
                        if (accept && i_last) begin
                            if (d_fast)
                                gap <= 1'b1;              // one idle cycle: the summary beat's FIFO slot
                            else begin
                                rs_row <= i_rs;
                                st     <= S_END;
                            end
                        end
                    S_END:
                        if (accd_f) begin
                            cstep <= 3'd0;
                            st    <= S_MW;
                        end
                    S_MW:
                        if (mcap && (mcap_step == cstep))
                            st <= (cstep == d_last_step) ? S_SUMW : S_ML;
                    S_ML: begin
                        cstep <= cstep + 3'd1;
                        st    <= S_MW;
                    end
                    S_SUMW:
                        if (!d_out || outd_f)
                            st <= (op == OP_LN_STAT) ? S_SUM1 : S_SUM;
                    S_SUM1:
                        if (room) begin
                            sum_we <= 1'b1;
                            sum_wd <= {2'd2, 32'd0, rs1};
                            st     <= S_SUM;
                        end
                    S_SUM:
                        if (room && !sum_we) begin
                            sum_we <= 1'b1;
                            sum_wd <= {2'd1, 16'd0, 16'd0, d_hasrs ? rs0 : 32'd0};
                            st     <= S_ROW;
                            accd_f <= 1'b0;
                            outd_f <= 1'b0;
                        end
                    default:
                        st <= S_ROW;
                endcase
            end
        end
    end

endmodule


// ------------------------------------------------------------------
// Fixed delay on a block-RAM ring: o_q after clock edge t is i_d sampled at edge t - D (3 <= D <= 2^AW + 1),
// the same as D flops in a row.  Free running (written and read every cycle); the read register and the
// output register are the RAM's own, so a wide, long delay costs one BRAM72K instead of W x D flops.
// ------------------------------------------------------------------
module vu_delay_ram
#(
    parameter integer W  = 96,
    parameter integer D  = 13,
    parameter integer AW = 5
)
(
    input  wire         i_clk,
    input  wire [W-1:0] i_d,
    output reg  [W-1:0] o_q
);
    reg [W-1:0]  mem [0:(1<<AW)-1] /* synthesis syn_ramstyle = "block_ram" */;   // not LRAM2K (MLP72 sites)
    reg [AW-1:0] wp = '0;
    reg [W-1:0]  rd;
    initial if (D < 3 || D > (1 << AW) + 1) $fatal(1, "vu_delay_ram: D out of range");
    always @(posedge i_clk) begin
        mem[wp] <= i_d;
        wp      <= wp + 1'b1;
        rd      <= mem[wp - AW'(D - 1)];
        o_q     <= rd;
    end
endmodule
