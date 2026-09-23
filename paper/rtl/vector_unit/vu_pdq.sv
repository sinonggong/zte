// Pre-dequant (PDQ) in front of a vector lane: operator fusion 2 (docs/PI0_VECTOR_OP_FUSION_20260918.md §3).
//
// Why: the MLP's GEGLU stage was three lane passes -- DEQUANT(gate acc), DEQUANT(up acc), GEGLU -- and after the
// double-buffered GEMM it is the largest stage of the chunk (full array, design clocks: prefix + expert geglu_quant
// 0.19 of 0.77 s [P]).  With PDQ the GEGLU record takes the two int32 accumulators directly and this block computes,
// per element, bit for bit what the two DEQUANT records wrote:
//     x' = bf16( (norm64(x) * c) * e )      gate: c = s_row (row), e = s_col of the gate columns
//     d' = bf16( (norm64(d) * c) * b )      up:   same s_row,    b = s_col of the up columns
// (vector_unit_ref.op_dequant without bias: mul(mul(norm64(acc), s_row), s_col), to_bf16; the lane's own DEQUANT runs
// U -> ADD1(+0) -> MUL1 -> MUL2 -> ADD2(+0) -> ROUND, and adding 0 is exact.)  The lane then runs GEGLU on x', d'.
//
// Stream: valid / ready on both sides.  i_en (static per record) selects PDQ; off, the block is a wire (no latency).
// On: a fixed 15-cycle pipeline (input register | U 3 | MUL 4 | MUL 4 | ROUND 3) into a FIFO of FD entries; the input is ready while
// the elements in flight plus the FIFO fit (credits), so the lane's own back-pressure (its row gap, its output FIFO)
// never drops an element.  o_empty: nothing in flight or queued (the node waits for it with the lane's idle).
// Row data other than x / d (last, mask, rs, b, c, e) ride along unchanged.
module vu_pdq #(
    parameter integer FD_LOG2 = 5
) (
    input  wire        i_clk,
    input  wire        i_rstn,
    input  wire        i_en,
    // from the node
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
    // to the lane
    output wire        o_valid,
    input  wire        i_ready,
    output wire        o_last,
    output wire        o_mask,
    output wire [47:0] o_x,
    output wire [15:0] o_rs,
    output wire [31:0] o_b,
    output wire [31:0] o_c,
    output wire [31:0] o_d,
    output wire [31:0] o_e,
    output wire        o_empty
);
    import vu_pkg::*;
    localparam integer L_U = 3, L_MUL = 4, L_RND = 3;
    localparam integer LAT = 1 + L_U + 2 * L_MUL + L_RND;         // 15: input register, U, two multiplies, round
    localparam integer FD  = 1 << FD_LOG2;
    localparam integer SBW = 1 + 1 + 16 + 32 + 32 + 32;          // side band: last, mask, rs, b, c, e

    // ------------------------------------------------------------ credits
    reg  [LAT-1:0]   vld;
    reg  [FD_LOG2:0] fcnt;                                       // FIFO entries
    reg  [5:0]       infl;                                       // elements in the pipeline
    reg  [FD_LOG2:0] wp, rp;
    reg              q_nonempty;                                 // registered (wp != rp) of the next cycle: exact
    reg              p_ready;                                    // registered credit check, one element of margin
    wire             acc_in  = i_valid && o_ready;
    wire             f_push  = vld[LAT-1];
    wire             f_pop   = i_en && q_nonempty && i_ready;       // bypass traffic never touches the FIFO
    wire [FD_LOG2:0] wp_n    = wp + (FD_LOG2+1)'(f_push), rp_n = rp + (FD_LOG2+1)'(f_pop);

    // ------------------------------------------------------------ U: norm64 of x and d, fp32 unpack of the scales
    // registered in three steps like the lane (magnitude | leading zeros | shift), so the timing matches its U
    reg  [53:0] xa1, da1, xa2, da2;
    reg         xs1, ds1, xs2, ds2, xz2, dz2;
    reg  [5:0]  xl2, dl2;
    reg  [36:0] xu, du, c1, c2, cu, e1, e2, eu, b1, b2, bu;
    // input register: the operands arrive through the slot memories' phase mux and the constant mux; unpacking
    // them in the same cycle was the vector node's longest path at 337 MHz (2026-09-22)
    reg  [47:0] x_r;
    reg  [31:0] d_r, b_r, c_r, e_r;
    always @(posedge i_clk) begin
        x_r <= i_x; d_r <= i_d; b_r <= i_b; c_r <= i_c; e_r <= i_e;
    end
    wire [53:0] x54 = {{6{x_r[47]}}, x_r};
    wire [53:0] d54 = {{22{d_r[31]}}, d_r};
    wire [6:0]  xlz = lzc64({xa1, 10'h3FF});
    wire [6:0]  dlz = lzc64({da1, 10'h3FF});
    wire [53:0] xn3 = xa2 << xl2;
    wire [53:0] dn3 = da2 << dl2;
    always @(posedge i_clk) begin
        xa1 <= x54[53] ? (54'd0 - x54) : x54;  xs1 <= x54[53];
        da1 <= d54[53] ? (54'd0 - d54) : d54;  ds1 <= d54[53];
        c1  <= fp32_to_xf(c_r);  e1 <= fp32_to_xf(e_r);  b1 <= fp32_to_xf(b_r);
        xa2 <= xa1; xs2 <= xs1; xz2 <= (xa1 == 54'd0); xl2 <= xlz[5:0];
        da2 <= da1; ds2 <= ds1; dz2 <= (da1 == 54'd0); dl2 <= dlz[5:0];
        c2  <= c1;  e2 <= e1;  b2 <= b1;
        xu  <= xz2 ? XF_ZERO : {xs2, 12'd53 - {6'd0, xl2}, xn3[53:30]};
        du  <= dz2 ? XF_ZERO : {ds2, 12'd53 - {6'd0, dl2}, dn3[53:30]};
        cu  <= c2;  eu <= e2;  bu <= b2;
    end

    // ------------------------------------------------------------ MUL (x s_row), MUL (x s_col), ROUND to bf16
    wire [36:0] xm1, dm1, xm2, dm2;
    reg  [36:0] ed [0:L_MUL-1], bd [0:L_MUL-1];                  // s_col operands wait for the first product
    integer k;
    always @(posedge i_clk) begin
        ed[0] <= eu; bd[0] <= bu;
        for (k = 1; k < L_MUL; k = k + 1) begin ed[k] <= ed[k-1]; bd[k] <= bd[k-1]; end
    end
    vu_mul u_mx1 (.i_clk(i_clk), .i_x(xu),  .i_y(cu),            .o_z(xm1));
    vu_mul u_md1 (.i_clk(i_clk), .i_x(du),  .i_y(cu),            .o_z(dm1));
    vu_mul u_mx2 (.i_clk(i_clk), .i_x(xm1), .i_y(ed[L_MUL-1]),   .o_z(xm2));
    vu_mul u_md2 (.i_clk(i_clk), .i_x(dm1), .i_y(bd[L_MUL-1]),   .o_z(dm2));
    wire [31:0] xcode, dcode;
    vu_round u_rx (.i_clk(i_clk), .i_fmt(FMT_BF16), .i_x(xm2), .o_code(xcode));
    vu_round u_rd (.i_clk(i_clk), .i_fmt(FMT_BF16), .i_x(dm2), .o_code(dcode));

    // ------------------------------------------------------------ side band delay line and valid
    reg  [SBW-1:0] sb [0:LAT-1];
    always @(posedge i_clk) begin
        sb[0] <= {i_last, i_mask, i_rs, i_b, i_c, i_e};
        for (k = 1; k < LAT; k = k + 1) sb[k] <= sb[k-1];
        if (!i_rstn) vld <= '0;
        else         vld <= {vld[LAT-2:0], acc_in && i_en};
    end

    // ------------------------------------------------------------ FIFO (registered-output RAM would add a cycle;
    // FD entries of 48 + 32 + side band are small enough for distributed RAM / registers)
    // registered read of the next head (rp_n), with a bypass for the entry being written this cycle
    reg  [SBW+31:0] fmem [0:FD-1] /* synthesis syn_ramstyle = "block_ram" */;
    wire [SBW+31:0] wdata = {sb[LAT-1], dcode[15:0], xcode[15:0]};
    reg  [SBW+31:0] ram_q, byp_d;
    reg             byp_q;
    always @(posedge i_clk) begin
        if (f_push) fmem[wp[FD_LOG2-1:0]] <= wdata;
        ram_q <= fmem[rp_n[FD_LOG2-1:0]];
        byp_d <= wdata;
        byp_q <= f_push && (wp[FD_LOG2-1:0] == rp_n[FD_LOG2-1:0]);
        if (!i_rstn) begin
            wp <= '0; rp <= '0; fcnt <= '0; infl <= 6'd0; q_nonempty <= 1'b0; p_ready <= 1'b0;
        end else begin
            wp <= wp_n;
            rp <= rp_n;
            fcnt <= fcnt + (FD_LOG2+1)'(f_push) - (FD_LOG2+1)'(f_pop);
            infl <= infl + 6'(acc_in && i_en) - 6'(f_push);
            q_nonempty <= (wp_n != rp_n);
            p_ready    <= (32'(infl) + 32'(fcnt)) < (FD - 4);
        end
    end
    wire [SBW+31:0] head = byp_q ? byp_d : ram_q;

    // ------------------------------------------------------------ outputs: PDQ or straight through
    assign o_ready = i_en ? p_ready : i_ready;
    assign o_valid = i_en ? q_nonempty : i_valid;
    assign o_last  = i_en ? head[SBW+31] : i_last;
    assign o_mask  = i_en ? head[SBW+30] : i_mask;
    assign o_rs    = i_en ? head[SBW+29 -: 16] : i_rs;
    assign o_b     = i_en ? head[SBW+13 -: 32] : i_b;
    assign o_c     = i_en ? head[SBW-19 -: 32] : i_c;
    assign o_e     = i_en ? head[SBW-51 -: 32] : i_e;
    assign o_x     = i_en ? {32'd0, head[15:0]} : i_x;
    assign o_d     = i_en ? {16'd0, head[31:16]} : i_d;
    assign o_empty = (infl == 6'd0) && !q_nonempty;
endmodule
