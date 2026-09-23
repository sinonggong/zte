// ------------------------------------------------------------------
// Four 2048 x 36-bit BRAM72K function tables with linear interpolation
// (pure datapath, free running).
//
//   word = {V[20:0] Q1.20, D[14:0] = V[i+1]-V[i] signed}
//   y    = (V << 12) + D * frac          Q1.32, >= 0   (i_zero forces 0)
//   z    = xf(y) * 2^eoff
//
// Every read is a registered BRAM read (no combinational ROM).  The caller
// presents fn/addr/frac/eoff/zero from registers.  Latency 4 to o_raw,
// 6 to o_z.   == vector_unit_ref.tbl_interp + raw_to_xf
// ------------------------------------------------------------------
`include "vu_pkg.sv"

module vu_tbl
#(
    parameter F_GELU  = "vu_tbl_gelu.mem",
    parameter F_SIGM  = "vu_tbl_sigm.mem",
    parameter F_EXP   = "vu_tbl_exp.mem",
    parameter F_RSQRT = "vu_tbl_rsqrt.mem"
)
(
    input  wire        i_clk,
    input  wire [1:0]  i_fn,
    input  wire [10:0] i_addr,
    input  wire [11:0] i_frac,
    input  wire [11:0] i_eoff,
    input  wire        i_zero,
    output reg  [33:0] o_raw,
    output reg  [36:0] o_z
);
    import vu_pkg::*;

    reg [35:0] rom_gelu  [0:2047] /* synthesis syn_ramstyle = "block_ram" */;
    reg [35:0] rom_sigm  [0:2047] /* synthesis syn_ramstyle = "block_ram" */;
    reg [35:0] rom_exp   [0:2047] /* synthesis syn_ramstyle = "block_ram" */;
    reg [35:0] rom_rsqrt [0:2047] /* synthesis syn_ramstyle = "block_ram" */;

    initial begin
        $readmemh(F_GELU,  rom_gelu);
        $readmemh(F_SIGM,  rom_sigm);
        $readmemh(F_EXP,   rom_exp);
        $readmemh(F_RSQRT, rom_rsqrt);
    end

    reg [35:0]        q_gelu, q_sigm, q_exp, q_rsqrt, q2;
    reg [1:0]         fn1;
    reg [11:0]        frac1, frac2, e1, e2, e3, e4, e5;
    reg               z1, z2, z3;
`ifdef VU_MUL_LOGIC
    reg signed [26:0] prod3 /* synthesis syn_multstyle = "logic" */;
`else
    reg signed [26:0] prod3;
`endif
    reg [20:0]        v3;
    reg [33:0]        r5;
    reg [5:0]         lz5;
    reg               zr5;

    wire [6:0]        lz_w = lzc64({o_raw, 30'h3FFFFFFF});
    wire [33:0]       n6   = r5 << lz5;

    always @(posedge i_clk) begin
        // 1: BRAM registered read
        q_gelu  <= rom_gelu[i_addr];
        q_sigm  <= rom_sigm[i_addr];
        q_exp   <= rom_exp[i_addr];
        q_rsqrt <= rom_rsqrt[i_addr];
        fn1 <= i_fn;  frac1 <= i_frac;  e1 <= i_eoff;  z1 <= i_zero;

        // 2: BRAM output register + function select
        case (fn1)
            FN_GELU: q2 <= q_gelu;
            FN_SIGM: q2 <= q_sigm;
            FN_EXP:  q2 <= q_exp;
            default: q2 <= q_rsqrt;
        endcase
        frac2 <= frac1;  e2 <= e1;  z2 <= z1;

        // 3: slope x fraction
        prod3 <= $signed(q2[14:0]) * $signed({1'b0, frac2});
        v3    <= q2[35:15];
        e3    <= e2;  z3 <= z2;

        // 4: interpolated Q1.32
        o_raw <= z3 ? 34'd0 : ({1'b0, v3, 12'd0} + {{7{prod3[26]}}, prod3});
        e4    <= e3;

        // 5: leading zeros
        lz5 <= lz_w[5:0];
        zr5 <= (o_raw == 34'd0);
        r5  <= o_raw;
        e5  <= e4;

        // 6: normalise
        o_z <= zr5 ? XF_ZERO : {1'b0, e5 + 12'd1 - {6'd0, lz5}, n6[33:10]};
    end

endmodule
