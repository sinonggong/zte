// ------------------------------------------------------------------
// xf multiply, latency 4 (pure datapath): operand reg | 24x24 product |
// product pipeline reg | normalise (truncate to 24 bits).
// mul(x, XF_ONE) == x bit-exactly (lane bypass).  == vector_unit_ref.mul
// The product is plain RTL; synthesis infers the multiplier (MLP72 or
// fabric, see paper/synth/run_vu_lane_synth.sh MULSTYLE).
// ------------------------------------------------------------------
`include "vu_pkg.sv"

module vu_mul
(
    input  wire        i_clk,
    input  wire [36:0] i_x,
    input  wire [36:0] i_y,
    output reg  [36:0] o_z
);
    import vu_pkg::*;

    reg [23:0]        xm1, ym1;
    reg signed [12:0] es1, es2, es3;
    reg               s1, s2, s3, z1, z2, z3;
`ifdef VU_MUL_LOGIC
    reg [47:0]        p2 /* synthesis syn_multstyle = "logic" */;
`else
    reg [47:0]        p2;
`endif
    reg [47:0]        p3;

    always @(posedge i_clk) begin
        xm1 <= i_x[23:0];
        ym1 <= i_y[23:0];
        es1 <= $signed({i_x[35], i_x[35:24]}) + $signed({i_y[35], i_y[35:24]});
        s1  <= i_x[36] ^ i_y[36];
        z1  <= (i_x[23:0] == 24'd0) || (i_y[23:0] == 24'd0);

        p2  <= xm1 * ym1;
        es2 <= es1;  s2 <= s1;  z2 <= z1;

        p3  <= p2;
        es3 <= es2;  s3 <= s2;  z3 <= z2;

        if (z3)
            o_z <= XF_ZERO;
        else if (p3[47])
            o_z <= {s3, es3[11:0] + 12'd1, p3[47:24]};
        else
            o_z <= {s3, es3[11:0], p3[46:23]};
    end

endmodule
