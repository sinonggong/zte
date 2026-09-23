// ------------------------------------------------------------------
// xf -> stream code, latency 3 (pure datapath).
//   FMT_BF16: round-to-nearest-even to 8 mantissa bits, saturate to the
//             largest finite (0x7F7F), below the smallest normal -> 0
//   FMT_FP32: exact (the 24-bit mantissa is an fp32 mantissa), same limits
//   FMT_INT8: round-to-nearest-even to an integer, clamp +-127, sign-extended
//   FMT_UINT8: the same, clamped to 0..255 with no sign (negatives read as 0)
// == vector_unit_ref.to_bf16 / to_fp32 / to_int8
// Negative control: +define+VU_NEG_ROUND_HALF_UP rounds bf16 ties up.
// ------------------------------------------------------------------
`include "vu_pkg.sv"

module vu_round
(
    input  wire        i_clk,
    input  wire [1:0]  i_fmt,
    input  wire [36:0] i_x,
    output reg  [31:0] o_code
);
    import vu_pkg::*;

    // 1 -------------------------------------------------------------
    wire [23:0]        m   = i_x[23:0];
    wire signed [12:0] e13 = $signed({i_x[35], i_x[35:24]});
`ifdef VU_NEG_ROUND_HALF_UP
    wire               inc16 = m[15];
`else
    wire               inc16 = m[15] & ((|m[14:0]) | m[16]);
`endif
    wire [8:0]         m9  = {1'b0, m[23:16]} + {8'd0, inc16};
    wire signed [12:0] sh13 = 13'sd23 - e13;

    reg [1:0]          f1;
    reg                s1, z1, sat1;
    reg [7:0]          m8_1;
    reg signed [12:0]  eb1, ef1;
    reg [23:0]         m24_1;
    reg [5:0]          sh1;
    always @(posedge i_clk) begin
        f1    <= i_fmt;
        s1    <= i_x[36];
        z1    <= (m == 24'd0);
        m8_1  <= m9[8] ? 8'h80 : m9[7:0];
        eb1   <= e13 + 13'sd127 + {12'd0, m9[8]};
        ef1   <= e13 + 13'sd127;
        m24_1 <= m;
        sat1  <= (e13 >= ((i_fmt == FMT_UINT8) ? 13'sd8 : 13'sd7));
        sh1   <= (sh13 > 13'sd48) ? 6'd48 : (sh13 < 13'sd1) ? 6'd1 : sh13[5:0];
    end

    // 2 -------------------------------------------------------------
    wire [47:0]        w2 = {m24_1, 24'd0} >> sh1;
    reg [1:0]          f2;
    reg                s2, z2, sat2, inc8_2;
    reg [7:0]          q2;
    reg [31:0]         c2;
    always @(posedge i_clk) begin
        f2     <= f1;
        s2     <= s1;
        z2     <= z1;
        sat2   <= sat1;
        q2     <= w2[31:24];
        inc8_2 <= w2[23] & ((|w2[22:0]) | w2[24]);
        if (f1 == FMT_BF16) begin
            if (z1 || (eb1 <= 13'sd0))
                c2 <= 32'd0;
            else if (eb1 >= 13'sd255)
                c2 <= {16'd0, s1, 15'h7F7F};
            else
                c2 <= {16'd0, s1, eb1[7:0], m8_1[6:0]};
        end else begin
            if (z1 || (ef1 <= 13'sd0))
                c2 <= 32'd0;
            else if (ef1 >= 13'sd255)
                c2 <= {s1, 31'h7F7FFFFF};
            else
                c2 <= {s1, ef1[7:0], m24_1[22:0]};
        end
    end

    // 3 -------------------------------------------------------------
    wire [8:0]         q9   = {1'b0, q2} + {8'd0, inc8_2};
    wire [8:0]         top9 = (f2 == FMT_UINT8) ? 9'd255 : 9'd127;
    wire [7:0]         q8   = z2 ? 8'd0 : (sat2 || (q9 > top9)) ? top9[7:0] : q9[7:0];
    always @(posedge i_clk) begin
        if (f2 == FMT_INT8)
            o_code <= s2 ? {{24{q8 != 8'd0}}, 8'd0 - q8} : {24'd0, q8};
        else if (f2 == FMT_UINT8)
            o_code <= s2 ? 32'd0 : {24'd0, q8};
        else
            o_code <= c2;
    end

endmodule
