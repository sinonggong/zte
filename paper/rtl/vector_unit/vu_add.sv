// ------------------------------------------------------------------
// xf add, latency 6 (pure datapath, free running):
//   1 compare magnitudes, exponent difference | 2 swap, clamp shift |
//   3 align (3 guard bits, truncating) | 4 add/sub | 5 leading zeros |
//   6 normalise (truncate to 24 bits)
// add(x, XF_ZERO) == x bit-exactly, which is how the lane bypasses it.
// == vector_unit_ref.add
// ------------------------------------------------------------------
`include "vu_pkg.sv"

module vu_add
(
    input  wire        i_clk,
    input  wire [36:0] i_x,
    input  wire [36:0] i_y,
    output reg  [36:0] o_z
);
    import vu_pkg::*;

    // 1 -------------------------------------------------------------
    reg [36:0]        x1, y1;
    reg               sw1;
    reg signed [12:0] dxy1;
    always @(posedge i_clk) begin
        x1   <= i_x;
        y1   <= i_y;
        sw1  <= ($signed(i_y[35:24]) > $signed(i_x[35:24])) ||
                ((i_y[35:24] == i_x[35:24]) && (i_y[23:0] > i_x[23:0]));
        dxy1 <= $signed({i_x[35], i_x[35:24]}) - $signed({i_y[35], i_y[35:24]});
    end

    // 2 -------------------------------------------------------------
    wire signed [12:0] dd = sw1 ? -dxy1 : dxy1;       // >= 0
    reg               ls2, sub2;
    reg [11:0]        le2;
    reg [23:0]        lm2, sm2;
    reg [4:0]         d2;
    always @(posedge i_clk) begin
        ls2  <= sw1 ? y1[36] : x1[36];
        le2  <= sw1 ? y1[35:24] : x1[35:24];
        lm2  <= sw1 ? y1[23:0] : x1[23:0];
        sm2  <= sw1 ? x1[23:0] : y1[23:0];
        sub2 <= x1[36] ^ y1[36];
        d2   <= (dd > 13'sd27) ? 5'd27 : dd[4:0];
    end

    // 3 -------------------------------------------------------------
    reg               ls3, sub3;
    reg [11:0]        le3;
    reg [26:0]        a3, b3;
    always @(posedge i_clk) begin
        a3   <= {lm2, 3'd0};
        b3   <= {sm2, 3'd0} >> d2;
        ls3  <= ls2;  le3 <= le2;  sub3 <= sub2;
    end

    // 4 -------------------------------------------------------------
    reg               ls4;
    reg [11:0]        le4;
    reg [27:0]        r4;
    always @(posedge i_clk) begin
        r4   <= sub3 ? ({1'b0, a3} - {1'b0, b3}) : ({1'b0, a3} + {1'b0, b3});
        ls4  <= ls3;  le4 <= le3;
    end

    // 5 -------------------------------------------------------------
    wire [6:0]        lz_w = lzc64({r4, 36'hFFFFFFFFF});
    reg               ls5, z5;
    reg [11:0]        le5;
    reg [27:0]        r5;
    reg [4:0]         lz5;
    always @(posedge i_clk) begin
        lz5  <= lz_w[4:0];
        z5   <= (r4 == 28'd0);
        r5   <= r4;
        ls5  <= ls4;  le5 <= le4;
    end

    // 6 -------------------------------------------------------------
    wire [27:0]       n6 = r5 << lz5;
    always @(posedge i_clk)
        o_z <= z5 ? XF_ZERO : {ls5, le5 + 12'd1 - {7'd0, lz5}, n6[27:4]};

endmodule
