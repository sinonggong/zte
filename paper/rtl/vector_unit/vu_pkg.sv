// ------------------------------------------------------------------
// pi0 vector-unit lane: shared constants and combinational helpers.
//
// xf = {s, e[11:0] signed, m[23:0]}, value = m * 2^(e-23), m[23] = 1 for
// non-zero; zero is m = 0 with e = EZ.  Bit-exact spec:
// paper/sw/vector_unit_ref.py; op table: paper/rtl/vector_unit/OPS.md.
// ------------------------------------------------------------------
`ifndef VU_PKG_SV
`define VU_PKG_SV

package vu_pkg;

    localparam integer XW = 37;

    localparam [11:0] EZ        = 12'h800;                       // -2048
    localparam [36:0] XF_ZERO   = {1'b0, 12'h800, 24'h000000};
    localparam [36:0] XF_ONE    = {1'b0, 12'h000, 24'h800000};   // 1.0
    localparam [36:0] XF_C127   = {1'b0, 12'h006, 24'hFE0000};   // 127.0
    localparam [36:0] XF_INV127 = {1'b0, 12'hFF9, 24'h810204};   // fp32(1/127) = 0x3C010204
    localparam [36:0] XF_C255   = {1'b0, 12'h007, 24'hFF0000};   // 255.0 (SMAX_Q8 with 255 levels)
    localparam [36:0] XF_INV255 = {1'b0, 12'hFF8, 24'h808081};   // fp32(1/255) = 0x3B808081
    localparam [36:0] XF_EPS    = {1'b0, 12'hFEC, 24'h8637BD};   // fp32(1e-6)  = 0x358637BD

    localparam [3:0] OP_ADD       = 4'd0;
    localparam [3:0] OP_RMS_STAT  = 4'd1;
    localparam [3:0] OP_RMS_APPLY = 4'd2;
    localparam [3:0] OP_LN_STAT   = 4'd3;
    localparam [3:0] OP_LN_APPLY  = 4'd4;
    localparam [3:0] OP_ROPE_A    = 4'd5;
    localparam [3:0] OP_ROPE_B    = 4'd6;
    localparam [3:0] OP_GELU      = 4'd7;
    localparam [3:0] OP_GEGLU     = 4'd8;
    localparam [3:0] OP_SILU      = 4'd9;
    localparam [3:0] OP_SMAX_SUM  = 4'd10;
    localparam [3:0] OP_SMAX_OUT  = 4'd11;
    localparam [3:0] OP_SMAX_Q8   = 4'd12;
    localparam [3:0] OP_QUANT     = 4'd13;
    localparam [3:0] OP_DEQUANT   = 4'd14;
    localparam [3:0] OP_EULER     = 4'd15;

    localparam [1:0] FN_GELU  = 2'd0;
    localparam [1:0] FN_SIGM  = 2'd1;
    localparam [1:0] FN_EXP   = 2'd2;
    localparam [1:0] FN_RSQRT = 2'd3;

    localparam [1:0] FMT_BF16 = 2'd0;
    localparam [1:0] FMT_FP32 = 2'd1;
    localparam [1:0] FMT_INT8 = 2'd2;
    localparam [1:0] FMT_UINT8 = 2'd3;   // 0..255, no sign (uint8 softmax probabilities)

    function automatic [36:0] bf16_to_xf(input [15:0] c);
        if (c[14:7] == 8'd0)
            bf16_to_xf = XF_ZERO;
        else
            bf16_to_xf = {c[15], {4'd0, c[14:7]} - 12'd127, 1'b1, c[6:0], 16'd0};
    endfunction

    function automatic [36:0] fp32_to_xf(input [31:0] c);
        if (c[30:23] == 8'd0)
            fp32_to_xf = XF_ZERO;
        else
            fp32_to_xf = {c[31], {4'd0, c[30:23]} - 12'd127, 1'b1, c[22:0]};
    endfunction

    function automatic [36:0] xf_neg(input [36:0] x);
        xf_neg = (x[23:0] == 24'd0) ? x : {~x[36], x[35:0]};
    endfunction

    function automatic [31:0] xf_to_fp32(input [36:0] x);
        reg signed [12:0] e8;
        begin
            e8 = $signed({x[35], x[35:24]}) + 13'sd127;
            if ((x[23:0] == 24'd0) || (e8 <= 13'sd0))
                xf_to_fp32 = 32'd0;
            else if (e8 >= 13'sd255)
                xf_to_fp32 = {x[36], 31'h7F7FFFFF};
            else
                xf_to_fp32 = {x[36], e8[7:0], x[22:0]};
        end
    endfunction

    // leading-zero count of a 64-bit value (64 for 0), balanced tree.
    // Narrower values are left-aligned by the caller and padded with ones,
    // so the count saturates at their width.
    function automatic [6:0] lzc64(input [63:0] v);
        reg [31:0]      v1;
        reg [31:0]      z1;
        reg [15:0]      v2;
        reg [15:0][1:0] z2;
        reg [7:0]       v3;
        reg [7:0][2:0]  z3;
        reg [3:0]       v4;
        reg [3:0][3:0]  z4;
        reg [1:0]       v5;
        reg [1:0][4:0]  z5;
        reg             v6;
        reg [5:0]       z6;
        integer         i;
        begin
            for (i = 0; i < 32; i = i + 1) begin
                v1[i] = v[2*i+1] | v[2*i];
                z1[i] = ~v[2*i+1];
            end
            for (i = 0; i < 16; i = i + 1) begin
                v2[i] = v1[2*i+1] | v1[2*i];
                z2[i] = v1[2*i+1] ? {1'b0, z1[2*i+1]} : {1'b1, z1[2*i]};
            end
            for (i = 0; i < 8; i = i + 1) begin
                v3[i] = v2[2*i+1] | v2[2*i];
                z3[i] = v2[2*i+1] ? {1'b0, z2[2*i+1]} : {1'b1, z2[2*i]};
            end
            for (i = 0; i < 4; i = i + 1) begin
                v4[i] = v3[2*i+1] | v3[2*i];
                z4[i] = v3[2*i+1] ? {1'b0, z3[2*i+1]} : {1'b1, z3[2*i]};
            end
            for (i = 0; i < 2; i = i + 1) begin
                v5[i] = v4[2*i+1] | v4[2*i];
                z5[i] = v4[2*i+1] ? {1'b0, z4[2*i+1]} : {1'b1, z4[2*i]};
            end
            v6 = v5[1] | v5[0];
            z6 = v5[1] ? {1'b0, z5[1]} : {1'b1, z5[0]};
            lzc64 = v6 ? {1'b0, z6} : 7'd64;
        end
    endfunction

    // bf16 signed greater-than on codes (no NaN handling; +0 > -0)
    function automatic bf16_gt(input [15:0] a, input [15:0] b);
        if (a == b)
            bf16_gt = 1'b0;
        else if (a[15] != b[15])
            bf16_gt = b[15];
        else if (!a[15])
            bf16_gt = a[14:0] > b[14:0];
        else
            bf16_gt = a[14:0] < b[14:0];
    endfunction

endpackage

`endif
