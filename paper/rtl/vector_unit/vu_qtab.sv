// ------------------------------------------------------------------
// QUANT row-constant table: 256 x 64-bit words {gR, MR, gS, MS} indexed by
// {exponent parity, 7-bit mantissa} of a row amax code (vu_tbl_quant.mem,
// written by vector_unit_ref.py --tables, derived from quant_setup).
// Two registered reads of one ROM: port A for the element path (R), port B
// for the summary beat (s_row).  Kept in BRAM: Synplify otherwise packs the
// 8-input ROM into LRAM2K, and every used LRAM2K blocks its paired MLP72.
// ------------------------------------------------------------------
module vu_qtab
#(
    parameter F_QUANT = "vu_tbl_quant.mem"
)
(
    input  wire        i_clk,
    input  wire [7:0]  i_addr_a,
    input  wire [7:0]  i_addr_b,
    output reg  [63:0] o_a,
    output reg  [63:0] o_b
) /* synthesis syn_romstyle = "block_rom" */;

    reg [63:0] rom [0:255] /* synthesis syn_romstyle = "block_rom" */;

    initial $readmemh(F_QUANT, rom);

    always @(posedge i_clk) begin
        o_a <= rom[i_addr_a];
        o_b <= rom[i_addr_b];
    end

endmodule
