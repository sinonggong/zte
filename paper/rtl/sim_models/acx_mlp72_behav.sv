// Behavioural, cycle-accurate SIMULATION model of the ACX_MLP72 subset used by
// paper/rtl/mlp72_int8_chain.sv.  Same module name, parameter names (the initd_*
// LRAM init parameters excepted) and port list as the vendor primitive
// (ACE 10.3.1 libraries/speedster7t/sim/speedster7t_sim_BRAM72K.sv:5596-5882),
// so the chain compiles against it unchanged.  The vendor core is encrypted;
// this model is written from the documents, not from the core:
//   ug086:NNNN  pdftotext of UG086 "Speedster7t Component Library User Guide"
//   ug088:NNNN  pdftotext of UG088 "Machine Learning Processor User Guide"
//   acx_integer.sv  ACE libraries/speedster7t/common/acx_integer.sv (_ACX_INT_MLP_FABRIC)
// Every behaviour the documents do not pin down carries a tag [An] and is listed
// in paper/rtl/sim_models/ASSUMPTIONS.md.
//
// Modelled subset (anything else stops with $fatal, or, when it only selects an
// unmodelled source, drives a POISON pattern so a wrong use shows up in the sums):
//   input selection   mux_sel_mult{a,b}_{l,h} = MLP_DIN, BRAM_DIN, BRAM_DOUT lo/hi, FWDI
//                     (Table 103, ug086:4381-4418); LRAM_DOUT sources -> POISON
//   stage 0           del_mult{a,b}_{l,h}; fwdo_mult* = stage-0 register output [A1]
//   byte selection    Int8 x4 only: bytesel_00_07=5'h01, bytesel_08_15=6'h21 (Table 107, ug086:4621-4631)
//   stage 1           del_mult00a .. del_mult12_15b on the multiplier inputs (ug086:4897-4899)
//   multipliers       multmode 00 signed 8x8, 01 unsigned, 12/13 mixed sign, 11 NO OP [A4]
//   adder tree        ADD03+ADD47 -> ADD07 (+/-, bypass) -> ADD0_7_REG; ADD811+ADD1215 -> ADD815
//                     -> ADD8_15_REG (Table 123, ug086:5029-5046); ADD015 / add_00_15_sel [A5]
//   AB accumulator    fpadd_ab_dinb_sel 000/001/100, load_ab/sub_ab through del_rndsubload_ab_reg [A3],
//                     add_accum_ab_bypass [A6], ACCUM_AB_REG (feedback always from the register)
//   CD accumulator    fpadd_cd_dina_sel [A7], fpadd_cd_dinb_sel 000/001/100, load/sub through
//                     del_rndsubload_reg, add_accum_cd_bypass, OUT_REG slices (feedback from the register)
//   outputs           out_reg_din_sel 011 [A9], dout_mlp_sel (all), outmode_sel 00/10,
//                     fwdo_dout = dout_mlp_sel value (Table 124, ug086:5142-5158),
//                     mlpram_mlp_dout, mlpram_din = din, mlpram_we = {expb, load_ab}
// Not modelled: LRAM, fp / block-fp, SNOADD and int16/7/6/4/3 modes, asynchronous
// resets, falling-edge clock, DFT, initd_* / mem_init_file.
//
// Compile-time switches that flip one assumption (for sensitivity runs):
//   +define+ACX_MLP72_BEHAV_FWDO_PRE_REG   fwdo_mult* = mux selection before the stage-0 register
`ifndef SYNTHESIS

// One optional delay stage (UG086 "Delay Stage Structure", ug086:4196-4280).
module acx_mlp72_behav_dreg #(
    parameter integer W = 1
) (
    input  wire         clk,
    input  wire         del,      // del_*: 1 register in circuit, 0 bypassed
    input  wire [3:0]   cesel,    // 0 -> 1'b0, 1..12 -> ce[cesel-1], 13 -> 1'b1 (Table 101)
    input  wire [2:0]   rstsel,   // 0 -> 1'b0, 1..4 -> rstn[rstsel-1], 5 -> 1'b1 (Table 101)
    input  wire [11:0]  ce,
    input  wire [3:0]   rstn,
    input  wire [W-1:0] d,
    output reg  [W-1:0] q_reg,    // the register itself (accumulator feedback uses it)
    output wire [W-1:0] q         // the stage output
);
    wire ce_i   = (cesel == 4'd0)  ? 1'b0 :
                  (cesel == 4'd13) ? 1'b1 :
                  (cesel <= 4'd12) ? ce[cesel - 4'd1] : 1'b0;
    wire rstn_i = (rstsel == 3'd0) ? 1'b0 :
                  (rstsel == 3'd5) ? 1'b1 :
                  (rstsel <= 3'd4) ? rstn[rstsel - 3'd1] : 1'b1;
    initial q_reg = '0;
    // Synchronous reset (rst_mode_* default 1'b0, ug086:4270-4274); reset wins over ce [A10].
    always @(posedge clk)
        if (!rstn_i)   q_reg <= '0;
        else if (ce_i) q_reg <= d;
    assign q = del ? q_reg : d;
endmodule

module ACX_MLP72
  #(
    parameter         clk_polarity                   = "rise",
    parameter         lram_rdclk_polarity            = "rise",
    parameter         lram_wrclk_polarity            = "rise",
    parameter [  2:0] mux_sel_multa_h                = 3'h0,
    parameter [  1:0] mux_sel_multa_l                = 2'h0,
    parameter [  2:0] mux_sel_multb_h                = 3'h0,
    parameter [  1:0] mux_sel_multb_l                = 2'h0,
    parameter [  4:0] bytesel_00_07                  = 5'h0,
    parameter [  5:0] bytesel_08_15                  = 6'h0,
    parameter [  4:0] multmode_00_07                 = 5'h0,
    parameter [  4:0] multmode_08_15                 = 5'h0,
    parameter [  3:0] cesel_multa_h                  = 4'h0,
    parameter [  3:0] cesel_multa_l                  = 4'h0,
    parameter [  3:0] cesel_multb_h                  = 4'h0,
    parameter [  3:0] cesel_multb_l                  = 4'h0,
    parameter [  3:0] cesel_mult00a                  = 4'h0,
    parameter [  3:0] cesel_mult00b                  = 4'h0,
    parameter [  3:0] cesel_mult01a                  = 4'h0,
    parameter [  3:0] cesel_mult01b                  = 4'h0,
    parameter [  3:0] cesel_mult02a                  = 4'h0,
    parameter [  3:0] cesel_mult02b                  = 4'h0,
    parameter [  3:0] cesel_mult03a                  = 4'h0,
    parameter [  3:0] cesel_mult03b                  = 4'h0,
    parameter [  3:0] cesel_mult04_07a               = 4'h0,
    parameter [  3:0] cesel_mult04_07b               = 4'h0,
    parameter [  3:0] cesel_mult08_11a               = 4'h0,
    parameter [  3:0] cesel_mult08_11b               = 4'h0,
    parameter [  3:0] cesel_mult12_15a               = 4'h0,
    parameter [  3:0] cesel_mult12_15b               = 4'h0,
    parameter [  3:0] cesel_expta_reg                = 4'h0,
    parameter [  3:0] cesel_exptb_reg                = 4'h0,
    parameter [  3:0] cesel_exptc_reg                = 4'h0,
    parameter [  3:0] cesel_exptd_reg                = 4'h0,
    parameter [  3:0] cesel_add_00_07_reg            = 4'h0,
    parameter [  3:0] cesel_add_08_15_reg            = 4'h0,
    parameter [  3:0] cesel_rndsubload_reg           = 4'h0,
    parameter [  3:0] cesel_rndsubload_ab_reg        = 4'h0,
    parameter [  3:0] cesel_out_reg_00_15            = 4'h0,
    parameter [  3:0] cesel_out_reg_16_31            = 4'h0,
    parameter [  3:0] cesel_out_reg_32_47            = 4'h0,
    parameter [  3:0] cesel_out_reg_48_63            = 4'h0,
    parameter [  3:0] cesel_accum_ab_reg             = 4'h0,
    parameter [  3:0] cesel_expb_din_reg             = 4'h0,
    parameter [  3:0] cesel_fpmult_ab_reg            = 4'h0,
    parameter [  3:0] cesel_fpmult_ab_pipe_reg       = 4'h0,
    parameter [  3:0] cesel_fpmult_cd_pipe_reg       = 4'h0,
    parameter [  3:0] cesel_fp_format_ab_reg         = 4'h0,
    parameter [  3:0] cesel_fp_format_cd_reg         = 4'h0,
    parameter [  2:0] rstsel_multa_h                 = 3'h0,
    parameter [  2:0] rstsel_multa_l                 = 3'h0,
    parameter [  2:0] rstsel_multb_h                 = 3'h0,
    parameter [  2:0] rstsel_multb_l                 = 3'h0,
    parameter [  2:0] rstsel_mult00a                 = 3'h0,
    parameter [  2:0] rstsel_mult00b                 = 3'h0,
    parameter [  2:0] rstsel_mult01a                 = 3'h0,
    parameter [  2:0] rstsel_mult01b                 = 3'h0,
    parameter [  2:0] rstsel_mult02a                 = 3'h0,
    parameter [  2:0] rstsel_mult02b                 = 3'h0,
    parameter [  2:0] rstsel_mult03a                 = 3'h0,
    parameter [  2:0] rstsel_mult03b                 = 3'h0,
    parameter [  2:0] rstsel_mult04_07a              = 3'h0,
    parameter [  2:0] rstsel_mult04_07b              = 3'h0,
    parameter [  2:0] rstsel_mult08_11a              = 3'h0,
    parameter [  2:0] rstsel_mult08_11b              = 3'h0,
    parameter [  2:0] rstsel_mult12_15a              = 3'h0,
    parameter [  2:0] rstsel_mult12_15b              = 3'h0,
    parameter [  2:0] rstsel_expta_reg               = 3'h0,
    parameter [  2:0] rstsel_exptb_reg               = 3'h0,
    parameter [  2:0] rstsel_exptc_reg               = 3'h0,
    parameter [  2:0] rstsel_exptd_reg               = 3'h0,
    parameter [  2:0] rstsel_add_00_07_reg           = 3'h0,
    parameter [  2:0] rstsel_add_08_15_reg           = 3'h0,
    parameter [  2:0] rstsel_rndsubload_reg          = 3'h0,
    parameter [  2:0] rstsel_rndsubload_ab_reg       = 3'h0,
    parameter [  2:0] rstsel_out_reg_00_15           = 3'h0,
    parameter [  2:0] rstsel_out_reg_16_31           = 3'h0,
    parameter [  2:0] rstsel_out_reg_32_47           = 3'h0,
    parameter [  2:0] rstsel_out_reg_48_63           = 3'h0,
    parameter [  2:0] rstsel_accum_ab_reg            = 3'h0,
    parameter [  2:0] rstsel_expb_din_reg            = 3'h0,
    parameter [  2:0] rstsel_fpmult_ab_reg           = 3'h0,
    parameter [  2:0] rstsel_fpmult_ab_pipe_reg      = 3'h0,
    parameter [  2:0] rstsel_fpmult_cd_pipe_reg      = 3'h0,
    parameter [  2:0] rstsel_fp_format_ab_reg        = 3'h0,
    parameter [  2:0] rstsel_fp_format_cd_reg        = 3'h0,
    parameter         rst_mode_mult00a               = 1'h0,
    parameter         rst_mode_mult00b               = 1'h0,
    parameter         rst_mode_mult01a               = 1'h0,
    parameter         rst_mode_mult01b               = 1'h0,
    parameter         rst_mode_mult02a               = 1'h0,
    parameter         rst_mode_mult02b               = 1'h0,
    parameter         rst_mode_mult03a               = 1'h0,
    parameter         rst_mode_mult03b               = 1'h0,
    parameter         rst_mode_out_reg_00_15         = 1'h0,
    parameter         rst_mode_out_reg_16_31         = 1'h0,
    parameter         rst_mode_out_reg_32_47         = 1'h0,
    parameter         rst_mode_out_reg_48_63         = 1'h0,
    parameter         del_multa_h                    = 1'h0,
    parameter         del_multa_l                    = 1'h0,
    parameter         del_multb_h                    = 1'h0,
    parameter         del_multb_l                    = 1'h0,
    parameter         del_mult00a                    = 1'h0,
    parameter         del_mult00b                    = 1'h0,
    parameter         del_mult01a                    = 1'h0,
    parameter         del_mult01b                    = 1'h0,
    parameter         del_mult02a                    = 1'h0,
    parameter         del_mult02b                    = 1'h0,
    parameter         del_mult03a                    = 1'h0,
    parameter         del_mult03b                    = 1'h0,
    parameter         del_mult04_07a                 = 1'h0,
    parameter         del_mult04_07b                 = 1'h0,
    parameter         del_mult08_11a                 = 1'h0,
    parameter         del_mult08_11b                 = 1'h0,
    parameter         del_mult12_15a                 = 1'h0,
    parameter         del_mult12_15b                 = 1'h0,
    parameter         del_add_00_07_reg              = 1'h0,
    parameter         del_add_08_15_reg              = 1'h0,
    parameter         del_fpmult_ab_reg              = 1'h0,
    parameter         del_fpmult_ab_pipe_reg         = 1'h0,
    parameter         del_fpmult_cd_pipe_reg         = 1'h0,
    parameter         del_fp_format_ab_reg           = 1'h0,
    parameter         del_fp_format_cd_reg           = 1'h0,
    parameter         add_00_07_bypass               = 1'h0,
    parameter         add_00_07_sub                  = 1'b0,
    parameter         add_08_15_bypass               = 1'h0,
    parameter         add_08_15_sub                  = 1'b0,
    parameter         add_00_15_sel                  = 1'h0,
    parameter [  1:0] del_expa_reg                   = 2'h0,
    parameter [  1:0] del_expb_reg                   = 2'h0,
    parameter [  1:0] del_expc_reg                   = 2'h0,
    parameter [  1:0] del_expd_reg                   = 2'h0,
    parameter [  2:0] del_rndsubload_reg             = 3'h0,
    parameter [  2:0] del_rndsubload_ab_reg          = 3'h0,
    parameter         del_out_reg_00_15              = 1'h0,
    parameter         del_out_reg_16_31              = 1'h0,
    parameter         del_out_reg_32_47              = 1'h0,
    parameter         del_out_reg_48_63              = 1'h0,
    parameter         del_accum_ab_reg               = 1'h0,
    parameter         del_expb_din_reg               = 1'h0,
    parameter         lram_sync_mode                 = 1'h0,
    parameter         lram_reg_dout                  = 1'h0,
    parameter         lram_sr_assertion              = 1'h0,
    parameter         lram_fifo_enable               = 1'h0,
    parameter [  6:0] lram_fifo_wrptr_rstval         = 7'h0,
    parameter [  6:0] lram_fifo_rdptr_rstval         = 7'h0,
    parameter [  6:0] lram_fifo_wrptr_maxval         = 7'h0,
    parameter [  6:0] lram_fifo_rdptr_maxval         = 7'h0,
    parameter [  1:0] lram_input_control_mode        = 2'h0,
    parameter [  1:0] lram_output_control_mode       = 2'h0,
    parameter [  1:0] lram_write_data_mode           = 2'h0,
    parameter         lram_enable_write_via_bram     = 1'h0,
    parameter         lram_accum_data_input_sel      = 1'h0,
    parameter         lram_out2multb_l               = 1'h0,
    parameter         lram_out2multb_h               = 1'h0,
    parameter [  1:0] lram_write_width               = 2'h0,
    parameter [  1:0] lram_read_width                = 2'h0,
    parameter         lram_clear_enable              = 1'b0,
    parameter         lram_fifo_ignore_flags         = 1'b0,
    parameter         lram_fifo_sync_mode            = 1'h0,
    parameter         lram_fifo_num_sync_stages_w2r  = 1'h0,
    parameter         lram_fifo_num_sync_stages_r2w  = 1'h0,
    parameter [  1:0] lram_fifo_out_modeb            = 2'h0,
    parameter         lram_fifo_fast_ef              = 1'h0,
    parameter [  6:0] lram_fifo_afull_threshold      = 7'h0,
    parameter [  6:0] lram_fifo_aempty_threshold     = 7'h0,
    parameter         lram_fifo_fwft_mode            = 1'h0,
    parameter [  3:0] lram_clk_pulse_sel             = 4'h0,
    parameter         lram_clk_sel_wr                = 1'h0,
    parameter         lram_clk_sel_rd                = 1'h0,
    parameter         fpmult_ab_bypass               = 1'h0,
    parameter         fpmult_ab_blockfp              = 1'h0,
    parameter [  2:0] fpmult_ab_blockfp_mode         = 3'h0,
    parameter         fpmult_ab_exp_size             = 1'h0,
    parameter         fpmult_cd_bypass               = 1'h0,
    parameter         fpmult_cd_blockfp              = 1'h0,
    parameter [  2:0] fpmult_cd_blockfp_mode         = 3'h0,
    parameter         fpmult_cd_exp_size             = 1'h0,
    parameter         rndsubload_share               = 1'h0,
    parameter         fpadd_cd_dina_sel              = 1'h0,
    parameter [  2:0] fpadd_cd_dinb_sel              = 3'h7,
    parameter [  2:0] fpadd_ab_dinb_sel              = 3'h7,
    parameter         fpadd_ab_nornd                 = 1'b0,
    parameter         fpadd_cd_nornd                 = 1'b0,
    parameter         fpadd_abcd_sel                 = 1'b0,
    parameter         add_accum_ab_bypass            = 1'h0,
    parameter         add_accum_cd_bypass            = 1'h0,
    parameter [  1:0] fpadd_ab_output_format         = 2'h0,
    parameter [  1:0] fpadd_cd_output_format         = 2'h0,
    parameter [  2:0] out_reg_din_sel                = 3'h0,
    parameter         accum_ab_reg_din_sel           = 1'h0,
    parameter [  1:0] dout_mlp_sel                   = 2'h0,
    parameter [  1:0] outmode_sel                    = 2'h0,
    parameter         mem_init_file                  = "",
    parameter         location                       = ""
    )
   (
    input wire [ 71: 0] din,
    input wire          load_ab,
    input wire          sub_ab,
    input wire          sub,
    input wire          load,
    input wire [ 11:0]  ce,
    input wire [ 3:0]   rstn,
    input wire [ 7:0]   expb,
    input wire          clk,
    input wire          lram_wrclk,
    input wire          lram_rdclk,

    output [ 71:0]      dout,
    output              empty,
    output              full,
    output              almost_empty,
    output              almost_full,
    output              write_error,
    output              read_error,
    output              sbit_error,
    output              dbit_error,

    input wire [ 71:0]  fwdi_multa_h,
    input wire [ 71:0]  fwdi_multa_l,
    input wire [ 71:0]  fwdi_multb_h,
    input wire [ 71:0]  fwdi_multb_l,
    input wire [ 47:0]  fwdi_dout,

    input wire [ 71:0]  mlpram_bramdin2mlpdin,
    input wire [143:0]  mlpram_bramdout2mlp,
    input wire          mlpram_sbit_error,
    input wire          mlpram_dbit_error,
    input wire [  5:0]  mlpram_wraddr,
    input wire [143:0]  mlpram_din2mlpdout,
    input wire          mlpram_wren,
    input wire [  5:0]  mlpram_rdaddr,
    input wire          mlpram_rden,

    output [ 71:0]      fwdo_multa_h,
    output [ 71:0]      fwdo_multa_l,
    output [ 71:0]      fwdo_multb_h,
    output [ 71:0]      fwdo_multb_l,
    output [ 47:0]      fwdo_dout,
    output [ 71:0]      mlpram_din,
    output [ 8:0]       mlpram_we,
    output [143:0]      mlpram_dout,
    output [ 95:0]      mlpram_mlp_dout
   );

    // A POISON byte is 8'hD5 = -43, so a poisoned operand gives non-zero products.
    localparam logic [71:0] POISON72 = {8'hBA, {8{8'hD5}}};
    localparam logic [47:0] POISON48 = 48'h0BAD_0BAD_0BAD;

    // ------------------------------------------------------------------ config check
    function automatic bit mm_ok(input logic [4:0] m);
        return (m == 5'h00) || (m == 5'h01) || (m == 5'h11) || (m == 5'h12) || (m == 5'h13);
    endfunction
    initial begin
        if (clk_polarity != "rise")
            $fatal(1, "%m ACX_MLP72_BEHAV: only clk_polarity=\"rise\" is modelled");
        if (lram_out2multb_l != 1'b0 || lram_out2multb_h != 1'b0)
            $fatal(1, "%m ACX_MLP72_BEHAV: lram_out2multb_* (LRAM) not modelled");
        if (rndsubload_share != 1'b0)
            $fatal(1, "%m ACX_MLP72_BEHAV: rndsubload_share=1 not modelled");
        if (fpmult_ab_bypass != 1'b1 || fpmult_cd_bypass != 1'b1 || accum_ab_reg_din_sel != 1'b0)
            $fatal(1, "%m ACX_MLP72_BEHAV: only integer mode (fpmult_*_bypass=1, accum_ab_reg_din_sel=0) is modelled");
        if (del_fpmult_ab_reg || del_fpmult_ab_pipe_reg || del_fpmult_cd_pipe_reg ||
            del_fp_format_ab_reg || del_fp_format_cd_reg || del_expb_din_reg)
            $fatal(1, "%m ACX_MLP72_BEHAV: floating-point pipeline registers not modelled");
        if (bytesel_00_07 != 5'h01 || bytesel_08_15 != 6'h21)
            $fatal(1, "%m ACX_MLP72_BEHAV: only Int8 x4 byte selection (5'h01 / 6'h21) is modelled");
        if (!mm_ok(multmode_00_07) || !mm_ok(multmode_08_15))
            $fatal(1, "%m ACX_MLP72_BEHAV: multmode %h/%h not modelled", multmode_00_07, multmode_08_15);
        if (del_rndsubload_ab_reg > 3'd5 || del_rndsubload_reg > 3'd6)
            $fatal(1, "%m ACX_MLP72_BEHAV: del_rndsubload_ab_reg <= 5, del_rndsubload_reg <= 6");
        if (mux_sel_multa_l == 2'b01 || mux_sel_multb_l == 2'b01 ||
            mux_sel_multa_h == 3'b010 || mux_sel_multa_h == 3'b011 ||
            mux_sel_multb_h == 3'b010 || mux_sel_multb_h == 3'b011)
            $display("%m ACX_MLP72_BEHAV note: an LRAM_DOUT operand source is selected -> POISON");
    end

    // ------------------------------------------------------------------ input selection
    // UG086 Table 103 (ug086:4381-4418).  BRAM_DIN = mlpram_bramdin2mlpdin, BRAM_DOUT =
    // mlpram_bramdout2mlp (ug086:4495-4503).
    reg [71:0] sel_multa_l, sel_multa_h, sel_multb_l, sel_multb_h;
    always @* begin
        case (mux_sel_multa_l)
            2'b00:   sel_multa_l = din;
            2'b10:   sel_multa_l = mlpram_bramdout2mlp[71:0];
            2'b11:   sel_multa_l = fwdi_multa_l;
            default: sel_multa_l = POISON72;                       // LRAM_DOUT[71:0]
        endcase
        case (mux_sel_multa_h)
            3'b000:  sel_multa_h = din;
            3'b001:  sel_multa_h = mlpram_bramdin2mlpdin;
            3'b100:  sel_multa_h = mlpram_bramdout2mlp[71:0];
            3'b101:  sel_multa_h = mlpram_bramdout2mlp[143:72];
            3'b110:  sel_multa_h = fwdi_multa_l;
            3'b111:  sel_multa_h = fwdi_multa_h;
            default: sel_multa_h = POISON72;                       // LRAM_DOUT
        endcase
        case (mux_sel_multb_l)
            2'b00:   sel_multb_l = din;
            2'b10:   sel_multb_l = mlpram_bramdout2mlp[71:0];
            2'b11:   sel_multb_l = fwdi_multb_l;
            default: sel_multb_l = POISON72;
        endcase
        case (mux_sel_multb_h)
            3'b000:  sel_multb_h = din;
            3'b001:  sel_multb_h = mlpram_bramdin2mlpdin;
            3'b100:  sel_multb_h = mlpram_bramdout2mlp[71:0];
            3'b101:  sel_multb_h = mlpram_bramdout2mlp[143:72];
            3'b110:  sel_multb_h = fwdi_multb_l;
            3'b111:  sel_multb_h = fwdi_multb_h;
            default: sel_multb_h = POISON72;
        endcase
    end

    // ------------------------------------------------------------------ stage 0 registers
    wire [71:0] s0_multa_l, s0_multa_h, s0_multb_l, s0_multb_h;
    acx_mlp72_behav_dreg #(72) u_s0_multa_l (.clk(clk), .del(del_multa_l), .cesel(cesel_multa_l),
        .rstsel(rstsel_multa_l), .ce(ce), .rstn(rstn), .d(sel_multa_l), .q_reg(), .q(s0_multa_l));
    acx_mlp72_behav_dreg #(72) u_s0_multa_h (.clk(clk), .del(del_multa_h), .cesel(cesel_multa_h),
        .rstsel(rstsel_multa_h), .ce(ce), .rstn(rstn), .d(sel_multa_h), .q_reg(), .q(s0_multa_h));
    acx_mlp72_behav_dreg #(72) u_s0_multb_l (.clk(clk), .del(del_multb_l), .cesel(cesel_multb_l),
        .rstsel(rstsel_multb_l), .ce(ce), .rstn(rstn), .d(sel_multb_l), .q_reg(), .q(s0_multb_l));
    acx_mlp72_behav_dreg #(72) u_s0_multb_h (.clk(clk), .del(del_multb_h), .cesel(cesel_multb_h),
        .rstsel(rstsel_multb_h), .ce(ce), .rstn(rstn), .d(sel_multb_h), .q_reg(), .q(s0_multb_h));

    // [A1] The forward cascade carries the stage-0 register OUTPUT.  UG086 only says
    // "the selection from mux_sel" (ug086:4322-4323, 4533-4547); UG088 ch.6 says that
    // with del_mult* enabled "each MLP72 in the column [processes] the data one cycle
    // after the MLP72 below it" and that data traverses the chain at ~750 MHz
    // (ug088:772-777).  Tapping before the register would make the column one
    // combinational path with no per-MLP delay, contradicting both statements.
`ifdef ACX_MLP72_BEHAV_FWDO_PRE_REG
    assign fwdo_multa_l = sel_multa_l;
    assign fwdo_multa_h = sel_multa_h;
    assign fwdo_multb_l = sel_multb_l;
    assign fwdo_multb_h = sel_multb_h;
`else
    assign fwdo_multa_l = s0_multa_l;
    assign fwdo_multa_h = s0_multa_h;
    assign fwdo_multb_l = s0_multb_l;
    assign fwdo_multb_h = s0_multb_h;
`endif

    // ------------------------------------------------------------------ byte selection
    // Int8 x4 (Table 107, ug086:4621-4631): multiplier k takes byte k of multa_l/multb_l
    // for k = 0..7 and byte k-8 of multa_h/multb_h for k = 8..15; byte k of a bus is
    // [8k+7:8k]; bits [71:64] are unused.
    wire [127:0] byteA = {s0_multa_h[63:0], s0_multa_l[63:0]};
    wire [127:0] byteB = {s0_multb_h[63:0], s0_multb_l[63:0]};

    // ------------------------------------------------------------------ stage 1 registers
    // "The input to each integer multiplier supports an optional delay stage.  For
    // multipliers[3:0] each individual input has its own delay stage control ... For
    // multipliers[15:4], the delay stages are controlled in banks of 4" (ug086:4897-4899).
    wire [127:0] s1A, s1B;
    acx_mlp72_behav_dreg #(8)  u_s1_00a (.clk(clk), .del(del_mult00a), .cesel(cesel_mult00a), .rstsel(rstsel_mult00a), .ce(ce), .rstn(rstn), .d(byteA[7:0]),    .q_reg(), .q(s1A[7:0]));
    acx_mlp72_behav_dreg #(8)  u_s1_01a (.clk(clk), .del(del_mult01a), .cesel(cesel_mult01a), .rstsel(rstsel_mult01a), .ce(ce), .rstn(rstn), .d(byteA[15:8]),   .q_reg(), .q(s1A[15:8]));
    acx_mlp72_behav_dreg #(8)  u_s1_02a (.clk(clk), .del(del_mult02a), .cesel(cesel_mult02a), .rstsel(rstsel_mult02a), .ce(ce), .rstn(rstn), .d(byteA[23:16]),  .q_reg(), .q(s1A[23:16]));
    acx_mlp72_behav_dreg #(8)  u_s1_03a (.clk(clk), .del(del_mult03a), .cesel(cesel_mult03a), .rstsel(rstsel_mult03a), .ce(ce), .rstn(rstn), .d(byteA[31:24]),  .q_reg(), .q(s1A[31:24]));
    acx_mlp72_behav_dreg #(32) u_s1_04a (.clk(clk), .del(del_mult04_07a), .cesel(cesel_mult04_07a), .rstsel(rstsel_mult04_07a), .ce(ce), .rstn(rstn), .d(byteA[63:32]),  .q_reg(), .q(s1A[63:32]));
    acx_mlp72_behav_dreg #(32) u_s1_08a (.clk(clk), .del(del_mult08_11a), .cesel(cesel_mult08_11a), .rstsel(rstsel_mult08_11a), .ce(ce), .rstn(rstn), .d(byteA[95:64]),  .q_reg(), .q(s1A[95:64]));
    acx_mlp72_behav_dreg #(32) u_s1_12a (.clk(clk), .del(del_mult12_15a), .cesel(cesel_mult12_15a), .rstsel(rstsel_mult12_15a), .ce(ce), .rstn(rstn), .d(byteA[127:96]), .q_reg(), .q(s1A[127:96]));
    acx_mlp72_behav_dreg #(8)  u_s1_00b (.clk(clk), .del(del_mult00b), .cesel(cesel_mult00b), .rstsel(rstsel_mult00b), .ce(ce), .rstn(rstn), .d(byteB[7:0]),    .q_reg(), .q(s1B[7:0]));
    acx_mlp72_behav_dreg #(8)  u_s1_01b (.clk(clk), .del(del_mult01b), .cesel(cesel_mult01b), .rstsel(rstsel_mult01b), .ce(ce), .rstn(rstn), .d(byteB[15:8]),   .q_reg(), .q(s1B[15:8]));
    acx_mlp72_behav_dreg #(8)  u_s1_02b (.clk(clk), .del(del_mult02b), .cesel(cesel_mult02b), .rstsel(rstsel_mult02b), .ce(ce), .rstn(rstn), .d(byteB[23:16]),  .q_reg(), .q(s1B[23:16]));
    acx_mlp72_behav_dreg #(8)  u_s1_03b (.clk(clk), .del(del_mult03b), .cesel(cesel_mult03b), .rstsel(rstsel_mult03b), .ce(ce), .rstn(rstn), .d(byteB[31:24]),  .q_reg(), .q(s1B[31:24]));
    acx_mlp72_behav_dreg #(32) u_s1_04b (.clk(clk), .del(del_mult04_07b), .cesel(cesel_mult04_07b), .rstsel(rstsel_mult04_07b), .ce(ce), .rstn(rstn), .d(byteB[63:32]),  .q_reg(), .q(s1B[63:32]));
    acx_mlp72_behav_dreg #(32) u_s1_08b (.clk(clk), .del(del_mult08_11b), .cesel(cesel_mult08_11b), .rstsel(rstsel_mult08_11b), .ce(ce), .rstn(rstn), .d(byteB[95:64]),  .q_reg(), .q(s1B[95:64]));
    acx_mlp72_behav_dreg #(32) u_s1_12b (.clk(clk), .del(del_mult12_15b), .cesel(cesel_mult12_15b), .rstsel(rstsel_mult12_15b), .ce(ce), .rstn(rstn), .d(byteB[127:96]), .q_reg(), .q(s1B[127:96]));

    // ------------------------------------------------------------------ multipliers
    // Table 123 (ug086:4979-5026).  [A4] NO OP (5'h11) is modelled as a zero product.
    function automatic logic [47:0] f_mult8(input logic [4:0] mode, input logic [7:0] a,
                                            input logic [7:0] b);
        logic signed [17:0] p;
        case (mode)
            5'h00:   p = $signed({{10{a[7]}}, a}) * $signed({{10{b[7]}}, b});   // SIGNED 8x8
            5'h01:   p = $signed({10'd0, a})      * $signed({10'd0, b});        // UNSIGNED 8x8
            5'h12:   p = $signed({{10{a[7]}}, a}) * $signed({10'd0, b});        // A signed, B unsigned
            5'h13:   p = $signed({10'd0, a})      * $signed({{10{b[7]}}, b});   // A unsigned, B signed
            default: p = '0;                                                     // 5'h11 NO OP
        endcase
        return {{30{p[17]}}, p};
    endfunction

    wire [47:0] prod [0:15];
    genvar k;
    for (k = 0; k < 16; k = k + 1) begin : g_mult
        assign prod[k] = f_mult8((k < 8) ? multmode_00_07 : multmode_08_15, s1A[8*k +: 8], s1B[8*k +: 8]);
    end

    // ------------------------------------------------------------------ adder tree, stage 2
    // "Within each bank, there are 8 multipliers which are summed as two groups of 4.
    // These intermediate sums are then optionally summed, or subtracted" (ug086:4894-4896);
    // add_00_07_bypass selects ADD03 or ADD07 into ADD0_7_REG (ug086:5029-5046).
    // [A5] Sums are carried at 48 bits with no internal saturation or truncation.
    wire [47:0] add03  = prod[0]  + prod[1]  + prod[2]  + prod[3];
    wire [47:0] add47  = prod[4]  + prod[5]  + prod[6]  + prod[7];
    wire [47:0] add811 = prod[8]  + prod[9]  + prod[10] + prod[11];
    wire [47:0] add1215= prod[12] + prod[13] + prod[14] + prod[15];
    wire [47:0] add07  = add_00_07_sub ? (add03  - add47)   : (add03  + add47);
    wire [47:0] add815 = add_08_15_sub ? (add811 - add1215) : (add811 + add1215);
    wire [47:0] s2_lo, s2_hi;
    acx_mlp72_behav_dreg #(48) u_add_00_07_reg (.clk(clk), .del(del_add_00_07_reg), .cesel(cesel_add_00_07_reg),
        .rstsel(rstsel_add_00_07_reg), .ce(ce), .rstn(rstn), .d(add_00_07_bypass ? add03 : add07), .q_reg(), .q(s2_lo));
    acx_mlp72_behav_dreg #(48) u_add_08_15_reg (.clk(clk), .del(del_add_08_15_reg), .cesel(cesel_add_08_15_reg),
        .rstsel(rstsel_add_08_15_reg), .ce(ce), .rstn(rstn), .d(add_08_15_bypass ? add811 : add815), .q_reg(), .q(s2_hi));

    // add_00_15_sel: "0 - ADD0_7_REG output ... 1 - ADD015 output is routed toward
    // FPMULT_AB_REG" (ug086:5080-5082).  [A5] ADD015 = ADD0_7_REG + ADD8_15_REG, after the
    // stage-2 registers, and no register between it and the accumulators in integer mode
    // with del_fpmult_ab_reg = 0.
    wire [47:0] add015    = s2_lo + s2_hi;
    wire [47:0] ab_in     = add_00_15_sel ? add015 : s2_lo;
    // [A7] The CD half's own integer input is the add[15:8] bank sum (Table 125 "load ...
    // with the add[15:8] sum", ug086:5188-5190).
    wire [47:0] cd_int_in = s2_hi;

    // ------------------------------------------------------------------ load / sub delay match
    // [A3] del_rndsubload_ab_reg / del_rndsubload_reg are pure delay lines of that many
    // registers (ranges 0..5 / 0..6, acx_integer.sv:324-325).  acx_integer.sv:130-131 sets
    // them to the number of data registers between the MLP inputs and the accumulator, so
    // the load pin is driven in the same cycle as the operands at the MLP inputs.
    wire [1:0] rsl_ab [0:5];
    wire [1:0] rsl_cd [0:6];
    assign rsl_ab[0] = {load_ab, sub_ab};
    assign rsl_cd[0] = {load, sub};
    for (k = 0; k < 5; k = k + 1) begin : g_rsl_ab
        acx_mlp72_behav_dreg #(2) u (.clk(clk), .del(1'b1), .cesel(cesel_rndsubload_ab_reg),
            .rstsel(rstsel_rndsubload_ab_reg), .ce(ce), .rstn(rstn), .d(rsl_ab[k]), .q_reg(), .q(rsl_ab[k+1]));
    end
    for (k = 0; k < 6; k = k + 1) begin : g_rsl_cd
        acx_mlp72_behav_dreg #(2) u (.clk(clk), .del(1'b1), .cesel(cesel_rndsubload_reg),
            .rstsel(rstsel_rndsubload_reg), .ce(ce), .rstn(rstn), .d(rsl_cd[k]), .q_reg(), .q(rsl_cd[k+1]));
    end
    wire load_ab_d = rsl_ab[del_rndsubload_ab_reg][1];
    wire sub_ab_d  = rsl_ab[del_rndsubload_ab_reg][0];
    wire load_cd_d = rsl_cd[del_rndsubload_reg][1];
    wire sub_cd_d  = rsl_cd[del_rndsubload_reg][0];

    // ------------------------------------------------------------------ AB accumulator
    wire [47:0] accum_ab_q, accum_ab;
    reg  [47:0] ab_dinb;
    always @* begin
        case (fpadd_ab_dinb_sel)                                   // ug086:5110-5116
            3'b000:  ab_dinb = accum_ab_q;                         // ACCUM_AB_REG (always registered)
            3'b001:  ab_dinb = fwdi_dout;                          // FWDI_DOUT[47:0]
            3'b100:  ab_dinb = {24'h0, fwdi_dout[47:24]};          // FWDI_DOUT[47:24]
            default: ab_dinb = POISON48;                           // LRAM sources
        endcase
    end
    // load_ab: "load the accumulator with the output of the add_00_15_sel multiplexer"
    // (ug086:5193-5194).  [A8] sub_ab: dinb - dina; load wins over sub.
    wire [47:0] ab_add = load_ab_d ? ab_in : (sub_ab_d ? (ab_dinb - ab_in) : (ab_dinb + ab_in));
    // [A6] add_accum_ab_bypass = 1 passes the adder input (add_00_15_sel mux) straight to
    // ACCUM_AB_REG; dinb, load_ab and sub_ab have no effect (acx_integer.sv:109, 288 use
    // bypass for "no accumulate").
    wire [47:0] ab_d   = add_accum_ab_bypass ? ab_in : ab_add;
    // The feedback path always uses the register, even when the output bypasses it
    // (acx_integer.sv:210-211).
    acx_mlp72_behav_dreg #(48) u_accum_ab_reg (.clk(clk), .del(del_accum_ab_reg), .cesel(cesel_accum_ab_reg),
        .rstsel(rstsel_accum_ab_reg), .ce(ce), .rstn(rstn), .d(ab_d), .q_reg(accum_ab_q), .q(accum_ab));

    // ------------------------------------------------------------------ CD accumulator
    // [A7] fpadd_cd_dina_sel = 1 takes "the value from (A*B) Accumulator" (ug086:5095-5098)
    // = the optionally registered ACCUM_AB_REG output (in-repo precedent:
    // src/rtl/matrix_engine/stack_stage_fp.sv:278 adds del_accum_ab to the CD load delay).
    wire [47:0] cd_dina = fpadd_cd_dina_sel ? accum_ab : cd_int_in;
    wire [63:0] out_reg_q, out_reg;
    reg  [47:0] cd_dinb;
    always @* begin
        case (fpadd_cd_dinb_sel)                                   // ug086:5101-5107
            3'b000:  cd_dinb = out_reg_q[47:0];                    // ACCUM_CD_REG = OUT_REG (registered)
            3'b001:  cd_dinb = fwdi_dout;                          // FWDI_DOUT[47:0]
            3'b100:  cd_dinb = accum_ab;                           // AB accumulator output
            default: cd_dinb = POISON48;                           // LRAM / reserved
        endcase
    end
    wire [47:0] cd_add = load_cd_d ? cd_dina : (sub_cd_d ? (cd_dinb - cd_dina) : (cd_dinb + cd_dina));
    wire [47:0] cd_out = add_accum_cd_bypass ? cd_dina : cd_add;   // [A6] same reading as AB

    reg [63:0] out_reg_d;
    always @* begin
        case (out_reg_din_sel)                                     // ug086:5129-5135
            3'b011:  out_reg_d = {16'h0000, cd_out};               // [A9] integer CD accumulator, [63:48] = 0
            default: out_reg_d = {16'h0BAD, POISON48};             // Mult8x4 / fp / A+-B / Mult16x2
        endcase
    end
    acx_mlp72_behav_dreg #(16) u_out_reg_00_15 (.clk(clk), .del(del_out_reg_00_15), .cesel(cesel_out_reg_00_15),
        .rstsel(rstsel_out_reg_00_15), .ce(ce), .rstn(rstn), .d(out_reg_d[15:0]),  .q_reg(out_reg_q[15:0]),  .q(out_reg[15:0]));
    acx_mlp72_behav_dreg #(16) u_out_reg_16_31 (.clk(clk), .del(del_out_reg_16_31), .cesel(cesel_out_reg_16_31),
        .rstsel(rstsel_out_reg_16_31), .ce(ce), .rstn(rstn), .d(out_reg_d[31:16]), .q_reg(out_reg_q[31:16]), .q(out_reg[31:16]));
    acx_mlp72_behav_dreg #(16) u_out_reg_32_47 (.clk(clk), .del(del_out_reg_32_47), .cesel(cesel_out_reg_32_47),
        .rstsel(rstsel_out_reg_32_47), .ce(ce), .rstn(rstn), .d(out_reg_d[47:32]), .q_reg(out_reg_q[47:32]), .q(out_reg[47:32]));
    acx_mlp72_behav_dreg #(16) u_out_reg_48_63 (.clk(clk), .del(del_out_reg_48_63), .cesel(cesel_out_reg_48_63),
        .rstsel(rstsel_out_reg_48_63), .ce(ce), .rstn(rstn), .d(out_reg_d[63:48]), .q_reg(out_reg_q[63:48]), .q(out_reg[63:48]));

    // ------------------------------------------------------------------ outputs
    // dout_mlp_sel "Select values for the forward DOUT cascade path" (ug086:5142-5158);
    // outmode_sel 00 = "72-bit output of value selected by dout_mlp_sel" (ug086:5161-5167).
    reg [71:0] dout_mlp;
    always @* begin
        case (dout_mlp_sel)
            2'b00:   dout_mlp = {8'h00, out_reg};                          // [A9] [71:64] = 0
            2'b01:   dout_mlp = {24'h0, accum_ab[23:0], out_reg[23:0]};
            2'b10:   dout_mlp = {24'h0, accum_ab};
            default: dout_mlp = {accum_ab[35:0], out_reg[35:0]};
        endcase
    end
    assign fwdo_dout = dout_mlp[47:0];
    assign dout      = (outmode_sel == 2'b00) ? dout_mlp :
                       (outmode_sel == 2'b10) ? mlpram_bramdout2mlp[143:72] : POISON72;
    // Bits[47:0] result, Bits[95:48] AB sum path (Table 125, ug086:5214-5218).
    assign mlpram_mlp_dout = {accum_ab, dout_mlp[47:0]};

    // Shared site pins toward the co-sited BRAM72K (vendor wrapper
    // speedster7t_sim_BRAM72K.sv:5884-5885): the MLP's din is the BRAM's upper write
    // data and {expb, load_ab} are its upper 9 write enables.
    assign mlpram_din  = din;
    assign mlpram_we   = {expb, load_ab};
    assign mlpram_dout = 144'h0;                                   // LRAM output: not modelled
    // ECC pass-through from a wide BRAM72K (acx_integer.sv:421-423).
    assign sbit_error  = mlpram_sbit_error;
    assign dbit_error  = mlpram_dbit_error;
    assign empty = 1'b1, full = 1'b0, almost_empty = 1'b1, almost_full = 1'b0,
           write_error = 1'b0, read_error = 1'b0;

endmodule

`endif
