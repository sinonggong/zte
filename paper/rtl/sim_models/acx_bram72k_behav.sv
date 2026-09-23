// Behavioural, cycle-accurate SIMULATION model of ACX_BRAM72K in the mode used by
// paper/rtl/mlp72_int8_chain.sv: 512 x 144-bit words written from the fabric
// (din + the co-sited MLP72's din/expb/load_ab pins), read synchronously with
// the optional output register, the read data also on the dedicated
// mlpram_dout2mlp path, and the read address optionally taken from the forward
// read-address cascade (fwdi/fwdo_ram_rd_addr).  Same module name, parameter
// names (initd_* excepted) and port list as the vendor primitive (ACE 10.3.1
// libraries/speedster7t/sim/speedster7t_sim_BRAM72K.sv:2995-4219).  The vendor
// memory core is encrypted; sources:
//   ug086:NNNN  pdftotext of UG086 (ACX_BRAM72K_SDP section; the full ACX_BRAM72K
//               with its cascade ports is not documented in UG086)
//   ug088:NNNN  pdftotext of UG088 chapter 6
//   wrapper     speedster7t_sim_BRAM72K.sv (ACX_BRAM72K glue, unencrypted) and
//               speedster7t_sim_BRAM72K_COMMON.sv (address-source decode)
//   GEN_DEEP    libraries/speedster7t/macros/ACX_BRAM72K_GEN_DEEP.sv
// Tags [Bn] are listed in paper/rtl/sim_models/ASSUMPTIONS.md.
//
// Not modelled (-> $fatal): widths other than 144, FIFO, ECC, block addressing,
// write data from the cascade or from the MLP result, falling-edge clocks,
// clk_sel, memory initialisation.  Unmodelled outputs are driven to 0.
//
// Compile-time switch for sensitivity runs:
//   +define+ACX_BRAM72K_BEHAV_DOUT2MLP_PRE_OUTREG   mlpram_dout2mlp taps the latch (one cycle earlier)
`ifndef SYNTHESIS

module ACX_BRAM72K
  #(
    parameter         rdclk_polarity            = "rise",
    parameter         wrclk_polarity            = "rise",
    parameter         mlpclk_polarity           = "rise",
    parameter [  3:0] write_width               = 4'h0,
    parameter [  3:0] read_width                = 4'h0,
    parameter         enable_wide_fabric_input  = 1'h0,
    parameter         clk_sel_wr                = 2'h0,
    parameter         clk_sel_rd                = 2'h0,
    parameter         outreg_enable             = 1'h0,
    parameter         outreg_sr_assertion       = 1'h0,
    parameter         blk_addr_enable           = 1'h0,
    parameter [  6:0] blk_addr_value            = 7'h0,
    parameter [  6:0] blk_wraddr_mask           = 7'h0,
    parameter [  6:0] blk_rdaddr_mask           = 7'h0,
    parameter         fabric_bram_bisten_top    = 1'h0,
    parameter         fabric_bram_bisten_bot    = 1'h0,
    parameter         dftclken                  = 1'h0,
    parameter         dftmask                   = 1'h0,
    parameter         ds                        = 1'h0,
    parameter         fiso                      = 1'h1,
    parameter         ls                        = 1'h0,
    parameter         pipemeb                   = 1'h0,
    parameter         rmea_512x128              = 1'h0,
    parameter         rmeb_512x128              = 1'h0,
    parameter         rmea_128x64               = 1'h0,
    parameter         rmeb_128x64               = 1'h0,
    parameter [  2:0] rma_512x128               = 3'h0,
    parameter [  2:0] rmb_512x128               = 3'h0,
    parameter [  2:0] rma_128x64                = 3'h0,
    parameter [  2:0] rmb_128x64                = 1'h0,
    parameter         rsc_128x64_reg_bypass     = 1'h0,
    parameter         rscen                     = 1'h1,
    parameter         sd                        = 1'h0,
    parameter         test1a                    = 1'h0,
    parameter         test1b                    = 1'h0,
    parameter         test_rnma                 = 1'h0,
    parameter [  3:0] wrmem_input_sel           = 4'h0,
    parameter [  3:0] rdmem_input_sel           = 4'h0,
    parameter         mlpram_din2mlpdout_sel    = 1'b0,
    parameter         ce_fwdi_ram_wr_addr       = 1'h0,
    parameter         ce_fwdi_ram_rd_addr       = 1'h0,
    parameter         del_fwdi_ram_wr_addr      = 1'h0,
    parameter         del_fwdi_ram_wr_data      = 1'h0,
    parameter         del_fwdi_ram_rd_addr      = 1'h0,
    parameter         fifo_enable               = 1'h0,
    parameter [ 14:0] fifo_wrptr_rstval         = 15'h0,
    parameter [ 14:0] fifo_rdptr_rstval         = 15'h0,
    parameter [ 14:0] fifo_wrptr_maxval         = 15'h7FFF,
    parameter [ 14:0] fifo_rdptr_maxval         = 15'h7FFF,
    parameter         dout_sel                  = 1'h0,
    parameter         fifo_sync_mode            = 1'h0,
    parameter         fifo_num_sync_stages_w2r  = 1'h0,
    parameter         fifo_num_sync_stages_r2w  = 1'h0,
    parameter [ 14:0] fifo_afull_threshold      = 15'h4,
    parameter [ 14:0] fifo_aempty_threshold     = 15'h4,
    parameter         fifo_fwft_mode            = 1'h0,
    parameter         fast_ef                   = 1'h0,
    parameter         fifo_ignore_flags         = 1'h0,
    parameter         wide_lram_enable          = 1'h0,
    parameter         enable_lram_read          = 1'h0,
    parameter         ecc_bypass_encode         = 1'h1,
    parameter         ecc_bypass_decode         = 1'h1,
    parameter         enable_revi_rd_data       = 1'h0,
    parameter         mem_init_file             = "",
    parameter         mem_init_file_size_width   = 72,
    parameter         mem_init_file_size_depth   = 1024,
    parameter         mem_init_file_offset_width = 0,
    parameter         mem_init_file_offset_depth = 0,
    parameter         mem_init_generate_mode     = 1'b0,
    parameter         location                  = ""
   )
   (
    input  [ 71:0] din,
    input          wrmsel,
    input  [  9:0] wraddrhi,
    input  [  8:0] we,
    input          wren,
    input          wrclk,
    input          rdmsel,
    input  [  9:0] rdaddrhi,
    input          rden,
    input          rdclk,
    input          outreg_rstn,
    input          outlatch_rstn,
    input          outreg_ce,
    input  [ 71:0] mlpram_din,
    input  [  8:0] mlpram_we,
    input  [143:0] mlpram_dout,
    input  [ 95:0] mlpram_mlp_dout,
    input          mlpclk,
    input  [  6:0] revi_wblk_addr,
    input  [  6:0] revi_rblk_addr,
    input  [ 13:0] fwdi_ram_wr_addr,
    input  [  6:0] fwdi_ram_wblk_addr,
    input          fwdi_ram_wren,
    input  [ 17:0] fwdi_ram_we,
    input          fwdi_ram_wrmsel,
    input  [143:0] fwdi_ram_wr_data,
    input  [ 13:0] fwdi_ram_rd_addr,
    input  [  6:0] fwdi_ram_rblk_addr,
    input          fwdi_ram_rden,
    input          fwdi_ram_rdmsel,
    input  [ 13:0] revi_ram_rd_addr,
    input  [  6:0] revi_ram_rblk_addr,
    input          revi_ram_rden,
    input  [143:0] revi_ram_rd_data,
    input          revi_ram_rdval,
    input          revi_ram_rdmsel,

    output [ 71:0] dout,
    output         full,
    output         almost_full,
    output         empty,
    output         almost_empty,
    output         write_error,
    output         read_error,
    output         sbit_error,
    output         dbit_error,
    output         mlpram_sbit_error,
    output         mlpram_dbit_error,
    output [  5:0] mlpram_wraddr,
    output [143:0] mlpram_din2mlpdout,
    output         mlpram_wren,
    output [  5:0] mlpram_rdaddr,
    output         mlpram_rden,
    output [ 71:0] mlpram_din2mlpdin,
    output [143:0] mlpram_dout2mlp,
    output [  6:0] revo_wblk_addr,
    output [  6:0] revo_rblk_addr,
    output [ 13:0] revo_ram_rd_addr,
    output [  6:0] revo_ram_rblk_addr,
    output         revo_ram_rden,
    output [143:0] revo_ram_rd_data,
    output         revo_ram_rdval,
    output         revo_ram_rdmsel,
    output [ 13:0] fwdo_ram_wr_addr,
    output [  6:0] fwdo_ram_wblk_addr,
    output         fwdo_ram_wren,
    output [ 17:0] fwdo_ram_we,
    output         fwdo_ram_wrmsel,
    output [143:0] fwdo_ram_wr_data,
    output [ 13:0] fwdo_ram_rd_addr,
    output [  6:0] fwdo_ram_rblk_addr,
    output         fwdo_ram_rden,
    output         fwdo_ram_rdmsel
   );

    localparam logic [71:0] POISON72 = {8'hBA, {8{8'hD5}}};

    // [B1] 144-bit hardware width codes: 4'h2 = sixteen 9-bit bytes, 4'h3 = eighteen 8-bit
    // bytes (the chain uses 4'h2 and drives all byte enables alike, so the difference is
    // not observable in the chain).
    localparam integer BYTEW = (write_width == 4'h3) ? 8 : 9;

    // Read-address source decode of the vendor sim wrapper
    // (speedster7t_sim_BRAM72K_COMMON.sv:1740-1742): codes 0, 1, 3, 8 and > 9 take the
    // address from the fabric rdaddrhi.  [B3] The remaining codes (2, 4, 5, 6, 7, 9) are
    // modelled as "address from fwdi_ram_rd_addr".  UNDOCUMENTED: UG086 does not describe
    // rdmem_input_sel, and src/rtl/matrix_engine/acx_bram_gen_deep_direct.sv uses 4'h2 on
    // BRAMs whose address arrives from the REVERSE cascade (revi, top-down), so the
    // forward-cascade code may be another value.
    localparam bit RD_FROM_FABRIC = (rdmem_input_sel == 4'd0) || (rdmem_input_sel == 4'd1) ||
                                    (rdmem_input_sel == 4'd3) || (rdmem_input_sel == 4'd8) ||
                                    (rdmem_input_sel >  4'd9);

    initial begin
        if (!((write_width == 4'h2 || write_width == 4'h3) && read_width == write_width))
            $fatal(1, "%m ACX_BRAM72K_BEHAV: only 144-bit read and write widths (4'h2/4'h3, equal) are modelled");
        if (fifo_enable || blk_addr_enable || !ecc_bypass_encode || !ecc_bypass_decode)
            $fatal(1, "%m ACX_BRAM72K_BEHAV: FIFO / block addressing / ECC not modelled");
        if (wrmem_input_sel == 4'd2 || wrmem_input_sel == 4'd4 ||
            wrmem_input_sel == 4'd10 || wrmem_input_sel == 4'd11)
            $fatal(1, "%m ACX_BRAM72K_BEHAV: write data from the cascade / MLP result not modelled");
        if (rdclk_polarity != "rise" || wrclk_polarity != "rise" || clk_sel_wr != 0 || clk_sel_rd != 0)
            $fatal(1, "%m ACX_BRAM72K_BEHAV: only rising-edge wrclk/rdclk, clk_sel = 0 modelled");
        if (outreg_sr_assertion != 1'b0)
            $fatal(1, "%m ACX_BRAM72K_BEHAV: only synchronous outreg reset modelled");
        if (!RD_FROM_FABRIC && (ce_fwdi_ram_rd_addr != del_fwdi_ram_rd_addr))
            $display("%m ACX_BRAM72K_BEHAV note: ce_fwdi_ram_rd_addr != del_fwdi_ram_rd_addr; modelled as del only [B4]");
    end

    // ------------------------------------------------------------------ memory, write port
    reg [143:0] mem [0:511];
    integer i;
    initial for (i = 0; i < 512; i = i + 1) mem[i] = {POISON72, POISON72};

    // Upper half of a 144-bit write: data from the MLP72's din (mlpram_din) and byte enables
    // from the MLP72's {expb, load_ab} (mlpram_we), but ONLY with enable_wide_fabric_input = 1;
    // otherwise the wrapper ties both to ACX_FLOAT (speedster7t_sim_BRAM72K.sv:4250-4271).
    // An unconnected upper half is modelled as never written (memory keeps POISON).
    wire [71:0]  w_din_hi = enable_wide_fabric_input ? mlpram_din : POISON72;
    wire [8:0]   w_we_hi  = enable_wide_fabric_input ? mlpram_we  : 9'h000;
    wire [143:0] w_din    = {w_din_hi, din};
    // [B1] Byte enables of a 144-bit write: we[8:0] for din[71:0], mlpram_we[8:0] for
    // din[143:72] = the SDP macro's we[17:9].  ACX_BRAM72K_SDP: "byte_width=9 ... we[7:0]
    // selects the lower 9-bit bytes and we[16:9] the higher ... we[8] and we[17] are ignored";
    // byte_width=8: we[17:0] one per 8-bit byte (ug086:10764-10778).  GEN_DEEP drives
    // byte_en[9] into load_ab and byte_en[17:10] into expb (ACX_BRAM72K_GEN_DEEP.sv:507-511,
    // 949, 955).
    wire [17:0]  w_we     = {w_we_hi, we};
    // 144 x 512: wraddr[13:5] <= user_wraddr[8:0] (Table 210, ug086:11011) = wraddrhi[9:1].
    wire [8:0]   w_addr   = wraddrhi[9:1];

    reg  [143:0] w_new;
    integer b;
    always @(posedge wrclk) begin
        if (wren && !wrmsel) begin
            w_new = mem[w_addr];
            if (BYTEW == 9) begin
                for (b = 0; b < 8; b = b + 1) begin
                    if (w_we[b])     w_new[9*b +: 9]      = w_din[9*b +: 9];
                    if (w_we[9 + b]) w_new[72 + 9*b +: 9] = w_din[72 + 9*b +: 9];
                end
            end else begin
                for (b = 0; b < 9; b = b + 1) begin
                    if (w_we[b])     w_new[8*b +: 8]      = w_din[8*b +: 8];
                    if (w_we[9 + b]) w_new[72 + 8*b +: 8] = w_din[72 + 8*b +: 8];
                end
            end
            mem[w_addr] <= w_new;
        end
    end

    // ------------------------------------------------------------------ read address
    // UG088 ch.6: the mlp_conv2d reference cascades the read address up the column "with a
    // delay stage enabled between each BRAM", so each MLP72 has data and weights "correctly
    // phased with each MLP72 operating one cycle behind the one below it" (ug088:792-797).
    // [B4] del_fwdi_ram_rd_addr = 1 puts one rdclk register on fwdi_ram_rd_addr/rden/rdmsel,
    // loaded every cycle.  [B5] fwdo_ram_rd_addr/rden/rdmsel carry the address this BRAM
    // actually uses (after that register, or the fabric address), so the next BRAM sees it
    // one cycle later.  [B3] 14-bit cascade address = {rdaddrhi[9:0], rdaddrlo[3:0]}; a
    // 144-bit read uses [13:5] (the SDP sim maps rdaddr_mapped = {rdmsel, rdaddr[13:4],
    // 4'h0}, speedster7t_sim_BRAM72K_SDP.sv:1143).
    reg  [13:0] casc_addr_q;
    reg         casc_rden_q, casc_rdmsel_q;
    initial begin casc_addr_q = '0; casc_rden_q = 1'b0; casc_rdmsel_q = 1'b0; end
    always @(posedge rdclk) begin
        casc_addr_q   <= fwdi_ram_rd_addr;
        casc_rden_q   <= fwdi_ram_rden;
        casc_rdmsel_q <= fwdi_ram_rdmsel;
    end
    wire [13:0] eff_addr   = RD_FROM_FABRIC ? {rdaddrhi, 4'h0} :
                             (del_fwdi_ram_rd_addr ? casc_addr_q : fwdi_ram_rd_addr);
    wire        eff_rden   = RD_FROM_FABRIC ? rden :
                             (del_fwdi_ram_rd_addr ? casc_rden_q : fwdi_ram_rden);
    wire        eff_rdmsel = RD_FROM_FABRIC ? rdmsel :
                             (del_fwdi_ram_rd_addr ? casc_rdmsel_q : fwdi_ram_rdmsel);
    assign fwdo_ram_rd_addr = eff_addr;
    assign fwdo_ram_rden    = eff_rden;
    assign fwdo_ram_rdmsel  = eff_rdmsel;

    // ------------------------------------------------------------------ read data
    // "the read address is registered and the stored data is latched into the output latches
    // on the following clock cycle ... [outreg_enable=1] an additional register after the
    // latch ... two cycles of latency" (ug086:11126-11132).  [B6] Read-during-write of the
    // same word returns the old word (not exercised: the chain never reads a word being
    // written).
    reg [143:0] rd_latch, rd_oreg;
    initial begin rd_latch = '0; rd_oreg = '0; end
    always @(posedge rdclk) begin
        if (!outlatch_rstn)                rd_latch <= '0;
        else if (eff_rden && !eff_rdmsel)  rd_latch <= mem[eff_addr[13:5]];
    end
    always @(posedge rdclk) begin
        if (!outreg_rstn)    rd_oreg <= '0;
        else if (outreg_ce)  rd_oreg <= rd_latch;
    end
    wire [143:0] rd_out = outreg_enable ? rd_oreg : rd_latch;
    assign dout = rd_out[71:0];
    // [B2] mlpram_dout2mlp ("BRAM_DOUT[143:0]", ug086:4501-4503) is the same point as dout:
    // after the output register when it is enabled.
`ifdef ACX_BRAM72K_BEHAV_DOUT2MLP_PRE_OUTREG
    assign mlpram_dout2mlp = rd_latch;
`else
    assign mlpram_dout2mlp = rd_out;
`endif

    // ------------------------------------------------------------------ other outputs
    assign mlpram_din2mlpdin  = din;                                      // wrapper :4279
    assign mlpram_din2mlpdout = mlpram_din2mlpdout_sel ? {72'h0, din} : 144'h0; // wrapper :4274-4277 (write cascade not modelled)
    assign {full, almost_full, empty, almost_empty, write_error, read_error, sbit_error, dbit_error} = 8'b0010_0000;
    assign {mlpram_sbit_error, mlpram_dbit_error} = 2'b00;
    assign mlpram_wraddr = 6'h0, mlpram_wren = 1'b0, mlpram_rdaddr = 6'h0, mlpram_rden = 1'b0;
    assign revo_wblk_addr = 7'h0, revo_rblk_addr = 7'h0, revo_ram_rd_addr = 14'h0, revo_ram_rblk_addr = 7'h0;
    assign revo_ram_rden = 1'b0, revo_ram_rd_data = 144'h0, revo_ram_rdval = 1'b0, revo_ram_rdmsel = 1'b0;
    assign fwdo_ram_wr_addr = 14'h0, fwdo_ram_wblk_addr = 7'h0, fwdo_ram_wren = 1'b0, fwdo_ram_we = 18'h0;
    assign fwdo_ram_wrmsel = 1'b0, fwdo_ram_wr_data = 144'h0, fwdo_ram_rblk_addr = 7'h0;

endmodule

`endif
