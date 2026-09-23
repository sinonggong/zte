// One hard-block INT8 GEMM chain for phase B of
// docs/PI0_FULL_CHIP_ARCHITECTURE_20260910.md: a feeder ACX_MLP72, N_STAGE
// multiplying ACX_MLP72s and one accumulating ACX_MLP72 in one column; the
// feeder and the multiplying MLP72s use their co-sited ACX_BRAM72K.
//
//   m = 0            feeder: its BRAM72K holds activation words; the MLP72 selects
//                    the 144-bit BRAM output onto multa_l / multa_h, registers it
//                    (stage 0) and forwards it up the fwd_multa cascade.  NO OP.
//   m = 1..N_STAGE   multiplying stage: multa_l/h from the cascade, multb_l/h from
//                    the co-sited BRAM (this stage's weight word), 16 signed 8x8
//                    products summed inside the DSP (ADD015, AB accumulator
//                    bypassed, ACCUM_AB_REG); the CD adder adds the partial sum of
//                    the stage below (FWDI_DOUT) into OUT_REG, which goes up on
//                    fwdo_dout.  Stage 1 bypasses the CD adder (nothing below it).
//   m = N_STAGE+1    accumulator: zero operands; AB = FWDI_DOUT (the finished word
//                    sum); the CD accumulator sums the words of a row (load on the
//                    row's first word, OUT_REG clock enable = word valid) and drives
//                    the chain result.  Its BRAM72K site is left free.
//
// Why the per-stage add sits in CD, not AB (found with the behavioural models in
// paper/rtl/sim_models/, 2026-09-16).  A 144-bit BRAM72K write takes its upper
// 72 data bits from the MLP72's din and its upper byte enables from the MLP72's
// {expb, load_ab} pins, and only with enable_wide_fabric_input = 1 (vendor
// wrapper speedster7t_sim_BRAM72K.sv: "assign mlpram_we = {expb, load_ab}",
// "if (enable_wide_fabric_input) w_din = {mlpram_din, din} else {72{Open}}";
// ACX_BRAM72K_GEN_DEEP drives load_ab with the wide write enable).  The first
// version left enable_wide_fabric_input at 0 (bytes 8..15 of every word never
// written) and used the AB accumulator for the cascade add with load_ab = 0.
// load_ab must now be the weight write enable, and "load_ab ... load the
// accumulator with the add_00_15_sel output" (UG086 Table 125) would drop the
// stage below's partial sum whenever a tile is written during computation.
// With AB bypassed, load_ab has no arithmetic effect.  Cost: one MLP72 per chain.
//
// A 144-bit word is {8'h0, byte15 .. byte8, 8'h0, byte7 .. byte0}: bytes 0..7
// feed multipliers 0..7 from the low bus, bytes 8..15 multipliers 8..15 from the
// high bus (UG086 Table 107).
//
// Protocol: one word per cycle on i_raddr with i_valid; i_first / i_last mark a
// row's first and last word (both on a one-word row); i_valid may drop inside a
// row.  o_sum_valid pulses once per row with the exact 48-bit sum over the row's
// words, all stages and all 16 products, LATENCY cycles after the row's last
// word was on the inputs.
//
// Timing, cycle c = the cycle a word is on the inputs (BRAM_RD_LAT = 2):
//   c+1                   raddr_q[0]
//   c+1+m                 read address at BRAM m (fabric pipeline or fwd_ram_rd_addr cascade)
//   c+1+BRAM_RD_LAT+m     BRAM m dout at MLP m's inputs, together with the forwarded activation
//   then +1 stage-0 reg, +2 stage-1 (mult inputs), +3 stage-2 (bank adders), +4 ACCUM_AB_REG,
//   +5 OUT_REG of MLP m.  Accumulator: ACCUM_AB_REG one cycle after stage N's OUT_REG
//   (T_CD), CD add in that cycle, OUT_REG at T_CD+1, o_sum at T_CD+2 = LATENCY.
//
// Assumptions this relies on are in paper/rtl/sim_models/ASSUMPTIONS.md (the
// fwdo_mult* tap after the stage-0 register, BRAM data to the MLP after the output
// register, the load delay register count, AB/CD bypass semantics, and for
// ADDR_CASCADE = 1 the undocumented rdmem_input_sel encoding).  Bit-exact against
// those models (paper/rtl/tb_mlp72_int8_chain.sv); not verified on silicon.

module mlp72_int8_chain #(
    parameter integer N_STAGE      = 16,  // multiplying MLP72s between feeder and accumulator
    parameter integer ADDR_BITS    = 9,   // <= 9: 512 x 144-bit words per BRAM72K
    parameter integer ADDR_CASCADE = 0,   // 1: stage read address up the BRAM72K fwd_ram_rd_addr cascade
    parameter integer BRAM_RD_LAT  = 2    // BRAM72K read address -> mlpram_dout2mlp (outreg_enable = 1)
) (
    input  wire                  i_clk,
    input  wire                  i_rstn,

    // BRAM writes on a slow clock: bit 0 of i_wen writes the feeder's
    // activation words, bit s stage s's weight words.  Writes to words that are
    // not being read may happen during computation.
    input  wire                  i_wclk,
    input  wire [143:0]          i_wdata,
    input  wire [ADDR_BITS-1:0]  i_waddr,
    input  wire [N_STAGE:0]      i_wen,

    input  wire                  i_valid,
    input  wire [ADDR_BITS-1:0]  i_raddr,
    input  wire                  i_first,
    input  wire                  i_last,

    output reg  [47:0]           o_sum,
    output reg                   o_sum_valid
);

    // Connect to Open to leave a pin unconnected instead of tied off.
    wire Open;
    ACX_FLOAT u_float (Open);

    // ---- cycle bookkeeping (see header) ----
    localparam integer ACC_LOAD_DEL = 4;                        // accumulator: stage 0, 1, 2 + ACCUM_AB_REG
    localparam integer T_CD         = N_STAGE + BRAM_RD_LAT + 7; // accumulator CD add, cycles after the input cycle
    localparam integer T_LOAD       = T_CD - ACC_LOAD_DEL;       // accumulator load pin
    localparam integer LATENCY      = T_CD + 2;                  // o_sum / o_sum_valid

    // ---- fabric: read address and the row control pipeline ----
    reg [ADDR_BITS-1:0] raddr_q [0:N_STAGE];
    integer p;
    always @(posedge i_clk) begin
        raddr_q[0] <= i_raddr;
        for (p = 1; p <= N_STAGE; p = p + 1)
            raddr_q[p] <= raddr_q[p-1];      // unused (removed) with ADDR_CASCADE = 1
    end

    // bit k of a pipe = the control bit of the word that was on the inputs k cycles ago
    reg [T_CD+1:1] v_pipe, f_pipe, l_pipe;
    always @(posedge i_clk) begin
        if (!i_rstn) begin
            v_pipe      <= '0;
            f_pipe      <= '0;
            l_pipe      <= '0;
            o_sum_valid <= 1'b0;
        end else begin
            v_pipe      <= {v_pipe[T_CD:1], i_valid};
            f_pipe      <= {f_pipe[T_CD:1], i_valid & i_first};
            l_pipe      <= {l_pipe[T_CD:1], i_valid & i_last};
            o_sum_valid <= l_pipe[T_CD+1];
        end
    end
    wire acc_load = f_pipe[T_LOAD];
    wire acc_ce   = v_pipe[T_CD];

    wire [71:0] acc_dout;
    always @(posedge i_clk)
        o_sum <= acc_dout[47:0];

    // ---- cascades: MLP m drives index m+1 ----
    wire [71:0] fwd_multa_l [0:N_STAGE+2];
    wire [71:0] fwd_multa_h [0:N_STAGE+2];
    wire [71:0] fwd_multb_l [0:N_STAGE+2];
    wire [71:0] fwd_multb_h [0:N_STAGE+2];
    wire [47:0] fwd_dout    [0:N_STAGE+2];
    wire [13:0] fwd_rd_addr [0:N_STAGE+1];
    wire        fwd_rden    [0:N_STAGE+1];
    wire        fwd_rdmsel  [0:N_STAGE+1];
    assign fwd_multa_l[0] = {72{Open}};
    assign fwd_multa_h[0] = {72{Open}};
    assign fwd_multb_l[0] = {72{Open}};
    assign fwd_multb_h[0] = {72{Open}};
    assign fwd_dout[0]    = {48{Open}};
    assign fwd_rd_addr[0] = {14{Open}};
    assign fwd_rden[0]    = Open;
    assign fwd_rdmsel[0]  = Open;

    genvar m;
    generate
        for (m = 0; m <= N_STAGE + 1; m = m + 1) begin : g_stage
            localparam logic IS_FEEDER = (m == 0);
            localparam logic IS_ACC    = (m == N_STAGE + 1);
            localparam logic IS_MULT   = !IS_FEEDER && !IS_ACC;
            localparam logic IS_MULT1  = (m == 1) && IS_MULT;
            localparam logic CASC_IN   = (ADDR_CASCADE != 0) && IS_MULT;
            // stage 1, 2, ACCUM_AB_REG and OUT_REG in circuit (not on the feeder)
            localparam logic       REG_S = !IS_FEEDER;
            localparam logic [3:0] CE_S  = REG_S ? 4'd13 : 4'd0;
            localparam logic [2:0] RST_S = REG_S ? 3'd5  : 3'd0;

            // ---- co-sited BRAM72K <-> MLP72 dedicated paths ----
            wire [143:0] bram_dout2mlp;
            wire [143:0] bram_din2mlpdout;
            wire [71:0]  bram_din2mlpdin;
            wire [5:0]   bram_wraddr2mlp;
            wire [5:0]   bram_rdaddr2mlp;
            wire         bram_wren2mlp;
            wire         bram_rden2mlp;
            wire         bram_sbit2mlp;
            wire         bram_dbit2mlp;
            wire [71:0]  mlp_din2bram;
            wire [8:0]   mlp_we2bram;
            wire [143:0] mlp_dout2bram;
            wire [95:0]  mlp_mlpdout2bram;
            // MLP pins shared with the BRAM's wide write
            wire [71:0]  mlp_din;
            wire         mlp_load_ab;
            wire [7:0]   mlp_expb;

            if (IS_ACC) begin : g_no_bram
                assign bram_dout2mlp    = {144{Open}};
                assign bram_din2mlpdout = {144{Open}};
                assign bram_din2mlpdin  = {72{Open}};
                assign bram_wraddr2mlp  = {6{Open}};
                assign bram_rdaddr2mlp  = {6{Open}};
                assign bram_wren2mlp    = Open;
                assign bram_rden2mlp    = Open;
                assign bram_sbit2mlp    = Open;
                assign bram_dbit2mlp    = Open;
                assign mlp_din          = 72'h0;          // zero operands on MLP_DIN
                assign mlp_load_ab      = 1'b0;
                assign mlp_expb         = {8{Open}};
            end else begin : g_bram
                // Wide (144-bit) write: din carries [71:0]; [143:72] and the upper byte
                // enables reach the BRAM through the MLP's din and {expb, load_ab}.
                assign mlp_din     = i_wdata[143:72];
                assign mlp_load_ab = i_wen[m];
                assign mlp_expb    = {8{i_wen[m]}};

                wire [8:0]  wa9;
                wire [8:0]  ra9;
                assign wa9 = i_waddr;                                   // zero-extended
                assign ra9 = raddr_q[(ADDR_CASCADE != 0) ? 0 : m];
                wire [9:0]  bram_rdaddrhi;
                wire [13:0] bram_fwdi_rd_addr;
                wire        bram_fwdi_rden;
                wire        bram_fwdi_rdmsel;
                if (CASC_IN) begin : g_addr_cascade
                    assign bram_rdaddrhi     = {10{Open}};
                    assign bram_fwdi_rd_addr = fwd_rd_addr[m];
                    assign bram_fwdi_rden    = fwd_rden[m];
                    assign bram_fwdi_rdmsel  = fwd_rdmsel[m];
                end else begin : g_addr_fabric
                    assign bram_rdaddrhi     = {ra9, 1'b0};             // 144 x 512: address [9:1]
                    assign bram_fwdi_rd_addr = {14{Open}};
                    assign bram_fwdi_rden    = Open;
                    assign bram_fwdi_rdmsel  = Open;
                end

                ACX_BRAM72K #(
                    .write_width              (4'h2),   // 144 bits, 9-bit bytes
                    .read_width               (4'h2),
                    .enable_wide_fabric_input (1'b1),   // upper half from mlpram_din / mlpram_we
                    .outreg_enable            (1'b1),
                    .wrmem_input_sel          (4'h0),
                    // ADDR_CASCADE: the feeder's BRAM takes the fabric address and starts the
                    // cascade, every stage BRAM takes fwdi_ram_rd_addr through one register.
                    // 4'h1 / 4'h2 follow ACX_BRAM72K_GEN_DEEP's entry / follower codes;
                    // UNDOCUMENTED, see sim_models/ASSUMPTIONS.md [B3].
                    .rdmem_input_sel          ((ADDR_CASCADE == 0) ? 4'h0 : (IS_FEEDER ? 4'h1 : 4'h2)),
                    .del_fwdi_ram_rd_addr     (CASC_IN),
                    .ce_fwdi_ram_rd_addr      (CASC_IN)
                ) u_bram (
                    .din                (i_wdata[71:0]),
                    .wrmsel             (1'b0),
                    .wraddrhi           ({wa9, 1'b0}),
                    .we                 ({9{i_wen[m]}}),
                    .wren               (i_wen[m]),
                    .wrclk              (i_wclk),
                    .rdmsel             (1'b0),
                    .rdaddrhi           (bram_rdaddrhi),
                    .rden               (1'b1),
                    .rdclk              (i_clk),
                    .outreg_rstn        (1'b1),
                    .outlatch_rstn      (1'b1),
                    .outreg_ce          (1'b1),
                    .mlpram_din         (mlp_din2bram),
                    .mlpram_we          (mlp_we2bram),
                    .mlpram_dout        (mlp_dout2bram),
                    .mlpram_mlp_dout    (mlp_mlpdout2bram),
                    .mlpclk             (i_clk),
                    .revi_wblk_addr     ({7{Open}}),
                    .revi_rblk_addr     ({7{Open}}),
                    .fwdi_ram_wr_addr   ({14{Open}}),
                    .fwdi_ram_wblk_addr ({7{Open}}),
                    .fwdi_ram_wren      (Open),
                    .fwdi_ram_we        ({18{Open}}),
                    .fwdi_ram_wrmsel    (Open),
                    .fwdi_ram_wr_data   ({144{Open}}),
                    .fwdi_ram_rd_addr   (bram_fwdi_rd_addr),
                    .fwdi_ram_rblk_addr ({7{Open}}),
                    .fwdi_ram_rden      (bram_fwdi_rden),
                    .fwdi_ram_rdmsel    (bram_fwdi_rdmsel),
                    .revi_ram_rd_addr   ({14{Open}}),
                    .revi_ram_rblk_addr ({7{Open}}),
                    .revi_ram_rden      (Open),
                    .revi_ram_rd_data   ({144{Open}}),
                    .revi_ram_rdval     (Open),
                    .revi_ram_rdmsel    (Open),
                    .dout               (),
                    .full               (),
                    .almost_full        (),
                    .empty              (),
                    .almost_empty       (),
                    .write_error        (),
                    .read_error         (),
                    .sbit_error         (),
                    .dbit_error         (),
                    .mlpram_sbit_error  (bram_sbit2mlp),
                    .mlpram_dbit_error  (bram_dbit2mlp),
                    .mlpram_wraddr      (bram_wraddr2mlp),
                    .mlpram_din2mlpdout (bram_din2mlpdout),
                    .mlpram_wren        (bram_wren2mlp),
                    .mlpram_rdaddr      (bram_rdaddr2mlp),
                    .mlpram_rden        (bram_rden2mlp),
                    .mlpram_din2mlpdin  (bram_din2mlpdin),
                    .mlpram_dout2mlp    (bram_dout2mlp),
                    .revo_wblk_addr     (),
                    .revo_rblk_addr     (),
                    .revo_ram_rd_addr   (),
                    .revo_ram_rblk_addr (),
                    .revo_ram_rden      (),
                    .revo_ram_rd_data   (),
                    .revo_ram_rdval     (),
                    .revo_ram_rdmsel    (),
                    .fwdo_ram_wr_addr   (),
                    .fwdo_ram_wblk_addr (),
                    .fwdo_ram_wren      (),
                    .fwdo_ram_we        (),
                    .fwdo_ram_wrmsel    (),
                    .fwdo_ram_wr_data   (),
                    .fwdo_ram_rd_addr   (fwd_rd_addr[m+1]),
                    .fwdo_ram_rblk_addr (),
                    .fwdo_ram_rden      (fwd_rden[m+1]),
                    .fwdo_ram_rdmsel    (fwd_rdmsel[m+1])
                );
            end

            wire [71:0] mlp_dout;
            if (IS_ACC) begin : g_acc_dout
                assign acc_dout = mlp_dout;
            end

            ACX_MLP72 #(
                // input selection (UG086 Table 103)
                .mux_sel_multa_l (IS_FEEDER ? 2'b10  : IS_MULT ? 2'b11  : 2'b00),   // BRAM_DOUT[71:0]   | FWDI_MULTA_L | MLP_DIN
                .mux_sel_multa_h (IS_FEEDER ? 3'b101 : IS_MULT ? 3'b111 : 3'b000),  // BRAM_DOUT[143:72] | FWDI_MULTA_H | MLP_DIN
                .mux_sel_multb_l (IS_MULT ? 2'b10  : 2'b00),                          // BRAM_DOUT[71:0]   (weights)    | MLP_DIN
                .mux_sel_multb_h (IS_MULT ? 3'b101 : 3'b000),                         // BRAM_DOUT[143:72]              | MLP_DIN
                // Int8 x4 mode: 16 multipliers (Table 107)
                .bytesel_00_07   (5'h01),
                .bytesel_08_15   (6'h21),
                .multmode_00_07  (IS_FEEDER ? 5'h11 : 5'h00),    // NO OP | SIGNED 8x8
                .multmode_08_15  (IS_FEEDER ? 5'h11 : 5'h00),
                // adder tree: all 16 products
                .add_00_07_bypass(1'b0),
                .add_00_07_sub   (1'b0),
                .add_08_15_bypass(1'b0),
                .add_08_15_sub   (1'b0),
                .add_00_15_sel   (1'b1),
                // integer mode
                .fpmult_ab_bypass(1'b1),
                .fpmult_cd_bypass(1'b1),
                .accum_ab_reg_din_sel(1'b0),
                // AB: bypassed (16-product sum -> ACCUM_AB_REG) except on the accumulator,
                // where it adds FWDI_DOUT to the zero product sum
                .fpadd_ab_dinb_sel  (IS_ACC ? 3'b001 : 3'b000),
                .add_accum_ab_bypass(IS_ACC ? 1'b0   : 1'b1),
                // CD: ACCUM_AB_REG + FWDI_DOUT on stages 2..N (bypass on stage 1 and the
                // feeder), ACCUM_AB_REG + OUT_REG feedback on the accumulator
                .fpadd_cd_dina_sel  (1'b1),
                .fpadd_cd_dinb_sel  (IS_MULT ? 3'b001 : 3'b000),
                .add_accum_cd_bypass((IS_FEEDER || IS_MULT1) ? 1'b1 : 1'b0),
                .rndsubload_share   (1'b0),
                // outputs: OUT_REG up the dout cascade and on dout
                .out_reg_din_sel (3'b011),
                .dout_mlp_sel    (2'b00),
                .outmode_sel     (2'b00),
                // stage 0 registers (the cascade inputs)
                .del_multa_h     (1'b1),
                .del_multa_l     (1'b1),
                .del_multb_h     (REG_S),
                .del_multb_l     (REG_S),
                .del_expb_din_reg(1'b0),
                // stage 1 registers (multiplier inputs)
                .del_mult00a     (REG_S), .del_mult00b(REG_S),
                .del_mult01a     (REG_S), .del_mult01b(REG_S),
                .del_mult02a     (REG_S), .del_mult02b(REG_S),
                .del_mult03a     (REG_S), .del_mult03b(REG_S),
                .del_mult04_07a  (REG_S), .del_mult04_07b(REG_S),
                .del_mult08_11a  (REG_S), .del_mult08_11b(REG_S),
                .del_mult12_15a  (REG_S), .del_mult12_15b(REG_S),
                // stage 2 registers (bank adders)
                .del_add_00_07_reg(REG_S),
                .del_add_08_15_reg(REG_S),
                // load delay match (accumulator only): stage 0 + 1 + 2 + ACCUM_AB_REG
                .del_rndsubload_ab_reg(3'd0),
                .del_rndsubload_reg   (IS_ACC ? 3'(ACC_LOAD_DEL) : 3'd0),
                // stage 4 registers
                .del_accum_ab_reg (REG_S),
                .del_out_reg_00_15(REG_S),
                .del_out_reg_16_31(REG_S),
                .del_out_reg_32_47(REG_S),
                .del_out_reg_48_63(1'b0),
                // clock enables: 4'd13 = tied high, 4'd9 = ce[8] (accumulator OUT_REG);
                // resets: 3'd5 = tied inactive, 3'd1 = rstn[0] (accumulator OUT_REG)
                .cesel_multa_h(4'd13), .cesel_multa_l(4'd13),
                .cesel_multb_h(CE_S),  .cesel_multb_l(CE_S),
                .cesel_expb_din_reg(4'd0),
                .cesel_mult00a(CE_S), .cesel_mult00b(CE_S),
                .cesel_mult01a(CE_S), .cesel_mult01b(CE_S),
                .cesel_mult02a(CE_S), .cesel_mult02b(CE_S),
                .cesel_mult03a(CE_S), .cesel_mult03b(CE_S),
                .cesel_mult04_07a(CE_S), .cesel_mult04_07b(CE_S),
                .cesel_mult08_11a(CE_S), .cesel_mult08_11b(CE_S),
                .cesel_mult12_15a(CE_S), .cesel_mult12_15b(CE_S),
                .cesel_add_00_07_reg(CE_S), .cesel_add_08_15_reg(CE_S),
                .cesel_rndsubload_ab_reg(4'd0),
                .cesel_rndsubload_reg(IS_ACC ? 4'd13 : 4'd0),
                .cesel_accum_ab_reg(CE_S),
                .cesel_out_reg_00_15(IS_ACC ? 4'd9 : CE_S),
                .cesel_out_reg_16_31(IS_ACC ? 4'd9 : CE_S),
                .cesel_out_reg_32_47(IS_ACC ? 4'd9 : CE_S),
                .cesel_out_reg_48_63(4'd0),
                .rstsel_multa_h(3'd5), .rstsel_multa_l(3'd5),
                .rstsel_multb_h(RST_S), .rstsel_multb_l(RST_S),
                .rstsel_expb_din_reg(3'd0),
                .rstsel_mult00a(RST_S), .rstsel_mult00b(RST_S),
                .rstsel_mult01a(RST_S), .rstsel_mult01b(RST_S),
                .rstsel_mult02a(RST_S), .rstsel_mult02b(RST_S),
                .rstsel_mult03a(RST_S), .rstsel_mult03b(RST_S),
                .rstsel_mult04_07a(RST_S), .rstsel_mult04_07b(RST_S),
                .rstsel_mult08_11a(RST_S), .rstsel_mult08_11b(RST_S),
                .rstsel_mult12_15a(RST_S), .rstsel_mult12_15b(RST_S),
                .rstsel_add_00_07_reg(RST_S), .rstsel_add_08_15_reg(RST_S),
                .rstsel_rndsubload_ab_reg(3'd0),
                .rstsel_rndsubload_reg(IS_ACC ? 3'd5 : 3'd0),
                .rstsel_accum_ab_reg(RST_S),
                .rstsel_out_reg_00_15(IS_ACC ? 3'd1 : RST_S),
                .rstsel_out_reg_16_31(IS_ACC ? 3'd1 : RST_S),
                .rstsel_out_reg_32_47(IS_ACC ? 3'd1 : RST_S),
                .rstsel_out_reg_48_63(3'd0)
            ) u_mlp (
                .clk                   (i_clk),
                .din                   (mlp_din),
                .mlpram_bramdin2mlpdin (bram_din2mlpdin),
                .load_ab               (mlp_load_ab),
                .load                  (IS_ACC ? acc_load : 1'b0),
                .sub_ab                (1'b0),
                .sub                   (1'b0),
                .ce                    ({3'b111, (IS_ACC ? acc_ce : 1'b1), 8'hFF}),
                .rstn                  ({3'b111, i_rstn}),
                .expb                  (mlp_expb),
                .dout                  (mlp_dout),
                // co-sited BRAM72K
                .mlpram_din            (mlp_din2bram),
                .mlpram_we             (mlp_we2bram),
                .mlpram_dout           (mlp_dout2bram),
                .mlpram_mlp_dout       (mlp_mlpdout2bram),
                .mlpram_bramdout2mlp   (bram_dout2mlp),
                .mlpram_din2mlpdout    (bram_din2mlpdout),
                .mlpram_wraddr         (bram_wraddr2mlp),
                .mlpram_wren           (bram_wren2mlp),
                .mlpram_rdaddr         (bram_rdaddr2mlp),
                .mlpram_rden           (bram_rden2mlp),
                .mlpram_sbit_error     (bram_sbit2mlp),
                .mlpram_dbit_error     (bram_dbit2mlp),
                .sbit_error            (),
                .dbit_error            (),
                // cascade up the column
                .fwdi_multa_h          (fwd_multa_h[m]),
                .fwdi_multa_l          (fwd_multa_l[m]),
                .fwdi_multb_h          (fwd_multb_h[m]),
                .fwdi_multb_l          (fwd_multb_l[m]),
                .fwdi_dout             (fwd_dout[m]),
                .fwdo_multa_h          (fwd_multa_h[m+1]),
                .fwdo_multa_l          (fwd_multa_l[m+1]),
                .fwdo_multb_h          (fwd_multb_h[m+1]),
                .fwdo_multb_l          (fwd_multb_l[m+1]),
                .fwdo_dout             (fwd_dout[m+1]),
                // LRAM FIFO unused
                .lram_wrclk            (Open),
                .lram_rdclk            (Open),
                .empty                 (),
                .full                  (),
                .almost_empty          (),
                .almost_full           (),
                .write_error           (),
                .read_error            ()
            );
        end
    endgenerate

endmodule
