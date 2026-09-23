// Column-parallel hard-block INT8 GEMM chain: a feeder ACX_MLP72 and N_STAGE
// multiplying ACX_MLP72s in one column, each with its co-sited ACX_BRAM72K.
//
// Why column-parallel.  mlp72_int8_chain.sv sums the partial sums of all stages
// for the same activation word, i.e. it computes sum_k act[k] . (sum_s wt[s,k]):
// every stage sees the same word (one feeder, one fwd_multa cascade, aligned
// address and sum pipelines), so N stages compute what one stage with the
// summed weights computes, and a GEMM column costs one cycle per word whatever
// N is.  The activation cascade carries one 16-byte word per cycle, so the only
// way to use all 16 x N multipliers is to let every stage multiply the same
// word by a DIFFERENT output column and keep its own sum:
//
//   m = 0            feeder: its BRAM72K holds activation words; the MLP72 selects
//                    the 144-bit BRAM output, registers it (stage 0) and forwards
//                    it up the fwd_multa cascade.  NO OP.
//   m = 1..N_STAGE   multiplying stage: multa from the cascade, multb from the
//                    co-sited BRAM (the weight word of this stage's column at the
//                    same address), 16 signed 8x8 products summed inside the DSP
//                    (ADD015 -> ACCUM_AB_REG, AB accumulator bypassed); the CD
//                    accumulator sums the row's words (ACCUM_AB_REG + OUT_REG,
//                    load on the row's first word, OUT_REG clock enable = word
//                    valid), so OUT_REG holds this stage's column sum.
//
// One row pass of W words gives N_STAGE column results.  A tile is therefore
// N_STAGE x floor(512 / W) columns (128 at K = 1024 bytes, 64 at K = 2048), and
// a column costs W / N_STAGE cycles: the multipliers are all busy.  The dout
// cascade is not used.  Resources: N_STAGE + 1 MLP72 and BRAM72K.
//
// Two read-address streams.  The feeder reads the activation row (i_act_raddr)
// and every stage reads the weight column (i_wt_raddr, delayed one cycle per
// stage); row r meets column p only because the two streams differ.  With
// ADDR_CASCADE = 1 the stage addresses go up the BRAM72K rd-address cascade from
// stage 1 (the entry BRAM, fabric address) and the feeder keeps its own fabric
// address.  colpar_row_sequencer.sv generates both streams.
//
// Result capture.  OUT_REG of stage m holds the row sum from T_CD(m)+1 until the
// next valid word reaches its CD adder.  A fabric register per stage captures it
// in the SECOND cycle of that window (T_CD(m)+2), so the MLP72 dout -> capture
// path is a 2-cycle multicycle path; this needs at least one idle cycle between
// rows.  o_sums_valid follows the last stage's capture; all N_STAGE captures are
// still stable in that cycle if the next row is at least N_STAGE cycles long
// (words + idle cycles in front of it).  The schedule compiler pads short rows.
//
// Row protocol constraints (checked by the testbench, not by the RTL):
//   (1) >= 1 idle cycle (i_valid = 0) between the last word of a row and the
//       first word of the next;
//   (2) cycles from one row's last word to the next row's last word >= N_STAGE.
//
// The BRAM72K write path is unchanged from mlp72_int8_chain.sv (144-bit writes
// with enable_wide_fabric_input = 1; load_ab / expb are the upper byte write
// enables and have no arithmetic effect with AB bypassed).  Assumptions are in
// paper/rtl/sim_models/ASSUMPTIONS.md; this uses the same subset as the
// accumulator MLP72 of mlp72_int8_chain.sv (A3, A6, A7, A10) plus the
// multiplying-stage subset.  Not verified on silicon.
//
// Timing, cycle c = the cycle a word is on the inputs:
//   c+1                   act_raddr_q (feeder), wt_raddr_q[0]
//   c+1+m                 read address at BRAM m
//   c+1+BRAM_RD_LAT+m     BRAM m dout at MLP m's inputs, with the forwarded activation
//   c+2+BRAM_RD_LAT+m     stage-0 reg; +1 stage-1 (mult inputs); +2 stage-2 (bank adders)
//   T_CD(m) = m+BRAM_RD_LAT+5   ACCUM_AB_REG valid, CD add
//   T_CD(m)+1             OUT_REG = running column sum
//   T_CD(m)+2             fabric capture (row's last word only)
//   LATENCY = T_CD(N_STAGE)+3   o_sums / o_sums_valid

module mlp72_int8_colpar_chain #(
    parameter integer N_STAGE      = 16,  // multiplying MLP72s above the feeder
    parameter integer ADDR_BITS    = 9,   // <= 9: 512 x 144-bit words per BRAM72K
    parameter integer ADDR_CASCADE = 0,   // 1: stage read address up the BRAM72K fwd_ram_rd_addr cascade
    parameter integer BRAM_RD_LAT  = 2,   // BRAM72K read address -> mlpram_dout2mlp (outreg_enable = 1)
    // The weight-load bus (i_wdata / i_waddr / i_wen) is a FABRIC broadcast to every stage, not a
    // hard cascade like the read path.  At 16 stages its 17 loads route; at 32 they do not -- ACE
    // gives up with "Router couldn't resolve all overflows" on wdata bits, at 1.55 % occupancy.
    // WR_PIPE_EVERY > 0 registers the bus once per that many stages on i_wclk, turning one net with
    // N_STAGE+1 loads into a chain of short point-to-point nets.  The writes then land WR_PIPE_DEPTH
    // i_wclk cycles later, so whoever owns the loader must hold "done" back by the same amount
    // (colpar_chain_node.sv does).  0 = the original flat broadcast.
    parameter integer WR_PIPE_EVERY = 0,
    // o_sums_valid is the capture ENABLE of the result port's 48 x N_STAGE bank, so at 32 stages one
    // array-clock net drives 1,536 loads and the path misses 750 MHz by 0.266 ns.  VALID_COPIES > 1
    // emits that many preserved copies of the same flop, one per slice of the bank.
    parameter integer VALID_COPIES = 1,
    // With VALID_COPIES > 1, the last VALID_TAIL bits of l_pipe are kept once per copy (syn_preserve), so each
    // copy's enable reaches its bank slice through its own short chain of registers.  With one shared
    // l_pipe[T_LAST] the placer put the copies beside their slices and the source wherever: in an 18-node array
    // that net ran 84 tile rows (1.8 ns) and missed 725 MHz by 0.56 ns while the node alone closed 750 MHz
    // (docs/PI0_CLOCK_PLAN_20260917.md §3.1).  Cycle-exact with VALID_TAIL = 0.
    parameter integer VALID_TAIL   = 3,
    parameter [4:0]   MULT_MODE    = 5'h00 // stage multipliers (UG086 Table 123): 5'h00 signed x signed;
                                          // 5'h13 activation (multa) unsigned x weight (multb) signed,
                                          // for uint8 attention probabilities x int8 V
) (
    input  wire                     i_clk,
    input  wire                     i_rstn,

    // BRAM writes on a slow clock: bit 0 of i_wen writes the feeder's
    // activation words, bit s stage s's weight words.
    input  wire                     i_wclk,
    input  wire [143:0]             i_wdata,
    input  wire [ADDR_BITS-1:0]     i_waddr,
    input  wire [N_STAGE:0]         i_wen,

    input  wire                     i_valid,
    input  wire [ADDR_BITS-1:0]     i_act_raddr,   // feeder: activation word address
    input  wire [ADDR_BITS-1:0]     i_wt_raddr,    // stages: weight word address
    input  wire                     i_first,
    input  wire                     i_last,

    output wire [48*N_STAGE-1:0]    o_sums,        // stage s (1..N_STAGE) at [48*(s-1) +: 48]
    output wire [VALID_COPIES-1:0] o_sums_valid        // registered copies, see g_vtail / g_vplain
);

    wire Open;
    ACX_FLOAT u_float (Open);

    // ---- cycle bookkeeping (see header) ----
    localparam integer LOAD_DEL = 4;                           // stage 0, 1, 2 + ACCUM_AB_REG
`ifdef COLPAR_NEG_CAP_LATE   // negative control: capture one cycle late (fails on 1-cycle row gaps)
    localparam integer CAP_OFF  = 3;
`else
    localparam integer CAP_OFF  = 2;                           // second cycle of the OUT_REG hold window
`endif
    localparam integer T_CD1    = BRAM_RD_LAT + 6;             // T_CD(1)
    localparam integer T_LAST   = T_CD1 + N_STAGE - 1 + CAP_OFF; // last stage's capture cycle
    localparam integer LATENCY  = T_LAST + 1;

    // ---- fabric: read address and the row control pipeline ----
    reg [ADDR_BITS-1:0] act_raddr_q;
    reg [ADDR_BITS-1:0] wt_raddr_q [0:N_STAGE];
    integer p;
    always @(posedge i_clk) begin
        act_raddr_q   <= i_act_raddr;
`ifdef COLPAR_NEG_SHARED_ADDR   // negative control: the stages read the activation address
        wt_raddr_q[0] <= i_act_raddr;
`else
        wt_raddr_q[0] <= i_wt_raddr;
`endif
        for (p = 1; p <= N_STAGE; p = p + 1)
            wt_raddr_q[p] <= wt_raddr_q[p-1];  // stages 2.. unused (removed) with ADDR_CASCADE = 1
    end

    // bit k of a pipe = the control bit of the word that was on the inputs k cycles ago
    reg [T_LAST:1] v_pipe, f_pipe, l_pipe;
    always @(posedge i_clk) begin
        if (!i_rstn) begin
            v_pipe       <= '0;
            f_pipe       <= '0;
            l_pipe       <= '0;
        end else begin
            v_pipe       <= {v_pipe[T_LAST-1:1], i_valid};
            f_pipe       <= {f_pipe[T_LAST-1:1], i_valid & i_first};
            l_pipe       <= {l_pipe[T_LAST-1:1], i_valid & i_last};
        end
    end

    // o_sums_valid: one register per copy.  With VALID_TAIL, copy c also owns the last VT bits of l_pipe
    // (copy c's tail bit k = l_pipe[T_LAST - VT + k]), so its enable arrives through a short local chain.
    generate if (VALID_COPIES > 1 && VALID_TAIL > 0) begin : g_vtail
        localparam integer VT = (VALID_TAIL < T_LAST) ? VALID_TAIL : T_LAST - 1;
        (* syn_preserve = 1 *) reg [VALID_COPIES*VT-1:0] l_tail;
        (* syn_preserve = 1 *) reg [VALID_COPIES-1:0]    sv;
        integer c;
        always @(posedge i_clk) begin
            if (!i_rstn) begin
                l_tail <= '0;
                sv     <= '0;
            end else begin
                for (c = 0; c < VALID_COPIES; c = c + 1) begin
                    l_tail[c*VT +: VT] <= {l_tail[c*VT +: VT], l_pipe[T_LAST-VT]};   // shift up, drop the top bit
                    sv[c]              <= l_tail[c*VT + VT-1];
                end
            end
        end
        assign o_sums_valid = sv;
    end else begin : g_vplain
        (* syn_preserve = 1 *) reg [VALID_COPIES-1:0] sv;
        always @(posedge i_clk)
            if (!i_rstn) sv <= '0;
            else         sv <= {VALID_COPIES{l_pipe[T_LAST]}};
        assign o_sums_valid = sv;
    end endgenerate

    // ---- weight-load bus pipeline (fabric clock) ----
    localparam integer WR_SEG   = (WR_PIPE_EVERY == 0) ? 1
                                : (N_STAGE + WR_PIPE_EVERY) / WR_PIPE_EVERY;   // segments, stage 0..N_STAGE
    localparam integer WR_DEPTH = (WR_PIPE_EVERY == 0) ? 0 : WR_SEG;
    reg  [143:0]          wd_p [0:WR_SEG-1];
    reg  [ADDR_BITS-1:0]  wa_p [0:WR_SEG-1];
    reg  [N_STAGE:0]      we_p [0:WR_SEG-1];
    integer ws;
    always @(posedge i_wclk) begin
        wd_p[0] <= i_wdata;
        wa_p[0] <= i_waddr;
        we_p[0] <= i_wen;
        for (ws = 1; ws < WR_SEG; ws = ws + 1) begin
            wd_p[ws] <= wd_p[ws-1];
            wa_p[ws] <= wa_p[ws-1];
            we_p[ws] <= we_p[ws-1];
        end
    end

    // ---- cascades: MLP m drives index m+1 ----
    wire [71:0] fwd_multa_l [0:N_STAGE+1];
    wire [71:0] fwd_multa_h [0:N_STAGE+1];
    wire [71:0] fwd_multb_l [0:N_STAGE+1];
    wire [71:0] fwd_multb_h [0:N_STAGE+1];
    wire [47:0] fwd_dout    [0:N_STAGE+1];
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
        for (m = 0; m <= N_STAGE; m = m + 1) begin : g_stage
            localparam logic IS_FEEDER = (m == 0);
            localparam logic IS_MULT   = !IS_FEEDER;
            localparam logic CASC_IN   = (ADDR_CASCADE != 0) && (m >= 2);   // address from the BRAM below
            localparam logic CASC_HEAD = (ADDR_CASCADE != 0) && (m == 1);   // cascade entry: fabric address
            localparam integer T_CD    = T_CD1 + m - 1;            // meaningful for m >= 1
            // stage 1, 2, ACCUM_AB_REG and OUT_REG in circuit (not on the feeder)
            localparam logic       REG_S  = IS_MULT;
            localparam logic [3:0] CE_S   = REG_S ? 4'd13 : 4'd0;
            localparam logic [2:0] RST_S  = REG_S ? 3'd5  : 3'd0;

`ifdef COLPAR_NEG_LOAD_LATE  // negative control: load pin one cycle late
            wire row_load = IS_MULT ? f_pipe[T_CD - LOAD_DEL + 1] : 1'b0;
`else
            wire row_load = IS_MULT ? f_pipe[T_CD - LOAD_DEL] : 1'b0;
`endif
            wire word_ce  = IS_MULT ? v_pipe[T_CD] : 1'b1;

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

            // Wide (144-bit) write: din carries [71:0]; [143:72] and the upper byte
            // enables reach the BRAM through the MLP's din and {expb, load_ab}.
            localparam integer WSEG = (WR_PIPE_EVERY == 0) ? 0 : (m / WR_PIPE_EVERY);
            wire [143:0]          st_wdata = (WR_PIPE_EVERY == 0) ? i_wdata : wd_p[WSEG];
            wire [ADDR_BITS-1:0]  st_waddr = (WR_PIPE_EVERY == 0) ? i_waddr : wa_p[WSEG];
            wire                  st_wen   = (WR_PIPE_EVERY == 0) ? i_wen[m] : we_p[WSEG][m];

            wire [71:0] mlp_din     = st_wdata[143:72];
            wire        mlp_load_ab = st_wen;
            wire [7:0]  mlp_expb    = {8{st_wen}};

            wire [8:0]  wa9 = st_waddr;                                 // zero-extended
            wire [8:0]  ra9 = IS_FEEDER ? act_raddr_q : wt_raddr_q[m];
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
                assign bram_rdaddrhi     = {ra9, 1'b0};                 // 144 x 512: address [9:1]
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
                // UNDOCUMENTED cascade codes, see sim_models/ASSUMPTIONS.md [B3].
                .rdmem_input_sel          (CASC_IN ? 4'h2 : (CASC_HEAD ? 4'h1 : 4'h0)),
                .del_fwdi_ram_rd_addr     (CASC_IN),
                .ce_fwdi_ram_rd_addr      (CASC_IN)
            ) u_bram (
                .din                (st_wdata[71:0]),
                .wrmsel             (1'b0),
                .wraddrhi           ({wa9, 1'b0}),
                .we                 ({9{st_wen}}),
                .wren               (st_wen),
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

            wire [71:0] mlp_dout;

            ACX_MLP72 #(
                // input selection (UG086 Table 103)
                .mux_sel_multa_l (IS_FEEDER ? 2'b10  : 2'b11),   // BRAM_DOUT[71:0]   | FWDI_MULTA_L
                .mux_sel_multa_h (IS_FEEDER ? 3'b101 : 3'b111),  // BRAM_DOUT[143:72] | FWDI_MULTA_H
                .mux_sel_multb_l (IS_MULT ? 2'b10  : 2'b00),     // BRAM_DOUT[71:0]   (weights) | MLP_DIN
                .mux_sel_multb_h (IS_MULT ? 3'b101 : 3'b000),    // BRAM_DOUT[143:72]           | MLP_DIN
                // Int8 x4 mode: 16 multipliers (Table 107)
                .bytesel_00_07   (5'h01),
                .bytesel_08_15   (6'h21),
                .multmode_00_07  (IS_FEEDER ? 5'h11 : MULT_MODE),    // NO OP | SIGNED 8x8 (or MULT_MODE)
                .multmode_08_15  (IS_FEEDER ? 5'h11 : MULT_MODE),
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
                // AB bypassed: 16-product sum -> ACCUM_AB_REG
                .fpadd_ab_dinb_sel  (3'b000),
                .add_accum_ab_bypass(1'b1),
                // CD: ACCUM_AB_REG + OUT_REG feedback (row accumulator); bypass on the feeder
                .fpadd_cd_dina_sel  (1'b1),
`ifdef COLPAR_NEG_CD_FWDI    // negative control: CD adds the stage below (the stage-sum chain)
                .fpadd_cd_dinb_sel  ((m >= 2) ? 3'b001 : 3'b000),
`else
                .fpadd_cd_dinb_sel  (3'b000),
`endif
                .add_accum_cd_bypass(IS_FEEDER ? 1'b1 : 1'b0),
                .rndsubload_share   (1'b0),
                // outputs: OUT_REG on dout
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
                // load delay match: stage 0 + 1 + 2 + ACCUM_AB_REG
                .del_rndsubload_ab_reg(3'd0),
                .del_rndsubload_reg   (IS_MULT ? 3'(LOAD_DEL) : 3'd0),
                // stage 4 registers
                .del_accum_ab_reg (REG_S),
                .del_out_reg_00_15(REG_S),
                .del_out_reg_16_31(REG_S),
                .del_out_reg_32_47(REG_S),
                .del_out_reg_48_63(1'b0),
                // clock enables: 4'd13 = tied high, 4'd9 = ce[8] (OUT_REG = word valid);
                // resets: 3'd5 = tied inactive, 3'd1 = rstn[0] (OUT_REG)
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
                .cesel_rndsubload_reg(IS_MULT ? 4'd13 : 4'd0),
                .cesel_accum_ab_reg(CE_S),
                .cesel_out_reg_00_15(IS_MULT ? 4'd9 : 4'd0),
                .cesel_out_reg_16_31(IS_MULT ? 4'd9 : 4'd0),
                .cesel_out_reg_32_47(IS_MULT ? 4'd9 : 4'd0),
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
                .rstsel_rndsubload_reg(IS_MULT ? 3'd5 : 3'd0),
                .rstsel_accum_ab_reg(RST_S),
                .rstsel_out_reg_00_15(IS_MULT ? 3'd1 : 3'd0),
                .rstsel_out_reg_16_31(IS_MULT ? 3'd1 : 3'd0),
                .rstsel_out_reg_32_47(IS_MULT ? 3'd1 : 3'd0),
                .rstsel_out_reg_48_63(3'd0)
            ) u_mlp (
                .clk                   (i_clk),
                .din                   (mlp_din),
                .mlpram_bramdin2mlpdin (bram_din2mlpdin),
                .load_ab               (mlp_load_ab),
                .load                  (row_load),
                .sub_ab                (1'b0),
                .sub                   (1'b0),
                .ce                    ({3'b111, word_ce, 8'hFF}),
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
                // cascade up the column (the dout cascade is carried but unused)
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

            // ---- fabric capture of this stage's column sum (2-cycle multicycle path) ----
            if (IS_MULT) begin : g_cap
                reg [47:0] cap;
                always @(posedge i_clk)
                    if (l_pipe[T_CD + CAP_OFF])
                        cap <= mlp_dout[47:0];
                assign o_sums[48*(m-1) +: 48] = cap;
            end
        end
    endgenerate

endmodule
