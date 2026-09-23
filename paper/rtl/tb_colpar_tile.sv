`timescale 1ps/1ps
// Tile-level GEMM check: colpar_row_sequencer.sv driving mlp72_int8_colpar_chain.sv, whose results
// cross to a slower fabric clock through colpar_result_port.sv, against the behavioural
// ACX_MLP72 / ACX_BRAM72K models (paper/rtl/sim_models/) and paper/sw/colpar_tile_golden.py
// (an int64 matmul per stage).  Run: paper/rtl/run_colpar_tile_sim.sh.
//
// Every BRAM72K word is written once with the array clock paused; then each tile is started
// through the sequencer's quasi-static command interface.  Two checkers compare every result
// in order (column p outer, row r inner, stage 1..N): one on the chain output (array clock,
// 1 ns) and one on the result port output (fabric clock, 3 ns).  The fabric side raises a
// random FIFO almost-full, which must stall the sequencer between row passes without
// losing a result.  The sequencer's output is checked against the chain's row protocol.
module tb_colpar_tile;
    parameter integer N_STAGE      = 4;
    parameter integer ADDR_CASCADE = 0;
    parameter integer ADDR_BITS    = 9;
    parameter [4:0]   MULT_MODE    = 5'h00;
    localparam integer WORDS    = 1 << ADDR_BITS;
    localparam integer MAXTILES = 64;
    localparam integer MAXPULSE = 8192;

    reg i_clk = 1'b0, i_wclk = 1'b0, i_fclk = 1'b0, i_rstn = 1'b0;
    reg clk_run = 1'b1;
    always #500  if (clk_run) i_clk = ~i_clk;
    always #850  i_wclk = ~i_wclk;
    always #1500 i_fclk = ~i_fclk;

    reg  [143:0]           i_wdata = '0;
    reg  [ADDR_BITS-1:0]   i_waddr = '0;
    reg  [N_STAGE:0]       i_wen   = '0;

    reg                    start    = 1'b0;
    reg  [ADDR_BITS-1:0]   act_base = '0;
    reg  [ADDR_BITS-1:0]   wt_base  = '0;
    reg  [ADDR_BITS:0]     words    = '0;
    reg  [ADDR_BITS:0]     nrows    = '0;
    reg  [ADDR_BITS:0]     passes   = '0;
    reg  [4:0]             gap      = '0;
    reg                    afull    = 1'b0;
    wire                   hold;
    wire                   s_valid, s_first, s_last, busy, done;
    wire [ADDR_BITS-1:0]   s_act, s_wt;
    wire [48*N_STAGE-1:0]  o_sums, f_sums;
    wire                   o_sums_valid, f_valid, f_overrun;

    colpar_row_sequencer #(.ADDR_BITS(ADDR_BITS)) u_seq (
        .i_clk(i_clk), .i_rstn(i_rstn), .i_start(start), .i_hold(hold),
        .i_act_base(act_base), .i_wt_base(wt_base), .i_words(words), .i_rows(nrows),
        .i_passes(passes), .i_gap(gap),
        .o_valid(s_valid), .o_first(s_first), .o_last(s_last),
        .o_act_raddr(s_act), .o_wt_raddr(s_wt), .o_busy(busy), .o_done(done));

    mlp72_int8_colpar_chain #(
        .N_STAGE(N_STAGE), .ADDR_BITS(ADDR_BITS), .ADDR_CASCADE(ADDR_CASCADE), .MULT_MODE(MULT_MODE)
    ) u_chain (
        .i_clk(i_clk), .i_rstn(i_rstn), .i_wclk(i_wclk), .i_wdata(i_wdata), .i_waddr(i_waddr),
        .i_wen(i_wen), .i_valid(s_valid), .i_act_raddr(s_act), .i_wt_raddr(s_wt),
        .i_first(s_first), .i_last(s_last), .o_sums(o_sums), .o_sums_valid(o_sums_valid));

    colpar_result_port #(.N_STAGE(N_STAGE)) u_port (
        .i_clk_array(i_clk), .i_rstn_array(i_rstn), .i_sums(o_sums), .i_sums_valid(o_sums_valid),
        .o_hold_array(hold),
        .i_clk_fabric(i_fclk), .i_rstn_fabric(i_rstn), .i_fifo_afull(afull),
        .o_sums(f_sums), .o_valid(f_valid), .o_overrun(f_overrun));

    reg [143:0] act   [0:WORDS-1];
    reg [143:0] wt    [0:N_STAGE*WORDS-1];
    reg [31:0]  tiles [0:7*MAXTILES-1];
    reg [47:0]  exp_sum [0:N_STAGE*MAXPULSE-1];
    string  vec;
    integer ntiles, npulses;

    // ------------------------------------------------------------------ checkers
    integer cyc = 0, prev_last = -(1 << 20), n_proto = 0, n_hold_cycles = 0;
    always @(posedge i_clk) begin
        cyc <= cyc + 1;
        if (busy && hold) n_hold_cycles = n_hold_cycles + 1;
        if (i_rstn && s_valid && s_first && cyc - prev_last < 2) begin
            if (n_proto < 10) $display("PROTOCOL: row starts %0d cycle(s) after the previous last word", cyc - prev_last);
            n_proto = n_proto + 1;
        end
        if (i_rstn && s_valid && s_last) begin
            if (cyc - prev_last < N_STAGE) begin
                if (n_proto < 10) $display("PROTOCOL: last words %0d cycles apart (< N_STAGE)", cyc - prev_last);
                n_proto = n_proto + 1;
            end
            prev_last = cyc;
        end
    end

    // array side: the chain output
    integer n_got = 0, n_rows_wrong = 0, n_sums_wrong = 0, s;
    reg     row_bad;
    always @(negedge i_clk) begin
        if (i_rstn && o_sums_valid) begin
            if (n_got >= npulses) begin
                $display("ERROR: extra o_sums_valid at cycle %0d", cyc);
                n_rows_wrong = n_rows_wrong + 1;
            end else begin
                row_bad = 1'b0;
                for (s = 0; s < N_STAGE; s = s + 1) begin
                    if (o_sums[48*s +: 48] !== exp_sum[N_STAGE*n_got + s]) begin
                        if (n_sums_wrong < 10)
                            $display("MISMATCH result %0d stage %0d: got %0d expected %0d", n_got, s + 1,
                                     $signed(o_sums[48*s +: 48]), $signed(exp_sum[N_STAGE*n_got + s]));
                        n_sums_wrong = n_sums_wrong + 1;
                        row_bad = 1'b1;
                    end
                end
                if (row_bad) n_rows_wrong = n_rows_wrong + 1;
            end
            n_got = n_got + 1;
        end
    end

    // fabric side: the result port output
    integer f_got = 0, f_wrong = 0, fs;
    reg     f_bad;
    always @(negedge i_fclk) begin
        if (i_rstn && f_valid) begin
            if (f_got >= npulses) begin
                $display("ERROR: extra fabric result");
                f_wrong = f_wrong + 1;
            end else begin
                f_bad = 1'b0;
                for (fs = 0; fs < N_STAGE; fs = fs + 1)
                    if (f_sums[48*fs +: 48] !== exp_sum[N_STAGE*f_got + fs]) f_bad = 1'b1;
                if (f_bad) begin
                    if (f_wrong < 10) $display("MISMATCH fabric result %0d", f_got);
                    f_wrong = f_wrong + 1;
                end
            end
            f_got = f_got + 1;
        end
    end

    // fabric FIFO almost-full: random bursts
    always @(negedge i_fclk)
        if ($urandom_range(15, 0) == 0) afull <= ~afull;

    // ------------------------------------------------------------------ drivers
    task automatic write_all();
        integer a, m;
        for (m = 0; m <= N_STAGE; m = m + 1) begin
            for (a = 0; a < WORDS; a = a + 1) begin
                @(negedge i_wclk);
                i_waddr  = ADDR_BITS'(a);
                i_wen    = '0;
                i_wen[m] = 1'b1;
                i_wdata  = (m == 0) ? act[a] : wt[(m-1)*WORDS + a];
            end
        end
        @(negedge i_wclk);
        i_wen = '0;
    endtask

    integer t;
    initial begin
        if (!$value$plusargs("vec=%s", vec)) $fatal(1, "+vec=<vector dir> required");
        if (!$value$plusargs("ntiles=%d", ntiles)) $fatal(1, "+ntiles=<n> required");
        if (!$value$plusargs("npulses=%d", npulses)) $fatal(1, "+npulses=<n> required");
        if (ntiles > MAXTILES || npulses > MAXPULSE) $fatal(1, "vector set too large");
        $readmemh({vec, "/act.memh"},   act);
        $readmemh({vec, "/wt.memh"},    wt);
        $readmemh({vec, "/tiles.memh"}, tiles);
        $readmemh({vec, "/exp.memh"},   exp_sum);

        repeat (8) @(posedge i_clk);
        @(negedge i_clk);
        i_rstn = 1'b1;
        repeat (4) @(posedge i_clk);

        clk_run = 1'b0;
        write_all();
        clk_run = 1'b1;
        repeat (8) @(posedge i_clk);

        for (t = 0; t < ntiles; t = t + 1) begin
            @(negedge i_clk);
            act_base = ADDR_BITS'(tiles[7*t]);
            wt_base  = ADDR_BITS'(tiles[7*t+1]);
            words    = (ADDR_BITS+1)'(tiles[7*t+2]);
            nrows    = (ADDR_BITS+1)'(tiles[7*t+3]);
            passes   = (ADDR_BITS+1)'(tiles[7*t+4]);
            gap      = 5'(tiles[7*t+5]);
            start    = 1'b1;
            wait (busy === 1'b1);
            @(negedge i_clk);
            start = 1'b0;
            wait (done === 1'b1);
            repeat (N_STAGE + 64) @(negedge i_clk);
        end

        if (n_got != npulses) begin
            $display("ERROR: %0d o_sums_valid pulses for %0d row passes", n_got, npulses);
            n_rows_wrong = n_rows_wrong + 1;
        end
        if (f_got != npulses) begin
            $display("ERROR: %0d fabric results for %0d row passes", f_got, npulses);
            f_wrong = f_wrong + 1;
        end
        $display("RESULT %s N_STAGE=%0d ADDR_CASCADE=%0d tiles=%0d row_passes=%0d checked=%0d rows_wrong=%0d sums_wrong=%0d fabric_checked=%0d fabric_wrong=%0d overrun=%0d hold_cycles=%0d protocol=%0d",
                 (n_rows_wrong == 0 && f_wrong == 0 && !f_overrun && n_proto == 0) ? "PASS" : "FAIL",
                 N_STAGE, ADDR_CASCADE, ntiles, npulses, n_got, n_rows_wrong, n_sums_wrong,
                 f_got, f_wrong, f_overrun, n_hold_cycles, n_proto);
        $finish;
    end

    initial begin
        #(64'd400_000_000_000);
        $display("RESULT FAIL timeout");
        $fatal(1, "timeout");
    end
endmodule
