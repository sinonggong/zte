`timescale 1ps/1ps
// Bit-exact testbench for paper/rtl/mlp72_int8_colpar_chain.sv against the
// behavioural ACX_MLP72 / ACX_BRAM72K models (paper/rtl/sim_models/), driven by
// vectors from paper/sw/mlp72_colpar_golden.py.  Run:
// paper/rtl/run_colpar_sim.sh.
//
// Phases as in tb_mlp72_int8_chain.sv:
//   phase 1  write every BRAM72K word (set 0) with i_clk paused;
//   phase 2  compute on the lower half while the upper half of every BRAM72K is
//            rewritten (set 1);
//   phase 3  compute on the upper half.
// Each o_sums_valid is checked against the golden per-stage 48-bit column sums.
// The golden's idle gaps satisfy the RTL's row protocol (>= 1 idle cycle between
// rows, >= N_STAGE cycles between consecutive last words); +inrow_gaps=1 adds
// random idle cycles inside rows.  The checker also flags protocol violations
// of the stimulus itself.
module tb_mlp72_int8_colpar_chain;
    parameter integer N_STAGE      = 4;
    parameter integer ADDR_CASCADE = 0;
    parameter integer ADDR_BITS    = 9;
    localparam integer WORDS   = 1 << ADDR_BITS;
    localparam integer HALF    = WORDS / 2;
    localparam integer MAXROWS = 1024;

    reg i_clk = 1'b0, i_wclk = 1'b0, i_rstn = 1'b0;
    reg clk_run = 1'b1;
    always #500 if (clk_run) i_clk = ~i_clk;
    always #850 i_wclk = ~i_wclk;

    reg  [143:0]             i_wdata = '0;
    reg  [ADDR_BITS-1:0]     i_waddr = '0;
    reg  [N_STAGE:0]         i_wen   = '0;
    reg                      i_valid = 1'b0;
    reg  [ADDR_BITS-1:0]     i_act_raddr = '0;
    reg  [ADDR_BITS-1:0]     i_wt_raddr  = '0;
    reg                      i_first = 1'b0;
    reg                      i_last  = 1'b0;
    wire [48*N_STAGE-1:0]    o_sums;
    wire                     o_sums_valid;

    mlp72_int8_colpar_chain #(
        .N_STAGE(N_STAGE), .ADDR_BITS(ADDR_BITS), .ADDR_CASCADE(ADDR_CASCADE)
    ) dut (.*);

    reg [143:0] act0 [0:WORDS-1];
    reg [143:0] act1 [0:WORDS-1];
    reg [143:0] wt0  [0:N_STAGE*WORDS-1];
    reg [143:0] wt1  [0:N_STAGE*WORDS-1];
    reg [31:0]  rows [0:5*MAXROWS-1];
    reg [47:0]  exp_sum [0:N_STAGE*MAXROWS-1];
    string  vec;
    integer nrows, inrow_gaps;

    // ------------------------------------------------------------------ checker
    integer cyc = 0;
    always @(posedge i_clk) cyc <= cyc + 1;

    integer last_cyc [0:MAXROWS-1];
    integer n_got = 0, n_err = 0, n_stage_err = 0, n_proto = 0;
    integer lat, lat_min = 1 << 30, lat_max = -1;
    integer s;
    reg     row_bad;
    always @(negedge i_clk) begin
        if (i_rstn && o_sums_valid) begin
            if (n_got >= nrows) begin
                $display("ERROR: extra o_sums_valid at cycle %0d", cyc);
                n_err = n_err + 1;
            end else begin
                lat = cyc - last_cyc[n_got];
                if (lat < lat_min) lat_min = lat;
                if (lat > lat_max) lat_max = lat;
                row_bad = 1'b0;
                for (s = 0; s < N_STAGE; s = s + 1) begin
                    if (o_sums[48*s +: 48] !== exp_sum[N_STAGE*n_got + s]) begin
                        if (n_stage_err < 10)
                            $display("MISMATCH row %0d stage %0d (phase %0d, act %0d, wt %0d, %0d words): got %h (%0d) expected %h (%0d)",
                                     n_got, s + 1, rows[5*n_got], rows[5*n_got+1], rows[5*n_got+2], rows[5*n_got+3],
                                     o_sums[48*s +: 48], $signed(o_sums[48*s +: 48]),
                                     exp_sum[N_STAGE*n_got + s], $signed(exp_sum[N_STAGE*n_got + s]));
                        n_stage_err = n_stage_err + 1;
                        row_bad = 1'b1;
                    end
                end
                if (row_bad) n_err = n_err + 1;
            end
            n_got = n_got + 1;
        end
    end

    // ------------------------------------------------------------------ drivers
    task automatic idle_cycle();
        @(negedge i_clk);
        i_valid = 1'b0; i_first = 1'b0; i_last = 1'b0;
        i_act_raddr = ADDR_BITS'($urandom);
        i_wt_raddr  = ADDR_BITS'($urandom);
    endtask

    integer prev_last = -(1 << 20);
    task automatic play_rows(input integer phase);
        integer r, w, astart, wstart, nw, gap;
        for (r = 0; r < nrows; r = r + 1) begin
            if (rows[5*r] == phase) begin
                astart = rows[5*r+1]; wstart = rows[5*r+2]; nw = rows[5*r+3]; gap = rows[5*r+4];
                repeat (gap) idle_cycle();
                for (w = 0; w < nw; w = w + 1) begin
                    while (inrow_gaps != 0 && w != 0 && $urandom_range(7, 0) == 0) idle_cycle();
                    @(negedge i_clk);
                    i_valid = 1'b1;
                    i_act_raddr = ADDR_BITS'(astart + w);
                    i_wt_raddr  = ADDR_BITS'(wstart + w);
                    i_first = (w == 0);
                    i_last  = (w == nw - 1);
                    if (w == 0 && cyc - prev_last < 2) begin
                        $display("PROTOCOL: row %0d starts %0d cycle(s) after the previous last word", r, cyc - prev_last);
                        n_proto = n_proto + 1;
                    end
                    if (w == nw - 1) begin
                        if (cyc - prev_last < N_STAGE) begin
                            $display("PROTOCOL: row %0d last word %0d cycles after the previous one (< N_STAGE)", r, cyc - prev_last);
                            n_proto = n_proto + 1;
                        end
                        last_cyc[r] = cyc;
                        prev_last   = cyc;
                    end
                end
            end
        end
        idle_cycle();
    endtask

    task automatic write_words(input integer set, input integer lo, input integer hi,
                               input integer addr_outer);
        integer a, m, i1, i2, n1, n2;
        n1 = addr_outer ? (hi - lo) : (N_STAGE + 1);
        n2 = addr_outer ? (N_STAGE + 1) : (hi - lo);
        for (i1 = 0; i1 < n1; i1 = i1 + 1) begin
            for (i2 = 0; i2 < n2; i2 = i2 + 1) begin
                a = lo + (addr_outer ? i1 : i2);
                m = addr_outer ? i2 : i1;
                @(negedge i_wclk);
                i_waddr = ADDR_BITS'(a);
                i_wen   = '0;
                i_wen[m] = 1'b1;
                if (m == 0) i_wdata = set ? act1[a] : act0[a];
                else        i_wdata = set ? wt1[(m-1)*WORDS + a] : wt0[(m-1)*WORDS + a];
            end
        end
        @(negedge i_wclk);
        i_wen = '0;
    endtask

    initial begin
        if (!$value$plusargs("vec=%s", vec)) $fatal(1, "+vec=<vector dir> required");
        if (!$value$plusargs("nrows=%d", nrows)) $fatal(1, "+nrows=<n> required");
        if (!$value$plusargs("inrow_gaps=%d", inrow_gaps)) inrow_gaps = 1;
        if (nrows > MAXROWS) $fatal(1, "nrows > MAXROWS");
        $readmemh({vec, "/act0.memh"}, act0);
        $readmemh({vec, "/act1.memh"}, act1);
        $readmemh({vec, "/wt0.memh"},  wt0);
        $readmemh({vec, "/wt1.memh"},  wt1);
        $readmemh({vec, "/rows.memh"}, rows);
        $readmemh({vec, "/exp.memh"},  exp_sum);

        repeat (8) @(posedge i_clk);
        @(negedge i_clk);
        i_rstn = 1'b1;
        repeat (4) @(posedge i_clk);

        clk_run = 1'b0;
        write_words(0, 0, WORDS, 0);
        clk_run = 1'b1;
        repeat (8) @(posedge i_clk);

        fork
            write_words(1, HALF, WORDS, 1);
            play_rows(2);
        join
        play_rows(3);
        repeat (N_STAGE + 64) idle_cycle();

        if (n_got != nrows) begin
            $display("ERROR: %0d o_sums_valid pulses for %0d rows", n_got, nrows);
            n_err = n_err + 1;
        end
        $display("RESULT %s N_STAGE=%0d ADDR_CASCADE=%0d rows=%0d checked=%0d rows_wrong=%0d stage_sums_wrong=%0d protocol=%0d latency_cycles=%0d..%0d",
                 (n_err == 0 && n_proto == 0) ? "PASS" : "FAIL", N_STAGE, ADDR_CASCADE, nrows, n_got,
                 n_err, n_stage_err, n_proto, lat_min, lat_max);
        $finish;
    end

    initial begin
        #(64'd400_000_000_000);
        $display("RESULT FAIL timeout");
        $fatal(1, "timeout");
    end
endmodule
