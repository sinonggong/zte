`timescale 1ps/1ps
// Bit-exact testbench for paper/rtl/mlp72_int8_chain.sv against the behavioural
// ACX_MLP72 / ACX_BRAM72K models (paper/rtl/sim_models/), driven by vectors from
// paper/sw/mlp72_chain_golden.py.  Run: paper/rtl/run_chain_sim.sh.
//
//   phase 1  write every word of the feeder (activations) and of every stage
//            (weights), content set 0; i_clk is paused.
//   phase 2  compute the phase-2 rows (lower half of the address space) while the
//            upper half of every BRAM72K is rewritten with content set 1 at the
//            same time (the architecture's tile fill during computation: the MLP72
//            load_ab / expb pins, which are the wide write enables, toggle).
//   phase 3  compute the phase-3 rows (upper half, content set 1).
// Rows are separated by the golden's idle gaps; with +inrow_gaps=1 random
// i_valid = 0 cycles are also inserted inside rows.  Each o_sum_valid is checked
// in order against the golden 48-bit sum.  Latency = cycle in which o_sum_valid
// is high minus the cycle in which the row's last word was on the inputs.
module tb_mlp72_int8_chain;
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

    reg  [143:0]          i_wdata = '0;
    reg  [ADDR_BITS-1:0]  i_waddr = '0;
    reg  [N_STAGE:0]      i_wen   = '0;
    reg                   i_valid = 1'b0;
    reg  [ADDR_BITS-1:0]  i_raddr = '0;
    reg                   i_first = 1'b0;
    reg                   i_last  = 1'b0;
    wire [47:0]           o_sum;
    wire                  o_sum_valid;

    mlp72_int8_chain #(
        .N_STAGE(N_STAGE), .ADDR_BITS(ADDR_BITS), .ADDR_CASCADE(ADDR_CASCADE)
    ) dut (.*);

    reg [143:0] act0 [0:WORDS-1];
    reg [143:0] act1 [0:WORDS-1];
    reg [143:0] wt0  [0:N_STAGE*WORDS-1];
    reg [143:0] wt1  [0:N_STAGE*WORDS-1];
    reg [31:0]  rows [0:4*MAXROWS-1];
    reg [47:0]  exp_sum [0:MAXROWS-1];
    string  vec;
    integer nrows, inrow_gaps;

    // ------------------------------------------------------------------ checker
    integer cyc = 0;
    always @(posedge i_clk) cyc <= cyc + 1;

    integer last_cyc [0:MAXROWS-1];
    integer n_got = 0, n_err = 0, lat, lat_min = 1 << 30, lat_max = -1;
    always @(negedge i_clk) begin
        if (i_rstn && o_sum_valid) begin
            if (n_got >= nrows) begin
                $display("ERROR: extra o_sum_valid at cycle %0d (o_sum=%h)", cyc, o_sum);
                n_err = n_err + 1;
            end else begin
                lat = cyc - last_cyc[n_got];
                if (lat < lat_min) lat_min = lat;
                if (lat > lat_max) lat_max = lat;
                if (o_sum !== exp_sum[n_got]) begin
                    if (n_err < 10)
                        $display("MISMATCH row %0d (phase %0d, start %0d, %0d words): got %h (%0d) expected %h (%0d)",
                                 n_got, rows[4*n_got], rows[4*n_got+1], rows[4*n_got+2],
                                 o_sum, $signed(o_sum), exp_sum[n_got], $signed(exp_sum[n_got]));
                    n_err = n_err + 1;
                end
            end
            n_got = n_got + 1;
        end
    end

    // ------------------------------------------------------------------ drivers
    task automatic idle_cycle();
        @(negedge i_clk);
        i_valid = 1'b0; i_first = 1'b0; i_last = 1'b0;
        i_raddr = ADDR_BITS'($urandom);
    endtask

    task automatic play_rows(input integer phase);
        integer r, w, start, nw, gap;
        for (r = 0; r < nrows; r = r + 1) begin
            if (rows[4*r] == phase) begin
                start = rows[4*r+1]; nw = rows[4*r+2]; gap = rows[4*r+3];
                repeat (gap) idle_cycle();
                for (w = 0; w < nw; w = w + 1) begin
                    while (inrow_gaps != 0 && w != 0 && $urandom_range(7, 0) == 0) idle_cycle();
                    @(negedge i_clk);
                    i_valid = 1'b1;
                    i_raddr = ADDR_BITS'(start + w);
                    i_first = (w == 0);
                    i_last  = (w == nw - 1);
                    if (w == nw - 1) last_cyc[r] = cyc;
                end
            end
        end
        idle_cycle();
    endtask

    // m outer (phase 1) or address outer (phase 2, so every BRAM is written
    // throughout the phase-2 computation)
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

        // phase 1 (i_clk paused)
        clk_run = 1'b0;
        write_words(0, 0, WORDS, 0);
        clk_run = 1'b1;
        repeat (8) @(posedge i_clk);

        // phase 2: compute on the lower half while rewriting the upper half
        fork
            write_words(1, HALF, WORDS, 1);
            play_rows(2);
        join
        // phase 3
        play_rows(3);
        repeat (N_STAGE + 64) idle_cycle();

        if (n_got != nrows) begin
            $display("ERROR: %0d o_sum_valid pulses for %0d rows", n_got, nrows);
            n_err = n_err + 1;
        end
        $display("RESULT %s N_STAGE=%0d ADDR_CASCADE=%0d rows=%0d checked=%0d errors=%0d latency_cycles=%0d..%0d",
                 (n_err == 0) ? "PASS" : "FAIL", N_STAGE, ADDR_CASCADE, nrows, n_got, n_err, lat_min, lat_max);
        $finish;
    end

    initial begin
        #(64'd400_000_000_000);
        $display("RESULT FAIL timeout");
        $fatal(1, "timeout");
    end
endmodule
