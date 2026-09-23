`timescale 1ps/1ps
// End-to-end check of one chain node's data path: GDDR6 -> colpar_tile_loader (AXI4 read
// bursts, pipelined) -> chain BRAM72Ks -> colpar_row_sequencer -> mlp72_int8_colpar_chain ->
// colpar_result_port -> fabric clock -> colpar_result_writer (AXI4 write bursts, pipelined) ->
// GDDR6, against paper/sw/colpar_tile_golden.py (int64 matmul) and the behavioural ACX_MLP72 /
// ACX_BRAM72K models.  Run: paper/rtl/run_colpar_node_sim.sh.
//
// GDDR6 is tb_axi_gddr6_model.sv (outstanding requests, random handshakes): one instance on the
// loader's clock, one on the writer's.
//   feeder image  at BASE_ACT, 512 words
//   stage images  at BASE_WT + (s-1) * 8192 bytes
// The loader fills the feeder with one command and all stages with a second one; the tiles then
// run as in tb_colpar_tile.sv.  Every result is checked on the fabric side, and the GDDR6 write
// side must end up holding every record (32-bit sums) in order at BASE_OUT.  The writer FIFO is
// small, so its almost-full holds the sequencer.
module tb_colpar_node;
    parameter integer N_STAGE         = 4;
    parameter integer ADDR_CASCADE    = 0;
    parameter integer ADDR_BITS       = 9;
    parameter integer BEATS_PER_BURST = 16;
    parameter integer MAX_OUT         = 8;
    parameter integer WR_BURST_BEATS  = 16;
    parameter integer WR_SLIM         = 1;    // result writer takes records straight from the port's bank
    localparam integer WORDS    = 1 << ADDR_BITS;
    localparam integer MAXTILES = 64;
    localparam integer MAXPULSE = 8192;
    localparam [41:0]  BASE_ACT = 42'h0_1000_0000;
    localparam [41:0]  BASE_WT  = 42'h0_2000_0000;
    localparam [41:0]  BASE_OUT = 42'h0_3000_0000;
    localparam integer BPR      = (32 * N_STAGE + 255) / 256;

    reg i_clk = 1'b0, i_wclk = 1'b0, i_fclk = 1'b0, i_rstn = 1'b0;
    always #500  i_clk  = ~i_clk;
    always #1700 i_wclk = ~i_wclk;
    always #1500 i_fclk = ~i_fclk;

    wire [143:0]           i_wdata;
    wire [ADDR_BITS-1:0]   i_waddr;
    wire [N_STAGE:0]       i_wen;

    reg                    start    = 1'b0;
    reg  [ADDR_BITS-1:0]   act_base = '0;
    reg  [ADDR_BITS-1:0]   wt_base  = '0;
    reg  [ADDR_BITS:0]     words    = '0;
    reg  [ADDR_BITS:0]     nrows    = '0;
    reg  [ADDR_BITS:0]     passes   = '0;
    reg  [4:0]             gap      = '0;
    reg                    flush    = 1'b0;
    wire                   afull, hold;
    wire                   s_valid, s_first, s_last, busy, done;
    wire [ADDR_BITS-1:0]   s_act, s_wt;
    wire [48*N_STAGE-1:0]  o_sums, f_sums;
    wire                   o_sums_valid, f_valid, f_overrun;

    reg [143:0] act   [0:WORDS-1];
    reg [143:0] wt    [0:N_STAGE*WORDS-1];
    reg [31:0]  tiles [0:7*MAXTILES-1];
    reg [47:0]  exp_sum [0:N_STAGE*MAXPULSE-1];
    string  vec;
    integer ntiles, npulses;

    function automatic [255:0] beat_at(input [41:0] a);
        integer i, sidx;
        reg [143:0] w0, w1;
        reg [41:0]  rel;
        if (a >= BASE_WT) begin
            rel  = a - BASE_WT;
            sidx = int'(rel / 42'd8192) % N_STAGE;
            i    = int'((rel % 42'd8192) / 42'd32);
            w0   = wt[sidx*WORDS + (2*i) % WORDS];
            w1   = wt[sidx*WORDS + (2*i + 1) % WORDS];
        end else begin
            i  = int'((a - BASE_ACT) / 42'd32);
            w0 = act[(2*i) % WORDS];
            w1 = act[(2*i + 1) % WORDS];
        end
        beat_at = {w1[135:72], w1[63:0], w0[135:72], w0[63:0]};
    endfunction

    // ------------------------------------------------------------------ loader + GDDR6 read side
    reg                    ld_arm = 1'b0;
    reg  [41:0]            ld_base = '0;
    reg  [4:0]             ld_first = '0, ld_nreg = '0;
    reg  [ADDR_BITS:0]     ld_nwords = '0;
    wire                   ld_busy, ld_done, ld_error;
    wire                   arvalid, arready, rvalid, rready, rlast;
    wire [41:0]            araddr, rd_addr;
    wire [7:0]             arlen;
    wire [2:0]             arsize;
    wire [1:0]             arburst;
    wire [255:0]           rdata, rd_beat;
    wire [31:0]            rm_rbursts, rm_err, rm_max_rq;
    assign rd_beat = beat_at(rd_addr);

    colpar_tile_loader #(
        .N_STAGE(N_STAGE), .ADDR_BITS(ADDR_BITS), .BEATS_PER_BURST(BEATS_PER_BURST), .MAX_OUTSTANDING(MAX_OUT)
    ) u_ld (
        .i_clk(i_wclk), .i_rstn(i_rstn), .i_arm(ld_arm), .i_base(ld_base), .i_first_target(ld_first),
        .i_n_regions(ld_nreg), .i_n_words(ld_nwords), .i_n_segs((ADDR_BITS+1)'(1)), .i_seg_step(14'd0),
        .i_tgt_step(14'(ld_nwords) + 14'(ld_nwords[0])), .i_wbase('0),
        .o_arvalid(arvalid), .i_arready(arready), .o_araddr(araddr), .o_arlen(arlen),
        .o_arsize(arsize), .o_arburst(arburst), .o_rready(rready), .i_rvalid(rvalid),
        .i_rdata(rdata), .i_rresp(2'b00), .i_rlast(rlast),
        .o_wdata(i_wdata), .o_waddr(i_waddr), .o_wen(i_wen),
        .o_busy(ld_busy), .o_done(ld_done), .o_error(ld_error));

    tb_axi_gddr6_model #(.MAX_BEATS(BEATS_PER_BURST)) u_rmem (
        .i_clk(i_wclk), .i_rstn(i_rstn),
        .i_arvalid(arvalid), .o_arready(arready), .i_araddr(araddr), .i_arlen(arlen),
        .i_arsize(arsize), .i_arburst(arburst), .o_rvalid(rvalid), .i_rready(rready),
        .o_rdata(rdata), .o_rlast(rlast), .o_rd_addr(rd_addr), .i_rd_beat(rd_beat),
        .i_awvalid(1'b0), .o_awready(), .i_awaddr(42'd0), .i_awlen(8'd0), .i_awsize(3'd0),
        .i_awburst(2'd0), .i_wvalid(1'b0), .o_wready(), .i_wdata(256'd0), .i_wstrb(32'd0),
        .i_wlast(1'b0), .o_bvalid(), .i_bready(1'b0),
        .o_wr_fire(), .o_wr_addr(), .o_wr_data(),
        .o_n_rbursts(rm_rbursts), .o_n_wbursts(), .o_n_err(rm_err), .o_max_rq(rm_max_rq), .o_max_wq());

    // ------------------------------------------------------------------ node
    colpar_row_sequencer #(.ADDR_BITS(ADDR_BITS)) u_seq (
        .i_clk(i_clk), .i_rstn(i_rstn), .i_start(start), .i_hold(hold),
        .i_act_base(act_base), .i_wt_base(wt_base), .i_words(words), .i_rows(nrows),
        .i_passes(passes), .i_gap(gap),
        .o_valid(s_valid), .o_first(s_first), .o_last(s_last),
        .o_act_raddr(s_act), .o_wt_raddr(s_wt), .o_busy(busy), .o_done(done));

    mlp72_int8_colpar_chain #(
        .N_STAGE(N_STAGE), .ADDR_BITS(ADDR_BITS), .ADDR_CASCADE(ADDR_CASCADE)
    ) u_chain (
        .i_clk(i_clk), .i_rstn(i_rstn), .i_wclk(i_wclk), .i_wdata(i_wdata), .i_waddr(i_waddr),
        .i_wen(i_wen), .i_valid(s_valid), .i_act_raddr(s_act), .i_wt_raddr(s_wt),
        .i_first(s_first), .i_last(s_last), .o_sums(o_sums), .o_sums_valid(o_sums_valid));

    colpar_result_port #(.N_STAGE(N_STAGE), .FABRIC_COPY(WR_SLIM ? 0 : 1)) u_port (
        .i_clk_array(i_clk), .i_rstn_array(i_rstn), .i_sums(o_sums), .i_sums_valid(o_sums_valid),
        .o_hold_array(hold),
        .i_clk_fabric(i_fclk), .i_rstn_fabric(i_rstn), .i_fifo_afull(afull),
        .o_sums(f_sums), .o_valid(f_valid), .o_overrun(f_overrun));

    // ------------------------------------------------------------------ writer + GDDR6 write side
    wire                   awvalid, awready, wvalid, wready, wlast, bvalid, bready;
    wire                   wr_fire, wr_idle, wr_error;
    wire [41:0]            awaddr, wr_addr;
    wire [7:0]             awlen;
    wire [2:0]             awsize;
    wire [1:0]             awburst;
    wire [255:0]           wdata, wr_data;
    wire [31:0]            wstrb, wm_wbursts, wm_err, wm_max_wq;

    colpar_result_writer #(
        .N_STAGE(N_STAGE), .SUM_BITS(32), .FIFO_LOG2(5), .BURST_BEATS(WR_BURST_BEATS),
        .MAX_OUTSTANDING(MAX_OUT), .AFULL_MARGIN(6), .RECORD_FIFO(WR_SLIM)
    ) u_wr (
        .i_clk(i_fclk), .i_rstn(i_rstn), .i_out_base(BASE_OUT), .i_blk_beats(10'd0), .i_gap_bytes(24'd0), .i_restart(1'b0), .i_flush(flush),
        .i_sums(f_sums), .i_sums_valid(f_valid), .o_afull(afull),
        .o_awvalid(awvalid), .i_awready(awready), .o_awaddr(awaddr), .o_awlen(awlen),
        .o_awsize(awsize), .o_awburst(awburst), .o_wvalid(wvalid), .i_wready(wready),
        .o_wdata(wdata), .o_wstrb(wstrb), .o_wlast(wlast), .i_bvalid(bvalid),
        .o_bready(bready), .i_bresp(2'b00), .o_idle(wr_idle), .o_error(wr_error));

    tb_axi_gddr6_model #(.MAX_BEATS(WR_BURST_BEATS)) u_wmem (
        .i_clk(i_fclk), .i_rstn(i_rstn),
        .i_arvalid(1'b0), .o_arready(), .i_araddr(42'd0), .i_arlen(8'd0), .i_arsize(3'd0),
        .i_arburst(2'd0), .o_rvalid(), .i_rready(1'b0), .o_rdata(), .o_rlast(), .o_rd_addr(),
        .i_rd_beat(256'd0),
        .i_awvalid(awvalid), .o_awready(awready), .i_awaddr(awaddr), .i_awlen(awlen),
        .i_awsize(awsize), .i_awburst(awburst), .i_wvalid(wvalid), .o_wready(wready),
        .i_wdata(wdata), .i_wstrb(wstrb), .i_wlast(wlast), .o_bvalid(bvalid), .i_bready(bready),
        .o_wr_fire(wr_fire), .o_wr_addr(wr_addr), .o_wr_data(wr_data),
        .o_n_rbursts(), .o_n_wbursts(wm_wbursts), .o_n_err(wm_err), .o_max_rq(), .o_max_wq(wm_max_wq));

    logic [255:0] gmem [longint];
    always @(posedge i_fclk) if (wr_fire) gmem[longint'(wr_addr >> 5)] = wr_data;

    integer n_hold_cycles = 0;
    always @(posedge i_clk) if (busy && hold) n_hold_cycles = n_hold_cycles + 1;

    // ------------------------------------------------------------------ fabric-side checker
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

    task automatic load(input [41:0] base, input integer first, input integer nreg, input integer nwords);
        @(negedge i_wclk);
        ld_base   = base;
        ld_first  = 5'(first);
        ld_nreg   = 5'(nreg);
        ld_nwords = (ADDR_BITS+1)'(nwords);
        ld_arm    = 1'b1;
        @(negedge i_wclk);
        ld_arm = 1'b0;
        @(posedge ld_done);
        @(negedge i_wclk);
    endtask

    integer t, g_wrong = 0;
    initial begin
        if (!$value$plusargs("vec=%s", vec)) $fatal(1, "+vec=<vector dir> required");
        if (!$value$plusargs("ntiles=%d", ntiles)) $fatal(1, "+ntiles=<n> required");
        if (!$value$plusargs("npulses=%d", npulses)) $fatal(1, "+npulses=<n> required");
        if (ntiles > MAXTILES || npulses > MAXPULSE) $fatal(1, "vector set too large");
        $readmemh({vec, "/act.memh"},   act);
        $readmemh({vec, "/wt.memh"},    wt);
        $readmemh({vec, "/tiles.memh"}, tiles);
        $readmemh({vec, "/exp.memh"},   exp_sum);

        repeat (8) @(posedge i_wclk);
        @(negedge i_wclk);
        i_rstn = 1'b1;
        repeat (4) @(posedge i_wclk);

        load(BASE_ACT, 0, 1, WORDS);            // feeder
        load(BASE_WT, 1, N_STAGE, WORDS);       // all stages in one command
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

        if (f_got != npulses) begin
            $display("ERROR: %0d fabric results for %0d row passes", f_got, npulses);
            f_wrong = f_wrong + 1;
        end

        // drain the writer and compare GDDR6
        @(negedge i_fclk);
        flush = 1'b1;
        repeat (4) @(negedge i_fclk);
        wait (wr_idle === 1'b1);
        repeat (4) @(negedge i_fclk);
        begin : g_cmp
            integer k, b, st2;
            longint idx0;
            reg [3*256-1:0] rec;
            idx0 = longint'(BASE_OUT >> 5);
            for (k = 0; k < npulses; k = k + 1) begin
                rec = '0;
                for (b = 0; b < BPR; b = b + 1) begin
                    if (gmem.exists(idx0 + k*BPR + b)) rec[b*256 +: 256] = gmem[idx0 + k*BPR + b];
                    else rec = 'x;
                end
                for (st2 = 0; st2 < N_STAGE; st2 = st2 + 1)
                    if ({{16{rec[32*st2+31]}}, rec[32*st2 +: 32]} !== exp_sum[N_STAGE*k + st2]) begin
                        if (g_wrong < 10) $display("MISMATCH GDDR6 record %0d stage %0d", k, st2 + 1);
                        g_wrong = g_wrong + 1;
                    end
            end
            if (gmem.num() != npulses * BPR) begin
                $display("ERROR: GDDR6 holds %0d beats, expected %0d", gmem.num(), npulses * BPR);
                g_wrong = g_wrong + 1;
            end
        end
        $display("RESULT %s N_STAGE=%0d BEATS_PER_BURST=%0d MAX_OUT=%0d rd_bursts=%0d rd_max_outstanding=%0d loader_error=%0d row_passes=%0d fabric_wrong=%0d overrun=%0d wr_bursts=%0d wr_max_outstanding=%0d wr_error=%0d axi_err=%0d gddr6_wrong=%0d hold_cycles=%0d",
                 (f_wrong == 0 && !f_overrun && !ld_error && rm_err == 0 && !wr_error && wm_err == 0 && g_wrong == 0) ? "PASS" : "FAIL",
                 N_STAGE, BEATS_PER_BURST, MAX_OUT, rm_rbursts, rm_max_rq, ld_error, npulses, f_wrong,
                 f_overrun, wm_wbursts, wm_max_wq, wr_error, rm_err + wm_err, g_wrong, n_hold_cycles);
        $finish;
    end

    initial begin
        #(64'd800_000_000_000);
        $display("RESULT FAIL timeout");
        $fatal(1, "timeout");
    end
endmodule
