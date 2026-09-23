`timescale 1ps/1ps
// Program-level check of colpar_chain_node.sv: a static command program (prog.memh from
// paper/sw/colpar_tile_golden.py) sets the output address, loads the feeder and all stage images
// from GDDR6, runs four tiles, reloads the feeder with a second activation image, runs three more
// tiles and flushes.  GDDR6 is tb_axi_gddr6_model.sv (outstanding requests, random handshakes);
// afterwards it must hold every record (32-bit sums) in order (exp_prog.memh, int64 matmul).
// The command stream has random valid gaps; the writer FIFO is small, so back-pressure holds the
// sequencer.  Addresses must match the golden: BASE_ACT, BASE_ACT2, BASE_WT, BASE_OUT.
module tb_colpar_node_prog;
    parameter integer N_STAGE        = 4;
    parameter integer ADDR_CASCADE   = 0;
    parameter integer ADDR_BITS      = 9;
    parameter integer RD_BURST_BEATS = 16;
    parameter integer MAX_OUT        = 8;
    parameter integer PROG_GDDR6     = 0;     // 1: the node fetches prog.memh from GDDR6 at BASE_PROG
    parameter integer WR_PIPE        = 0;     // pipeline the weight-load broadcast once per this many stages
    parameter integer VALID_COPIES   = 1;     // copies of the result-port capture enable
    parameter integer WR_SLIM        = 1;     // result writer: record FIFO straight from the port's bank
    localparam integer WORDS    = 1 << ADDR_BITS;
    localparam integer MAXPULSE = 8192;
    localparam integer MAXPROG  = 256;
    localparam [41:0]  BASE_PROG = 42'h0_0800_0000;
    localparam [41:0]  BASE_ACT  = 42'h0_1000_0000;
    localparam [41:0]  BASE_ACT2 = 42'h0_1800_0000;
    localparam [41:0]  BASE_WT   = 42'h0_2000_0000;
    localparam [41:0]  BASE_OUT  = 42'h0_3000_0000;
    localparam integer BPR       = (32 * N_STAGE + 255) / 256;

    reg i_clk = 1'b0, i_fclk = 1'b0, i_rstn = 1'b0;
    always #500  i_clk  = ~i_clk;
    always #1500 i_fclk = ~i_fclk;

    // ------------------------------------------------------------------ program stream
    reg [127:0] prog [0:MAXPROG-1];
    integer     pc = 0;
    reg         cmd_valid = 1'b0;
    reg         prog_start = 1'b0;
    wire        cmd_ready;
    wire        halt, node_error;
    always @(posedge i_fclk) begin
        if (i_rstn) begin
            if (cmd_valid && cmd_ready) pc <= pc + 1;
            cmd_valid <= ($urandom_range(3, 0) != 0);
        end
    end

    // ------------------------------------------------------------------ vectors
    reg [143:0] act   [0:WORDS-1];
    reg [143:0] act_b [0:WORDS-1];
    reg [143:0] wt    [0:N_STAGE*WORDS-1];
    reg [47:0]  exp_sum [0:N_STAGE*MAXPULSE-1];

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
        end else if (a < BASE_ACT) begin
            i       = int'((a - BASE_PROG) / 42'd32);
            beat_at = {prog[(2*i + 1) % MAXPROG], prog[(2*i) % MAXPROG]};
            return beat_at;
        end else if (a >= BASE_ACT2) begin
            i  = int'((a - BASE_ACT2) / 42'd32);
            w0 = act_b[(2*i) % WORDS];
            w1 = act_b[(2*i + 1) % WORDS];
        end else begin
            i  = int'((a - BASE_ACT) / 42'd32);
            w0 = act[(2*i) % WORDS];
            w1 = act[(2*i + 1) % WORDS];
        end
        beat_at = {w1[135:72], w1[63:0], w0[135:72], w0[63:0]};
    endfunction

    // ------------------------------------------------------------------ node + GDDR6
    wire                   arvalid, arready, rvalid, rready, rlast;
    wire                   awvalid, awready, wvalid, wready, wlast, bvalid, bready, wr_fire;
    wire [41:0]            araddr, awaddr, rd_addr, wr_addr;
    wire [7:0]             arlen, awlen;
    wire [2:0]             arsize, awsize;
    wire [1:0]             arburst, awburst;
    wire [255:0]           rdata, rd_beat, wdata, wr_data;
    wire [31:0]            wstrb, m_rbursts, m_wbursts, m_err, m_max_rq, m_max_wq;
    assign rd_beat = beat_at(rd_addr);

    colpar_chain_node #(
        .N_STAGE(N_STAGE), .ADDR_BITS(ADDR_BITS), .ADDR_CASCADE(ADDR_CASCADE),
        .RD_BURST_BEATS(RD_BURST_BEATS), .RD_MAX_OUTSTANDING(MAX_OUT),
        .WR_BURST_BEATS(16), .WR_MAX_OUTSTANDING(MAX_OUT), .WR_FIFO_LOG2((N_STAGE > 16) ? 6 : 5), .WR_AFULL_MARGIN(6),
        .PROG_FROM_GDDR6(PROG_GDDR6), .WR_PIPE_EVERY(WR_PIPE), .VALID_COPIES(VALID_COPIES), .WR_SLIM(WR_SLIM)
    ) dut (
        .i_clk_array(i_clk), .i_clk_fabric(i_fclk), .i_rstn(i_rstn),
        .i_prog_start(prog_start), .i_prog_base(BASE_PROG),
        .i_cmd(prog[pc % MAXPROG]), .i_cmd_valid(cmd_valid && PROG_GDDR6 == 0), .o_cmd_ready(cmd_ready),
        .o_arvalid(arvalid), .i_arready(arready), .o_araddr(araddr), .o_arlen(arlen),
        .o_arsize(arsize), .o_arburst(arburst), .o_rready(rready), .i_rvalid(rvalid),
        .i_rdata(rdata), .i_rresp(2'b00), .i_rlast(rlast),
        .o_awvalid(awvalid), .i_awready(awready), .o_awaddr(awaddr), .o_awlen(awlen),
        .o_awsize(awsize), .o_awburst(awburst), .o_wvalid(wvalid), .i_wready(wready),
        .o_wdata(wdata), .o_wstrb(wstrb), .o_wlast(wlast), .i_bvalid(bvalid),
        .o_bready(bready), .i_bresp(2'b00),
        .o_halt(halt), .o_error(node_error));

    tb_axi_gddr6_model #(.MAX_BEATS((RD_BURST_BEATS > 16) ? RD_BURST_BEATS : 16)) u_mem (
        .i_clk(i_fclk), .i_rstn(i_rstn),
        .i_arvalid(arvalid), .o_arready(arready), .i_araddr(araddr), .i_arlen(arlen),
        .i_arsize(arsize), .i_arburst(arburst), .o_rvalid(rvalid), .i_rready(rready),
        .o_rdata(rdata), .o_rlast(rlast), .o_rd_addr(rd_addr), .i_rd_beat(rd_beat),
        .i_awvalid(awvalid), .o_awready(awready), .i_awaddr(awaddr), .i_awlen(awlen),
        .i_awsize(awsize), .i_awburst(awburst), .i_wvalid(wvalid), .o_wready(wready),
        .i_wdata(wdata), .i_wstrb(wstrb), .i_wlast(wlast), .o_bvalid(bvalid), .i_bready(bready),
        .o_wr_fire(wr_fire), .o_wr_addr(wr_addr), .o_wr_data(wr_data),
        .o_n_rbursts(m_rbursts), .o_n_wbursts(m_wbursts), .o_n_err(m_err),
        .o_max_rq(m_max_rq), .o_max_wq(m_max_wq));

    logic [255:0] gmem [longint];
    always @(posedge i_fclk) if (wr_fire && i_rstn) gmem[longint'(wr_addr >> 5)] = wr_data;   // i_rstn: see tb_vu_node.sv

    // ------------------------------------------------------------------ run and compare
    string  vec;
    integer npulses, g_wrong = 0;
    initial begin
        if (!$value$plusargs("vec=%s", vec)) $fatal(1, "+vec=<vector dir> required");
        if (!$value$plusargs("npulses=%d", npulses)) $fatal(1, "+npulses=<n> required");
        if (npulses > MAXPULSE) $fatal(1, "vector set too large");
        $readmemh({vec, "/act.memh"},      act);
        $readmemh({vec, "/act_b.memh"},    act_b);
        $readmemh({vec, "/wt.memh"},       wt);
        $readmemh({vec, "/prog.memh"},     prog);
        $readmemh({vec, "/exp_prog.memh"}, exp_sum);

        repeat (8) @(posedge i_fclk);
        @(negedge i_fclk);
        i_rstn = 1'b1;

        if (PROG_GDDR6 != 0) begin
            repeat (4) @(negedge i_fclk);
            prog_start = 1'b1;
            @(negedge i_fclk);
            prog_start = 1'b0;
        end
        wait (halt === 1'b1);
        repeat (8) @(negedge i_fclk);

        begin : g_cmp
            integer k, b, st2;
            longint idx0;
            reg [BPR*256-1:0] rec;          // one record (4 beats at 32 stages)
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
        $display("RESULT %s WR_PIPE=%0d VC=%0d PROG_GDDR6=%0d N_STAGE=%0d RD_BURST_BEATS=%0d MAX_OUT=%0d commands=%0d rd_bursts=%0d rd_max_outstanding=%0d wr_bursts=%0d wr_max_outstanding=%0d records=%0d node_error=%0d axi_err=%0d gddr6_wrong=%0d",
                 (g_wrong == 0 && !node_error && m_err == 0) ? "PASS" : "FAIL",
                 WR_PIPE, VALID_COPIES, PROG_GDDR6, N_STAGE, RD_BURST_BEATS, MAX_OUT, pc, m_rbursts, m_max_rq, m_wbursts, m_max_wq,
                 npulses, node_error, m_err, g_wrong);
        $finish;
    end

    initial begin
        #(64'd40_000_000_000);
        $display("RESULT FAIL timeout (pc=%0d) ctrl_st=%0d rec_cnt=%0d rec_exp=%0d wr_idle=%b afull=%b blk_left=%0d assigned=%0d queued=%0d",
                 pc, dut.u_ctrl.st, dut.u_ctrl.rec_cnt, dut.u_ctrl.rec_expected, dut.wr_idle,
                 dut.afull, dut.u_wr.blk_left, dut.u_wr.assigned, dut.u_wr.queued);
        $fatal(1, "timeout");
    end
endmodule
