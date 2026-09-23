`timescale 1ps/1ps
// End-to-end check of vu_node.sv (program fetch, operand slot loading by shape, chunked lane runs,
// element and summary writers) against paper/sw/vu_node_golden.py, which builds a GDDR6 image
// (operand tensors, op records) with the bit-exact vector-unit reference and the expected output
// beats.  GDDR6 is tb_axi_gddr6_model.sv (outstanding requests, random handshakes).
//   +vec=<dir>: mem_in.hex (beat-index beat per line), mem_exp.hex (beat-index beat), prog_base.txt
// Every expected beat must be written exactly; nothing else may be written.
module tb_vu_node;
    wire [127:0] dbg_w;                                // vu_node_ml's debug snapshot (0 for the one-lane node)
    always @(dbg_w) if (dbg_w !== 128'd0 && dbg_w !== 'x) $display("DBG op-end %h", dbg_w);
    parameter F_GELU  = "vu_tbl_gelu.mem";
    parameter F_SIGM  = "vu_tbl_sigm.mem";
    parameter F_EXP   = "vu_tbl_exp.mem";
    parameter F_RSQRT = "vu_tbl_rsqrt.mem";
    parameter F_QUANT = "vu_tbl_quant.mem";
    parameter integer MAX_OUT = 8;

    reg i_clk = 1'b0, i_rstn = 1'b0;
    always #1500 i_clk = ~i_clk;

    logic [255:0] rmem [longint];
    logic [255:0] emem [longint];
    logic [255:0] gmem [longint];

    wire          arvalid, arready, rvalid, rready, rlast;
    wire          awvalid, awready, wvalid, wready, wlast, bvalid, bready, wr_fire;
    wire [41:0]   araddr, awaddr, rd_addr, wr_addr;
    wire [7:0]    arlen, awlen;
    wire [2:0]    arsize, awsize;
    wire [1:0]    arburst, awburst;
    wire [255:0]  rdata, wdata, wr_data;
    reg  [255:0]  rd_beat;
    wire [31:0]   wstrb, m_rbursts, m_wbursts, m_err, m_max_rq, m_max_wq;
    wire          halt, node_error;
    reg           prog_start = 1'b0;
    reg  [41:0]   prog_base = '0;

    always @* rd_beat = rmem.exists(longint'(rd_addr >> 5)) ? rmem[longint'(rd_addr >> 5)] : 256'd0;

    // N_LANE = 0: the one-lane reference vu_node.sv; N_LANE >= 1: vu_node_ml.sv with N_LANE lanes
    parameter integer N_LANE = 0;
    // slot memory size: a lane holds rows of up to 2^SLOT_BITS elements (11 -> 2048; the full-size
    // pi0 MLP streams 4096-wide rows and needs 12)
    parameter integer SLOT_BITS = 11;
    // vu_node_ml.sv operand loaders (1, or a divisor of N_LANE: parallel loading of lane groups)
    parameter integer N_LD = 1;
    longint cyc = 0, cyc_load = 0, cyc_run = 0, cyc_drain = 0;
    reg     started = 1'b0;
    generate
        if (N_LANE == 0) begin : g_one
            vu_node #(
                .F_GELU(F_GELU), .F_SIGM(F_SIGM), .F_EXP(F_EXP), .F_RSQRT(F_RSQRT), .F_QUANT(F_QUANT),
                .SLOT_BITS(SLOT_BITS),
                .RD_MAX_OUTSTANDING(MAX_OUT), .WR_MAX_OUTSTANDING(MAX_OUT), .WR_FIFO_LOG2(5)
            ) dut (
                .i_clk(i_clk), .i_rstn(i_rstn), .i_prog_start(prog_start), .i_prog_base(prog_base),
                .o_arvalid(arvalid), .i_arready(arready), .o_araddr(araddr), .o_arlen(arlen),
                .o_arsize(arsize), .o_arburst(arburst), .o_rready(rready), .i_rvalid(rvalid),
                .i_rdata(rdata), .i_rresp(2'b00), .i_rlast(rlast),
                .o_awvalid(awvalid), .i_awready(awready), .o_awaddr(awaddr), .o_awlen(awlen),
                .o_awsize(awsize), .o_awburst(awburst), .o_wvalid(wvalid), .i_wready(wready),
                .o_wdata(wdata), .o_wstrb(wstrb), .o_wlast(wlast), .i_bvalid(bvalid),
                .o_bready(bready), .i_bresp(2'b00),
                .o_halt(halt), .o_error(node_error));
            // cycle accounting by control state (vu_node.sv encoding)
            always @(posedge i_clk) if (started && !halt) begin
                cyc = cyc + 1;
                if (dut.st >= 4'd2 && dut.st <= 4'd6) cyc_load = cyc_load + 1;
                if (dut.st == 4'd7) cyc_run = cyc_run + 1;
                if (dut.st == 4'd8 || dut.st == 4'd9 || dut.st == 4'd10) cyc_drain = cyc_drain + 1;
            end
        end else begin : g_ml
            vu_node_ml #(
                .F_GELU(F_GELU), .F_SIGM(F_SIGM), .F_EXP(F_EXP), .F_RSQRT(F_RSQRT), .F_QUANT(F_QUANT),
                .N_LANE(N_LANE), .SLOT_BITS(SLOT_BITS), .N_LD(N_LD),
                .RD_MAX_OUTSTANDING(MAX_OUT), .WR_MAX_OUTSTANDING(MAX_OUT), .WR_FIFO_LOG2(5)
            ) dut (
                .i_clk(i_clk), .i_rstn(i_rstn), .i_prog_start(prog_start), .i_prog_base(prog_base),
                .o_arvalid(arvalid), .i_arready(arready), .o_araddr(araddr), .o_arlen(arlen),
                .o_arsize(arsize), .o_arburst(arburst), .o_rready(rready), .i_rvalid(rvalid),
                .i_rdata(rdata), .i_rresp(2'b00), .i_rlast(rlast),
                .o_awvalid(awvalid), .i_awready(awready), .o_awaddr(awaddr), .o_awlen(awlen),
                .o_awsize(awsize), .o_awburst(awburst), .o_wvalid(wvalid), .i_wready(wready),
                .o_wdata(wdata), .o_wstrb(wstrb), .o_wlast(wlast), .i_bvalid(bvalid),
                .o_bready(bready), .i_bresp(2'b00),
                .o_halt(halt), .o_error(node_error), .o_error_bits(), .o_dbg(dbg_w));
            // +debug: node state and lane 0's input / PDQ state every 5000 cycles
            bit dbg_on = 1'b0;
            initial dbg_on = $test$plusargs("debug");
            always @(posedge i_clk) if (dbg_on && started && !halt && (cyc % 5000) == 0)
                for (int gi = 0; gi < 2 && gi < N_LANE; gi++)
                    $display("DBG cyc=%0d st=%0d pdq_en=%b lane%0d: in_valid=%b ready=%b l_ready=%b l_idle=%b p_valid=%b infl=%0d fcnt=%0d wp=%0d rp=%0d run_done=%b",
                             cyc, dut.st, dut.pdq_en, gi,
                             gi == 0 ? dut.g_lane[0].in_valid : dut.g_lane[N_LANE > 1 ? 1 : 0].in_valid,
                             gi == 0 ? dut.g_lane[0].lane_ready : dut.g_lane[N_LANE > 1 ? 1 : 0].lane_ready,
                             gi == 0 ? dut.g_lane[0].l_ready : dut.g_lane[N_LANE > 1 ? 1 : 0].l_ready,
                             gi == 0 ? dut.g_lane[0].l_idle : dut.g_lane[N_LANE > 1 ? 1 : 0].l_idle,
                             gi == 0 ? dut.g_lane[0].p_valid : dut.g_lane[N_LANE > 1 ? 1 : 0].p_valid,
                             gi == 0 ? dut.g_lane[0].g_pdq.u_pdq.infl : dut.g_lane[N_LANE > 1 ? 1 : 0].g_pdq.u_pdq.infl,
                             gi == 0 ? dut.g_lane[0].g_pdq.u_pdq.fcnt : dut.g_lane[N_LANE > 1 ? 1 : 0].g_pdq.u_pdq.fcnt,
                             gi == 0 ? dut.g_lane[0].g_pdq.u_pdq.wp : dut.g_lane[N_LANE > 1 ? 1 : 0].g_pdq.u_pdq.wp,
                             gi == 0 ? dut.g_lane[0].g_pdq.u_pdq.rp : dut.g_lane[N_LANE > 1 ? 1 : 0].g_pdq.u_pdq.rp,
                             gi == 0 ? dut.g_lane[0].run_done : dut.g_lane[N_LANE > 1 ? 1 : 0].run_done);
            // cycle accounting by control state (vu_node_ml.sv encoding)
            always @(posedge i_clk) if (started && !halt) begin
                cyc = cyc + 1;
                if ((dut.st >= 5'd4 && dut.st <= 5'd6) || dut.st == 5'd11 || dut.st == 5'd12) cyc_load = cyc_load + 1;
                if (dut.st == 5'd13) cyc_run = cyc_run + 1;
                if (dut.st >= 5'd14 && dut.st <= 5'd16) cyc_drain = cyc_drain + 1;
            end
        end
    endgenerate

    tb_axi_gddr6_model #(.MAX_BEATS(16)) u_mem (
        .i_clk(i_clk), .i_rstn(i_rstn),
        .i_arvalid(arvalid), .o_arready(arready), .i_araddr(araddr), .i_arlen(arlen),
        .i_arsize(arsize), .i_arburst(arburst), .o_rvalid(rvalid), .i_rready(rready),
        .o_rdata(rdata), .o_rlast(rlast), .o_rd_addr(rd_addr), .i_rd_beat(rd_beat),
        .i_awvalid(awvalid), .o_awready(awready), .i_awaddr(awaddr), .i_awlen(awlen),
        .i_awsize(awsize), .i_awburst(awburst), .i_wvalid(wvalid), .o_wready(wready),
        .i_wdata(wdata), .i_wstrb(wstrb), .i_wlast(wlast), .o_bvalid(bvalid), .i_bready(bready),
        .o_wr_fire(wr_fire), .o_wr_addr(wr_addr), .o_wr_data(wr_data),
        .o_n_rbursts(m_rbursts), .o_n_wbursts(m_wbursts), .o_n_err(m_err),
        .o_max_rq(m_max_rq), .o_max_wq(m_max_wq));

    integer n_dup = 0;
    always @(posedge i_clk) begin
        if (wr_fire && i_rstn) begin      // i_rstn: with +verilator+rand+reset the model's outputs are random before the first edge
            if (gmem.exists(longint'(wr_addr >> 5))) n_dup = n_dup + 1;
            gmem[longint'(wr_addr >> 5)] = wr_data;
        end
    end

    string  vec;
    integer fd, rc, n_in = 0, n_exp = 0, n_wrong = 0, n_extra = 0;
    longint idx;
    logic [255:0] beat;
    initial begin
        if (!$value$plusargs("vec=%s", vec)) $fatal(1, "+vec=<dir> required");
        fd = $fopen({vec, "/mem_in.hex"}, "r");
        if (fd == 0) $fatal(1, "no mem_in.hex");
        while (!$feof(fd)) begin
            rc = $fscanf(fd, "%h %h\n", idx, beat);
            if (rc == 2) begin rmem[idx] = beat; n_in = n_in + 1; end
        end
        $fclose(fd);
        fd = $fopen({vec, "/mem_exp.hex"}, "r");
        if (fd == 0) $fatal(1, "no mem_exp.hex");
        while (!$feof(fd)) begin
            rc = $fscanf(fd, "%h %h\n", idx, beat);
            if (rc == 2) begin emem[idx] = beat; n_exp = n_exp + 1; end
        end
        $fclose(fd);
        fd = $fopen({vec, "/prog_base.txt"}, "r");
        rc = $fscanf(fd, "%h", prog_base);
        $fclose(fd);

        repeat (8) @(posedge i_clk);
        @(negedge i_clk);
        i_rstn = 1'b1;
        repeat (4) @(negedge i_clk);
        prog_start = 1'b1;
        started    = 1'b1;
        @(negedge i_clk);
        prog_start = 1'b0;

        wait (halt === 1'b1);
        if (N_LANE != 0) $display("DBG final %h", dbg_w);
        repeat (16) @(negedge i_clk);

        foreach (emem[i]) begin
            if (!gmem.exists(i)) begin
                if (n_wrong < 10) $display("MISSING beat %h", i);
                n_wrong = n_wrong + 1;
            end else if (gmem[i] !== emem[i]) begin
                if (n_wrong < 10) $display("MISMATCH beat %h: got %h expected %h", i, gmem[i], emem[i]);
                n_wrong = n_wrong + 1;
            end
        end
        foreach (gmem[i]) if (!emem.exists(i)) begin
            n_extra = n_extra + 1;
            if (n_extra <= 4) $display("EXTRA beat at %h: %h", i * 32, gmem[i]);
        end
        $display("RESULT %s N_LANE=%0d MAX_OUT=%0d image_beats=%0d expected_beats=%0d written_beats=%0d wrong=%0d extra=%0d rewritten=%0d rd_bursts=%0d wr_bursts=%0d max_outstanding_rd=%0d wr=%0d axi_err=%0d node_error=%0d cycles=%0d load=%0d run=%0d drain_flush=%0d",
                 (n_wrong == 0 && n_extra == 0 && n_dup == 0 && m_err == 0 && !node_error) ? "PASS" : "FAIL",
                 N_LANE, MAX_OUT, n_in, n_exp, gmem.num(), n_wrong, n_extra, n_dup, m_rbursts, m_wbursts,
                 m_max_rq, m_max_wq, m_err, node_error, cyc, cyc_load, cyc_run, cyc_drain);
        $finish;
    end

    initial begin
        #(64'd4_000_000_000_000);
        $display("RESULT FAIL timeout");
        $fatal(1, "timeout");
    end
endmodule
