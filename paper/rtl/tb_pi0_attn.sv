`timescale 1ps/1ps
// One expert ATTENTION BLOCK of the real checkpoint across three nodes on shared GDDR6
// (paper/sw/pi0_attn_golden.py): a 4-lane vector node (vu_node_ml.sv), an int8 chain node and a uint8 x int8
// chain node for PV (colpar_chain_node.sv with MULT_MODE 5'h13).  The nodes share one NAP (colpar_nap_mux.sv)
// in front of tb_axi_gddr6_model.sv.  The host runs one stage at a time: it points the stage's node at that
// stage's program base and waits for its halt (stages.txt).  Every beat of every intermediate and output
// region must match the bit-exact reference, and nothing else may be written.
module tb_pi0_attn;
    parameter F_GELU  = "vu_tbl_gelu.mem";
    parameter F_SIGM  = "vu_tbl_sigm.mem";
    parameter F_EXP   = "vu_tbl_exp.mem";
    parameter F_RSQRT = "vu_tbl_rsqrt.mem";
    parameter F_QUANT = "vu_tbl_quant.mem";
    parameter integer N_LANE  = 4;
    parameter integer MAX_OUT = 8;
    parameter integer MAXSTAGE = 64;
    // vector node slot memories: rows of up to 2^SLOT_BITS elements (the full-size layer's 4096-wide MLP rows
    // need 12)
    parameter integer SLOT_BITS = 11;
    // vector node operand loaders (1, 2 or 4: parallel loading of lane groups, vu_rd_fanout.sv)
    parameter integer N_LD = 1;
    // chain nodes: multiplying stages (16 or 32; deep chains need the weight-bus pipeline and capture-enable copies)
    parameter integer N_STAGE = 16;
    parameter integer WR_PIPE_EVERY = (N_STAGE > 16) ? 4 : 0;
    parameter integer VALID_COPIES = (N_STAGE > 16) ? 4 : 1;

    reg i_clk = 1'b0, i_fclk = 1'b0, i_rstn = 1'b0;
    always #500  i_clk  = ~i_clk;      // 1 GHz array clock
    always #1500 i_fclk = ~i_fclk;     // 333 MHz fabric clock

    logic [255:0] rmem [longint];
    logic [255:0] emem [longint];
    logic [255:0] gmem [longint];

    wire [2:0]      s_arvalid, s_arready, s_rvalid, s_rready, s_awvalid, s_awready, s_wvalid, s_wready;
    wire [2:0]      s_wlast, s_bvalid, s_bready;
    wire [3*42-1:0] s_araddr, s_awaddr;
    wire [3*8-1:0]  s_arlen, s_awlen;
    wire [3*3-1:0]  s_arsize, s_awsize;
    wire [3*2-1:0]  s_arburst, s_awburst;
    wire [3*256-1:0] s_wdata;
    wire [3*32-1:0] s_wstrb;
    wire [255:0]    s_rdata;
    wire [1:0]      s_rresp, s_bresp;
    wire            s_rlast;
    wire [2:0]      halt, err;
    reg  [2:0]      start = 3'b000;
    reg  [41:0]     pbase [0:2];

    // ---- node 0: the vector node (4 lanes) ----
    vu_node_ml #(
        .F_GELU(F_GELU), .F_SIGM(F_SIGM), .F_EXP(F_EXP), .F_RSQRT(F_RSQRT), .F_QUANT(F_QUANT),
        .N_LANE(N_LANE), .SLOT_BITS(SLOT_BITS), .N_LD(N_LD), .RD_MAX_OUTSTANDING(MAX_OUT), .WR_MAX_OUTSTANDING(MAX_OUT),
        .WR_FIFO_LOG2(5)
    ) u_vn (
        .i_clk(i_fclk), .i_rstn(i_rstn), .i_prog_start(start[0]), .i_prog_base(pbase[0]),
        .o_arvalid(s_arvalid[0]), .i_arready(s_arready[0]), .o_araddr(s_araddr[0 +: 42]),
        .o_arlen(s_arlen[0 +: 8]), .o_arsize(s_arsize[0 +: 3]), .o_arburst(s_arburst[0 +: 2]),
        .o_rready(s_rready[0]), .i_rvalid(s_rvalid[0]), .i_rdata(s_rdata), .i_rresp(s_rresp), .i_rlast(s_rlast),
        .o_awvalid(s_awvalid[0]), .i_awready(s_awready[0]), .o_awaddr(s_awaddr[0 +: 42]),
        .o_awlen(s_awlen[0 +: 8]), .o_awsize(s_awsize[0 +: 3]), .o_awburst(s_awburst[0 +: 2]),
        .o_wvalid(s_wvalid[0]), .i_wready(s_wready[0]), .o_wdata(s_wdata[0 +: 256]),
        .o_wstrb(s_wstrb[0 +: 32]), .o_wlast(s_wlast[0]), .i_bvalid(s_bvalid[0]), .o_bready(s_bready[0]),
        .i_bresp(s_bresp), .o_halt(halt[0]), .o_error(err[0]));

    // ---- nodes 1 and 2: the int8 chain and the uint8 (PV) chain ----
    genvar gc;
    generate
        for (gc = 1; gc < 3; gc = gc + 1) begin : g_chain
            colpar_chain_node #(
                .N_STAGE(N_STAGE), .WR_PIPE_EVERY(WR_PIPE_EVERY), .VALID_COPIES(VALID_COPIES), .PROG_FROM_GDDR6(1), .WR_FIFO_LOG2((N_STAGE > 16) ? 6 : 5), .WR_AFULL_MARGIN(6),
                .RD_MAX_OUTSTANDING(MAX_OUT), .WR_MAX_OUTSTANDING(MAX_OUT),
                .MULT_MODE((gc == 2) ? 5'h13 : 5'h00)
            ) u_chain (
                .i_clk_array(i_clk), .i_clk_fabric(i_fclk), .i_rstn(i_rstn),
                .i_prog_start(start[gc]), .i_prog_base(pbase[gc]),
                .i_cmd(128'd0), .i_cmd_valid(1'b0), .o_cmd_ready(),
                .o_arvalid(s_arvalid[gc]), .i_arready(s_arready[gc]), .o_araddr(s_araddr[gc*42 +: 42]),
                .o_arlen(s_arlen[gc*8 +: 8]), .o_arsize(s_arsize[gc*3 +: 3]), .o_arburst(s_arburst[gc*2 +: 2]),
                .o_rready(s_rready[gc]), .i_rvalid(s_rvalid[gc]), .i_rdata(s_rdata), .i_rresp(s_rresp),
                .i_rlast(s_rlast),
                .o_awvalid(s_awvalid[gc]), .i_awready(s_awready[gc]), .o_awaddr(s_awaddr[gc*42 +: 42]),
                .o_awlen(s_awlen[gc*8 +: 8]), .o_awsize(s_awsize[gc*3 +: 3]), .o_awburst(s_awburst[gc*2 +: 2]),
                .o_wvalid(s_wvalid[gc]), .i_wready(s_wready[gc]), .o_wdata(s_wdata[gc*256 +: 256]),
                .o_wstrb(s_wstrb[gc*32 +: 32]), .o_wlast(s_wlast[gc]), .i_bvalid(s_bvalid[gc]),
                .o_bready(s_bready[gc]), .i_bresp(s_bresp), .o_halt(halt[gc]), .o_error(err[gc]));
        end
    endgenerate

    wire          m_arvalid, m_arready, m_rvalid, m_rready, m_rlast;
    wire          m_awvalid, m_awready, m_wvalid, m_wready, m_wlast, m_bvalid, m_bready, wr_fire;
    wire [41:0]   m_araddr, m_awaddr, rd_addr, wr_addr;
    wire [7:0]    m_arlen, m_awlen;
    wire [2:0]    m_arsize, m_awsize;
    wire [1:0]    m_arburst, m_awburst;
    wire [255:0]  m_rdata, m_wdata, wr_data;
    reg  [255:0]  rd_beat;
    wire [31:0]   m_wstrb, n_rb, n_wb, n_err, max_rq, max_wq;

    colpar_nap_mux #(.N_M(3)) u_mux (
        .i_clk(i_fclk), .i_rstn(i_rstn),
        .s_arvalid(s_arvalid), .s_arready(s_arready), .s_araddr(s_araddr), .s_arlen(s_arlen),
        .s_arsize(s_arsize), .s_arburst(s_arburst), .s_rvalid(s_rvalid), .s_rready(s_rready),
        .s_rdata(s_rdata), .s_rresp(s_rresp), .s_rlast(s_rlast),
        .s_awvalid(s_awvalid), .s_awready(s_awready), .s_awaddr(s_awaddr), .s_awlen(s_awlen),
        .s_awsize(s_awsize), .s_awburst(s_awburst), .s_wvalid(s_wvalid), .s_wready(s_wready),
        .s_wdata(s_wdata), .s_wstrb(s_wstrb), .s_wlast(s_wlast), .s_bvalid(s_bvalid),
        .s_bready(s_bready), .s_bresp(s_bresp),
        .m_arvalid(m_arvalid), .m_arready(m_arready), .m_araddr(m_araddr), .m_arlen(m_arlen),
        .m_arsize(m_arsize), .m_arburst(m_arburst), .m_rvalid(m_rvalid), .m_rready(m_rready),
        .m_rdata(m_rdata), .m_rresp(2'b00), .m_rlast(m_rlast),
        .m_awvalid(m_awvalid), .m_awready(m_awready), .m_awaddr(m_awaddr), .m_awlen(m_awlen),
        .m_awsize(m_awsize), .m_awburst(m_awburst), .m_wvalid(m_wvalid), .m_wready(m_wready),
        .m_wdata(m_wdata), .m_wstrb(m_wstrb), .m_wlast(m_wlast), .m_bvalid(m_bvalid),
        .m_bready(m_bready), .m_bresp(2'b00));

    // reads see the nodes' own writes; only rmem is read (an associative-array read in combinational
    // logic inserts default entries, which must not pollute gmem)
    always @* rd_beat = rmem.exists(longint'(rd_addr >> 5)) ? rmem[longint'(rd_addr >> 5)] : 256'd0;

    tb_axi_gddr6_model #(.MAX_BEATS(16)) u_mem (
        .i_clk(i_fclk), .i_rstn(i_rstn),
        .i_arvalid(m_arvalid), .o_arready(m_arready), .i_araddr(m_araddr), .i_arlen(m_arlen),
        .i_arsize(m_arsize), .i_arburst(m_arburst), .o_rvalid(m_rvalid), .i_rready(m_rready),
        .o_rdata(m_rdata), .o_rlast(m_rlast), .o_rd_addr(rd_addr), .i_rd_beat(rd_beat),
        .i_awvalid(m_awvalid), .o_awready(m_awready), .i_awaddr(m_awaddr), .i_awlen(m_awlen),
        .i_awsize(m_awsize), .i_awburst(m_awburst), .i_wvalid(m_wvalid), .o_wready(m_wready),
        .i_wdata(m_wdata), .i_wstrb(m_wstrb), .i_wlast(m_wlast), .o_bvalid(m_bvalid), .i_bready(m_bready),
        .o_wr_fire(wr_fire), .o_wr_addr(wr_addr), .o_wr_data(wr_data),
        .o_n_rbursts(n_rb), .o_n_wbursts(n_wb), .o_n_err(n_err), .o_max_rq(max_rq), .o_max_wq(max_wq));

    // data movement: beats read and written on the shared NAP, for per-stage accounting
    longint unsigned rd_beats = 0, wr_beats = 0;
    always @(posedge i_fclk) begin
        if (m_rvalid && m_rready) rd_beats <= rd_beats + 1;
        if (wr_fire) wr_beats <= wr_beats + 1;
    end

    always @(posedge i_fclk) begin
        if (wr_fire) begin
            gmem[longint'(wr_addr >> 5)] = wr_data;
            rmem[longint'(wr_addr >> 5)] = wr_data;
        end
    end

    task automatic run_stage(input integer n, input longint unsigned base);
        @(negedge i_fclk);
        pbase[n] = 42'(base);
        @(negedge i_fclk);
        start[n] = 1'b1;
        @(negedge i_fclk);
        start[n] = 1'b0;
        wait (halt[n] === 1'b0);          // the node clears halt when it restarts
        wait (halt[n] === 1'b1);
        repeat (16) @(negedge i_fclk);
    endtask

    string  vec;
    integer fd, rc, n_exp = 0, n_wrong = 0, n_extra = 0, n_stage = 0, i;
    integer st_node [0:MAXSTAGE-1];
    integer n_bar = 0, bar_node [0:7];
    longint bar_base [0:7];
    longint st_base [0:MAXSTAGE-1];
    longint idx;
    longint unsigned t_prev = 0, rb_prev = 0, wb_prev = 0;
    logic [255:0] beat;
    initial begin
        if (!$value$plusargs("vec=%s", vec)) $fatal(1, "+vec=<dir> required");
        fd = $fopen({vec, "/mem_in.hex"}, "r");
        while (!$feof(fd)) begin
            rc = $fscanf(fd, "%h %h\n", idx, beat);
            if (rc == 2) rmem[idx] = beat;
        end
        $fclose(fd);
        fd = $fopen({vec, "/mem_exp.hex"}, "r");
        while (!$feof(fd)) begin
            rc = $fscanf(fd, "%h %h\n", idx, beat);
            if (rc == 2) begin emem[idx] = beat; n_exp = n_exp + 1; end
        end
        $fclose(fd);
        fd = $fopen({vec, "/stages.txt"}, "r");
        while (!$feof(fd) && n_stage < MAXSTAGE) begin
            rc = $fscanf(fd, "%d %h\n", st_node[n_stage], st_base[n_stage]);
            if (rc == 2) n_stage = n_stage + 1;
        end
        $fclose(fd);

        repeat (8) @(posedge i_fclk);
        @(negedge i_fclk);
        i_rstn = 1'b1;
        repeat (4) @(negedge i_fclk);

        fd = $fopen({vec, "/barrier.txt"}, "r");
        if (fd != 0) begin                    // one program per node, sequenced by GDDR6 flags
            n_bar = 0;
            while (!$feof(fd) && n_bar < 8) begin
                rc = $fscanf(fd, "%d %h\n", bar_node[n_bar], bar_base[n_bar]);
                if (rc == 2) n_bar = n_bar + 1;
            end
            $fclose(fd);
            @(negedge i_fclk);
            for (i = 0; i < n_bar; i = i + 1) pbase[bar_node[i]] = 42'(bar_base[i]);
            @(negedge i_fclk);
            for (i = 0; i < n_bar; i = i + 1) start[bar_node[i]] = 1'b1;
            @(negedge i_fclk);
            start = 3'b000;
            for (i = 0; i < n_bar; i = i + 1) wait (halt[bar_node[i]] === 1'b1);
            repeat (16) @(negedge i_fclk);
            $display("BARRIER run: %0d node programs, no host between stages", n_bar);
        end else begin
            for (i = 0; i < n_stage; i = i + 1) begin
                run_stage(st_node[i], st_base[i]);
                $display("STAGE %0d node %0d fabric_cycles %0d rd_beats %0d wr_beats %0d (total rd_bursts %0d wr_bursts %0d)",
                         i, st_node[i], ($time - t_prev) / 3000, rd_beats - rb_prev, wr_beats - wb_prev, n_rb, n_wb);
                t_prev = $time; rb_prev = rd_beats; wb_prev = wr_beats;
                $fflush();
                if (err !== 3'b000) $display("NODE ERROR after stage %0d: err=%b", i, err);
            end
        end

        foreach (emem[j]) begin
            if (!gmem.exists(j)) begin
                if (n_wrong < 12) $display("MISSING beat %h", j);
                n_wrong = n_wrong + 1;
            end else if (gmem[j] !== emem[j]) begin
                if (n_wrong < 12) $display("MISMATCH beat %h: got %h expected %h", j, gmem[j], emem[j]);
                n_wrong = n_wrong + 1;
            end
        end
        foreach (gmem[j]) if (!emem.exists(j)) n_extra = n_extra + 1;
        fd = $fopen({vec, "/gmem.txt"}, "w");
        foreach (gmem[j]) $fwrite(fd, "%h %h\n", j, gmem[j]);
        $fclose(fd);
        $display("RESULT %s stages=%0d expected_beats=%0d written_beats=%0d wrong=%0d extra=%0d rd_bursts=%0d wr_bursts=%0d rd_beats=%0d wr_beats=%0d fabric_cycles=%0d axi_err=%0d node_error=%b",
                 (n_wrong == 0 && n_extra == 0 && n_err == 0 && err == 3'b000) ? "PASS" : "FAIL",
                 n_stage, n_exp, gmem.num(), n_wrong, n_extra, n_rb, n_wb, rd_beats, wr_beats, $time / 3000, n_err, err);
        $finish;
    end

    longint unsigned timeout_ps = 64'd400_000_000_000;
    initial begin
        void'($value$plusargs("timeout_ps=%d", timeout_ps));
        #(timeout_ps);
        $display("RESULT FAIL timeout (halt=%b)", halt);
        $fatal(1, "timeout");
    end
endmodule
