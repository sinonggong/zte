`timescale 1ps/1ps
// One pi0 linear layer across node types on shared GDDR6 (paper/sw/pi0_linear_golden.py):
//   vector node 0 QUANT -> chain node GEMM (reads the int8 code region directly) -> vector node 1 DEQUANT
// Three nodes, each running its own program from GDDR6, share one NAP (colpar_nap_mux.sv) in front of
// tb_axi_gddr6_model.sv.  The host sequence starts each node after the previous one halts.  Every beat of
// the intermediate regions (QUANT codes and row scales, chain sums) and of the final bf16 outputs must match
// the bit-exact reference; nothing else may be written.
module tb_pi0_linear;
    parameter F_GELU  = "vu_tbl_gelu.mem";
    parameter F_SIGM  = "vu_tbl_sigm.mem";
    parameter F_EXP   = "vu_tbl_exp.mem";
    parameter F_RSQRT = "vu_tbl_rsqrt.mem";
    parameter F_QUANT = "vu_tbl_quant.mem";

    reg i_clk = 1'b0, i_fclk = 1'b0, i_rstn = 1'b0;
    always #500  i_clk  = ~i_clk;
    always #1500 i_fclk = ~i_fclk;

    logic [255:0] rmem [longint];
    logic [255:0] emem [longint];
    logic [255:0] gmem [longint];

    // ---- three masters on the mux ----
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

    colpar_chain_node #(
        .N_STAGE(16), .PROG_FROM_GDDR6(1), .WR_FIFO_LOG2(5), .WR_AFULL_MARGIN(6)
    ) u_chain (
        .i_clk_array(i_clk), .i_clk_fabric(i_fclk), .i_rstn(i_rstn),
        .i_prog_start(start[1]), .i_prog_base(pbase[1]),
        .i_cmd(128'd0), .i_cmd_valid(1'b0), .o_cmd_ready(),
        .o_arvalid(s_arvalid[1]), .i_arready(s_arready[1]), .o_araddr(s_araddr[42 +: 42]),
        .o_arlen(s_arlen[8 +: 8]), .o_arsize(s_arsize[3 +: 3]), .o_arburst(s_arburst[2 +: 2]),
        .o_rready(s_rready[1]), .i_rvalid(s_rvalid[1]), .i_rdata(s_rdata), .i_rresp(s_rresp), .i_rlast(s_rlast),
        .o_awvalid(s_awvalid[1]), .i_awready(s_awready[1]), .o_awaddr(s_awaddr[42 +: 42]),
        .o_awlen(s_awlen[8 +: 8]), .o_awsize(s_awsize[3 +: 3]), .o_awburst(s_awburst[2 +: 2]),
        .o_wvalid(s_wvalid[1]), .i_wready(s_wready[1]), .o_wdata(s_wdata[256 +: 256]),
        .o_wstrb(s_wstrb[32 +: 32]), .o_wlast(s_wlast[1]), .i_bvalid(s_bvalid[1]), .o_bready(s_bready[1]),
        .i_bresp(s_bresp), .o_halt(halt[1]), .o_error(err[1]));

    genvar gv;
    generate
        for (gv = 0; gv < 3; gv = gv + 2) begin : g_vn
            vu_node #(
                .F_GELU(F_GELU), .F_SIGM(F_SIGM), .F_EXP(F_EXP), .F_RSQRT(F_RSQRT), .F_QUANT(F_QUANT),
                .WR_FIFO_LOG2(5)
            ) u_vn (
                .i_clk(i_fclk), .i_rstn(i_rstn), .i_prog_start(start[gv]), .i_prog_base(pbase[gv]),
                .o_arvalid(s_arvalid[gv]), .i_arready(s_arready[gv]), .o_araddr(s_araddr[gv*42 +: 42]),
                .o_arlen(s_arlen[gv*8 +: 8]), .o_arsize(s_arsize[gv*3 +: 3]), .o_arburst(s_arburst[gv*2 +: 2]),
                .o_rready(s_rready[gv]), .i_rvalid(s_rvalid[gv]), .i_rdata(s_rdata), .i_rresp(s_rresp),
                .i_rlast(s_rlast),
                .o_awvalid(s_awvalid[gv]), .i_awready(s_awready[gv]), .o_awaddr(s_awaddr[gv*42 +: 42]),
                .o_awlen(s_awlen[gv*8 +: 8]), .o_awsize(s_awsize[gv*3 +: 3]), .o_awburst(s_awburst[gv*2 +: 2]),
                .o_wvalid(s_wvalid[gv]), .i_wready(s_wready[gv]), .o_wdata(s_wdata[gv*256 +: 256]),
                .o_wstrb(s_wstrb[gv*32 +: 32]), .o_wlast(s_wlast[gv]), .i_bvalid(s_bvalid[gv]),
                .o_bready(s_bready[gv]), .i_bresp(s_bresp), .o_halt(halt[gv]), .o_error(err[gv]));
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

    // GDDR6 reads see the nodes' own writes (the QUANT codes feed the chain, its sums feed DEQUANT):
    // writes are mirrored into rmem and reads only touch rmem, because reading an associative array
    // in combinational logic can insert default entries, which must not pollute gmem
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

    integer n_dup = 0;
    always @(posedge i_fclk) begin
        if (wr_fire) begin
            if (gmem.exists(longint'(wr_addr >> 5))) n_dup = n_dup + 1;
            gmem[longint'(wr_addr >> 5)] = wr_data;
            rmem[longint'(wr_addr >> 5)] = wr_data;
        end
    end

    task automatic run_node(input integer n);
        @(negedge i_fclk);
        start[n] = 1'b1;
        @(negedge i_fclk);
        start[n] = 1'b0;
        wait (halt[n] === 1'b1);
        repeat (8) @(negedge i_fclk);
    endtask

    string  vec;
    integer fd, rc, n_exp = 0, n_wrong = 0, n_extra = 0;
    longint idx;
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
        fd = $fopen({vec, "/progs.txt"}, "r");
        rc = $fscanf(fd, "%h\n%h\n%h\n", pbase[0], pbase[1], pbase[2]);
        $fclose(fd);

        repeat (8) @(posedge i_fclk);
        @(negedge i_fclk);
        i_rstn = 1'b1;
        repeat (4) @(negedge i_fclk);

        run_node(0);   // QUANT
        run_node(1);   // chain GEMM
        run_node(2);   // DEQUANT

        foreach (emem[i]) begin
            if (!gmem.exists(i)) begin
                if (n_wrong < 10) $display("MISSING beat %h", i);
                n_wrong = n_wrong + 1;
            end else if (gmem[i] !== emem[i]) begin
                if (n_wrong < 10) $display("MISMATCH beat %h: got %h expected %h", i, gmem[i], emem[i]);
                n_wrong = n_wrong + 1;
            end
        end
        foreach (gmem[i]) if (!emem.exists(i)) n_extra = n_extra + 1;
        $display("RESULT %s expected_beats=%0d written_beats=%0d wrong=%0d extra=%0d rewritten=%0d rd_bursts=%0d wr_bursts=%0d axi_err=%0d node_error=%b",
                 (n_wrong == 0 && n_extra == 0 && n_dup == 0 && n_err == 0 && err == 3'b000) ? "PASS" : "FAIL",
                 n_exp, gmem.num(), n_wrong, n_extra, n_dup, n_rb, n_wb, n_err, err);
        $finish;
    end

    initial begin
        #(64'd4_000_000_000_000);
        $display("RESULT FAIL timeout (halt=%b)", halt);
        $fatal(1, "timeout");
    end
endmodule
