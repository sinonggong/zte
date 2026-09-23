`timescale 1ps/1ps
// One REAL-WIDTH GEMM across N_NODE chain nodes (paper/sw/pi0_wide_gemm_golden.py): the expert's
// o_proj, 51 x 2048 @ 2048 x 1024, split into sixteen 64-column tiles and, with N_NODE > 1, into row
// slices as well.  Each node runs its own program from GDDR6 over a shared NAP (colpar_nap_mux.sv):
// per tile it loads sixteen stage images, runs one column group over its slice of the token rows, and
// writes a 64-column BLOCK of the row-major 51 x 1024 result at its own row offset (OUT block beats /
// gap).  The host starts the nodes once; afterwards every beat of the result must match the int64
// matmul and nothing outside it may be written.
//
// A row split needs no new command: the opcode-6 column group already takes (act_base, T), and OUT
// already takes a base.  This testbench is what proves that at the real shape.
module tb_pi0_wide_gemm;
    parameter integer N_STAGE    = 16;
    parameter integer N_NODE     = 1;
    parameter integer MAX_OUT    = 8;
    parameter integer TIMEOUT_US = 8000;

    reg i_clk = 1'b0, i_fclk = 1'b0, i_rstn = 1'b0;
    always #500  i_clk  = ~i_clk;      // 1 GHz array clock
    always #1500 i_fclk = ~i_fclk;     // 333 MHz fabric clock

    logic [255:0] rmem [longint];
    logic [255:0] emem [longint];
    logic [255:0] gmem [longint];

    wire [N_NODE-1:0]                 s_arvalid, s_arready, s_rvalid, s_rready;
    wire [N_NODE-1:0]                 s_awvalid, s_awready, s_wvalid, s_wready, s_wlast;
    wire [N_NODE-1:0]                 s_bvalid, s_bready;
    wire [N_NODE*42-1:0]              s_araddr, s_awaddr;
    wire [N_NODE*8-1:0]               s_arlen, s_awlen;
    wire [N_NODE*3-1:0]               s_arsize, s_awsize;
    wire [N_NODE*2-1:0]               s_arburst, s_awburst;
    wire [N_NODE*256-1:0]             s_wdata;
    wire [N_NODE*32-1:0]              s_wstrb;
    wire [255:0]                      s_rdata;
    wire [1:0]                        s_rresp, s_bresp;
    wire                              s_rlast;
    wire [N_NODE-1:0]                 halt, node_err;
    reg  [N_NODE-1:0]                 start = '0;
    reg  [41:0]                       pbase [0:N_NODE-1];

    genvar g;
    generate
        for (g = 0; g < N_NODE; g = g + 1) begin : g_node
            colpar_chain_node #(
                .N_STAGE(N_STAGE), .ADDR_BITS(9), .PROG_FROM_GDDR6(1),
                .RD_MAX_OUTSTANDING(MAX_OUT), .WR_MAX_OUTSTANDING(MAX_OUT),
                .WR_FIFO_LOG2(5), .WR_AFULL_MARGIN(6)
            ) u_chain (
                .i_clk_array(i_clk), .i_clk_fabric(i_fclk), .i_rstn(i_rstn),
                .i_prog_start(start[g]), .i_prog_base(pbase[g]),
                .i_cmd(128'd0), .i_cmd_valid(1'b0), .o_cmd_ready(),
                .o_arvalid(s_arvalid[g]), .i_arready(s_arready[g]), .o_araddr(s_araddr[g*42 +: 42]),
                .o_arlen(s_arlen[g*8 +: 8]), .o_arsize(s_arsize[g*3 +: 3]), .o_arburst(s_arburst[g*2 +: 2]),
                .o_rready(s_rready[g]), .i_rvalid(s_rvalid[g]), .i_rdata(s_rdata), .i_rresp(s_rresp),
                .i_rlast(s_rlast),
                .o_awvalid(s_awvalid[g]), .i_awready(s_awready[g]), .o_awaddr(s_awaddr[g*42 +: 42]),
                .o_awlen(s_awlen[g*8 +: 8]), .o_awsize(s_awsize[g*3 +: 3]), .o_awburst(s_awburst[g*2 +: 2]),
                .o_wvalid(s_wvalid[g]), .i_wready(s_wready[g]), .o_wdata(s_wdata[g*256 +: 256]),
                .o_wstrb(s_wstrb[g*32 +: 32]), .o_wlast(s_wlast[g]), .i_bvalid(s_bvalid[g]),
                .o_bready(s_bready[g]), .i_bresp(s_bresp),
                .o_halt(halt[g]), .o_error(node_err[g]));
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

    colpar_nap_mux #(.N_M(N_NODE)) u_mux (
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

    // the nodes never read what they write here, so an unwritten beat reads as zero
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

    always @(posedge i_fclk) if (wr_fire) gmem[longint'(wr_addr >> 5)] = wr_data;

    string  vec;
    integer fd, rc, n_exp = 0, n_wrong = 0, n_extra = 0, n_prog = 0, i;
    longint idx, base;
    integer node;
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
        while (!$feof(fd) && n_prog < N_NODE) begin
            rc = $fscanf(fd, "%d %h\n", node, base);
            if (rc == 2) begin pbase[node] = 42'(base); n_prog = n_prog + 1; end
        end
        $fclose(fd);
        if (n_prog != N_NODE) $fatal(1, "stages.txt has %0d programs, the testbench has %0d nodes",
                                     n_prog, N_NODE);

        repeat (8) @(posedge i_fclk);
        @(negedge i_fclk);
        i_rstn = 1'b1;
        repeat (4) @(negedge i_fclk);
        start = {N_NODE{1'b1}};          // every node starts at once: no host between them
        @(negedge i_fclk);
        start = '0;
        for (i = 0; i < N_NODE; i = i + 1) wait (halt[i] === 1'b1);
        repeat (16) @(negedge i_fclk);

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
        $display("RESULT %s nodes=%0d expected_beats=%0d written_beats=%0d wrong=%0d extra=%0d rd_bursts=%0d wr_bursts=%0d rd_max_outstanding=%0d wr_max_outstanding=%0d axi_err=%0d node_error=%0d",
                 (n_wrong == 0 && n_extra == 0 && n_err == 0 && node_err == '0) ? "PASS" : "FAIL",
                 N_NODE, n_exp, gmem.num(), n_wrong, n_extra, n_rb, n_wb, max_rq, max_wq, n_err, node_err);
        $finish;
    end

    initial begin
        #(64'(TIMEOUT_US) * 64'd1_000_000);
        $display("RESULT FAIL timeout (halt=%b)", halt);
        $fatal(1, "timeout");
    end
endmodule
