`timescale 1ps/1ps
// NM colpar_chain_node instances share one NAP through colpar_nap_mux.sv.  Each node runs its
// own program (paper/sw/colpar_tile_golden.py --addr-offset i<<32) against one GDDR6 model
// (tb_axi_gddr6_model.sv); node i's images and outputs live in the address window i<<32.  All
// programs run concurrently with random command-stream, AXI and back-pressure timing; at the
// end every node's records (32-bit sums) must be in its window, in order (int64 matmul).
// The mux is burst-granular, so each node's pipelined requests are serialised through it.
module tb_colpar_nap_share;
    parameter integer NM             = 3;
    parameter integer N_STAGE        = 4;
    parameter integer ADDR_BITS      = 9;
    parameter integer RD_BURST_BEATS = 16;
    localparam integer WORDS    = 1 << ADDR_BITS;
    localparam integer MAXP     = 2048;             // row passes per program, at most
    localparam integer MAXPROG  = 32;
    localparam [31:0]  BASE_ACT  = 32'h1000_0000;
    localparam [31:0]  BASE_ACT2 = 32'h1800_0000;
    localparam [31:0]  BASE_WT   = 32'h2000_0000;
    localparam [31:0]  BASE_OUT  = 32'h3000_0000;
    localparam integer BPR       = (32 * N_STAGE + 255) / 256;

    reg i_clk = 1'b0, i_fclk = 1'b0, i_rstn = 1'b0;
    always #500  i_clk  = ~i_clk;
    always #1500 i_fclk = ~i_fclk;

    // ------------------------------------------------------------------ vectors
    reg [143:0] act   [0:NM*WORDS-1];
    reg [143:0] act_b [0:NM*WORDS-1];
    reg [143:0] wt    [0:NM*N_STAGE*WORDS-1];
    reg [127:0] prog  [0:NM*MAXPROG-1];
    reg [47:0]  exp_sum [0:NM*N_STAGE*MAXP-1];

    function automatic [255:0] beat_at(input [41:0] a);
        integer node, i, sidx;
        reg [143:0] w0, w1;
        reg [31:0]  lo;
        node = int'(a[41:32]) % NM;
        lo   = a[31:0];
        if (lo >= BASE_WT) begin
            sidx = int'((lo - BASE_WT) / 32'd8192) % N_STAGE;
            i    = int'(((lo - BASE_WT) % 32'd8192) / 32'd32);
            w0   = wt[(node*N_STAGE + sidx)*WORDS + (2*i) % WORDS];
            w1   = wt[(node*N_STAGE + sidx)*WORDS + (2*i + 1) % WORDS];
        end else if (lo >= BASE_ACT2) begin
            i  = int'((lo - BASE_ACT2) / 32'd32);
            w0 = act_b[node*WORDS + (2*i) % WORDS];
            w1 = act_b[node*WORDS + (2*i + 1) % WORDS];
        end else begin
            i  = int'((lo - BASE_ACT) / 32'd32);
            w0 = act[node*WORDS + (2*i) % WORDS];
            w1 = act[node*WORDS + (2*i + 1) % WORDS];
        end
        beat_at = {w1[135:72], w1[63:0], w0[135:72], w0[63:0]};
    endfunction

    // ------------------------------------------------------------------ nodes + mux
    wire [NM-1:0]        s_arvalid, s_arready, s_rvalid, s_rready;
    wire [NM*42-1:0]     s_araddr, s_awaddr;
    wire [NM*8-1:0]      s_arlen, s_awlen;
    wire [NM*3-1:0]      s_arsize, s_awsize;
    wire [NM*2-1:0]      s_arburst, s_awburst;
    wire [255:0]         s_rdata;
    wire [1:0]           s_rresp, s_bresp;
    wire                 s_rlast;
    wire [NM-1:0]        s_awvalid, s_awready, s_wvalid, s_wready, s_wlast, s_bvalid, s_bready;
    wire [NM*256-1:0]    s_wdata;
    wire [NM*32-1:0]     s_wstrb;
    wire [NM-1:0]        halt, node_error, cmd_ready;
    reg  [NM-1:0]        cmd_valid = '0;
    integer              pc [0:NM-1];

    wire                 m_arvalid, m_arready, m_rvalid, m_rready, m_rlast;
    wire                 m_awvalid, m_awready, m_wvalid, m_wready, m_wlast, m_bvalid, m_bready, wr_fire;
    wire [41:0]          m_araddr, m_awaddr, rd_addr, wr_addr;
    wire [7:0]           m_arlen, m_awlen;
    wire [2:0]           m_arsize, m_awsize;
    wire [1:0]           m_arburst, m_awburst;
    wire [255:0]         m_rdata, m_wdata, rd_beat, wr_data;
    wire [31:0]          m_wstrb, n_rbursts, n_wbursts, n_err, max_rq, max_wq;
    assign rd_beat = beat_at(rd_addr);

    genvar gi;
    generate
        for (gi = 0; gi < NM; gi = gi + 1) begin : g_node
            always @(posedge i_fclk) begin
                if (!i_rstn) begin
                    pc[gi] <= 0;
                end else begin
                    if (cmd_valid[gi] && cmd_ready[gi]) pc[gi] <= pc[gi] + 1;
                    cmd_valid[gi] <= ($urandom_range(3, 0) != 0);
                end
            end
            colpar_chain_node #(
                .N_STAGE(N_STAGE), .ADDR_BITS(ADDR_BITS), .RD_BURST_BEATS(RD_BURST_BEATS),
                .RD_MAX_OUTSTANDING(8), .WR_BURST_BEATS(16), .WR_MAX_OUTSTANDING(8),
                .WR_FIFO_LOG2(5), .WR_AFULL_MARGIN(6)
            ) u_node (
                .i_clk_array(i_clk), .i_clk_fabric(i_fclk), .i_rstn(i_rstn),
                .i_cmd(prog[gi*MAXPROG + pc[gi]]), .i_cmd_valid(cmd_valid[gi]), .o_cmd_ready(cmd_ready[gi]),
                .o_arvalid(s_arvalid[gi]), .i_arready(s_arready[gi]), .o_araddr(s_araddr[gi*42 +: 42]),
                .o_arlen(s_arlen[gi*8 +: 8]), .o_arsize(s_arsize[gi*3 +: 3]), .o_arburst(s_arburst[gi*2 +: 2]),
                .o_rready(s_rready[gi]), .i_rvalid(s_rvalid[gi]), .i_rdata(s_rdata), .i_rresp(s_rresp),
                .i_rlast(s_rlast),
                .o_awvalid(s_awvalid[gi]), .i_awready(s_awready[gi]), .o_awaddr(s_awaddr[gi*42 +: 42]),
                .o_awlen(s_awlen[gi*8 +: 8]), .o_awsize(s_awsize[gi*3 +: 3]), .o_awburst(s_awburst[gi*2 +: 2]),
                .o_wvalid(s_wvalid[gi]), .i_wready(s_wready[gi]), .o_wdata(s_wdata[gi*256 +: 256]),
                .o_wstrb(s_wstrb[gi*32 +: 32]), .o_wlast(s_wlast[gi]), .i_bvalid(s_bvalid[gi]),
                .o_bready(s_bready[gi]), .i_bresp(s_bresp),
                .o_halt(halt[gi]), .o_error(node_error[gi]));
        end
    endgenerate

    colpar_nap_mux #(.N_M(NM)) u_mux (
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

    tb_axi_gddr6_model #(.MAX_BEATS((RD_BURST_BEATS > 16) ? RD_BURST_BEATS : 16)) u_mem (
        .i_clk(i_fclk), .i_rstn(i_rstn),
        .i_arvalid(m_arvalid), .o_arready(m_arready), .i_araddr(m_araddr), .i_arlen(m_arlen),
        .i_arsize(m_arsize), .i_arburst(m_arburst), .o_rvalid(m_rvalid), .i_rready(m_rready),
        .o_rdata(m_rdata), .o_rlast(m_rlast), .o_rd_addr(rd_addr), .i_rd_beat(rd_beat),
        .i_awvalid(m_awvalid), .o_awready(m_awready), .i_awaddr(m_awaddr), .i_awlen(m_awlen),
        .i_awsize(m_awsize), .i_awburst(m_awburst), .i_wvalid(m_wvalid), .o_wready(m_wready),
        .i_wdata(m_wdata), .i_wstrb(m_wstrb), .i_wlast(m_wlast), .o_bvalid(m_bvalid), .i_bready(m_bready),
        .o_wr_fire(wr_fire), .o_wr_addr(wr_addr), .o_wr_data(wr_data),
        .o_n_rbursts(n_rbursts), .o_n_wbursts(n_wbursts), .o_n_err(n_err),
        .o_max_rq(max_rq), .o_max_wq(max_wq));

    logic [255:0] gmem [longint];
    always @(posedge i_fclk) if (wr_fire) gmem[longint'(wr_addr >> 5)] = wr_data;

    // ------------------------------------------------------------------ run and compare
    string  vecdir;
    integer g_wrong = 0, npulses;
    initial begin
        integer n;
        if (!$value$plusargs("vecdir=%s", vecdir)) $fatal(1, "+vecdir=<prefix; node i reads <prefix>i/> required");
        if (!$value$plusargs("npulses=%d", npulses)) $fatal(1, "+npulses=<records per program> required");
        for (n = 0; n < NM; n = n + 1) begin
            $readmemh($sformatf("%s%0d/act.memh", vecdir, n),      act,     n*WORDS, (n+1)*WORDS - 1);
            $readmemh($sformatf("%s%0d/act_b.memh", vecdir, n),    act_b,   n*WORDS, (n+1)*WORDS - 1);
            $readmemh($sformatf("%s%0d/wt.memh", vecdir, n),       wt,      n*N_STAGE*WORDS, (n+1)*N_STAGE*WORDS - 1);
            $readmemh($sformatf("%s%0d/prog.memh", vecdir, n),     prog,    n*MAXPROG);
            $readmemh($sformatf("%s%0d/exp_prog.memh", vecdir, n), exp_sum, n*N_STAGE*MAXP);
        end

        repeat (8) @(posedge i_fclk);
        @(negedge i_fclk);
        i_rstn = 1'b1;

        wait (&halt === 1'b1);
        repeat (8) @(negedge i_fclk);

        begin : g_cmp
            integer k, b, st2, node;
            longint idx0;
            reg [3*256-1:0] rec;
            for (node = 0; node < NM; node = node + 1) begin
                idx0 = longint'(({10'(node), 32'h0} + {10'd0, BASE_OUT}) >> 5);
                for (k = 0; k < npulses; k = k + 1) begin
                    rec = '0;
                    for (b = 0; b < BPR; b = b + 1) begin
                        if (gmem.exists(idx0 + k*BPR + b)) rec[b*256 +: 256] = gmem[idx0 + k*BPR + b];
                        else rec = 'x;
                    end
                    for (st2 = 0; st2 < N_STAGE; st2 = st2 + 1)
                        if ({{16{rec[32*st2+31]}}, rec[32*st2 +: 32]} !== exp_sum[(node*MAXP + k)*N_STAGE + st2]) begin
                            if (g_wrong < 10) $display("MISMATCH node %0d record %0d stage %0d", node, k, st2 + 1);
                            g_wrong = g_wrong + 1;
                        end
                end
            end
            if (gmem.num() != NM * npulses * BPR) begin
                $display("ERROR: GDDR6 holds %0d beats, expected %0d", gmem.num(), NM * npulses * BPR);
                g_wrong = g_wrong + 1;
            end
        end
        $display("RESULT %s NM=%0d N_STAGE=%0d RD_BURST_BEATS=%0d rd_bursts=%0d wr_bursts=%0d max_outstanding_rd=%0d wr=%0d records=%0d node_error=%b axi_err=%0d gddr6_wrong=%0d",
                 (g_wrong == 0 && node_error == '0 && n_err == 0) ? "PASS" : "FAIL",
                 NM, N_STAGE, RD_BURST_BEATS, n_rbursts, n_wbursts, max_rq, max_wq, NM * npulses,
                 node_error, n_err, g_wrong);
        $finish;
    end

    initial begin
        #(64'd2_000_000_000_000);
        $display("RESULT FAIL timeout");
        $fatal(1, "timeout");
    end
endmodule
