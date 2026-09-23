// pi0_host_gddr_bridge.sv against a byte-strobed GDDR6 slave model with random ready / valid delays.
// The host side is driven the way PCIe PIO arrives: 64-bit writes (an 8-byte strobe inside a 32-byte beat),
// 32-bit reads of one beat, plus a 4-beat full-strobe burst.  Two pages (the second above 8 GB) must not alias.
`timescale 1ns/1ps
module tb_pi0_host_gddr_bridge;
    reg clk = 0; always #2 clk = ~clk;
    reg rstn = 0;
    reg [41:0] page = 42'h0;

    reg         t_arvalid = 0; wire t_arready; reg [27:0] t_araddr = 0; reg [7:0] t_arlen = 0;
    wire        t_rvalid; reg t_rready = 1; wire [255:0] t_rdata; wire [1:0] t_rresp; wire t_rlast;
    reg         t_awvalid = 0; wire t_awready; reg [27:0] t_awaddr = 0; reg [7:0] t_awlen = 0;
    reg         t_wvalid = 0; wire t_wready; reg [255:0] t_wdata = 0; reg [31:0] t_wstrb = 0; reg t_wlast = 0;
    wire        t_bvalid; reg t_bready = 1; wire [1:0] t_bresp;
    wire        m_arvalid, m_rready, m_awvalid, m_wvalid, m_wlast, m_bready;
    wire [41:0] m_araddr, m_awaddr; wire [7:0] m_arlen, m_awlen; wire [2:0] m_arsize, m_awsize; wire [1:0] m_arburst, m_awburst;
    wire [255:0] m_wdata; wire [31:0] m_wstrb;
    reg         m_arready = 0, m_rvalid = 0, m_rlast = 0, m_awready = 0, m_wready = 0, m_bvalid = 0;
    reg [255:0] m_rdata = 0;
    wire [31:0] n_rd, n_wr;

    pi0_host_gddr_bridge dut (
        .i_clk(clk), .i_rstn(rstn), .i_page_base(page),
        .t_arvalid(t_arvalid), .t_arready(t_arready), .t_araddr(t_araddr), .t_arlen(t_arlen), .t_arsize(3'd5), .t_arburst(2'b01),
        .t_rvalid(t_rvalid), .t_rready(t_rready), .t_rdata(t_rdata), .t_rresp(t_rresp), .t_rlast(t_rlast),
        .t_awvalid(t_awvalid), .t_awready(t_awready), .t_awaddr(t_awaddr), .t_awlen(t_awlen), .t_awsize(3'd5), .t_awburst(2'b01),
        .t_wvalid(t_wvalid), .t_wready(t_wready), .t_wdata(t_wdata), .t_wstrb(t_wstrb), .t_wlast(t_wlast),
        .t_bvalid(t_bvalid), .t_bready(t_bready), .t_bresp(t_bresp),
        .m_arvalid(m_arvalid), .m_arready(m_arready), .m_araddr(m_araddr), .m_arlen(m_arlen), .m_arsize(m_arsize), .m_arburst(m_arburst),
        .m_rvalid(m_rvalid), .m_rready(m_rready), .m_rdata(m_rdata), .m_rresp(2'b00), .m_rlast(m_rlast),
        .m_awvalid(m_awvalid), .m_awready(m_awready), .m_awaddr(m_awaddr), .m_awlen(m_awlen), .m_awsize(m_awsize), .m_awburst(m_awburst),
        .m_wvalid(m_wvalid), .m_wready(m_wready), .m_wdata(m_wdata), .m_wstrb(m_wstrb), .m_wlast(m_wlast),
        .m_bvalid(m_bvalid), .m_bready(m_bready), .m_bresp(2'b00),
        .o_n_reads(n_rd), .o_n_writes(n_wr));

    // ---------------------------------------------------------------- byte-strobed slave with random delays
    byte unsigned mem [longint];
    function automatic [255:0] beat_at(input longint a);
        for (int i = 0; i < 32; i++) beat_at[8*i +: 8] = mem.exists(a + i) ? mem[a + i] : 8'h00;
    endfunction
    longint wa, ra; int wl, rl;
    initial begin
        forever begin
            @(posedge clk);
            // write address
            if (m_awvalid && !m_awready && ($urandom_range(3, 0) != 0)) begin m_awready <= 1; end
            else m_awready <= 0;
        end
    end
    always @(posedge clk) begin
        if (m_awvalid && m_awready) begin wa = longint'(m_awaddr); wl = int'(m_awlen) + 1; end
        m_wready <= ($urandom_range(2, 0) != 0);
        if (m_wvalid && m_wready) begin
            for (int i = 0; i < 32; i++) if (m_wstrb[i]) mem[wa + i] = m_wdata[8*i +: 8];
            wa = wa + 32;
            if (m_wlast) m_bvalid <= 1;
        end
        if (m_bvalid && m_bready) m_bvalid <= 0;
        // reads
        m_arready <= m_arvalid && !m_arready && ($urandom_range(3, 0) != 0);
        if (m_arvalid && m_arready) begin ra = longint'(m_araddr); rl = int'(m_arlen) + 1; end
        if (!m_rvalid || m_rready) begin
            if (rl > 0 && ($urandom_range(3, 0) != 0)) begin
                m_rvalid <= 1; m_rdata <= beat_at(ra); m_rlast <= (rl == 1); ra = ra + 32; rl = rl - 1;
            end else m_rvalid <= 0;
        end
    end

    // ---------------------------------------------------------------- host side
    integer fails = 0;
    task automatic wr64(input [27:0] a, input [63:0] v);        // a PIO 64-bit write
        int lane; lane = int'(a[4:3]);
        @(negedge clk); t_awvalid = 1; t_awaddr = {a[27:5], 5'd0}; t_awlen = 0;
        t_wvalid = 1; t_wdata = {4{v}}; t_wstrb = 32'hff << (8 * lane); t_wlast = 1;
        fork
            begin while (!t_awready) @(negedge clk); @(negedge clk); t_awvalid = 0; end
            begin while (!t_wready) @(negedge clk); @(negedge clk); t_wvalid = 0; end
        join
        while (!t_bvalid) @(negedge clk);
        @(negedge clk);
    endtask
    task automatic wrburst4(input [27:0] a, input [1023:0] v);   // a 4-beat full-strobe burst
        @(negedge clk); t_awvalid = 1; t_awaddr = a; t_awlen = 3;
        while (!t_awready) @(negedge clk); @(negedge clk); t_awvalid = 0;
        for (int b = 0; b < 4; b++) begin
            t_wvalid = 1; t_wdata = v[256*b +: 256]; t_wstrb = '1; t_wlast = (b == 3);
            while (!t_wready) @(negedge clk);
            @(negedge clk);
        end
        t_wvalid = 0; t_wlast = 0;
        while (!t_bvalid) @(negedge clk);
        @(negedge clk);
    endtask
    task automatic rdbeat(input [27:0] a, output [255:0] v);
        @(negedge clk); t_arvalid = 1; t_araddr = {a[27:5], 5'd0}; t_arlen = 0;
        while (!t_arready) @(negedge clk); @(negedge clk); t_arvalid = 0;
        while (!t_rvalid) @(negedge clk);
        v = t_rdata;
        @(negedge clk);
    endtask

    reg [255:0] r; reg [1023:0] big;
    initial begin
        repeat (4) @(negedge clk); rstn = 1; repeat (2) @(negedge clk);
        page = 42'h0_0000_0000;
        for (int k = 0; k < 4; k++) wr64(28'h00_1000 + 28'(8 * k), 64'h1111_0000_0000_0000 + 64'(k));
        page = 42'h2_4000_0000;                                  // 9 GB
        for (int k = 0; k < 4; k++) wr64(28'h00_1000 + 28'(8 * k), 64'h9999_0000_0000_0000 + 64'(k));
        for (int b = 0; b < 32; b++) big[32*b +: 32] = 32'hC0DE_0000 + 32'(b);
        wrburst4(28'hFFF_FF80, big);                             // the last 128 bytes of the 256 MB window
        rdbeat(28'h00_1000, r);
        if (r !== {64'h9999_0000_0000_0003, 64'h9999_0000_0000_0002, 64'h9999_0000_0000_0001, 64'h9999_0000_0000_0000}) begin
            fails++; $display("FAIL page 9 GB beat: %h", r); end
        page = 42'h0_0000_0000;
        rdbeat(28'h00_1000, r);
        if (r !== {64'h1111_0000_0000_0003, 64'h1111_0000_0000_0002, 64'h1111_0000_0000_0001, 64'h1111_0000_0000_0000}) begin
            fails++; $display("FAIL page 0 beat (aliased?): %h", r); end
        page = 42'h2_4000_0000;
        for (int b = 0; b < 4; b++) begin
            rdbeat(28'hFFF_FF80 + 28'(32 * b), r);
            if (r !== big[256*b +: 256]) begin fails++; $display("FAIL burst beat %0d: %h", b, r); end
        end
        if (!mem.exists(longint'(42'h2_4000_0000) + 64'h0FFF_FFE0)) begin fails++; $display("FAIL burst not at page + offset"); end
        $display("RESULT %s tb_pi0_host_gddr_bridge fails=%0d reads=%0d writes=%0d", fails == 0 ? "PASS" : "FAIL", fails, n_rd, n_wr);
        $finish;
    end
    initial begin #2000000; $display("RESULT FAIL timeout"); $finish; end
endmodule
