// Protocol and register check of pi0_chip_ctrl.sv as the host reaches it: 32-bit PCIe accesses become one
// 256-bit AXI beat with a 4-byte strobe on the NAP master's fabric side.  The testbench is the AXI initiator
// (AW/W/B and AR/R with single beats), writes program bases and start bits dword by dword, checks the byte
// merge, the kept broadcast mask, the go / soft-reset pulses and the per-node cycle counters against a
// behavioural node model (halts a fixed number of cycles after start).
`timescale 1ns/1ps
module tb_pi0_chip_ctrl;
    localparam integer N_NODE = 3;
    reg clk = 0; always #2 clk = ~clk;
    reg rstn = 0;
    reg         arvalid = 0; wire arready; reg [27:0] araddr = 0; reg [7:0] arlen = 0;
    wire        rvalid; reg rready = 1; wire [255:0] rdata; wire [1:0] rresp; wire rlast;
    reg         awvalid = 0; wire awready; reg [27:0] awaddr = 0;
    reg         wvalid = 0; wire wready; reg [255:0] wdata = 0; reg [31:0] wstrb = 0; reg wlast = 1;
    wire        bvalid; reg bready = 1; wire [1:0] bresp;
    wire [42*N_NODE-1:0] prog_base; wire [N_NODE-1:0] prog_start; wire soft_rst; wire [41:0] page_base;
    reg  [N_NODE-1:0] halt = '1, err = '0;
    reg  [31:0] run_left [0:N_NODE-1];

    pi0_chip_ctrl #(.N_NODE(N_NODE), .AXI_ADDR_WIDTH(28), .ID_WORD(24'h533001)) dut (
        .i_clk(clk), .i_rstn(rstn),
        .i_arvalid(arvalid), .o_arready(arready), .i_araddr(araddr), .i_arlen(arlen),
        .o_rvalid(rvalid), .i_rready(rready), .o_rdata(rdata), .o_rresp(rresp), .o_rlast(rlast),
        .i_awvalid(awvalid), .o_awready(awready), .i_awaddr(awaddr),
        .i_wvalid(wvalid), .o_wready(wready), .i_wdata(wdata), .i_wstrb(wstrb), .i_wlast(wlast),
        .o_bvalid(bvalid), .i_bready(bready), .o_bresp(bresp),
        .o_prog_base(prog_base), .o_prog_start(prog_start), .o_soft_rst(soft_rst), .o_page_base(page_base),
        .i_halt(halt), .i_error(err), .i_error_bits({N_NODE{8'h00}}), .i_dbg({N_NODE{128'h0}}));

    // node model: a start clears halt; halt returns after 100 + 10*n cycles
    integer n;
    always @(posedge clk) for (n = 0; n < N_NODE; n = n + 1) begin
        if (prog_start[n]) begin halt[n] <= 1'b0; run_left[n] <= 100 + 10*n; end
        else if (!halt[n]) begin
            if (run_left[n] == 1) halt[n] <= 1'b1;
            run_left[n] <= run_left[n] - 1;
        end
    end

    integer fails = 0;
    task automatic wr32(input [27:0] a, input [31:0] v);   // one 32-bit host write: address then the strobed beat
        integer lane; begin
            lane = a[4:2];
            @(negedge clk); awvalid = 1; awaddr = a;
            wvalid = 1; wdata = {8{v}}; wstrb = 32'hf << (4*lane); wlast = 1;
            // a handshake needs a posedge: the DUT samples valid at the edge, so drop it only at the negedge after
            fork
                begin do @(negedge clk); while (!(awvalid && awready)); awvalid = 0; end
                begin do @(negedge clk); while (!(wvalid && wready)); wvalid = 0; wstrb = 0; end
            join
            while (!bvalid) @(negedge clk);
            @(negedge clk);
        end
    endtask
    task automatic rd32(input [27:0] a, output [31:0] v);
        integer lane; begin
            lane = a[4:2];
            @(negedge clk); arvalid = 1; araddr = a; arlen = 0;
            do @(negedge clk); while (!(arvalid && arready)); arvalid = 0;
            while (!rvalid) @(negedge clk);
            v = rdata[32*lane +: 32];
            @(negedge clk);
        end
    endtask
    task automatic check(input [255:0] got, input [255:0] exp, input string what);
        if (got !== exp) begin fails = fails + 1; $display("FAIL %s: got %h exp %h", what, got, exp); end
    endtask

    reg [31:0] v; integer cyc0, cyc1;
    initial begin
        repeat (4) @(negedge clk); rstn = 1; repeat (2) @(negedge clk);
        // identify: read beat N_NODE dword 7
        rd32(28'(N_NODE*32 + 28), v); check(v, {24'h533001, 8'(N_NODE)}, "id word");
        // program base of node 1 in two dwords, then start
        wr32(28'(1*32 + 0), 32'h1234_5678); wr32(28'(1*32 + 4), 32'h0000_03ab);
        check(prog_base[42 +: 42], 42'h3ab_1234_5678, "node 1 base after two dword writes");
        check(prog_base[0 +: 42], 42'd0, "node 0 base untouched");
        rd32(28'(1*32 + 0), v); check(v, 32'h1234_5678, "read back base lo");
        rd32(28'(1*32 + 4), v); check(v, 32'h0000_03ab, "read back base hi");
        rd32(28'(1*32 + 8), v); check(v[0], 1'b1, "halt before start");
        // start node 1 (dword 2 bit 0 = bit 64)
        @(negedge clk); cyc0 = $time;
        wr32(28'(1*32 + 8), 32'h1);
        check(prog_base[42 +: 42], 42'h3ab_1234_5678, "base kept across the start write");
        rd32(28'(1*32 + 8), v); check(v[0], 1'b0, "running after start");
        while (halt[1] == 0) @(negedge clk);
        rd32(28'(1*32 + 8), v); check(v[0], 1'b1, "halted");
        rd32(28'(1*32 + 12), v); check(v, 32'd110, "cycles from start to halt");
        // broadcast: mask first (dword 3 = bits 96..127), then go (dword 0 bit 0)
        wr32(28'(N_NODE*32 + 12), 32'b101);         // nodes 0 and 2
        check(prog_start, 3'b000, "mask write does not start");
        wr32(28'(N_NODE*32 + 0), 32'h1);
        repeat (3) @(negedge clk);
        check(halt, 3'b010, "go started nodes 0 and 2 only");
        while (halt != 3'b111) @(negedge clk);
        rd32(28'(0*32 + 12), v); check(v, 32'd100, "node 0 cycles");
        rd32(28'(2*32 + 12), v); check(v, 32'd120, "node 2 cycles");
        rd32(28'(N_NODE*32 + 0), v); check(v[N_NODE-1:0], 3'b111, "halt vector");
        // soft reset pulse
        wr32(28'(N_NODE*32 + 0), 32'h2);
        // bridge page register at beat N_NODE + 1, two dword writes, read back
        wr32(28'((N_NODE + 1)*32 + 0), 32'h4000_0000); wr32(28'((N_NODE + 1)*32 + 4), 32'h0000_0002);
        check(page_base, 42'h2_4000_0000, "page base");
        rd32(28'((N_NODE + 1)*32 + 4), v); check(v, 32'h2, "page base read back");
        check(prog_base[42 +: 42], 42'h3ab_1234_5678, "node base untouched by the page write");
        // out-of-range beat clamps onto the broadcast beat
        rd32(28'(40*32 + 28), v); check(v, {24'h533001, 8'(N_NODE)}, "clamped beat reads the id word");
        $display("RESULT %s tb_pi0_chip_ctrl fails=%0d", fails == 0 ? "PASS" : "FAIL", fails);
        $finish;
    end
    reg seen_rst = 0; always @(posedge clk) if (soft_rst) seen_rst <= 1;
    final if (!seen_rst) $display("FAIL soft reset pulse never seen");
    initial begin #200000; $display("RESULT FAIL timeout"); $finish; end
endmodule
