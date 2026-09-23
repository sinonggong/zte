// axi_rd_stripe / axi_wr_stripe -> axi_id_reorder against an OUT-OF-ORDER memory model (it answers the open
// bursts of different IDs in random order, as the VP815 NoC does across GDDR6 channels).  Otherwise as tb_axi_stripe:
// axi_rd_stripe / axi_wr_stripe against a physical memory model: random bursts (1..16 beats, random logical beat
// addresses so many cross 4 KB edges, up to 8 reads / 4 writes in flight, random ready / valid delays).  Phase 1
// writes random data through axi_wr_stripe, phase 2 reads everything back through axi_rd_stripe.  Checked:
//   * every NoC burst stays inside one 4 KB stripe of one channel (physical address = stripe_addr(logical)),
//   * the data read back equal what was written (logical reference memory),
//   * RLAST reaches the master exactly at each of its bursts' last beat, one B per master write burst.
`timescale 1ns/1ps
module tb_axi_id_reorder;
    import axi_stripe_pkg::*;
    reg clk = 0; always #2 clk = ~clk;
    reg rstn = 0;
    // master side
    reg         s_arvalid = 0; wire s_arready; reg [41:0] s_araddr = 0; reg [7:0] s_arlen = 0;
    wire        s_rvalid; reg s_rready = 1; wire [255:0] s_rdata; wire [1:0] s_rresp; wire s_rlast;
    reg         s_awvalid = 0; wire s_awready; reg [41:0] s_awaddr = 0; reg [7:0] s_awlen = 0;
    reg         s_wvalid = 0; wire s_wready; reg [255:0] s_wdata = 0; reg s_wlast = 0;
    wire        s_bvalid; reg s_bready = 1; wire [1:0] s_bresp;
    // NoC side
    // stripe -> reorder
    wire        q_arvalid, q_arready, q_rvalid, q_rready, q_rlast, q_awvalid, q_awready, q_bvalid, q_bready;
    wire [41:0] q_araddr, q_awaddr; wire [7:0] q_arlen, q_awlen; wire [2:0] q_arsize, q_awsize; wire [1:0] q_arburst, q_awburst, q_rresp, q_bresp;
    wire [255:0] q_rdata;
    wire [7:0]  m_arid, m_awid; reg [7:0] m_rid = 0, m_bid = 0;
    wire        m_arvalid; reg m_arready = 0; wire [41:0] m_araddr; wire [7:0] m_arlen; wire [2:0] m_arsize; wire [1:0] m_arburst;
    reg         m_rvalid = 0; wire m_rready; reg [255:0] m_rdata = 0; reg m_rlast = 0;
    wire        m_awvalid; reg m_awready = 0; wire [41:0] m_awaddr; wire [7:0] m_awlen; wire [2:0] m_awsize; wire [1:0] m_awburst;
    wire        m_wvalid; reg m_wready = 0; wire [255:0] m_wdata; wire [31:0] m_wstrb; wire m_wlast;
    reg         m_bvalid = 0; wire m_bready;

    axi_rd_stripe u_rd (.i_clk(clk), .i_rstn(rstn), .i_en(1'b1),
        .s_arvalid(s_arvalid), .s_arready(s_arready), .s_araddr(s_araddr), .s_arlen(s_arlen), .s_arsize(3'd5), .s_arburst(2'b01),
        .s_rvalid(s_rvalid), .s_rready(s_rready), .s_rdata(s_rdata), .s_rresp(s_rresp), .s_rlast(s_rlast),
        .m_arvalid(q_arvalid), .m_arready(q_arready), .m_araddr(q_araddr), .m_arlen(q_arlen), .m_arsize(q_arsize), .m_arburst(q_arburst),
        .m_rvalid(q_rvalid), .m_rready(q_rready), .m_rdata(q_rdata), .m_rresp(q_rresp), .m_rlast(q_rlast));
    axi_wr_stripe u_wr (.i_clk(clk), .i_rstn(rstn), .i_en(1'b1),
        .s_awvalid(s_awvalid), .s_awready(s_awready), .s_awaddr(s_awaddr), .s_awlen(s_awlen), .s_awsize(3'd5), .s_awburst(2'b01),
        .s_wvalid(s_wvalid), .s_wready(s_wready), .s_wdata(s_wdata), .s_wstrb('1), .s_wlast(s_wlast),
        .s_bvalid(s_bvalid), .s_bready(s_bready), .s_bresp(s_bresp),
        .m_awvalid(q_awvalid), .m_awready(q_awready), .m_awaddr(q_awaddr), .m_awlen(q_awlen), .m_awsize(q_awsize), .m_awburst(q_awburst),
        .m_wvalid(m_wvalid), .m_wready(m_wready), .m_wdata(m_wdata), .m_wstrb(m_wstrb), .m_wlast(m_wlast),
        .m_bvalid(q_bvalid), .m_bready(q_bready), .m_bresp(q_bresp));
`ifdef REORDER_NEG_BYPASS      // negative control: no reorder stage (IDs tied 0, responses passed as they come): must FAIL
    assign m_arvalid = q_arvalid; assign q_arready = m_arready; assign m_araddr = q_araddr; assign m_arlen = q_arlen;
    assign m_arsize = q_arsize; assign m_arburst = q_arburst; assign m_arid = 8'd0;
    assign q_rvalid = m_rvalid; assign m_rready = q_rready; assign q_rdata = m_rdata; assign q_rresp = 2'b00; assign q_rlast = m_rlast;
    assign m_awvalid = q_awvalid; assign q_awready = m_awready; assign m_awaddr = q_awaddr; assign m_awlen = q_awlen;
    assign m_awsize = q_awsize; assign m_awburst = q_awburst; assign m_awid = 8'd0;
    assign q_bvalid = m_bvalid; assign m_bready = q_bready; assign q_bresp = 2'b00;
`else
    axi_id_reorder u_ro (.i_clk(clk), .i_rstn(rstn),
        .s_arvalid(q_arvalid), .s_arready(q_arready), .s_araddr(q_araddr), .s_arlen(q_arlen), .s_arsize(q_arsize), .s_arburst(q_arburst),
        .s_rvalid(q_rvalid), .s_rready(q_rready), .s_rdata(q_rdata), .s_rresp(q_rresp), .s_rlast(q_rlast),
        .s_awvalid(q_awvalid), .s_awready(q_awready), .s_awaddr(q_awaddr), .s_awlen(q_awlen), .s_awsize(q_awsize), .s_awburst(q_awburst),
        .s_bvalid(q_bvalid), .s_bready(q_bready), .s_bresp(q_bresp),
        .m_arvalid(m_arvalid), .m_arready(m_arready), .m_araddr(m_araddr), .m_arlen(m_arlen), .m_arsize(m_arsize), .m_arburst(m_arburst),
        .m_arid(m_arid), .m_rvalid(m_rvalid), .m_rready(m_rready), .m_rdata(m_rdata), .m_rresp(2'b00), .m_rlast(m_rlast), .m_rid(m_rid),
        .m_awvalid(m_awvalid), .m_awready(m_awready), .m_awaddr(m_awaddr), .m_awlen(m_awlen), .m_awsize(m_awsize), .m_awburst(m_awburst),
        .m_awid(m_awid), .m_bvalid(m_bvalid), .m_bready(m_bready), .m_bresp(2'b00), .m_bid(m_bid));
`endif

    integer errors = 0;
    function automatic bit burst_ok(input [41:0] a, input [7:0] len);
        // one channel, inside one 4 KB page of its offset space, and a legal channel-window address
        burst_ok = ((a[11:5] + len) <= 8'd127) && (a[41:37] == 5'd0) && (a[32:31] == 2'b00);
    endfunction

    // ---------------------------------------------------------------- NoC-side memory (physical beat address)
    logic [255:0] pmem [longint];
    longint rq_a [$]; int rq_n [$]; int rq_id [$];
    longint wq_a [$]; int wq_n [$]; int wq_id [$];
    int     bq_id [$];
    int     cur = -1;                                     // the read burst being returned (index into rq_*)
    always @(posedge clk) begin
        m_arready <= ($urandom_range(3, 0) != 0);
        if (m_arvalid && m_arready) begin
            if (!burst_ok(m_araddr, m_arlen)) begin errors++; $display("BAD AR %h len %0d", m_araddr, m_arlen); end
            rq_a.push_back(longint'(m_araddr >> 5)); rq_n.push_back(int'(m_arlen) + 1); rq_id.push_back(int'(m_arid));
        end
        m_awready <= ($urandom_range(3, 0) != 0);
        if (m_awvalid && m_awready) begin
            if (!burst_ok(m_awaddr, m_awlen)) begin errors++; $display("BAD AW %h len %0d", m_awaddr, m_awlen); end
            wq_a.push_back(longint'(m_awaddr >> 5)); wq_n.push_back(int'(m_awlen) + 1); wq_id.push_back(int'(m_awid));
        end
        m_wready <= ($urandom_range(2, 0) != 0) && (wq_a.size() > 0);
        if (m_wvalid && m_wready) begin
            pmem[wq_a[0]] = m_wdata;
            if ((wq_n[0] == 1) != m_wlast) begin errors++; $display("WLAST at the wrong beat"); end
            if (wq_n[0] == 1) begin bq_id.push_back(wq_id[0]); void'(wq_a.pop_front()); void'(wq_n.pop_front()); void'(wq_id.pop_front()); end
            else begin wq_a[0] = wq_a[0] + 1; wq_n[0] = wq_n[0] - 1; end
        end
        // B: any completed burst, in random order
        if (m_bvalid && m_bready) m_bvalid <= 1'b0;
        else if (!m_bvalid && bq_id.size() > 0 && ($urandom_range(2, 0) == 0)) begin
            int j; j = $urandom_range(bq_id.size() - 1, 0);
            m_bvalid <= 1'b1; m_bid <= 8'(bq_id[j]); bq_id.delete(j);
        end
        // R: pick a random open burst whenever none is in progress (bursts never interleave beat by beat)
        if (!m_rvalid || m_rready) begin
            if (cur < 0 && rq_a.size() > 0 && ($urandom_range(3, 0) != 0)) cur = $urandom_range(rq_a.size() - 1, 0);
            if (cur >= 0) begin
                m_rvalid <= 1'b1;
                m_rdata  <= pmem.exists(rq_a[cur]) ? pmem[rq_a[cur]] : 256'hDEAD;
                m_rlast  <= (rq_n[cur] == 1);
                m_rid    <= 8'(rq_id[cur]);
                if (rq_n[cur] == 1) begin rq_a.delete(cur); rq_n.delete(cur); rq_id.delete(cur); cur = -1; end
                else begin rq_a[cur] = rq_a[cur] + 1; rq_n[cur] = rq_n[cur] - 1; end
            end else m_rvalid <= 1'b0;
        end
    end

    // ---------------------------------------------------------------- master: writes, then reads
    logic [255:0] lmem [longint];                     // logical reference
    localparam int NB = 400;
    longint b_addr [NB]; int b_len [NB];
    int n_b = 0, n_rlast = 0, n_rbeats = 0;
    initial begin
        for (int i = 0; i < NB; i++) begin
            // bursts in a 256 KB logical window at random beat offsets (a third start near a 4 KB edge)
            longint base = longint'($urandom_range(0, 8191));
            if (i % 3 == 0) base = (base & ~longint'(127)) + 128 - longint'($urandom_range(1, 15));
            b_addr[i] = base; b_len[i] = $urandom_range(1, 16);
        end
        repeat (4) @(negedge clk); rstn = 1; repeat (2) @(negedge clk);
        // ---- writes: AW and W from separate threads (W may run ahead of its AW, as AXI allows)
        fork
            begin
                for (int i = 0; i < NB; i++) begin
                    @(negedge clk); s_awvalid = 1; s_awaddr = 42'(b_addr[i] << 5); s_awlen = 8'(b_len[i] - 1);
                    do @(posedge clk); while (!s_awready);
                    @(negedge clk); s_awvalid = 0;
                end
            end
            begin
                for (int i = 0; i < NB; i++)
                    for (int k = 0; k < b_len[i]; k++) begin
                        logic [255:0] d = {$urandom, $urandom, $urandom, $urandom, $urandom, $urandom, $urandom, $urandom};
                        @(negedge clk); s_wvalid = 1; s_wdata = d; s_wlast = (k == b_len[i] - 1);
                        do @(posedge clk); while (!s_wready);
                        lmem[b_addr[i] + k] = d;
                        @(negedge clk); s_wvalid = 0;
                    end
            end
            begin
                while (n_b < NB) begin @(posedge clk); if (s_bvalid && s_bready) n_b++; end
            end
        join
        // ---- reads: AR thread and R checker
        fork
            begin
                for (int i = 0; i < NB; i++) begin
                    @(negedge clk); s_arvalid = 1; s_araddr = 42'(b_addr[i] << 5); s_arlen = 8'(b_len[i] - 1);
                    do @(posedge clk); while (!s_arready);
                    @(negedge clk); s_arvalid = 0;
                end
            end
            begin
                for (int i = 0; i < NB; i++)
                    for (int k = 0; k < b_len[i]; k++) begin
                        do @(posedge clk); while (!(s_rvalid && s_rready));
                        n_rbeats++;
                        if (s_rdata !== lmem[b_addr[i] + k]) begin
                            errors++; if (errors < 8) $display("DATA burst %0d beat %0d: got %h", i, k, s_rdata[63:0]); end
                        if (s_rlast !== (k == b_len[i] - 1)) begin
                            errors++; if (errors < 8) $display("RLAST burst %0d beat %0d = %b", i, k, s_rlast); end
                        if (s_rlast) n_rlast++;
                    end
            end
        join
        repeat (20) @(posedge clk);
        if (n_b != NB || n_rlast != NB) errors++;
        $display("RESULT %s tb_axi_id_reorder errors=%0d bursts=%0d B=%0d RLAST=%0d beats=%0d", errors == 0 ? "PASS" : "FAIL",
                 errors, NB, n_b, n_rlast, n_rbeats);
        $finish;
    end
    initial begin #20000000; $display("RESULT FAIL tb_axi_id_reorder timeout (B %0d RLAST %0d)", n_b, n_rlast); $finish; end
endmodule
