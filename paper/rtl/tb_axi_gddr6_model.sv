`timescale 1ps/1ps
// GDDR6 AXI4 memory model for the chain-node test benches (one clock, one AXI ID).
//   read:  queues up to QDEPTH outstanding AR requests (random ARREADY) and serves their R
//          bursts in request order with random RVALID gaps; the beat data come from the test
//          bench combinationally: o_rd_addr -> i_rd_beat.
//   write: queues up to QDEPTH outstanding AW requests (random AWREADY); accepts W beats of
//          the oldest open burst only (random WREADY, never before its AW handshake); returns
//          one B per burst after a random delay; reports each written beat on o_wr_*.
// Checks (counted in o_n_err): ARSIZE/AWSIZE = 32 bytes, INCR bursts, burst length <=
// MAX_BEATS, WLAST on the last beat only, WSTRB all ones.  o_max_rq / o_max_wq give the most
// outstanding read / write requests seen.  Tie the unused side's valid inputs low.
// +nostall (plusarg): every ready / valid the model drives is asserted as early as it may be, no random gaps --
// the node-intrinsic cycle count under an ideal memory.  Without it the model is unchanged.
// +rd_lat=<n> / +wr_lat=<n> (plusargs, clock cycles, default 0): a burst's first R beat is not served before n cycles
// after its AR handshake, and its B not before n cycles after its last W beat -- the GDDR6 + NoC round trip that
// +nostall leaves out (a node that waits for one transfer before starting the next pays it every time).
module tb_axi_gddr6_model #(
    parameter integer MAX_BEATS = 16,
    parameter integer QDEPTH    = 32
) (
    input  wire          i_clk,
    input  wire          i_rstn,

    input  wire          i_arvalid,
    output reg           o_arready,
    input  wire [41:0]   i_araddr,
    input  wire [7:0]    i_arlen,
    input  wire [2:0]    i_arsize,
    input  wire [1:0]    i_arburst,
    output reg           o_rvalid,
    input  wire          i_rready,
    output reg  [255:0]  o_rdata,
    output reg           o_rlast,
    output wire [41:0]   o_rd_addr,
    input  wire [255:0]  i_rd_beat,

    input  wire          i_awvalid,
    output reg           o_awready,
    input  wire [41:0]   i_awaddr,
    input  wire [7:0]    i_awlen,
    input  wire [2:0]    i_awsize,
    input  wire [1:0]    i_awburst,
    input  wire          i_wvalid,
    output reg           o_wready,
    input  wire [255:0]  i_wdata,
    input  wire [31:0]   i_wstrb,
    input  wire          i_wlast,
    output reg           o_bvalid,
    input  wire          i_bready,

    output reg           o_wr_fire,
    output reg  [41:0]   o_wr_addr,
    output reg  [255:0]  o_wr_data,

    output wire [31:0]   o_n_rbursts,
    output wire [31:0]   o_n_wbursts,
    output wire [31:0]   o_n_err,
    output wire [31:0]   o_max_rq,
    output wire [31:0]   o_max_wq
);
    bit nostall = 1'b0;
    initial nostall = $test$plusargs("nostall");
    // +stall_pct=<n>: every ready / valid the model drives is held off with probability n % (default: the
    // fixed 25 % / 33 % below), to reproduce real NAP back-pressure
    integer stall_pct = -1;
    initial if (!$value$plusargs("stall_pct=%d", stall_pct)) stall_pct = -1;
    function automatic bit go4(input bit dummy);   // ready with the default 3/4, or 1 - stall_pct
        go4 = nostall || (stall_pct < 0 ? ($urandom_range(3, 0) != 0) : ($urandom_range(99, 0) >= stall_pct));
    endfunction
    function automatic bit go3(input bit dummy);   // wready with the default 2/3, or 1 - stall_pct
        go3 = nostall || (stall_pct < 0 ? ($urandom_range(2, 0) != 0) : ($urandom_range(99, 0) >= stall_pct));
    endfunction
    integer rd_lat = 0, wr_lat = 0;
    initial begin
        if (!$value$plusargs("rd_lat=%d", rd_lat)) rd_lat = 0;
        if (!$value$plusargs("wr_lat=%d", wr_lat)) wr_lat = 0;
    end
    longint    cyc = 0;
    longint    arq_t0 [0:QDEPTH-1];
    longint    bq_t0 [$];
    reg [41:0] arq_a [0:QDEPTH-1];
    reg [8:0]  arq_l [0:QDEPTH-1];
    reg [41:0] awq_a [0:QDEPTH-1];
    reg [8:0]  awq_l [0:QDEPTH-1];
    integer    arq_h = 0, arq_t = 0, awq_h = 0, awq_t = 0, b_pend = 0;
    integer    n_rbursts = 0, n_wbursts = 0, n_err = 0, max_rq = 0, max_wq = 0;
    reg [41:0] r_addr = '0, w_addr = '0;
    reg [8:0]  r_left = '0, w_left = '0;

    assign o_rd_addr   = r_addr;
    assign o_n_rbursts = n_rbursts;
    assign o_n_wbursts = n_wbursts;
    assign o_n_err     = n_err;
    assign o_max_rq    = max_rq;
    assign o_max_wq    = max_wq;

    always @(posedge i_clk) begin : p_model
        reg        w_fire, deq_w;
        reg [8:0]  w_left_next;
        if (!i_rstn) begin
            o_arready <= 1'b0;
            o_rvalid  <= 1'b0;
            o_awready <= 1'b0;
            o_wready  <= 1'b0;
            o_bvalid  <= 1'b0;
            o_wr_fire <= 1'b0;
            r_left    <= '0;
            w_left    <= '0;
        end else begin
            // ---- AR queue ----
            if (i_arvalid && o_arready) begin
                if (i_arsize != 3'd5 || i_arburst != 2'b01 || int'(i_arlen) + 1 > MAX_BEATS) n_err = n_err + 1;
                arq_a[arq_t % QDEPTH] = i_araddr;
                arq_l[arq_t % QDEPTH] = 9'(i_arlen) + 9'd1;
                arq_t0[arq_t % QDEPTH] = cyc + longint'(rd_lat);
                arq_t     = arq_t + 1;
                n_rbursts = n_rbursts + 1;
                if (arq_t - arq_h > max_rq) max_rq = arq_t - arq_h;
            end
            o_arready <= ((arq_t - arq_h) < QDEPTH - 1) && go4(1'b0);

            // ---- R ----
            if (o_rvalid && i_rready) o_rvalid <= 1'b0;
            if (r_left != 9'd0) begin
                if ((!o_rvalid || i_rready) && go4(1'b0)) begin
                    o_rvalid <= 1'b1;
                    o_rdata  <= i_rd_beat;
                    o_rlast  <= (r_left == 9'd1);
                    r_addr   <= r_addr + 42'd32;
                    r_left   <= r_left - 9'd1;
                end
            end else if (arq_t != arq_h && cyc >= arq_t0[arq_h % QDEPTH]) begin
                r_addr <= arq_a[arq_h % QDEPTH];
                r_left <= arq_l[arq_h % QDEPTH];
                arq_h  = arq_h + 1;
            end

            // ---- AW queue ----
            if (i_awvalid && o_awready) begin
                if (i_awsize != 3'd5 || i_awburst != 2'b01 || int'(i_awlen) + 1 > MAX_BEATS) n_err = n_err + 1;
                awq_a[awq_t % QDEPTH] = i_awaddr;
                awq_l[awq_t % QDEPTH] = 9'(i_awlen) + 9'd1;
                awq_t     = awq_t + 1;
                n_wbursts = n_wbursts + 1;
                if (awq_t - awq_h > max_wq) max_wq = awq_t - awq_h;
            end
            o_awready <= ((awq_t - awq_h) < QDEPTH - 1) && go4(1'b0);

            // ---- W ----
            w_fire    = i_wvalid && o_wready;
            o_wr_fire <= 1'b0;
            if (w_fire) begin
                if (i_wstrb != 32'hFFFF_FFFF || i_wlast != (w_left == 9'd1)) n_err = n_err + 1;
                o_wr_fire <= 1'b1;
                o_wr_addr <= w_addr;
                o_wr_data <= i_wdata;
                if (w_left == 9'd1) bq_t0.push_back(cyc + longint'(wr_lat));
            end
            deq_w       = (w_left == 9'd0) && (awq_t != awq_h);
            w_left_next = deq_w ? awq_l[awq_h % QDEPTH] : (w_fire ? w_left - 9'd1 : w_left);
            if (deq_w) begin
                w_addr <= awq_a[awq_h % QDEPTH];
                awq_h  = awq_h + 1;
            end else if (w_fire) begin
                w_addr <= w_addr + 42'd32;
            end
            w_left   <= w_left_next;
            o_wready <= (w_left_next != 9'd0) && go3(1'b0);

            // ---- B ----
            while (bq_t0.size() > 0 && cyc >= bq_t0[0]) begin void'(bq_t0.pop_front()); b_pend = b_pend + 1; end
            cyc = cyc + 1;
            if (o_bvalid && i_bready) o_bvalid <= 1'b0;
            if ((!o_bvalid || i_bready) && b_pend > 0 && go4(1'b0)) begin
                o_bvalid <= 1'b1;
                b_pend   = b_pend - 1;
            end
        end
    end
endmodule
