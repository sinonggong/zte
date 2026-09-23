// Host control path for the node array: the only thing the host touches between chunks.
//
// The host writes programs, weights and activations straight into GDDR6; the nodes fetch them
// themselves and sequence each other with GDDR6 flags (node_sync.sv), so per chunk the host only has
// to point each node at its program and set it going, then wait.  This block is that register window,
// sitting behind an ACX_NAP_AXI_MASTER (the NoC is the initiator; see nap_axi_ports.sv).
//
// One 32-byte beat per node, beat index = addr[27:5].  The host reaches it through PCIe BAR0 with 32-bit
// accesses, which arrive as one beat with a 4-byte strobe, so writes are byte-merged into the registers
// (wstrb) and the action bits fire on the write that carries them:
//
//   write beat n < N_NODE     [41:0] program base (bytes 0-5); byte 8 bit 64 = 1: start that node now
//   write beat N_NODE         bytes 12-16 [96 +: N_NODE] start mask (kept); byte 0 bit 0 = 1: start every
//                             node whose mask bit is set; bit 1 = 1: soft reset (o_soft_rst pulses once;
//                             the top stretches it into every node's reset)
//   read  beat n < N_NODE     [41:0] program base, [64] halt, [65] error, [79:72] the node's error detail
//                             bits, [127:96] fabric cycles from the node's last start to its halt,
//                             [255:128] the node's debug snapshot (vector nodes: end-of-op packer state)
//   read  beat N_NODE         [N_NODE-1:0] halt vector, [128 +: N_NODE] error vector,
//                             [231:224] N_NODE, [255:232] ID_WORD (so the host can tell which bitstream it holds)
//   write / read beat N_NODE + 1   [41:0] the host->GDDR6 bridge's page base (pi0_host_gddr_bridge.sv), bytes merged;
//                                  [64] GDDR6 channel striping on (STRIPE builds; set it before loading an image, never
//                                  while a node runs: nodes, programs and the host must use the same map)
//
// A write is a pulse in the fabric clock domain, which is the nodes' own program-fetch domain, so no
// crossing is needed.  Only single-beat accesses are used; a longer burst is accepted and only its
// last beat takes effect, which keeps the decode small.  AXI does not order AW against W, so the last
// write beat is held until the address has arrived rather than dropped.
module pi0_chip_ctrl #(
    parameter integer N_NODE         = 36,
    parameter integer AXI_ADDR_WIDTH = 28,
    parameter [23:0]  ID_WORD        = 24'h504930     // "PI0"; a build may override it
) (
    input  wire                       i_clk,
    input  wire                       i_rstn,

    // NAP target side (the NoC initiates)
    input  wire                       i_arvalid,
    output reg                        o_arready,
    input  wire [AXI_ADDR_WIDTH-1:0]  i_araddr,
    input  wire [7:0]                 i_arlen,
    output reg                        o_rvalid,
    input  wire                       i_rready,
    output reg  [255:0]               o_rdata,
    output wire [1:0]                 o_rresp,
    output reg                        o_rlast,

    input  wire                       i_awvalid,
    output reg                        o_awready,
    input  wire [AXI_ADDR_WIDTH-1:0]  i_awaddr,
    input  wire                       i_wvalid,
    output wire                       o_wready,
    input  wire [255:0]               i_wdata,
    input  wire [31:0]                i_wstrb,
    input  wire                       i_wlast,
    output reg                        o_bvalid,
    input  wire                       i_bready,
    output wire [1:0]                 o_bresp,

    // node side
    output reg  [42*N_NODE-1:0]       o_prog_base,
    output reg  [N_NODE-1:0]          o_prog_start,
    output reg                        o_soft_rst,      // one-cycle pulse
    output reg  [41:0]                o_page_base,     // host bridge page (beat N_NODE + 1)
    output reg                        o_stripe_en,     // beat N_NODE + 1 bit 64
    input  wire [N_NODE-1:0]          i_halt,
    input  wire [N_NODE-1:0]          i_error,
    input  wire [8*N_NODE-1:0]        i_error_bits,
    input  wire [128*N_NODE-1:0]      i_dbg            // per-node debug snapshot, read beat n [255:128]
);
    localparam integer IDX_BITS = $clog2(N_NODE + 2);

    assign o_rresp  = 2'b00;
    assign o_bresp  = 2'b00;
    assign o_wready = 1'b1;                       // the register file never back-pressures write data

    reg [IDX_BITS-1:0] wr_idx, rd_idx;
    reg                wr_armed, rd_armed, wd_held;
    reg [255:0]        wd_data;
    reg [31:0]         wd_strb;
    reg [7:0]          rd_left;
    reg [N_NODE-1:0]   start_mask;                // broadcast beat bytes 12-16, kept between writes
    reg [31:0]         cycles [0:N_NODE-1];       // per node: fabric cycles from halt dropping after a start to halt rising
    reg [N_NODE-1:0]   running, armed;             // armed: started, halt not yet seen low (a synchronised halt lags a few cycles)

    // the index is clamped on the FULL address slice, so a write past the window lands on the
    // broadcast beat instead of wrapping back onto some node
    wire [AXI_ADDR_WIDTH-6:0] aw_sel  = i_awaddr[AXI_ADDR_WIDTH-1:5];
    wire [AXI_ADDR_WIDTH-6:0] ar_sel  = i_araddr[AXI_ADDR_WIDTH-1:5];
    wire [IDX_BITS-1:0]       aw_idx  = (aw_sel > N_NODE + 1) ? IDX_BITS'(N_NODE) : IDX_BITS'(aw_sel);
    wire [IDX_BITS-1:0]       ar_idx  = (ar_sel > N_NODE + 1) ? IDX_BITS'(N_NODE) : IDX_BITS'(ar_sel);

    // the beat to apply: this cycle's, or one that arrived before its address
    wire                      wr_now  = wr_armed && (wd_held || (i_wvalid && i_wlast));
    wire [255:0]              wr_beat = wd_held ? wd_data : i_wdata;
    wire [31:0]               wr_strb = wd_held ? wd_strb : i_wstrb;
    // the strobed bytes of the write merged over a register's current value
    function automatic [255:0] merge(input [255:0] cur, input [255:0] nw, input [31:0] strb);
        integer b;
        begin
            merge = cur;
            for (b = 0; b < 32; b = b + 1) if (strb[b]) merge[8*b +: 8] = nw[8*b +: 8];
        end
    endfunction
    reg  [255:0] wr_cur;                                       // the addressed register as the host sees it
    always @* begin
        wr_cur = '0;
        if (wr_idx < IDX_BITS'(N_NODE))       wr_cur[41:0] = o_prog_base[42*wr_idx +: 42];
        else if (wr_idx == IDX_BITS'(N_NODE)) wr_cur[96 +: N_NODE] = start_mask;
        else begin                            wr_cur[41:0] = o_page_base; wr_cur[64] = o_stripe_en; end
    end
    wire [255:0] wr_new   = merge(wr_cur, wr_beat, wr_strb);
    wire [N_NODE-1:0] wr_mask = wr_new[96 +: N_NODE];
    wire         wr_start = wr_strb[8]  & wr_beat[64];       // the action bits count only when their byte is written
    wire         wr_go    = wr_strb[0]  & wr_beat[0];
    wire         wr_rst   = wr_strb[0]  & wr_beat[1];

    integer n;
    always @(posedge i_clk) begin
        o_prog_start <= '0;                       // one-cycle pulses
        o_soft_rst   <= 1'b0;
        o_arready    <= 1'b0;
        o_awready    <= 1'b0;

        if (!i_rstn) begin
            wr_armed  <= 1'b0;
            wd_held   <= 1'b0;
            rd_armed  <= 1'b0;
            o_bvalid  <= 1'b0;
            o_rvalid  <= 1'b0;
            o_rlast   <= 1'b0;
            o_prog_base <= '0;
            o_page_base <= '0;
            o_stripe_en <= 1'b0;
            start_mask  <= '0;
            running     <= '0;
            armed       <= '0;
        end else begin
            for (n = 0; n < N_NODE; n = n + 1) begin
                if (o_prog_start[n])      begin cycles[n] <= 32'd0; armed[n] <= 1'b1; running[n] <= 1'b0; end
                else if (armed[n] && !i_halt[n]) begin armed[n] <= 1'b0; running[n] <= 1'b1; cycles[n] <= 32'd1; end
                else if (running[n]) begin                    // counts the cycles the node reports not halted
                    if (i_halt[n]) running[n] <= 1'b0;
                    else           cycles[n] <= cycles[n] + 32'd1;
                end
            end
            // ---- write address ----
            if (i_awvalid && !o_awready && !wr_armed) begin
                o_awready <= 1'b1;
                wr_idx    <= aw_idx;
                wr_armed  <= 1'b1;
            end

            // ---- write data: the last beat of the burst is the one that counts, and it may arrive
            //      before its address, so hold it until the address is there ----
            if (i_wvalid && i_wlast && !wr_armed) begin
                wd_held <= 1'b1;
                wd_data <= i_wdata;
                wd_strb <= i_wstrb;
            end
            if (wr_now) begin
                wr_armed <= 1'b0;
                wd_held  <= 1'b0;
                o_bvalid <= 1'b1;
                if (wr_idx < IDX_BITS'(N_NODE)) begin
                    o_prog_base[42*wr_idx +: 42] <= wr_new[41:0];
                    o_prog_start[wr_idx]         <= wr_start;
                end else if (wr_idx == IDX_BITS'(N_NODE)) begin
                    start_mask <= wr_mask;
                    for (n = 0; n < N_NODE; n = n + 1)
                        o_prog_start[n] <= wr_go & wr_mask[n];
                    o_soft_rst <= wr_rst;
                end else begin
                    o_page_base <= wr_new[41:0];
                    o_stripe_en <= wr_new[64];
                end
            end
            if (o_bvalid && i_bready) o_bvalid <= 1'b0;

            // ---- read ----
            if (i_arvalid && !o_arready && !rd_armed) begin
                o_arready <= 1'b1;
                rd_idx    <= ar_idx;
                rd_left   <= i_arlen;
                rd_armed  <= 1'b1;
            end
            if (rd_armed && (!o_rvalid || i_rready)) begin
                o_rvalid <= 1'b1;
                o_rlast  <= (rd_left == 8'd0);
                o_rdata  <= '0;
                if (rd_idx < IDX_BITS'(N_NODE)) begin
                    o_rdata[41:0]   <= o_prog_base[42*rd_idx +: 42];
                    o_rdata[64]     <= i_halt[rd_idx];
                    o_rdata[65]     <= i_error[rd_idx];
                    o_rdata[79:72]  <= i_error_bits[8*rd_idx +: 8];
                    o_rdata[127:96] <= cycles[rd_idx];
                    o_rdata[255:128] <= i_dbg[128*rd_idx +: 128];
                end else if (rd_idx == IDX_BITS'(N_NODE + 1)) begin
                    o_rdata[41:0] <= o_page_base;
                    o_rdata[64]   <= o_stripe_en;
                end else begin
                    o_rdata[N_NODE-1:0]      <= i_halt;
                    o_rdata[128 +: N_NODE]   <= i_error;
                    o_rdata[231:224]         <= 8'(N_NODE);
                    o_rdata[255:232]         <= ID_WORD;
                end
                if (rd_left == 8'd0) rd_armed <= 1'b0;
                else                 rd_left  <= rd_left - 8'd1;
            end else if (o_rvalid && i_rready) begin
                o_rvalid <= 1'b0;
                o_rlast  <= 1'b0;
            end
        end
    end
endmodule
