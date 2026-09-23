// Node-to-node synchronisation through one GDDR6 word, so the host is not needed between stages.
//
// A producer ends its stage with POST(addr, value): a single 32-bit write into the low word of the beat at
// `addr` (the other 224 bits are written as zero, so a flag word owns its beat).  A consumer starts its stage
// with WAIT(addr, value): it reads that beat until the low 32 bits are >= value (unsigned), first after
// POLL_CYCLES and then after twice the previous interval, up to POLL_MAX cycles apart (a long stage costs
// ~log2(POLL_MAX / POLL_CYCLES) + wait / POLL_MAX reads instead of wait / POLL_CYCLES; each read is an AXI transaction),
// which makes a flag monotone within a chunk and a stale value from the previous chunk harmless as long as the
// host raises the epoch.  One outstanding transaction at a time; the node is otherwise idle while it syncs, so
// this master simply takes the node's AR/AW channels while o_busy is high.
//
// i_arm starts one operation: i_post selects POST, otherwise WAIT.  o_done pulses when it is finished.
module node_sync #(
    parameter integer AXI_ADDR_WIDTH = 42,
    parameter integer POLL_CYCLES    = 64,       // fabric cycles before the second poll of a flag
    // longest interval between polls (= POLL_CYCLES: fixed interval).  Was 4096: a WAIT then sees its flag up to
    // 4096 cycles late (36 us at the half array's 114.6 MHz fabric), on every stage hand-over of the chunk's
    // critical path (4,410 stages).  At 512 a waiting node reads one beat per 512 cycles -- 35 nodes cost < 0.1 %
    // of one channel -- and a hand-over is late by at most ~4.5 us.
    parameter integer POLL_MAX       = 512
) (
    input  wire                       i_clk,
    input  wire                       i_rstn,

    input  wire                       i_arm,
    input  wire                       i_post,
    input  wire [AXI_ADDR_WIDTH-1:0]  i_addr,    // beat aligned (32 bytes)
    input  wire [31:0]                i_value,
    output reg                        o_busy,
    output reg                        o_done,
    output reg                        o_error,   // a response other than OKAY

    output reg                        o_arvalid,
    input  wire                       i_arready,
    output wire [AXI_ADDR_WIDTH-1:0]  o_araddr,
    output wire [7:0]                 o_arlen,
    output wire [2:0]                 o_arsize,
    output wire [1:0]                 o_arburst,
    output wire                       o_rready,
    input  wire                       i_rvalid,
    input  wire [255:0]               i_rdata,
    input  wire [1:0]                 i_rresp,

    output reg                        o_awvalid,
    input  wire                       i_awready,
    output wire [AXI_ADDR_WIDTH-1:0]  o_awaddr,
    output wire [7:0]                 o_awlen,
    output wire [2:0]                 o_awsize,
    output wire [1:0]                 o_awburst,
    output reg                        o_wvalid,
    input  wire                       i_wready,
    output wire [255:0]               o_wdata,
    output wire [31:0]                o_wstrb,
    output wire                       o_wlast,
    input  wire                       i_bvalid,
    output wire                       o_bready,
    input  wire [1:0]                 i_bresp
);
    localparam [2:0] S_IDLE = 3'd0, S_AR = 3'd1, S_R = 3'd2, S_HOLD = 3'd3, S_AW = 3'd4, S_W = 3'd5, S_B = 3'd6;

    reg [2:0]                 st;
    reg [AXI_ADDR_WIDTH-1:0]  addr_q;
    reg [31:0]                value_q;
    reg [23:0]                wait_cnt, interval;

    assign o_araddr  = addr_q;
    assign o_awaddr  = addr_q;
    assign o_arlen   = 8'd0;
    assign o_awlen   = 8'd0;
    assign o_arsize  = 3'd5;
    assign o_awsize  = 3'd5;
    assign o_arburst = 2'b01;
    assign o_awburst = 2'b01;
    assign o_rready  = (st == S_R);
    assign o_bready  = (st == S_B);
    assign o_wdata   = {224'd0, value_q};
    assign o_wstrb   = 32'hFFFF_FFFF;
    assign o_wlast   = 1'b1;

    always @(posedge i_clk) begin
        o_done <= 1'b0;
        if (!i_rstn) begin
            st        <= S_IDLE;
            o_busy    <= 1'b0;
            o_error   <= 1'b0;
            o_arvalid <= 1'b0;
            o_awvalid <= 1'b0;
            o_wvalid  <= 1'b0;
        end else begin
            case (st)
            S_IDLE: if (i_arm) begin
                addr_q  <= i_addr;
                value_q <= i_value;
                o_busy  <= 1'b1;
                interval <= 24'(POLL_CYCLES);
                if (i_post) begin
                    o_awvalid <= 1'b1;
                    st        <= S_AW;
                end else begin
                    o_arvalid <= 1'b1;
                    st        <= S_AR;
                end
            end
            S_AR: if (i_arready) begin
                o_arvalid <= 1'b0;
                st        <= S_R;
            end
            S_R: if (i_rvalid) begin
                if (i_rresp != 2'b00) o_error <= 1'b1;
                if (i_rdata[31:0] >= value_q) begin
                    o_busy <= 1'b0;
                    o_done <= 1'b1;
                    st     <= S_IDLE;
                end else begin
                    wait_cnt <= interval;
                    interval <= (32'(interval) * 2 > POLL_MAX) ? 24'(POLL_MAX) : interval << 1;
                    st       <= S_HOLD;
                end
            end
            S_HOLD: if (wait_cnt == 24'd0) begin
                o_arvalid <= 1'b1;
                st        <= S_AR;
            end else begin
                wait_cnt <= wait_cnt - 24'd1;
            end
            S_AW: if (i_awready) begin
                o_awvalid <= 1'b0;
                o_wvalid  <= 1'b1;
                st        <= S_W;
            end
            S_W: if (i_wready) begin
                o_wvalid <= 1'b0;
                st       <= S_B;
            end
            S_B: if (i_bvalid) begin
                if (i_bresp != 2'b00) o_error <= 1'b1;
                o_busy <= 1'b0;
                o_done <= 1'b1;
                st     <= S_IDLE;
            end
            default: st <= S_IDLE;
            endcase
        end
    end
endmodule
