// Program fetcher: reads a chain node's command program from GDDR6 and presents it as the
// command stream of colpar_node_ctrl.sv, so the host only writes GDDR6 (weights, activations,
// programs) and pulses a start bit.
//
// Layout: 128-bit commands, two per 256-bit beat (command 2i in bits [127:0] of beat i, command
// 2i+1 in [255:128]), from i_base.  One beat per AXI burst (ARLEN = 0): commands are consumed
// once per load or tile, so per-beat fetching costs nothing that matters.
//
// Sharing the node's read channel: the fetcher issues an AR only while i_grant is high, which
// the node drives from "executor waiting for a command" (colpar_node_ctrl o_fetching): then the
// tile loader is idle with no request outstanding, and the executor cannot leave that state
// before this fetch has delivered a command.  It fetches only when both buffered commands are
// consumed and stops after the executor has taken an END (opcode 0 in the first command of a
// record; records are RECORD_WORDS commands long, 1 for the chain node, 6 for the vector node).
// i_start restarts at i_base.
// o_error (sticky): RRESP != OKAY or a missing RLAST.
module colpar_prog_fetch #(
    parameter integer AXI_ADDR_WIDTH = 42,
    parameter integer RECORD_WORDS   = 1
) (
    input  wire                       i_clk,
    input  wire                       i_rstn,
    input  wire                       i_start,
    input  wire [AXI_ADDR_WIDTH-1:0]  i_base,
    input  wire                       i_grant,

    output reg                        o_arvalid,
    input  wire                       i_arready,
    output reg  [AXI_ADDR_WIDTH-1:0]  o_araddr,
    output wire [7:0]                 o_arlen,
    output wire [2:0]                 o_arsize,
    output wire [1:0]                 o_arburst,
    input  wire                       i_rvalid,
    output wire                       o_rready,
    input  wire [255:0]               i_rdata,
    input  wire [1:0]                 i_rresp,
    input  wire                       i_rlast,

    output wire [127:0]               o_cmd,
    output wire                       o_cmd_valid,
    input  wire                       i_cmd_ready,

    output reg                        o_error
);
    assign o_arlen   = 8'd0;
    assign o_arsize  = 3'd5;
    assign o_arburst = 2'b01;

    reg [255:0] beat;
    reg [1:0]   avail;          // commands left in `beat`
    reg         idx;            // next command: 0 = low half, 1 = high half
    reg         active, waiting_r;
    reg [7:0]   rec_pos;        // position of the next command inside its record

    assign o_cmd       = idx ? beat[255:128] : beat[127:0];
    assign o_cmd_valid = active && (avail != 2'd0);
    assign o_rready    = waiting_r;

    always @(posedge i_clk) begin
        if (!i_rstn) begin
            active    <= 1'b0;
            o_arvalid <= 1'b0;
            waiting_r <= 1'b0;
            avail     <= 2'd0;
            idx       <= 1'b0;
            o_error   <= 1'b0;
            rec_pos   <= 8'd0;
        end else if (i_start) begin
            active    <= 1'b1;
            o_araddr  <= i_base;
            avail     <= 2'd0;
            idx       <= 1'b0;
            o_arvalid <= 1'b0;
            waiting_r <= 1'b0;
            rec_pos   <= 8'd0;
        end else if (active) begin
            if (o_cmd_valid && i_cmd_ready) begin
                rec_pos <= (rec_pos + 8'd1 == 8'(RECORD_WORDS)) ? 8'd0 : rec_pos + 8'd1;
                if (rec_pos == 8'd0 && o_cmd[127:124] == 4'd0) begin
                    active <= 1'b0;                     // END taken
                    avail  <= 2'd0;
                end else begin
                    avail <= avail - 2'd1;
                    idx   <= 1'b1;
                end
            end
            if (avail == 2'd0 && !o_arvalid && !waiting_r && i_grant)
                o_arvalid <= 1'b1;
            if (o_arvalid && i_arready) begin
                o_arvalid <= 1'b0;
                waiting_r <= 1'b1;
            end
            if (waiting_r && i_rvalid) begin
                beat      <= i_rdata;
                avail     <= 2'd2;
                idx       <= 1'b0;
                waiting_r <= 1'b0;
                o_araddr  <= o_araddr + AXI_ADDR_WIDTH'(32);
                if (i_rresp != 2'b00 || !i_rlast) o_error <= 1'b1;
            end
        end
    end
endmodule
