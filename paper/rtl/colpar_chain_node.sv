// One column-parallel chain node: command executor + tile loader + row sequencer + chain +
// result port + result writer.  Two clocks: the array clock (chain, sequencer, result bank)
// and the fabric/NAP clock (everything else, including the chain's BRAM72K writes).
// The node talks to GDDR6 through one AXI4 read master and one AXI4 write master (a NAP or
// a NAP-sharing mux) and runs a static command program (colpar_node_ctrl.sv).  With
// PROG_FROM_GDDR6 = 1 the program is read from GDDR6 at i_prog_base after i_prog_start
// (colpar_prog_fetch.sv, sharing the read channel while the executor waits for a command);
// otherwise it arrives on the i_cmd stream.
module colpar_chain_node #(
    parameter integer N_STAGE         = 16,
    parameter integer ADDR_BITS       = 9,
    parameter integer ADDR_CASCADE    = 0,
    parameter integer AXI_ADDR_WIDTH  = 42,
    parameter integer RD_BURST_BEATS  = 16,     // the NoC carries at most 16 beats (UG086)
    parameter integer RD_MAX_OUTSTANDING = 8,
    parameter integer WR_BURST_BEATS  = 16,
    parameter integer WR_MAX_OUTSTANDING = 8,
    parameter integer WR_SUM_BITS     = 32,
    parameter integer WR_FIFO_LOG2    = 8,
    parameter integer WR_AFULL_MARGIN = 8,
    parameter integer PROG_FROM_GDDR6 = 0,
    parameter [4:0]   MULT_MODE       = 5'h00,  // chain multipliers: 5'h00 int8 x int8, 5'h13 uint8 act x int8 wt
    // Pipeline the chain's weight-load broadcast once per this many stages (0 = flat, the original).
    // Needed above ~16 stages: the flat 144-bit bus to every stage does not route.  The writes land
    // that many fabric cycles later, so the loader's "done" is held back to match.
    parameter integer WR_PIPE_EVERY = 0,
    // copies of the result-port capture enable (see colpar_result_port.sv); 1 = the original
    parameter integer VALID_COPIES  = 1,
    // 1: the result writer stores each record straight from the port's array-clock bank into a record-wide
    // BRAM FIFO (no fabric copy, skid or serialiser); 0: the original path
    parameter integer WR_SLIM       = 1
) (
    input  wire                       i_clk_array,
    input  wire                       i_clk_fabric,
    input  wire                       i_rstn,

    input  wire                       i_prog_start,   // PROG_FROM_GDDR6 = 1
    input  wire [AXI_ADDR_WIDTH-1:0]  i_prog_base,
    input  wire [127:0]               i_cmd,          // PROG_FROM_GDDR6 = 0
    input  wire                       i_cmd_valid,
    output wire                       o_cmd_ready,

    output wire                       o_arvalid,
    input  wire                       i_arready,
    output wire [AXI_ADDR_WIDTH-1:0]  o_araddr,
    output wire [7:0]                 o_arlen,
    output wire [2:0]                 o_arsize,
    output wire [1:0]                 o_arburst,
    output wire                       o_rready,
    input  wire                       i_rvalid,
    input  wire [255:0]               i_rdata,
    input  wire [1:0]                 i_rresp,
    input  wire                       i_rlast,

    output wire                       o_awvalid,
    input  wire                       i_awready,
    output wire [AXI_ADDR_WIDTH-1:0]  o_awaddr,
    output wire [7:0]                 o_awlen,
    output wire [2:0]                 o_awsize,
    output wire [1:0]                 o_awburst,
    output wire                       o_wvalid,
    input  wire                       i_wready,
    output wire [255:0]               o_wdata,
    output wire [31:0]                o_wstrb,
    output wire                       o_wlast,
    input  wire                       i_bvalid,
    output wire                       o_bready,
    input  wire [1:0]                 i_bresp,

    output wire                       o_halt,
    output wire                       o_error,
    output wire [4:0]                 o_error_bits      // {sync, prog fetch, overrun, writer, loader}
);
    wire                      ld_arm, ld_done_raw, ld_busy, ld_error;
    // the chain's write-bus pipeline depth, kept identical to mlp72_int8_colpar_chain.sv
    localparam integer WR_DEPTH = (WR_PIPE_EVERY == 0) ? 0
                                : (N_STAGE + WR_PIPE_EVERY) / WR_PIPE_EVERY;
    wire                      ld_done;
    wire [ADDR_BITS-1:0]      ld_wbase;
    generate
        if (WR_DEPTH == 0) begin : g_done_direct
            assign ld_done = ld_done_raw;
        end else begin : g_done_delayed
            // the last stage's write is WR_DEPTH cycles behind the loader, so a TILE must not start
            // on the loader's own done or it would read a BRAM that has not been written yet
            reg [WR_DEPTH:0] done_d;          // one bit wider so WR_DEPTH = 1 is a legal slice
            always @(posedge i_clk_fabric) begin
                if (!i_rstn) done_d <= '0;
                else         done_d <= {done_d[WR_DEPTH-1:0], ld_done_raw};
            end
`ifdef COLPAR_NEG_NO_WR_PIPE_DELAY
            // negative control: report the load done on the loader's own timing, so a TILE starts
            // before the last pipelined stage has been written
            assign ld_done = ld_done_raw;
`else
            assign ld_done = done_d[WR_DEPTH-1];
`endif
        end
    endgenerate
    wire [AXI_ADDR_WIDTH-1:0] ld_base;
    wire [4:0]                ld_first, ld_nreg;
    wire [ADDR_BITS:0]        ld_nwords, ld_nsegs;
    wire [13:0]               ld_seg_step, ld_tgt_step;
    wire [143:0]              wdata;
    wire [ADDR_BITS-1:0]      waddr;
    wire [N_STAGE:0]          wen;

    wire                      sy_arm, sy_post, sy_done, sy_busy, sy_error;
    wire [AXI_ADDR_WIDTH-1:0] sy_addr;
    wire [31:0]               sy_value;
    wire                      sy_arvalid, sy_rready, sy_awvalid, sy_wvalid;
    wire [AXI_ADDR_WIDTH-1:0] sy_araddr, sy_awaddr;
    wire [7:0]                sy_arlen, sy_awlen;
    wire [2:0]                sy_arsize, sy_awsize;
    wire [1:0]                sy_arburst, sy_awburst;
    wire [255:0]              sy_wdata;
    wire [31:0]               sy_wstrb;
    wire                      sy_wlast, sy_bready;
    wire                      w_awvalid, w_wvalid, w_wlast, w_bready;
    wire [AXI_ADDR_WIDTH-1:0] w_awaddr;
    wire [7:0]                w_awlen;
    wire [2:0]                w_awsize;
    wire [1:0]                w_awburst;
    wire [255:0]              w_wdata;
    wire [31:0]               w_wstrb;

    wire                      seq_start, seq_busy, seq_done, hold;
    wire [ADDR_BITS-1:0]      act_base, wt_base;
    wire [ADDR_BITS:0]        words, rows, passes;
    wire [4:0]                gap;
    wire                      s_valid, s_first, s_last;
    wire [ADDR_BITS-1:0]      s_act, s_wt;

    wire [48*N_STAGE-1:0]     sums, fsums;
    wire [VALID_COPIES-1:0]   sums_valid;
    wire                      fvalid, overrun, afull;
    wire [AXI_ADDR_WIDTH-1:0] out_base;
    wire [9:0]                out_blk;
    wire [23:0]               out_gap;
    wire                      wr_restart, wr_flush, wr_idle, wr_error, wr_drained;

    // ---- command source and the node's read channel ----
    wire [127:0]              cmd;
    wire                      cmd_valid, cmd_ready, fetching, pf_error;
    wire                      l_arvalid, l_rready, f_arvalid, f_rready;
    wire [AXI_ADDR_WIDTH-1:0] l_araddr, f_araddr;
    wire [7:0]                l_arlen, f_arlen;
    wire [2:0]                l_arsize, f_arsize;
    wire [1:0]                l_arburst, f_arburst;
    generate
        if (PROG_FROM_GDDR6 != 0) begin : g_fetch
            colpar_prog_fetch #(.AXI_ADDR_WIDTH(AXI_ADDR_WIDTH)) u_fetch (
                .i_clk(i_clk_fabric), .i_rstn(i_rstn), .i_start(i_prog_start), .i_base(i_prog_base),
                .i_grant(fetching),
                .o_arvalid(f_arvalid), .i_arready(i_arready && fetching), .o_araddr(f_araddr),
                .o_arlen(f_arlen), .o_arsize(f_arsize), .o_arburst(f_arburst),
                .i_rvalid(i_rvalid && fetching), .o_rready(f_rready), .i_rdata(i_rdata),
                .i_rresp(i_rresp), .i_rlast(i_rlast),
                .o_cmd(cmd), .o_cmd_valid(cmd_valid), .i_cmd_ready(cmd_ready), .o_error(pf_error));
            assign o_cmd_ready = 1'b0;
        end else begin : g_stream
            assign cmd         = i_cmd;
            assign cmd_valid   = i_cmd_valid;
            assign o_cmd_ready = cmd_ready;
            assign f_arvalid   = 1'b0;
            assign f_araddr    = '0;
            assign f_arlen     = '0;
            assign f_arsize    = '0;
            assign f_arburst   = '0;
            assign f_rready    = 1'b0;
            assign pf_error    = 1'b0;
        end
    endgenerate
    // the executor only waits for a command with the loader idle, so `fetching` selects the owner
    assign o_arvalid = sy_busy ? sy_arvalid : fetching ? f_arvalid : l_arvalid;
    assign o_araddr  = sy_busy ? sy_araddr  : fetching ? f_araddr  : l_araddr;
    assign o_arlen   = sy_busy ? sy_arlen   : fetching ? f_arlen   : l_arlen;
    assign o_arsize  = sy_busy ? sy_arsize  : fetching ? f_arsize  : l_arsize;
    assign o_arburst = sy_busy ? sy_arburst : fetching ? f_arburst : l_arburst;
    assign o_rready  = sy_busy ? sy_rready  : fetching ? f_rready  : l_rready;

    colpar_node_ctrl #(.AXI_ADDR_WIDTH(AXI_ADDR_WIDTH), .ADDR_BITS(ADDR_BITS)) u_ctrl (
        .i_clk(i_clk_fabric), .i_rstn(i_rstn), .i_restart(i_prog_start && PROG_FROM_GDDR6 != 0),
        .i_cmd(cmd), .i_cmd_valid(cmd_valid), .o_cmd_ready(cmd_ready),
        .o_ld_arm(ld_arm), .o_ld_base(ld_base), .o_ld_first(ld_first), .o_ld_nreg(ld_nreg),
        .o_ld_nwords(ld_nwords), .o_ld_nsegs(ld_nsegs), .o_ld_seg_step(ld_seg_step), .o_ld_tgt_step(ld_tgt_step), .o_ld_wbase(ld_wbase),
        .i_ld_done(ld_done),
        .o_seq_start(seq_start), .o_act_base(act_base), .o_wt_base(wt_base), .o_words(words),
        .o_rows(rows), .o_passes(passes), .o_gap(gap),
        .i_seq_busy_async(seq_busy), .i_seq_done_async(seq_done),
        .i_rec_valid(fvalid), .o_out_base(out_base), .o_out_blk(out_blk), .o_out_gap(out_gap),
        .o_wr_restart(wr_restart),
        .o_wr_flush(wr_flush), .i_wr_idle(wr_idle), .i_wr_drained(wr_drained),
        .o_sync_arm(sy_arm), .o_sync_post(sy_post), .o_sync_addr(sy_addr), .o_sync_value(sy_value),
        .i_sync_done(sy_done),
        .o_halt(o_halt), .o_fetching(fetching));

    node_sync #(.AXI_ADDR_WIDTH(AXI_ADDR_WIDTH)) u_sync (
        .i_clk(i_clk_fabric), .i_rstn(i_rstn), .i_arm(sy_arm), .i_post(sy_post), .i_addr(sy_addr),
        .i_value(sy_value), .o_busy(sy_busy), .o_done(sy_done), .o_error(sy_error),
        .o_arvalid(sy_arvalid), .i_arready(i_arready && sy_busy), .o_araddr(sy_araddr), .o_arlen(sy_arlen),
        .o_arsize(sy_arsize), .o_arburst(sy_arburst), .o_rready(sy_rready), .i_rvalid(i_rvalid && sy_busy),
        .i_rdata(i_rdata), .i_rresp(i_rresp),
        .o_awvalid(sy_awvalid), .i_awready(i_awready && sy_busy), .o_awaddr(sy_awaddr), .o_awlen(sy_awlen),
        .o_awsize(sy_awsize), .o_awburst(sy_awburst), .o_wvalid(sy_wvalid), .i_wready(i_wready && sy_busy),
        .o_wdata(sy_wdata), .o_wstrb(sy_wstrb), .o_wlast(sy_wlast), .i_bvalid(i_bvalid && sy_busy),
        .o_bready(sy_bready), .i_bresp(i_bresp));

    colpar_tile_loader #(
        .N_STAGE(N_STAGE), .ADDR_BITS(ADDR_BITS), .AXI_ADDR_WIDTH(AXI_ADDR_WIDTH),
        .BEATS_PER_BURST(RD_BURST_BEATS), .MAX_OUTSTANDING(RD_MAX_OUTSTANDING)
    ) u_ld (
        .i_clk(i_clk_fabric), .i_rstn(i_rstn), .i_arm(ld_arm), .i_base(ld_base),
        .i_first_target(ld_first), .i_n_regions(ld_nreg), .i_n_words(ld_nwords), .i_n_segs(ld_nsegs),
        .i_seg_step(ld_seg_step), .i_tgt_step(ld_tgt_step), .i_wbase(ld_wbase),
        .o_arvalid(l_arvalid), .i_arready(i_arready && !fetching), .o_araddr(l_araddr), .o_arlen(l_arlen),
        .o_arsize(l_arsize), .o_arburst(l_arburst), .o_rready(l_rready), .i_rvalid(i_rvalid && !fetching),
        .i_rdata(i_rdata), .i_rresp(i_rresp), .i_rlast(i_rlast),
        .o_wdata(wdata), .o_waddr(waddr), .o_wen(wen),
        .o_busy(ld_busy), .o_done(ld_done_raw), .o_error(ld_error));

    colpar_row_sequencer #(.ADDR_BITS(ADDR_BITS)) u_seq (
        .i_clk(i_clk_array), .i_rstn(i_rstn), .i_start(seq_start), .i_hold(hold),
        .i_act_base(act_base), .i_wt_base(wt_base), .i_words(words), .i_rows(rows),
        .i_passes(passes), .i_gap(gap),
        .o_valid(s_valid), .o_first(s_first), .o_last(s_last),
        .o_act_raddr(s_act), .o_wt_raddr(s_wt), .o_busy(seq_busy), .o_done(seq_done));

    mlp72_int8_colpar_chain #(
        .N_STAGE(N_STAGE), .ADDR_BITS(ADDR_BITS), .ADDR_CASCADE(ADDR_CASCADE), .MULT_MODE(MULT_MODE),
        .WR_PIPE_EVERY(WR_PIPE_EVERY), .VALID_COPIES(VALID_COPIES)
    ) u_chain (
        .i_clk(i_clk_array), .i_rstn(i_rstn), .i_wclk(i_clk_fabric), .i_wdata(wdata),
        .i_waddr(waddr), .i_wen(wen), .i_valid(s_valid), .i_act_raddr(s_act), .i_wt_raddr(s_wt),
        .i_first(s_first), .i_last(s_last), .o_sums(sums), .o_sums_valid(sums_valid));

    colpar_result_port #(.N_STAGE(N_STAGE), .VALID_COPIES(VALID_COPIES), .FABRIC_COPY(WR_SLIM ? 0 : 1)) u_port (
        .i_clk_array(i_clk_array), .i_rstn_array(i_rstn), .i_sums(sums), .i_sums_valid(sums_valid),
        .o_hold_array(hold),
        .i_clk_fabric(i_clk_fabric), .i_rstn_fabric(i_rstn), .i_fifo_afull(afull),
        .o_sums(fsums), .o_valid(fvalid), .o_overrun(overrun));

    colpar_result_writer #(
        .N_STAGE(N_STAGE), .SUM_BITS(WR_SUM_BITS), .AXI_ADDR_WIDTH(AXI_ADDR_WIDTH), .FIFO_LOG2(WR_FIFO_LOG2),
        .BURST_BEATS(WR_BURST_BEATS), .MAX_OUTSTANDING(WR_MAX_OUTSTANDING), .AFULL_MARGIN(WR_AFULL_MARGIN),
        .RECORD_FIFO(WR_SLIM)
    ) u_wr (
        .i_clk(i_clk_fabric), .i_rstn(i_rstn), .i_out_base(out_base), .i_blk_beats(out_blk),
        .i_gap_bytes(out_gap), .i_restart(wr_restart),
        .i_flush(wr_flush), .i_sums(fsums), .i_sums_valid(fvalid), .o_afull(afull),
        .o_awvalid(w_awvalid), .i_awready(i_awready && !sy_busy), .o_awaddr(w_awaddr), .o_awlen(w_awlen),
        .o_awsize(w_awsize), .o_awburst(w_awburst), .o_wvalid(w_wvalid), .i_wready(i_wready && !sy_busy),
        .o_wdata(w_wdata), .o_wstrb(w_wstrb), .o_wlast(w_wlast), .i_bvalid(i_bvalid && !sy_busy),
        .o_bready(w_bready), .i_bresp(i_bresp), .o_idle(wr_idle), .o_drained(wr_drained), .o_error(wr_error));

    assign o_error = ld_error | wr_error | overrun | pf_error | sy_error;
    assign o_error_bits = {sy_error, pf_error, overrun, wr_error, ld_error};
    // the write channel belongs to the result writer, except while the sync master posts a flag
    assign o_awvalid = sy_busy ? sy_awvalid : w_awvalid;
    assign o_awaddr  = sy_busy ? sy_awaddr  : w_awaddr;
    assign o_awlen   = sy_busy ? sy_awlen   : w_awlen;
    assign o_awsize  = sy_busy ? sy_awsize  : w_awsize;
    assign o_awburst = sy_busy ? sy_awburst : w_awburst;
    assign o_wvalid  = sy_busy ? sy_wvalid  : w_wvalid;
    assign o_wdata   = sy_busy ? sy_wdata   : w_wdata;
    assign o_wstrb   = sy_busy ? sy_wstrb   : w_wstrb;
    assign o_wlast   = sy_busy ? sy_wlast   : w_wlast;
    assign o_bready  = sy_busy ? sy_bready  : w_bready;

endmodule
