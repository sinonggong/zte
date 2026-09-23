// Result crossing for one column-parallel chain: array clock (750 MHz) -> fabric clock.
//
// The chain's o_sums (N_STAGE x int48 column sums of one row pass) are valid for one
// cycle.  This port banks them on the array clock and hands them to the fabric
// clock with a toggle request, without an acknowledge:
//
//   array:  bank <= i_sums and req_tgl <= ~req_tgl on i_sums_valid
//   fabric: 3-flop synchroniser on req_tgl; on a change, o_sums <= bank, o_valid pulse
//
// Why no acknowledge.  Results arrive at most once per row pass, i.e. at least
// W + G >= N_STAGE array cycles apart (the chain's row protocol), so the bank is
// stable for that long.  The fabric side sees the toggle within SYNC_STAGES + 1
// of its cycles and samples the bank in the same cycle.  The compiler guarantees
//     (SYNC_STAGES + 1) * T_fabric + margin  <=  (W + G) * T_array
// by choosing G (with 750 / 250 MHz and 3 flops: 4 * 4 ns = 16 ns <= 16 * 1.333 ns
// = 21.3 ns).  The fabric side must accept every result (it writes into its own
// FIFO); back-pressure is its almost-full flag, synchronised to the array clock
// as o_hold_array for colpar_row_sequencer, which then stretches the gap before
// the next row pass.  Up to ceil(LATENCY / (W + G)) + 1 passes are already in
// flight when the hold arrives, so the FIFO threshold must leave that much room.
//
// Constraints (the RTL alone does not make these safe):
//   set_max_delay from bank to fabric o_sums of (W_min + G_min) * T_array - (SYNC_STAGES + 1) * T_fabric,
//   set_false_path on req_tgl -> sync[0] and on i_fifo_afull -> hold sync (async synchronisers).
// o_overrun flags a result that reached the bank before the fabric side had taken the
// previous one (only possible if the gap rule above is violated); it is sticky.
module colpar_result_port #(
    parameter integer N_STAGE      = 16,
    parameter integer SYNC_STAGES  = 3,
    // copies of the capture enable the chain hands over, one per slice of the bank; at 32 stages the
    // bank is 1,536 flops and a single enable net does not make 750 MHz
    parameter integer VALID_COPIES = 1,
    // 1: o_sums is a fabric-clock copy of the bank, o_valid a registered pulse (the original).
    // 0: o_sums IS the bank and o_valid pulses in the sampling cycle itself, for a consumer that stores the
    //    record in that cycle (colpar_result_writer RECORD_FIFO = 1): the same sampling edge and crossing
    //    budget, without 48 x N_STAGE fabric flops
    parameter integer FABRIC_COPY  = 1
) (
    // array clock
    input  wire                     i_clk_array,
    input  wire                     i_rstn_array,
    input  wire [48*N_STAGE-1:0]    i_sums,
    input  wire [VALID_COPIES-1:0]  i_sums_valid,
    output reg                      o_hold_array,

    // fabric clock
    input  wire                     i_clk_fabric,
    input  wire                     i_rstn_fabric,
    input  wire                     i_fifo_afull,
    output wire [48*N_STAGE-1:0]    o_sums,
    output wire                     o_valid,
    output reg                      o_overrun
);

    // ---- array side ----
    // The bank is split into VALID_COPIES slices, each captured by its own copy of the enable, so no
    // single array-clock net drives more than 48*N_STAGE/VALID_COPIES loads.
    localparam integer SLICE = (48 * N_STAGE + VALID_COPIES - 1) / VALID_COPIES;
    (* syn_preserve = 1 *) reg [SLICE*VALID_COPIES-1:0] bank;
    reg                                          req_tgl;
    (* syn_preserve = 1 *) reg [1:0]             hold_sync;
    genvar gv;
    generate
        for (gv = 0; gv < VALID_COPIES; gv = gv + 1) begin : g_bank
            wire [SLICE-1:0] slice_in = {{SLICE{1'b0}}, i_sums[48*N_STAGE-1:0]} >> (gv * SLICE);
            always @(posedge i_clk_array)
                if (i_sums_valid[gv]) bank[gv*SLICE +: SLICE] <= slice_in;
        end
    endgenerate
    always @(posedge i_clk_array) begin
        hold_sync    <= i_rstn_array ? {hold_sync[0], i_fifo_afull} : 2'b00;
        o_hold_array <= hold_sync[1];
        if (!i_rstn_array) begin
            req_tgl <= 1'b0;
        end else if (i_sums_valid[0]) begin
            req_tgl <= ~req_tgl;
        end
    end

    // ---- fabric side ----
    (* syn_preserve = 1 *) reg [SYNC_STAGES-1:0] req_sync;
    reg                                          req_seen;
    // Overrun detection: the array side's request count moves by two between
    // fabric samples only if two results arrived inside one synchroniser window;
    // a toggle cannot show that, so count requests on the array side and compare.
    reg [7:0] req_cnt_array;
    (* syn_preserve = 1 *) reg [7:0] req_cnt_gray;          // Gray-coded copy: one bit moves per count, so the
    always @(posedge i_clk_array)                           // fabric side never sees a value it never had
        if (!i_rstn_array) begin
            req_cnt_array <= 8'd0;
            req_cnt_gray  <= 8'd0;
        end else if (i_sums_valid[0]) begin
            req_cnt_array <= req_cnt_array + 8'd1;
            req_cnt_gray  <= (req_cnt_array + 8'd1) ^ ((req_cnt_array + 8'd1) >> 1);
        end

    reg [7:0] got_cnt;
    (* syn_preserve = 1 *) reg [7:0] cnt_sync0, cnt_sync1;
    // Gray -> binary of the synchronised count (on the first silicon the binary crossing fired a false overrun)
    wire [7:0] cnt_bin = {cnt_sync1[7], cnt_sync1[6] ^ cnt_sync1[7],
                          ^cnt_sync1[7:5], ^cnt_sync1[7:4], ^cnt_sync1[7:3], ^cnt_sync1[7:2], ^cnt_sync1[7:1], ^cnt_sync1[7:0]};
    wire [7:0] cnt_lead = cnt_bin - got_cnt;          // results the array has made that the fabric has not taken
    wire      sample = i_rstn_fabric && (req_sync[SYNC_STAGES-1] != req_seen);
    reg [48*N_STAGE-1:0] sums_q;
    reg                  valid_q;
    assign o_sums  = (FABRIC_COPY != 0) ? sums_q : bank[48*N_STAGE-1:0];
    assign o_valid = (FABRIC_COPY != 0) ? valid_q : sample;
    always @(posedge i_clk_fabric) begin
        // the synchroniser and count copies are reset too: unreset, a random power-up value of
        // req_sync[SYNC_STAGES-1] fires one spurious sample after reset (seen with Verilator's
        // +verilator+rand+reset+2, and as node_error on the first silicon)
        req_sync <= i_rstn_fabric ? {req_sync[SYNC_STAGES-2:0], req_tgl} : '0;
        valid_q  <= 1'b0;
        if (!i_rstn_fabric) begin
            req_seen  <= 1'b0;
            got_cnt   <= 8'd0;
            o_overrun <= 1'b0;
            cnt_sync0 <= 8'd0;
            cnt_sync1 <= 8'd0;
        end else if (sample) begin
            req_seen <= req_sync[SYNC_STAGES-1];
            if (FABRIC_COPY != 0) sums_q <= bank[48*N_STAGE-1:0];
            valid_q  <= 1'b1;
            got_cnt  <= got_cnt + 8'd1;
        end
        // overrun check over a Gray-coded crossing: two results inside one synchroniser window.
        // The lead is signed: the count and the toggle cross on unconstrained paths, and when the count's route is
        // slower the fabric side's got_cnt is briefly AHEAD of cnt_bin (lead -1 = 255 unsigned).  Read unsigned,
        // that fired a false overrun on the half array (s1m_250, chip nodes 1 and 2, 2026-09-18) on runs whose data
        // were bit-exact -- a real overrun loses a record, and the node would never finish its TILE.
        if (i_rstn_fabric) begin
            cnt_sync0 <= req_cnt_gray;
            cnt_sync1 <= cnt_sync0;
        end
        if (i_rstn_fabric && !cnt_lead[7] && cnt_lead > 8'd2)
            o_overrun <= 1'b1;
    end
endmodule
