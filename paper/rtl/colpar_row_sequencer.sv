// Row/pass sequencer for one column-parallel chain
// (paper/rtl/mlp72_int8_colpar_chain.sv), on the array clock.
//
// A tile is M activation rows against P weight columns per stage:
//   feeder BRAM72K   row r    at act_base + r*W .. + W-1
//   stage  BRAM72K   column p at wt_base  + p*W .. + W-1   (stage s holds its own P columns)
// Row pass (r, p) streams W words with the feeder address walking row r and the
// stage address walking column p, then G >= 1 idle cycles.  It yields one column
// sum per stage, so a tile yields M x P row passes = M x P x N_STAGE results, in
// the order r = 0..M-1 (outer), p = 0..P-1 (inner), stage 1..N_STAGE.  With the compiler's
// column assignment c = N_STAGE p + s the results of each row are in column order, so the
// output records form a row-major matrix that the vector node's DEQUANT reads directly.
//
// Built for the array clock (750 MHz target): every register sees one LUT level
// plus at most one short carry chain.  The word, row and pass counters are
// loaded with n-2 and count down; "last" is the counter's sign bit, so no
// comparator is needed.  The gap counter has its own copy of in_row (fan-out) and
// saturates once negative, so a held gap needs no extra mux.  The next row's
// start addresses (start + W) are computed the cycle after a start changes,
// which is early enough because a row pass lasts at least two cycles.
//
// i_hold (array-clock synchronous, e.g. colpar_result_port.o_hold_array) stretches
// the gap before the next row pass; passes already started finish normally.
//
// Command inputs are quasi-static: the slow domain sets them, raises i_start and
// holds everything until o_done (a level, cleared by the next start).  i_start is
// synchronised here; the command buses need a false-path/max-delay constraint.
module colpar_row_sequencer #(
    parameter integer ADDR_BITS = 9,
    parameter integer GAP_BITS  = 5
) (
    input  wire                   i_clk,
    input  wire                   i_rstn,
    input  wire                   i_start,
    input  wire                   i_hold,
    input  wire [ADDR_BITS-1:0]   i_act_base,
    input  wire [ADDR_BITS-1:0]   i_wt_base,
    input  wire [ADDR_BITS:0]     i_words,    // W >= 1
    input  wire [ADDR_BITS:0]     i_rows,     // M >= 1
    input  wire [ADDR_BITS:0]     i_passes,   // P >= 1
    input  wire [GAP_BITS-1:0]    i_gap,      // G >= 1
    output reg                    o_valid,
    output reg                    o_first,
    output reg                    o_last,
    output reg  [ADDR_BITS-1:0]   o_act_raddr,
    output reg  [ADDR_BITS-1:0]   o_wt_raddr,
    output reg                    o_busy,
    output reg                    o_done
);
    localparam integer CW = ADDR_BITS + 2;    // signed counters for W, M, P
    localparam integer GW = GAP_BITS + 1;     // signed gap counter

    reg [2:0]            start_q;
    wire                 start_rise = start_q[1] & ~start_q[2];

    reg [CW-1:0]         wm2, mm2, pm2;
    reg [GW-1:0]         gm2;
    reg [CW-1:0]         k_cnt, r_cnt, p_cnt;
    reg [GW-1:0]         g_cnt;
    reg                  in_row, first_flag, tile_end;
    (* syn_preserve = 1 *) reg in_row_g;     // copy of in_row that only drives the gap counter
    reg [ADDR_BITS-1:0]  words_q, act_base_q, wt_base_q;
    reg [ADDR_BITS-1:0]  act_start, wt_start, act_start_w, wt_start_w;
    reg [ADDR_BITS-1:0]  act_cur, wt_cur;

    // gap counter: loaded at a row's last word, otherwise counts down and stops once negative
    always @(posedge i_clk) begin
        if (start_rise)
            g_cnt <= {GW{1'b1}};                 // one idle cycle, then the first row
        else if (in_row_g && k_cnt[CW-1])
            g_cnt <= gm2;
        else
            g_cnt <= g_cnt + {GW{~g_cnt[GW-1]}};
    end

    always @(posedge i_clk) begin
        start_q     <= {start_q[1:0], i_start};
        act_start_w <= act_start + words_q;
        wt_start_w  <= wt_start + words_q;

        if (!i_rstn) begin
            start_q  <= 3'b000;
            o_valid  <= 1'b0;
            o_first  <= 1'b0;
            o_last   <= 1'b0;
            o_busy   <= 1'b0;
            o_done   <= 1'b0;
            in_row   <= 1'b0;
            in_row_g <= 1'b0;
            tile_end <= 1'b0;
        end else if (start_rise) begin
            wm2        <= CW'(i_words)  - CW'(2);
            mm2        <= CW'(i_rows)   - CW'(2);
            pm2        <= CW'(i_passes) - CW'(2);
            gm2        <= GW'(i_gap)    - GW'(2);
            r_cnt      <= CW'(i_rows)   - CW'(2);
            p_cnt      <= CW'(i_passes) - CW'(2);
            words_q    <= i_words[ADDR_BITS-1:0];
            act_base_q <= i_act_base;
            wt_base_q  <= i_wt_base;
            act_start  <= i_act_base;
            wt_start   <= i_wt_base;
            in_row     <= 1'b0;
            in_row_g   <= 1'b0;
            tile_end   <= 1'b0;
            o_busy     <= 1'b1;
            o_done     <= 1'b0;
            o_valid    <= 1'b0;
            o_first    <= 1'b0;
            o_last     <= 1'b0;
        end else if (o_busy) begin
            if (in_row) begin
                o_valid     <= 1'b1;
                o_first     <= first_flag;
                o_last      <= k_cnt[CW-1];
                o_act_raddr <= act_cur;
                o_wt_raddr  <= wt_cur;
                first_flag  <= 1'b0;
                act_cur     <= act_cur + 1'b1;
                wt_cur      <= wt_cur + 1'b1;
                k_cnt       <= k_cnt - 1'b1;
                if (k_cnt[CW-1]) begin               // the row's last word
                    in_row   <= 1'b0;
                    in_row_g <= 1'b0;
                    if (p_cnt[CW-1]) begin           // last column pass of this row
                        p_cnt     <= pm2;
                        wt_start  <= wt_base_q;
                        act_start <= act_start_w;
                        if (r_cnt[CW-1]) tile_end <= 1'b1;
                        else             r_cnt    <= r_cnt - 1'b1;
                    end else begin
                        p_cnt     <= p_cnt - 1'b1;
                        wt_start  <= wt_start_w;
                    end
                end
            end else begin
                o_valid <= 1'b0;
                o_first <= 1'b0;
                o_last  <= 1'b0;
                if (g_cnt[GW-1] && (tile_end || !i_hold)) begin   // gap over
                    if (tile_end) begin
                        o_busy <= 1'b0;
                        o_done <= 1'b1;
                    end else begin
                        in_row     <= 1'b1;
                        in_row_g   <= 1'b1;
                        first_flag <= 1'b1;
                        act_cur    <= act_start;
                        wt_cur     <= wt_start;
                        k_cnt      <= wm2;
                    end
                end
            end
        end
    end
endmodule
