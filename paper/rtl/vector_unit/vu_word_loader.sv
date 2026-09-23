// Word loader for the multi-lane vector node (vu_node_ml.sv): reads elements [i_first, i_first+i_count)
// of one operand tensor from GDDR6 (AXI4 read) and emits them as 4-element words, one word per cycle.
//
// Element layout in GDDR6 is that of vu_slot_loader.sv: element i occupies bits [i*WIDTH, (i+1)*WIDTH)
// from the tensor base (beat aligned), WIDTH 16 (bf16), 32 (fp32 / int32) or 64 (summary record
// {amax, max, rs0}).  Word m holds elements 4m .. 4m+3, so a word is a fixed slice of one beat (a beat
// holds 4 / 2 / 1 words).  Emitted words cover element words (i_first >> 2) .. ((i_first+i_count-1) >> 2);
// o_widx counts them from 0.  Elements outside the range but inside an emitted word are emitted too
// (the caller never reads them).
//
// o_wword = {e3, e2, e1, e0}, 32 bits per element: bf16 zero-extended, fp32 / int32 as stored, summary
// records reduced to the field i_field (0 rs0 [31:0], 1 max [47:32], 2 amax [63:48], zero-extended).
//
// Loading 4 elements per cycle instead of one is what lets one loader keep up with 4 lanes (each lane
// takes one element per cycle).  The next beat is taken in the cycle that emits the last word of the held
// one, so a bf16 beat costs 4 cycles and a 32-bit beat 2 (4 elements per cycle either way; without it,
// 5 and 3).  Negative control VU_WL_NEG_NO_SKID restores the extra cycle (same data, more cycles).
// The load geometry is derived over three cycles after i_arm (capture; beats and words; burst length), so the
// arm arithmetic never shares a cycle with the AXI state it sets up.  Bursts: at most BEATS_PER_BURST beats (NoC limit 16, UG086), up to
// MAX_OUTSTANDING AR requests ahead of their data (one AXI ID), as in vu_slot_loader.sv.
module vu_word_loader #(
    parameter integer AXI_ADDR_WIDTH  = 42,
    parameter integer CNT_BITS        = 13,          // elements per load <= 2^CNT_BITS
    parameter integer BEATS_PER_BURST = 16,
    parameter integer MAX_OUTSTANDING = 8
) (
    input  wire                       i_clk,
    input  wire                       i_rstn,

    input  wire                       i_arm,           // one cycle; fields held until o_done
    input  wire [AXI_ADDR_WIDTH-1:0]  i_base,          // byte address of element 0 of the tensor
    input  wire [31:0]                i_first,         // first element to load
    input  wire [CNT_BITS:0]          i_count,         // elements to load, 1 .. 2^CNT_BITS
    input  wire [1:0]                 i_wsel,          // 0: 16 bits, 1: 32 bits, 2: 64 bits
    input  wire [1:0]                 i_field,         // summary records: 0 rs0, 1 max, 2 amax

    output reg                        o_arvalid,
    input  wire                       i_arready,
    output reg  [AXI_ADDR_WIDTH-1:0]  o_araddr,
    output reg  [7:0]                 o_arlen,
    output wire [2:0]                 o_arsize,
    output wire [1:0]                 o_arburst,
    output wire                       o_rready,
    input  wire                       i_rvalid,
    input  wire [255:0]               i_rdata,
    input  wire [1:0]                 i_rresp,
    input  wire                       i_rlast,

    output reg                        o_we,
    output reg  [CNT_BITS-2:0]        o_widx,
    output reg  [127:0]               o_wword,

    output reg                        o_busy,
    output reg                        o_done,
    output reg                        o_error
);
    assign o_arsize  = 3'd5;
    assign o_arburst = 2'b01;

    // ---- load geometry, fixed at arm ----
    reg [1:0]   wsel, field;
    reg [1:0]   wpb_log2;                    // log2(words per beat): 2, 1, 0
    reg [31:0]  beats_total;
    reg [1:0]   skip_q;                      // leading words to skip in the first beat
    reg         arm1, arm2;                  // geometry pipeline after i_arm
    reg [31:0]  a_first, a_last;
    reg [AXI_ADDR_WIDTH-1:0] a_base;
    reg [31:0]  a_beats;

    function automatic [8:0] burst_of(input [31:0] beats_left);
        burst_of = (beats_left > BEATS_PER_BURST) ? 9'(BEATS_PER_BURST) : 9'(beats_left);
    endfunction

    // ---- AR issuer ----
    reg         ar_active;
    reg [31:0]  ar_beats_left;
    reg [8:0]   ar_len_q;
    reg [7:0]   outstanding;
    wire        ar_fire = o_arvalid && i_arready;

    // ---- R consumer: one word per cycle out of a held beat ----
    reg         rc_active;
    reg [31:0]  rc_beats_left;
    reg [8:0]   rc_burst_left;
    reg [255:0] beat;
    reg         beat_valid;
    reg [1:0]   kidx;                        // next word index inside `beat`
    reg [31:0]  written;
    reg [31:0]  words_left;
    wire [2:0]  wpb    = 3'd1 << wpb_log2;
    wire        last_in_beat = (3'(kidx) + 3'd1 == wpb);
    wire        final_word   = (words_left == 32'd1);
`ifdef VU_WL_NEG_NO_SKID
    wire        can_take = rc_active && !beat_valid;
`else
    wire        can_take = rc_active && (!beat_valid || (last_in_beat && !final_word));
`endif
    wire        r_fire = o_rready && i_rvalid;
    wire        r_end  = r_fire && i_rlast;
    assign o_rready = can_take;

    function automatic [31:0] fsel(input [63:0] r, input [1:0] f);
        fsel = (f == 2'd1) ? {16'd0, r[47:32]} : (f == 2'd2) ? {16'd0, r[63:48]} : r[31:0];
    endfunction

    wire [63:0]  s16  = beat[{kidx, 6'd0} +: 64];
    wire [127:0] s32  = beat[{kidx[0], 7'd0} +: 128];
    wire [127:0] word = (wsel == 2'd0) ? {16'd0, s16[63:48], 16'd0, s16[47:32], 16'd0, s16[31:16], 16'd0, s16[15:0]} :
                        (wsel == 2'd1) ? s32 :
                        {fsel(beat[255:192], field), fsel(beat[191:128], field), fsel(beat[127:64], field),
                         fsel(beat[63:0], field)};

    always @(posedge i_clk) begin
        o_done <= 1'b0;
        o_we   <= 1'b0;
        if (!i_rstn) begin
            o_arvalid   <= 1'b0;
            o_busy      <= 1'b0;
            o_error     <= 1'b0;
            ar_active   <= 1'b0;
            rc_active   <= 1'b0;
            beat_valid  <= 1'b0;
            outstanding <= 8'd0;
            arm1        <= 1'b0;
            arm2        <= 1'b0;
        end else begin
            outstanding <= outstanding + {7'd0, ar_fire} - {7'd0, r_end};

            // arm, cycle 1: capture the load
            arm1 <= 1'b0;
            arm2 <= 1'b0;
            if (i_arm && !o_busy) begin
                a_first  <= i_first;
                a_last   <= i_first + 32'(i_count) - 32'd1;
                a_base   <= i_base;
                wsel     <= i_wsel;
                field    <= i_field;
                wpb_log2 <= (i_wsel == 2'd0) ? 2'd2 : (i_wsel == 2'd1) ? 2'd1 : 2'd0;
                o_busy   <= 1'b1;
                arm1     <= 1'b1;
            end
            // cycle 2: beats and words
            if (arm1) begin : geom
                reg [5:0]  lg;
                reg [31:0] first_beat, last_beat;
                lg          = (wsel == 2'd0) ? 6'd4 : (wsel == 2'd1) ? 6'd3 : 6'd2;
                first_beat  = a_first >> lg;
                last_beat   = a_last >> lg;
                skip_q      <= 2'((a_first >> 2) & ((32'd1 << wpb_log2) - 32'd1));
                a_beats     <= last_beat - first_beat + 32'd1;
                o_araddr    <= a_base + AXI_ADDR_WIDTH'({first_beat, 5'd0});
                words_left  <= (a_last >> 2) - (a_first >> 2) + 32'd1;
                written     <= '0;
                beat_valid  <= 1'b0;
                arm2        <= 1'b1;
            end
            // cycle 3: bursts, then the load runs
            if (arm2) begin
                beats_total   <= a_beats;
                ar_active     <= 1'b1;
                ar_beats_left <= a_beats;
                rc_active     <= 1'b1;
                rc_beats_left <= a_beats;
                rc_burst_left <= burst_of(a_beats);
            end

            // issuer
            if (ar_active && !o_arvalid && outstanding < 8'(MAX_OUTSTANDING)) begin
                o_arvalid <= 1'b1;
                o_arlen   <= 8'(burst_of(ar_beats_left) - 9'd1);
                ar_len_q  <= burst_of(ar_beats_left);
            end
            if (ar_fire) begin
                o_arvalid     <= 1'b0;
                o_araddr      <= o_araddr + AXI_ADDR_WIDTH'({ar_len_q, 5'd0});
                ar_beats_left <= ar_beats_left - 32'(ar_len_q);
                if (ar_beats_left == 32'(ar_len_q)) ar_active <= 1'b0;
            end

            // consumer: emit the held beat's words one per cycle; take the next beat when none is held, or
            // in the cycle that emits the held beat's last word (the take below overrides beat_valid / kidx)
            if (beat_valid) begin
                o_we       <= 1'b1;
                o_widx     <= (CNT_BITS-1)'(written);
                o_wword    <= word;
                written    <= written + 32'd1;
                words_left <= words_left - 32'd1;
                kidx       <= kidx + 2'd1;
                if (final_word) begin
                    beat_valid <= 1'b0;
                    rc_active  <= 1'b0;
                    o_busy     <= 1'b0;
                    o_done     <= 1'b1;
                end else if (last_in_beat) begin
                    beat_valid <= 1'b0;
                end
            end
            if (can_take && i_rvalid) begin
                beat          <= i_rdata;
                beat_valid    <= 1'b1;
                kidx          <= (rc_beats_left == beats_total) ? skip_q : 2'd0;
                rc_beats_left <= rc_beats_left - 32'd1;
                rc_burst_left <= rc_burst_left - 9'd1;
                if (i_rresp != 2'b00 || i_rlast != (rc_burst_left == 9'd1)) o_error <= 1'b1;
                if (rc_burst_left == 9'd1) rc_burst_left <= burst_of(rc_beats_left - 32'd1);
            end
        end
    end
endmodule
