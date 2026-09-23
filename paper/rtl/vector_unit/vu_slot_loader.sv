// Operand slot loader for the vector node: reads N elements of one width from GDDR6 (AXI4 read,
// through the node's read channel) and writes them to a slot memory at addresses 0..N-1.
//
// Element layout in GDDR6: little-endian, WIDTH bits each, packed without gaps from byte address
// 0 of the tensor: element i occupies bits [i*WIDTH, (i+1)*WIDTH) of the byte stream, so a 256-bit
// beat holds 256/WIDTH elements.  A load starts at element i_first (it need not be on a beat
// boundary: the loader fetches the beat that holds it and skips the leading elements).
// Widths: 16 (bf16), 32 (fp32 / int32), 64 (summary records {amax, max, rs0}).
//
// Bursts: at most BEATS_PER_BURST beats (NoC limit 16, UG086), up to MAX_OUTSTANDING AR requests
// ahead of their data (one AXI ID).  Output: o_we / o_waddr / o_wdata (64 bits, zero-extended),
// one element per cycle; the caller routes them to the selected slot.
module vu_slot_loader #(
    parameter integer AXI_ADDR_WIDTH  = 42,
    parameter integer ADDR_BITS       = 11,          // slot depth 2^ADDR_BITS elements
    parameter integer BEATS_PER_BURST = 16,
    parameter integer MAX_OUTSTANDING = 8
) (
    input  wire                       i_clk,
    input  wire                       i_rstn,

    input  wire                       i_arm,           // one cycle; fields held until o_done
    input  wire [AXI_ADDR_WIDTH-1:0]  i_base,          // byte address of element 0 of the tensor
    input  wire [31:0]                i_first,         // first element to load
    input  wire [ADDR_BITS:0]         i_count,         // elements to load, 1 .. 2^ADDR_BITS
    input  wire [1:0]                 i_wsel,          // 0: 16 bits, 1: 32 bits, 2: 64 bits

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
    output reg  [ADDR_BITS-1:0]       o_waddr,
    output reg  [63:0]                o_wdata,

    output reg                        o_busy,
    output reg                        o_done,
    output reg                        o_error
);
    assign o_arsize  = 3'd5;
    assign o_arburst = 2'b01;

    // ---- load geometry, fixed at arm ----
    reg [1:0]   wsel;
    reg [5:0]   epb_log2;                    // log2(elements per beat): 4, 3, 2
    reg [ADDR_BITS:0] count_q;
    reg [31:0]  beats_total;                 // beats to read
    reg [4:0]   skip_q;                      // leading elements to skip in the first beat

    function automatic [8:0] burst_of(input [31:0] beats_left);
        burst_of = (beats_left > BEATS_PER_BURST) ? 9'(BEATS_PER_BURST) : 9'(beats_left);
    endfunction

    // ---- AR issuer ----
    reg         ar_active;
    reg [31:0]  ar_beats_left;
    reg [8:0]   ar_len_q;
    reg [7:0]   outstanding;
    wire        ar_fire = o_arvalid && i_arready;

    // ---- R consumer: one element per cycle out of a held beat ----
    reg         rc_active;
    reg [31:0]  rc_beats_left;
    reg [8:0]   rc_burst_left;
    reg [255:0] beat;
    reg         beat_valid;
    reg [5:0]   eidx;                        // next element index inside `beat`
    reg [ADDR_BITS:0] written;
    wire [5:0]  epb    = 6'd1 << epb_log2;
    wire        r_fire = o_rready && i_rvalid;
    wire        r_end  = r_fire && i_rlast;
    assign o_rready = rc_active && !beat_valid;

    wire [63:0] elem = (wsel == 2'd0) ? {48'd0, beat[{eidx, 4'd0} +: 16]} :
                       (wsel == 2'd1) ? {32'd0, beat[{eidx, 5'd0} +: 32]} :
                                        beat[{eidx, 6'd0} +: 64];

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
        end else begin
            outstanding <= outstanding + {7'd0, ar_fire} - {7'd0, r_end};

            if (i_arm && !o_busy) begin : arm
                reg [5:0]  lg;
                reg [31:0] first_beat, last_beat;
                lg          = (i_wsel == 2'd0) ? 6'd4 : (i_wsel == 2'd1) ? 6'd3 : 6'd2;
                first_beat  = i_first >> lg;
                last_beat   = (i_first + 32'(i_count) - 32'd1) >> lg;
                wsel        <= i_wsel;
                epb_log2    <= lg;
                count_q     <= i_count;
                skip_q      <= 5'(i_first & ((32'd1 << lg) - 32'd1));
                beats_total <= last_beat - first_beat + 32'd1;
                o_araddr    <= i_base + AXI_ADDR_WIDTH'({first_beat, 5'd0});
                ar_active   <= 1'b1;
                ar_beats_left <= last_beat - first_beat + 32'd1;
                rc_active   <= 1'b1;
                rc_beats_left <= last_beat - first_beat + 32'd1;
                rc_burst_left <= burst_of(last_beat - first_beat + 32'd1);
                written     <= '0;
                beat_valid  <= 1'b0;
                o_busy      <= 1'b1;
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

            // consumer: take a beat, then emit its elements one per cycle
            if (rc_active && !beat_valid && i_rvalid) begin
                beat          <= i_rdata;
                beat_valid    <= 1'b1;
                eidx          <= (rc_beats_left == beats_total) ? {1'b0, skip_q} : 6'd0;
                rc_beats_left <= rc_beats_left - 32'd1;
                rc_burst_left <= rc_burst_left - 9'd1;
                if (i_rresp != 2'b00 || i_rlast != (rc_burst_left == 9'd1)) o_error <= 1'b1;
                if (rc_burst_left == 9'd1) rc_burst_left <= burst_of(rc_beats_left - 32'd1);
            end else if (beat_valid) begin
                o_we    <= 1'b1;
                o_waddr <= written[ADDR_BITS-1:0];
                o_wdata <= elem;
                written <= written + 1'b1;
                eidx    <= eidx + 6'd1;
                if (written + 1'b1 == count_q) begin
                    beat_valid <= 1'b0;
                    rc_active  <= 1'b0;
                    o_busy     <= 1'b0;
                    o_done     <= 1'b1;
                end else if (eidx + 6'd1 == epb) begin
                    beat_valid <= 1'b0;
                end
            end
        end
    end
endmodule
