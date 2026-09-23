// Host -> GDDR6 paging bridge: the host's bulk path on the node-array bitstreams.
//
// Why: on these bitstreams the PCIe DMA engine cannot be used (its init through the compressed DBI gateway wedges
// the gateway) and a 32-bit BAR can map at most 512 MB of GDDR6, while a whole pi0 chunk needs ~10.5 GB (2.8 GB of
// weights and tables + per-stage regions).  So BAR1 maps a fabric NAP instead of GDDR6 (pci_express.acxip), and this
// block forwards every access to GDDR6 at  i_page_base + (address within the 256 MB NAP window).  The host moves
// the page through the register window (pi0_chip_ctrl write beat N_NODE + 1) -- one register write per 256 MB.
//
// Target side: what nap_axi_target (ACX_NAP_AXI_MASTER) delivers from the PCIe bridge -- single beats or short
// bursts with byte strobes, one transaction per direction at a time here (the wrapper echoes one AXI ID per
// direction, so a second one must not be accepted before the first answers).
// Initiator side: what nap_axi_initiator (ACX_NAP_AXI_SLAVE) needs toward GDDR6.  Burst length, size, strobes and
// data pass through unchanged; only the address is rebased.  Single clock (the fabric clock of both NAPs).
module pi0_host_gddr_bridge #(
    parameter integer TGT_AW = 28,
    parameter integer INI_AW = 42
) (
    input  wire                 i_clk,
    input  wire                 i_rstn,
    input  wire [INI_AW-1:0]    i_page_base,         // GDDR6 address of window offset 0 (static while accessed)

    // target side (from the NoC)
    input  wire                 t_arvalid,
    output wire                 t_arready,
    input  wire [TGT_AW-1:0]    t_araddr,
    input  wire [7:0]           t_arlen,
    input  wire [2:0]           t_arsize,
    input  wire [1:0]           t_arburst,
    output wire                 t_rvalid,
    input  wire                 t_rready,
    output wire [255:0]         t_rdata,
    output wire [1:0]           t_rresp,
    output wire                 t_rlast,
    input  wire                 t_awvalid,
    output wire                 t_awready,
    input  wire [TGT_AW-1:0]    t_awaddr,
    input  wire [7:0]           t_awlen,
    input  wire [2:0]           t_awsize,
    input  wire [1:0]           t_awburst,
    input  wire                 t_wvalid,
    output wire                 t_wready,
    input  wire [255:0]         t_wdata,
    input  wire [31:0]          t_wstrb,
    input  wire                 t_wlast,
    output wire                 t_bvalid,
    input  wire                 t_bready,
    output wire [1:0]           t_bresp,

    // initiator side (to GDDR6)
    output reg                  m_arvalid,
    input  wire                 m_arready,
    output reg  [INI_AW-1:0]    m_araddr,
    output reg  [7:0]           m_arlen,
    output reg  [2:0]           m_arsize,
    output reg  [1:0]           m_arburst,
    input  wire                 m_rvalid,
    output wire                 m_rready,
    input  wire [255:0]         m_rdata,
    input  wire [1:0]           m_rresp,
    input  wire                 m_rlast,
    output reg                  m_awvalid,
    input  wire                 m_awready,
    output reg  [INI_AW-1:0]    m_awaddr,
    output reg  [7:0]           m_awlen,
    output reg  [2:0]           m_awsize,
    output reg  [1:0]           m_awburst,
    output wire                 m_wvalid,
    input  wire                 m_wready,
    output wire [255:0]         m_wdata,
    output wire [31:0]          m_wstrb,
    output wire                 m_wlast,
    input  wire                 m_bvalid,
    output wire                 m_bready,
    input  wire [1:0]           m_bresp,

    output reg  [31:0]          o_n_reads,
    output reg  [31:0]          o_n_writes
);
    // ------------------------------------------------------------------ reads: AR -> forwarded AR -> R pass-through
    localparam [1:0] R_IDLE = 2'd0, R_AR = 2'd1, R_DATA = 2'd2;
    reg [1:0] rs;
    assign t_arready = (rs == R_IDLE);
    assign t_rvalid  = (rs == R_DATA) && m_rvalid;
    assign m_rready  = (rs == R_DATA) && t_rready;
    assign t_rdata   = m_rdata;
    assign t_rresp   = m_rresp;
    assign t_rlast   = m_rlast;
    always @(posedge i_clk) begin
        if (!i_rstn) begin
            rs        <= R_IDLE;
            m_arvalid <= 1'b0;
            o_n_reads <= 32'd0;
        end else begin
            case (rs)
            R_IDLE: if (t_arvalid) begin
                m_araddr  <= i_page_base + INI_AW'(t_araddr);
                m_arlen   <= t_arlen;
                m_arsize  <= t_arsize;
                m_arburst <= t_arburst;
                m_arvalid <= 1'b1;
                rs        <= R_AR;
            end
            R_AR: if (m_arready) begin
                m_arvalid <= 1'b0;
                rs        <= R_DATA;
            end
            R_DATA: if (m_rvalid && t_rready && m_rlast) begin
                rs        <= R_IDLE;
                o_n_reads <= o_n_reads + 32'd1;
            end
            default: rs <= R_IDLE;
            endcase
        end
    end

    // ------------------------------------------------------------------ writes: AW -> forwarded AW -> W pass-through -> B
    // W beats that arrive before the AW handshake wait at t_wready = 0 (AXI allows W before AW; nothing is dropped).
    localparam [1:0] W_IDLE = 2'd0, W_AW = 2'd1, W_DATA = 2'd2, W_RESP = 2'd3;
    reg [1:0] ws;
    reg       b_held;
    reg [1:0] b_resp_q;
    assign t_awready = (ws == W_IDLE);
    assign m_wvalid  = (ws == W_DATA) && t_wvalid;
    assign t_wready  = (ws == W_DATA) && m_wready;
    assign m_wdata   = t_wdata;
    assign m_wstrb   = t_wstrb;
    assign m_wlast   = t_wlast;
    assign m_bready  = (ws == W_RESP) && !b_held;
    assign t_bvalid  = b_held;
    assign t_bresp   = b_resp_q;
    always @(posedge i_clk) begin
        if (!i_rstn) begin
            ws         <= W_IDLE;
            m_awvalid  <= 1'b0;
            b_held     <= 1'b0;
            o_n_writes <= 32'd0;
        end else begin
            case (ws)
            W_IDLE: if (t_awvalid) begin
                m_awaddr  <= i_page_base + INI_AW'(t_awaddr);
                m_awlen   <= t_awlen;
                m_awsize  <= t_awsize;
                m_awburst <= t_awburst;
                m_awvalid <= 1'b1;
                ws        <= W_AW;
            end
            W_AW: if (m_awready) begin
                m_awvalid <= 1'b0;
                ws        <= W_DATA;
            end
            W_DATA: if (t_wvalid && m_wready && t_wlast) ws <= W_RESP;
            W_RESP: begin
                if (m_bvalid && !b_held) begin
                    b_held   <= 1'b1;
                    b_resp_q <= m_bresp;
                end
                if (b_held && t_bready) begin
                    b_held     <= 1'b0;
                    ws         <= W_IDLE;
                    o_n_writes <= o_n_writes + 32'd1;
                end
            end
            default: ws <= W_IDLE;
            endcase
        end
    end
endmodule
