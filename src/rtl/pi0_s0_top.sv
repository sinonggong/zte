// S0 test bitstream for the pi0 node array (docs/PI0_E2E_HARDWARE_DELIVERY_20260917.md §7): the deployed IO ring
// (PCIe endpoint, 8 GDDR6 controllers, Device Manager) around pi0_chip_top with a handful of nodes, on the
// three-PLL clock plan of docs/PI0_CLOCK_PLAN_20260917.md:
//
//   i_clk_array   725 MHz   PLL_SW_3 (pll_array.acxip)      MLP72 chains
//   i_clk_vec     333 MHz   PLL_SW_1 (pll_nap.acxip) out0    vector nodes and their NAPs
//   i_clk_fabric  250 MHz   PLL_SW_1 out1                    chain NAPs, host window
//   i_mcu_clk     100 MHz   PLL_SW_0                         Device Manager (GDDR6 training, PCIe PERST / DBI)
//   i_reg_clk     64.5 MHz  PLL_SW_0                         unused here, kept so PLL_SW_0 stays as deployed
//
// The module keeps the deployed top's name and every IO-ring port (src/ace/ioring_design/
// tc_ref_design_top_user_design_port_list.svh), so the project, BAR map (BAR0 -> NOC[3][4] = the host window,
// BAR3 -> Device Manager) and board scripts are unchanged.  It drops the tensor-core quads, top_ctrl and i_mlp_clk.
//
// Resets: a power-on counter on the fabric clock, plus the host's soft reset (write beat N_NODE bit 1 of the
// register window), stretched and synchronised into each node clock by the vendor reset processor.
`include "7t_interfaces.svh"
`include "speedster7t/common/speedster7t_jtap.sv"
`default_nettype wire
`ifdef SYNTHESIS
`include "speedster7t/macros/ACX_DEVICE_MANAGER.svp"
`else
`include "ACX_DEVICE_MANAGER.svp"
`endif

module tc_ref_design_top #(
    parameter F_GELU  = "vu_tbl_gelu.mem",
    parameter F_SIGM  = "vu_tbl_sigm.mem",
    parameter F_EXP   = "vu_tbl_exp.mem",
    parameter F_RSQRT = "vu_tbl_rsqrt.mem",
    parameter F_QUANT = "vu_tbl_quant.mem",
    parameter integer N_CHAIN  = 1,
    parameter integer N_PV     = 0,
    parameter integer N_VNODE  = 1,
    parameter integer N_LANE   = 2,
    parameter integer N_LD     = 2,
    parameter integer SLOT_BITS = 12,
    parameter integer N_STAGE  = 16,
    parameter integer N_DEEP   = 0,
    parameter integer DEEP_STAGE = 32,
    parameter integer NAPS_PER_NODE = 2,
    parameter integer MAX_OUT  = 8,
    parameter integer STRIPE   = 0,                     // GDDR6 channel striping (pi0_chip_top.sv): 0 off, 12 = 4 KB
    parameter [23:0]  ID_WORD  = 24'h533000,          // "S0" + build number in the low byte
    parameter  [3:0]  MCU_NAP_ROW = 4,                // Device Manager NAP = NOC[8][4] (BAR3 DBI gateway)
    parameter  [3:0]  MCU_NAP_COL = 8,
    parameter  [3:0]  CTRL_NAP_ROW = 4,               // host window NAP = NOC[3][4] (BAR0)
    parameter  [3:0]  CTRL_NAP_COL = 3,
    parameter  [3:0]  BRIDGE_NAP_ROW = 5,             // host -> GDDR6 bridge NAP = NOC[3][5] (BAR1, 256 MB)
    parameter  [3:0]  BRIDGE_NAP_COL = 3
) (
    input  wire                         i_reg_clk,
    input  wire                         i_mcu_clk,
    input  wire                         i_clk_array,
    input  wire                         i_clk_fabric,
    input  wire                         i_clk_vec,
    input  wire t_JTAG_INPUT            i_jtag_in,
    output wire t_JTAG_OUTPUT           o_jtag_out,
    input  wire                         pcie_perst_l,
    input  wire [5:0]                   pci_express_status_ltssm_state,
    input  wire [3:0]                   pci_express_status_flr_pf_active,
    input  wire                         pci_express_status_flr_vf_active,
    input  wire                         pll_pcie_lock
);

    // ---------------------------------------------------------------- Device Manager, as deployed
    wire adm_start;
    ACX_SYNCHRONIZER x_sync_adm_start (
        .clk  (i_mcu_clk),
        .din  (1'b1),
        .rstn (pll_pcie_lock),
        .dout (adm_start)
    );

    ACX_DEVICE_MANAGER #(
      .NAP_ROW(MCU_NAP_ROW),
      .NAP_COLUMN(MCU_NAP_COL),
      .ENABLE_PCIE_1_PERSTN      (1'b1),
      .ENABLE_PCIE_1_HOT_RSTN    (1'b1),
      .ENABLE_PCIE_1_DBI_GATEWAY (1'b1),
      .ENABLE_PCIE_RECONFIG_FPGA (1'b1)
    ) u_acx_ip_monitor (
        .i_clk                 (i_mcu_clk),
        .i_start               (adm_start),
        .i_pcie_0_perstn       (1'b1),
        .i_pcie_1_perstn       (pcie_perst_l),
        .i_pcie_0_ltssm_state  (6'h3f),
        .i_pcie_1_ltssm_state  (pci_express_status_ltssm_state),
        .o_pcie_0_reconfig_fpga_n (),
        .o_pcie_1_reconfig_fpga_n (),
        .o_status              (),
        .o_serdes_status       (),
        .i_jtag_in             (i_jtag_in),
        .o_jtag_out            (o_jtag_out),
        .o_jtap_bus            (),
        .i_tdo_bus             (1'b0)
    );

    // ---------------------------------------------------------------- resets
    // Power-on: 2^12 fabric cycles after configuration (the deployed top_ctrl does the same with reset_cnt[10]).
    // Soft: the host's request is stretched to 256 fabric cycles so every domain's synchroniser sees it.
    // Output pipelines (RESET_OVER_CLOCK 0), not the clock-network route: at 725 MHz the single reset-over-clock
    // synchroniser missed the MLP72 reset pins by 0.52 ns (S0 build s0o1, 2026-09-17).
    reg  [12:0] por_cnt = 13'd0;
    reg  [8:0]  soft_cnt = 9'd0;
    wire        soft_rst;
    always @(posedge i_clk_fabric) begin
        if (!por_cnt[12]) por_cnt <= por_cnt + 13'd1;
        if (soft_rst)          soft_cnt <= 9'd256;
        else if (soft_cnt != 0) soft_cnt <= soft_cnt - 9'd1;
    end
    wire rst_src_n = por_cnt[12] & (soft_cnt == 9'd0);

    wire rstn_fabric, rstn_array, rstn_vec;
    reset_processor_v2 #(
        .NUM_INPUT_RESETS(1), .IN_RST_PIPE_LENGTH(5), .SYNC_INPUT_RESETS(1), .OUT_RST_PIPE_LENGTH(4), .RESET_OVER_CLOCK(0)
    ) u_rst_fabric (.i_rstn_array(rst_src_n), .i_clk(i_clk_fabric), .o_rstn(rstn_fabric));
    reset_processor_v2 #(
        .NUM_INPUT_RESETS(1), .IN_RST_PIPE_LENGTH(5), .SYNC_INPUT_RESETS(1), .OUT_RST_PIPE_LENGTH(4), .RESET_OVER_CLOCK(0)
    ) u_rst_array (.i_rstn_array(rst_src_n), .i_clk(i_clk_array), .o_rstn(rstn_array));
    reset_processor_v2 #(
        .NUM_INPUT_RESETS(1), .IN_RST_PIPE_LENGTH(5), .SYNC_INPUT_RESETS(1), .OUT_RST_PIPE_LENGTH(4), .RESET_OVER_CLOCK(0)
    ) u_rst_vec (.i_rstn_array(rst_src_n), .i_clk(i_clk_vec), .o_rstn(rstn_vec));

    // ---------------------------------------------------------------- the array
    wire all_halt, any_error;
    pi0_chip_top #(
        .F_GELU(F_GELU), .F_SIGM(F_SIGM), .F_EXP(F_EXP), .F_RSQRT(F_RSQRT), .F_QUANT(F_QUANT),
        .N_CHAIN(N_CHAIN), .N_PV(N_PV), .N_VNODE(N_VNODE), .N_LANE(N_LANE), .N_STAGE(N_STAGE),
        .N_DEEP(N_DEEP), .DEEP_STAGE(DEEP_STAGE), .NAPS_PER_NODE(NAPS_PER_NODE),
        .SLOT_BITS(SLOT_BITS), .N_LD(N_LD), .MAX_OUT(MAX_OUT),
        .CTRL_NAP_COL(CTRL_NAP_COL), .CTRL_NAP_ROW(CTRL_NAP_ROW), .ID_WORD(ID_WORD),
        .HOST_BRIDGE(1), .BRIDGE_NAP_COL(BRIDGE_NAP_COL), .BRIDGE_NAP_ROW(BRIDGE_NAP_ROW), .STRIPE(STRIPE)
    ) u_array (
        .i_clk_array(i_clk_array), .i_clk_fabric(i_clk_fabric), .i_clk_vec(i_clk_vec),
        .i_rstn_array(rstn_array), .i_rstn_fabric(rstn_fabric), .i_rstn_vec(rstn_vec),
        .o_all_halt(all_halt), .o_any_error(any_error), .o_soft_rst(soft_rst));

endmodule : tc_ref_design_top
