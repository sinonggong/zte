//////////////////////////////////////
// ACE GENERATED VERILOG INCLUDE FILE
// Generated on: 2026.09.16 at 07:52:11 PDT
// By: ACE 10.5.2
// From project: tc_ref_design_top
//////////////////////////////////////
// User Design Port Binding Include File
//////////////////////////////////////

//////////////////////////////////////
// User Design Ports
//////////////////////////////////////
    // Ports for gddr6_0
    // Ports for gddr6_1
    // Ports for gddr6_2
    // Ports for gddr6_3
    // Ports for gddr6_4
    // Ports for gddr6_5
    // Ports for gddr6_6
    // Ports for gddr6_7
    // Ports for noc
    // Ports for pci_express
    // Status
`ACX_BIND_USER_DESIGN_PORT(pci_express_status_flr_pf_active[0], i_user_10_09_mlp_00[10])
`ACX_BIND_USER_DESIGN_PORT(pci_express_status_flr_pf_active[1], i_user_10_09_mlp_00[11])
`ACX_BIND_USER_DESIGN_PORT(pci_express_status_flr_pf_active[2], i_user_10_09_mlp_00[12])
`ACX_BIND_USER_DESIGN_PORT(pci_express_status_flr_pf_active[3], i_user_10_09_mlp_00[13])
`ACX_BIND_USER_DESIGN_PORT(pci_express_status_flr_vf_active, i_user_10_09_mlp_00[9])
`ACX_BIND_USER_DESIGN_PORT(pci_express_status_ltssm_state[0], i_user_10_09_lut_00[1])
`ACX_BIND_USER_DESIGN_PORT(pci_express_status_ltssm_state[1], i_user_10_09_lut_00[2])
`ACX_BIND_USER_DESIGN_PORT(pci_express_status_ltssm_state[2], i_user_10_09_lut_00[3])
`ACX_BIND_USER_DESIGN_PORT(pci_express_status_ltssm_state[3], i_user_10_09_lut_00[4])
`ACX_BIND_USER_DESIGN_PORT(pci_express_status_ltssm_state[4], i_user_10_09_lut_00[5])
`ACX_BIND_USER_DESIGN_PORT(pci_express_status_ltssm_state[5], i_user_10_09_lut_00[6])
    // Ports for pll
`ifdef ACX_CLK_SW_FULL
`ACX_BIND_USER_DESIGN_PORT(i_mcu_clk, i_user_06_00_trunk_00[1])
`ACX_BIND_USER_DESIGN_PORT(i_reg_clk, i_user_06_00_trunk_00[0])
`endif
    // Ports for pll_nap
`ifdef ACX_CLK_SW_FULL
`ACX_BIND_USER_DESIGN_PORT(i_mlp_clk, i_user_06_00_trunk_00[2])
`endif
    // Ports for pll_pcie
`ifdef ACX_CLK_NE_FULL
`ACX_BIND_USER_DESIGN_PORT(pll_pcie_lock, i_user_12_08_lut_17[2])
`endif
    // Ports for vp_clkio_ne
`ACX_BIND_USER_DESIGN_PORT(pcie_perst_l, i_user_06_09_trunk_00[127])
    // Ports for vp_clkio_se
    // Ports for vp_clkio_sw
    // Ports for vp_pll_se_2
`ifdef ACX_CLK_SE_FULL
`endif
    // Ports for vp_pll_sw_2
`ifdef ACX_CLK_SW_FULL
`endif

//////////////////////////////////////
// End IO Ring User Design Port Binding Include File
//////////////////////////////////////
