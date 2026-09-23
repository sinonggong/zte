//////////////////////////////////////
// ACE GENERATED VERILOG INCLUDE FILE
// Generated on: 2026.09.16 at 07:52:11 PDT
// By: ACE 10.5.2
// From project: tc_ref_design_top
//////////////////////////////////////
// IO Ring Simulation Defines Include File
// 
// This file must be included in your compilation
// prior to the Device Simulation Model (DSM) being compiled
//////////////////////////////////////


//////////////////////////////////////
// Switch to set SystemVerilog Direct Connect
// Interfaces in DSM to Monitor-Only Mode.
// This is required when using the IO Designer
// generated user design port bindings file.
//////////////////////////////////////
  `define ACX_ENABLE_DCI_MONITOR_MODE = 1;

//////////////////////////////////////
// Clock Selects in each IP
//////////////////////////////////////

//////////////////////////////////////
// GDDR6_0:
  // Ref clock
  `define ACX_GDDR6_0_REF_CLK_SEL = 19;
  // NoC clock
  `define ACX_GDDR6_0_NOC_CLK_SEL = 19;

//////////////////////////////////////
// GDDR6_1:
  // Ref clock
  `define ACX_GDDR6_1_REF_CLK_SEL = 19;
  // NoC clock
  `define ACX_GDDR6_1_NOC_CLK_SEL = 19;

//////////////////////////////////////
// GDDR6_2:
  // Ref clock
  `define ACX_GDDR6_2_REF_CLK_SEL = 19;
  // NoC clock
  `define ACX_GDDR6_2_NOC_CLK_SEL = 19;

//////////////////////////////////////
// GDDR6_3:
  // Ref clock
  `define ACX_GDDR6_3_REF_CLK_SEL = 19;
  // NoC clock
  `define ACX_GDDR6_3_NOC_CLK_SEL = 19;

//////////////////////////////////////
// GDDR6_4:
  // Ref clock
  `define ACX_GDDR6_4_REF_CLK_SEL = 0;
  // NoC clock
  `define ACX_GDDR6_4_NOC_CLK_SEL = 0;

//////////////////////////////////////
// GDDR6_5:
  // Ref clock
  `define ACX_GDDR6_5_REF_CLK_SEL = 0;
  // NoC clock
  `define ACX_GDDR6_5_NOC_CLK_SEL = 0;

//////////////////////////////////////
// GDDR6_6:
  // Ref clock
  `define ACX_GDDR6_6_REF_CLK_SEL = 0;
  // NoC clock
  `define ACX_GDDR6_6_NOC_CLK_SEL = 0;

//////////////////////////////////////
// GDDR6_7:
  // Ref clock
  `define ACX_GDDR6_7_REF_CLK_SEL = 0;
  // NoC clock
  `define ACX_GDDR6_7_NOC_CLK_SEL = 0;

//////////////////////////////////////
// PCIE_1:
  // AUX clock
  `define ACX_PCIE_1_AUX_CLK_SEL = 16;
  // Master clock
  `define ACX_PCIE_1_MASTER_CLK_SEL = 16;
  // Slave clock
  `define ACX_PCIE_1_SLAVE_CLK_SEL = 16;

//////////////////////////////////////
// NoC:
  // NoC Ref clock
  `define ENOC_CLK_SEL = 15;

//////////////////////////////////////
// Reset Selects in each IP
//////////////////////////////////////

//////////////////////////////////////
// GDDR6_0:
  `define ACX_GDDR6_0_RST_SEL = 16;

//////////////////////////////////////
// GDDR6_1:
  `define ACX_GDDR6_1_RST_SEL = 17;

//////////////////////////////////////
// GDDR6_2:
  `define ACX_GDDR6_2_RST_SEL = 18;

//////////////////////////////////////
// GDDR6_3:
  `define ACX_GDDR6_3_RST_SEL = 19;

//////////////////////////////////////
// GDDR6_4:
  `define ACX_GDDR6_4_RST_SEL = 20;

//////////////////////////////////////
// GDDR6_5:
  `define ACX_GDDR6_5_RST_SEL = 21;

//////////////////////////////////////
// GDDR6_6:
  `define ACX_GDDR6_6_RST_SEL = 22;

//////////////////////////////////////
// GDDR6_7:
  `define ACX_GDDR6_7_RST_SEL = 23;

//////////////////////////////////////
// PCIE_1:
  `define ACX_PCIE_1_RST_SEL = 15;

//////////////////////////////////////
// CLKIO_NE:
  `define ACX_CLKIO_NE_RST_SEL = 3;

//////////////////////////////////////
// CLKIO_SE:
  `define ACX_CLKIO_SE_RST_SEL = 1;

//////////////////////////////////////
// CLKIO_SW:
  `define ACX_CLKIO_SW_RST_SEL = 0;

//////////////////////////////////////
// End IO Ring Simulation Defines Include File
//////////////////////////////////////
