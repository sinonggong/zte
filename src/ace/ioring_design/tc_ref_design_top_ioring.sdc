#######################################
# ACE GENERATED SDC FILE
# Generated on: 2026.09.16 at 07:52:11 PDT
# By: ACE 10.5.2
# From project: tc_ref_design_top
#######################################
# IO Ring Boundary SDC File
#######################################

# Boundary clocks for gddr6_0

# Boundary clocks for gddr6_1

# Boundary clocks for gddr6_2

# Boundary clocks for gddr6_3

# Boundary clocks for gddr6_4

# Boundary clocks for gddr6_5

# Boundary clocks for gddr6_6

# Boundary clocks for gddr6_7

# Boundary clocks for noc

# Boundary clocks for pci_express

# Boundary clocks for pll
create_clock -period 15.5 {i_reg_clk}
# Frequency = 64.51612903225806 MHz
set_clock_uncertainty -setup 0.10770329614269007 [get_clocks {i_reg_clk}]

create_clock -period 10.0 {i_mcu_clk}
# Frequency = 100.0 MHz
set_clock_uncertainty -setup 0.10770329614269007 [get_clocks {i_mcu_clk}]


# Boundary clocks for pll_nap
create_clock -period 6.666666666666667 {i_mlp_clk}
# Frequency = 150.0 MHz
set_clock_uncertainty -setup 0.07774602526460402 [get_clocks {i_mlp_clk}]


# Boundary clocks for pll_pcie

# Boundary clocks for vp_clkio_ne

# Boundary clocks for vp_clkio_se

# Boundary clocks for vp_clkio_sw

# Boundary clocks for vp_pll_se_2

# Boundary clocks for vp_pll_sw_2

# Virtual clocks for IO Ring IPs

# See "/home/sngong/projects/pi0_achronix_ace1052/src/ace/ioring_design/tc_ref_design_top_ioring_clock_groups.sdc" to use as a template to define your clock groups.

######################################
# End IO Ring Boundary SDC File
######################################
