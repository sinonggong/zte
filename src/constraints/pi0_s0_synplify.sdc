# S0 node-array bitstream: synthesis clocks (the IO ring's generated sdc repeats the PLL clocks; keep in step with
# docs/PI0_CLOCK_PLAN_20260917.md)
create_clock -name i_clk_array  [get_ports i_clk_array]  -period 1.379
create_clock -name i_clk_vec    [get_ports i_clk_vec]    -period 3.0
create_clock -name i_clk_fabric [get_ports i_clk_fabric] -period 4.0
create_clock -name i_mcu_clk    [get_ports i_mcu_clk]    -period 10.0
create_clock -name i_reg_clk    [get_ports i_reg_clk]    -period 15.5
set_clock_groups -asynchronous -group {i_clk_array} -group {i_clk_vec} -group {i_clk_fabric} -group {i_mcu_clk} -group {i_reg_clk}
