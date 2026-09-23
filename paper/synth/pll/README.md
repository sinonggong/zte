# Node-array PLL files (D1.1, 2026-09-17)

As ACE 10.5.2 re-solved and saved them in dry run `t3_725` (`docs/PI0_CLOCK_PLAN_20260917.md`):

- `pll_nap_vec333_fab250.acxip`: PLL_SW_1, VCO 8000 (ref 1, fb 20); clkout0 `i_clk_vec` ODN 24 = 333.333 MHz,
  clkout1 `i_clk_fabric` ODN 32 = 250 MHz. Replaces `src/acxip/pll_nap.acxip` (same file name, so the acxprj is unchanged).
- `pll_array_725.acxip`: PLL_SW_3, VCO 5800 (ref 2, fb 29); clkout0 `i_clk_array` ODN 8 = 725 MHz. New file
  `src/acxip/pll_array.acxip` + `add_project_source_files -ip -project tc_ref_design_top {{./../acxip/pll_array.acxip}}`.
- `port_list_3pll.svh`: the top-level ports the regenerated IO ring expects.

`src/acxip/pll.acxip` (PLL_SW_0: reg / NoC 200 / mcu) stays as deployed.
