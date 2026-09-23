# -------------------------------------------------------------------------
# ACE timing constaint file
# All clock relationships, and IO timing constraints should be set
# in this file
# Clocks are set in the <design_name>_ioring.sdc file
# -------------------------------------------------------------------------

# -------------------------------------------------------------------------
# Example of adding new clocks, (in this example through a GPIO pin)
# -------------------------------------------------------------------------
# Set 500MHz target
# set INCLK_PERIOD 2.0
# create_clock -name gpio_clk [get_ports i_gpio_clk] -period $INCLK_PERIOD

# -------------------------------------------------------------------------
# Example of IO timing constraints
# -------------------------------------------------------------------------
# It is recommended that in_clk and out_clk are virtual clocks, based on the
# IO ports of their respective clocks.  This allows for the clock skew 
# into the device fabric.
# set_input_delay  -clock in_clk  -min  2   [get_ports din\[*\]]
# set_input_delay  -clock in_clk  -max  2.8 [get_ports din\[*\]]
# set_output_delay -clock out_clk -min -0.2 [get_ports dout\[*\]]
# set_output_delay -clock out_clk -max -0.6 [get_ports dout\[*\]]

# -------------------------------------------------------------------------
# Example of defining a generated clock
# -------------------------------------------------------------------------
# create_generated_clock -name clk_gate [ get_pins {i_clkgate/clk_out} ] -source  [get_ports {i_clk} ] -divide_by 1

# -------------------------------------------------------------------------
# Create asynchronous clock groups as more than one clock
# -------------------------------------------------------------------------
#set_clock_groups -asynchronous -group {i_clk} \
#                               -group {gddr6_1_dc0_clk} \
#                               -group {gddr6_2_dc0_clk} \
#                               -group {gddr6_5_dc0_clk} \
#                               -group {gddr6_6_dc0_clk} 

# -------------------------------------------------------------------------
# Example of optionally creating clocks based on the build
# -------------------------------------------------------------------------
# Auto detect if snapshot is in the design
# if { [get_ports tck] != "" } { 
#     set use_snapshot 1
# } else {
#     set use_snapshot 0
# }
# if { $use_snapshot==1 } {
#     create_clock -period 100.0 -name tck   [get_ports tck]
#     set_clock_groups -asynchronous -group {tck}
# }

# For reset over clock, if RESET_OVER_CLOCK_NEG is set in synplify_options.tcl
# then enable the following multicycle constraints
# set_multicycle_path -through [get_nets rstn] -to [get_clocks i_clk] -setup 2
# set_multicycle_path -through [get_nets rstn] -to [get_clocks i_clk] -hold 1
# PI0 PLACER PESSIMISM
#
# The hardware clock and the placer's target are deliberately different.
#
# Build A, impl_pi0_t2_q1r4_98f45c5e2a4f_20260816T190030Z, measured that every
# critical path left in this design is interconnect, not logic: sc_s10 is
# 0.291 ns of logic against 5.971 ns of net over 0 logic levels, sc_s40 is
# 0.053 against 6.058, sc_s20 is 0.392 against 10.799.  On paths like that the
# only thing that decides the frequency is how hard the placer tries to put
# the two endpoints near each other.
#
# It also measured what happens when you stop pushing it.  Moving the target
# from 750 MHz to 250 MHz took i_mlp_clk from 175.6 MHz DOWN to 160.1, and
# took i_mcu_clk from 141.7 to 129.5 - inside vendor IP that nobody had
# touched.  Given a target it believes it can reach, the placer optimises
# wirelength instead of critical-path distance.
#
# So the PLL is set to a rate the fabric can genuinely hold and the placer is
# lied to by exactly this much.  ACE reads this file AFTER
# src/ace/ioring_design/tc_ref_design_top_ioring.sdc (verified in build A's
# log: ioring.sdc, then ace_constraints.sdc, then the corner delay files), so
# these override the 0.042 ns the IO ring generates.  Slack is then reported
# against the pessimistic requirement, which means a positive number is real
# closure with this much margin already subtracted - it needs no
# interpretation and no footnote.
#
# HOW TO READ THE SLACK THIS PRODUCES, because it is easy to get backwards.
#
# These numbers are a PLACEMENT LEVER, not physics.  The real electrical
# uncertainty for i_mlp_clk is 0.0777 ns and the IO ring generates it from the
# PLL's own jitter; 2.5 ns is a fiction that exists only to stop the placer
# deciding it has already won.  Every slack ACE reports is therefore
# pessimistic by (2.5 - 0.0777) = 2.42 ns on i_mlp_clk and by 2.92 ns on
# i_nap_clk.
#
# So a POSITIVE reported slack means the design closes with at least that much
# margin ON TOP of 2.42 ns, and needs no interpretation at all.  A slightly
# NEGATIVE reported slack does NOT mean the design misses - it has to be
# corrected before it means anything.  Build E is the worked example:
#
#   reported          -0.314 ns at 150 MHz, and ACE's Upper Limit column
#                     says 143.2 MHz, which looks like a failure
#   required          6.667 clock + 1.143 network - 2.500 uncertainty
#                     + 0.041 CRPR + 0.012 setup = 5.363
#   arrival           5.671
#   corrected         6.667 + 1.143 - 0.0777 + 0.041 + 0.012 = 7.785 required
#                     7.785 - 5.671 - 0.006 statistical = +2.108 ns
#
# i.e. build E does meet 150 MHz, by 2.1 ns, and every other clock corrects
# positive too.  Do NOT convert a corrected slack back into an "achievable
# frequency" with ACE's 1/(T - slack) formula: uncertainty is a fixed number of
# nanoseconds and does not scale with the period, so that arithmetic inflates.
# The defensible claim is the narrow one - at the programmed frequency, with
# the real uncertainty, the worst path has this much slack.
#
# Signing off cleanly WITHOUT the correction: substitute the real 0.0777 here
# and re-run report_timing_final against the existing routed database.  Note
# that re-running the whole flow instead would place differently and probably
# worse, since low pessimism is what made build A place badly in the first
# place - the placement being signed off is the one these numbers produced.
#
# Raise the PLL and lower these together once a build reports slack to spare.
#
# 2026-08-25: that was tried, and it did not work.  The premise behind the whole
# 2.5 ns lie is that it pushes the PLACER harder.  It was never tested directly,
# and build U tested it: 200 MHz with these lowered to 1.786 / 1.571, chosen so
# that BOTH clocks receive exactly the required times build J routed at 175 MHz.
# It tracked build S -- the same 200 MHz at 2.5 -- point for point, 10,880
# against 10,872 overflows at iteration 84, and build S ends by diverging.
#
# The measured response says why:
#
#   U vs S: uncertainty 2.5 -> 1.786, same period       ->  0.5% HPWL change
#   U vs R: period 5.714 -> 5.000, SAME required time   ->  6.8% HPWL change
#
# The placer responds to the create_clock period, not to the uncertainty.  So
# these values are mostly making every reported slack pessimistic by 2.43 ns and
# forcing whoever reads one to correct it by hand; they are not buying placement
# quality.  Do not spend another build lowering them at a higher PLL.
#
# They are back at 2.5 / 3.0 because that is what builds G, H, J, R, Y and Z were
# all built with, and changing them would silently invalidate every A/B against
# those.  Builds here are deterministic, which is what makes an A/B at the same
# frequency exact -- and also what makes an unnoticed constraint change ruin one.
#
# The experiment still worth running is the cheap control this never got:
# 175 MHz at 1.786 against build R at 2.5.  If the data path is unchanged, the
# lie costs nothing but confusion and should be replaced with the real 0.0698 ns
# so slack can be read directly.
#
# If the PLL moves, these do NOT need to move with it -- that was the belief
# being tested, and it did not survive.  scripts/set_pi0_mlp_clock.py
# deliberately does not touch them.
set_clock_uncertainty -setup 2.500 [get_clocks i_mlp_clk]
set_clock_uncertainty -setup 3.000 [get_clocks i_nap_clk]
# Hold (2026-09-09, board session 08).  Without a hold uncertainty ACE fixes
# hold only to ~0 ps: the p175cdc final report's worst hold slacks were
# +0.004..+0.016 ns on i_nap_clk (pf_mem, the weight-prefetch BRAM) and
# i_mlp_clk (the requant quotient pipeline) -- no margin at all on silicon.
# With the reg->nap crossing fixed the board still lost a few 128-bit bridge
# words per run at random (findings 14c); demand 150 ps of hold margin on
# every internal clock so the router pads those paths.
set_clock_uncertainty -hold 0.150 [get_clocks i_mlp_clk]
set_clock_uncertainty -hold 0.150 [get_clocks i_reg_clk]

create_generated_clock -name i_nap_clk [ get_pins {u_acx_clkdiv/clk_out} ] -source  [get_ports {i_mlp_clk} ] -divide_by 2
# The nap clock only exists from here on: an uncertainty placed above this line is
# silently ignored ("clock i_nap_clk not found"); p150cdch shipped with the nap hold
# unpadded (pf_mem +0.016 ns) for exactly that reason.
set_clock_uncertainty -hold 0.150 [get_clocks i_nap_clk]
set_false_path -from [get_clocks i_reg_clk] -to [get_clocks i_mlp_clk]
set_false_path -from [get_clocks i_reg_clk] -to [get_clocks i_nap_clk]
set_false_path -from [get_clocks i_reg_clk] -to [get_clocks i_mcu_clk]
set_false_path -from [get_clocks i_mlp_clk] -to [get_clocks i_reg_clk]
set_false_path -from [get_clocks i_mlp_clk] -to [get_clocks i_mcu_clk]
set_false_path -from [get_clocks i_mcu_clk] -to [get_clocks i_reg_clk]
set_false_path -from [get_clocks i_mcu_clk] -to [get_clocks i_nap_clk]
set_false_path -from [get_clocks i_mcu_clk] -to [get_clocks i_mlp_clk]
set_false_path -from [get_clocks i_nap_clk] -to [get_clocks i_mcu_clk]
set_false_path -from [get_clocks i_nap_clk] -to [get_clocks i_reg_clk]
#set_multicycle_path -from [get_clocks i_nap_clk] -to [get_clocks i_mlp_clk] -end -setup 2
#set_multicycle_path -from [get_clocks i_nap_clk] -to [get_clocks i_mlp_clk] -end -hold 1
#set_multicycle_path -from [get_pins tc_gen_quad_0__i_tc_quad.tc_gen_noc_*__i_tc_core.rdata_hi_1__1__*/ck] -to [get_clocks i_mlp_clk] -end -setup 2
#set_multicycle_path -from [get_pins tc_gen_quad_0__i_tc_quad.tc_gen_noc_*__i_tc_core.rdata_lo_1__1__*/ck] -to [get_clocks i_mlp_clk] -end -setup 2
#set_multicycle_path -from [get_pins tc_gen_quad_0__i_tc_quad.tc_gen_noc_*__i_tc_core.rdata_vld_1__1_/ck] -to [get_clocks i_mlp_clk] -end -setup 2
#set_multicycle_path -from [get_pins tc_gen_quad_0__i_tc_quad.tc_gen_noc_*__i_tc_core.rdata_sop_1__1_/ck] -to [get_clocks i_mlp_clk] -end -setup 2
#set_multicycle_path -from [get_pins tc_gen_quad_0__i_tc_quad.tc_gen_noc_*__i_tc_core.rdata_eop_1__1_/ck] -to [get_clocks i_mlp_clk] -end -setup 2
#set_multicycle_path -from [get_pins tc_gen_quad_0__i_tc_quad.tc_gen_noc_*__i_tc_core.rdata_acc_zero_1_/ck] -to [get_clocks i_mlp_clk] -end -setup 2
#set_multicycle_path -from [get_pins tc_gen_quad_0__i_tc_quad.tc_gen_noc_*__i_tc_core.rdata_acc_fsum_1_/ck] -to [get_clocks i_mlp_clk] -end -setup 2
#set_multicycle_path -from [get_pins tc_gen_quad_0__i_tc_quad.tc_gen_noc_*__i_tc_core.rdata_wgt_pgsw_1_/ck] -to [get_clocks i_mlp_clk] -end -setup 2
#
#set_multicycle_path -from [get_pins tc_gen_quad_0__i_tc_quad.tc_gen_noc_*__i_tc_core.rdata_hi_1__1__*/ck] -to [get_clocks i_mlp_clk] -end -hold 1
#set_multicycle_path -from [get_pins tc_gen_quad_0__i_tc_quad.tc_gen_noc_*__i_tc_core.rdata_lo_1__1__*/ck] -to [get_clocks i_mlp_clk] -end -hold 1
#set_multicycle_path -from [get_pins tc_gen_quad_0__i_tc_quad.tc_gen_noc_*__i_tc_core.rdata_vld_1__1_/ck] -to [get_clocks i_mlp_clk] -end -hold 1
#set_multicycle_path -from [get_pins tc_gen_quad_0__i_tc_quad.tc_gen_noc_*__i_tc_core.rdata_sop_1__1_/ck] -to [get_clocks i_mlp_clk] -end -hold 1
#set_multicycle_path -from [get_pins tc_gen_quad_0__i_tc_quad.tc_gen_noc_*__i_tc_core.rdata_eop_1__1_/ck] -to [get_clocks i_mlp_clk] -end -hold 1
#set_multicycle_path -from [get_pins tc_gen_quad_0__i_tc_quad.tc_gen_noc_*__i_tc_core.rdata_acc_zero_1_/ck] -to [get_clocks i_mlp_clk] -end -hold 1
#set_multicycle_path -from [get_pins tc_gen_quad_0__i_tc_quad.tc_gen_noc_*__i_tc_core.rdata_acc_fsum_1_/ck] -to [get_clocks i_mlp_clk] -end -hold 1
#set_multicycle_path -from [get_pins tc_gen_quad_0__i_tc_quad.tc_gen_noc_*__i_tc_core.rdata_wgt_pgsw_1_/ck] -to [get_clocks i_mlp_clk] -end -hold 1

#set_multicycle_path -from [get_clocks i_mlp_clk] -to [get_clocks i_nap_clk] -start -setup 2
#set_multicycle_path -from [get_clocks i_mlp_clk] -to [get_clocks i_nap_clk] -start -hold 1
#set_false_path -from [get_cells i_top_ctrl.tc_clamshell_array] 
# ACE aborts the entire flow on the first "Empty -from argument specified",
# and several of these patterns legitimately match nothing: a register the
# mapper optimised away, or a quad that does not exist when NUM_TC_QUAD is
# less than 4.  That turned run_prepare into a hard failure 91 seconds in,
# after a 24-hour synthesis.  Apply each exception only when it matches.
proc pi0_false_path_from {pattern} {
    if {[catch {get_cells $pattern} cells]} {
        puts "INFO: false-path pattern could not be resolved, skipped: $pattern"
        return
    }
    if {[llength $cells] == 0} {
        puts "INFO: false-path pattern matched no cell, skipped: $pattern"
        return
    }
    set_false_path -from $cells
}

foreach pi0_fp_pattern {
    i_top_ctrl.tc_wgt_nonident_array
    i_top_ctrl.tc_rotate_ident_array
    i_top_ctrl.tc_init_nowr_array
    i_top_ctrl.tc_full_addr_array
    i_top_ctrl.tc_clamshell_array
    i_top_ctrl.tc_free_run_array
    i_top_ctrl.tc_chmap_sw_array_*
    i_top_ctrl.tc_chmap_nw_array_*
    i_top_ctrl.tc_chmap_ne_array_*
    i_top_ctrl.tc_chmap_se_array_*
    i_top_ctrl.tc_run_iter_array_*
    i_top_ctrl.tc_ost_burst_array_*
    tc_gen_quad_*__i_tc_quad.tc_gen_noc_*__i_nap_mem_wdata_*
    tc_gen_quad_*__i_tc_quad.tc_gen_noc_*__i_nap_mem_addr_*
} {
    pi0_false_path_from $pi0_fp_pattern
}
#set_property fanout_limit 1 [get_nets tc_gen_quad_0__i_tc_quad.mlp_rstn_array*]
#set_property fanout_limit 1 [get_nets tc_gen_quad_1__i_tc_quad.mlp_rstn_array*]
#set_property fanout_limit 1 [get_nets tc_gen_quad_2__i_tc_quad.mlp_rstn_array*]
#set_property fanout_limit 1 [get_nets tc_gen_quad_3__i_tc_quad.mlp_rstn_array*]




