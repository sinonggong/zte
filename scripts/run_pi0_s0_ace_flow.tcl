# ACE 10.5.2 flow for the S0 node-array test bitstream: IO-ring generation (three-PLL clock plan), Synplify
# through the override project paper/synth/s0/pi0_s0_synth.prj, prepare / place / route, bitstream.
# Env: PI0_REPO_ROOT, PI0_ACE_IMPL, PI0_S0_SYN_PRJ (the Synplify project), PI0_ACE_SEED (optional).
# Modelled on scripts/run_pi0_int8_ace_full_flow.tcl; the deployed design's constraints are swapped for the S0
# ones per impl, and the legacy ./../ioring files are already gone from the acxprj (10.5.2 accepts one IO-ring
# User Mode 0 hex).
foreach required_env {PI0_REPO_ROOT PI0_ACE_IMPL PI0_S0_SYN_PRJ} {
    if {![info exists ::env($required_env)]} { error "$required_env is not set" }
}
set repo [file normalize $::env(PI0_REPO_ROOT)]
set impl_name $::env(PI0_ACE_IMPL)
set project_name tc_ref_design_top
set project_file "$repo/src/ace/tc_ref_design_top.acxprj"
set synplify_project [file normalize $::env(PI0_S0_SYN_PRJ)]
if {![regexp {^[A-Za-z0-9][A-Za-z0-9_.-]*$} $impl_name]} { error "PI0_ACE_IMPL must be a simple identifier: $impl_name" }
if {![file exists $project_file]} { error "ACE project is missing: $project_file" }
if {![file exists $synplify_project]} { error "Synplify project is missing: $synplify_project" }

restore_project $project_file -activeimpl impl_1 -no_db
if {[lsearch -exact [get_impl_names -project $project_name] $impl_name] < 0} {
    if {[file exists "$repo/src/ace/$impl_name"]} { error "implementation output exists but is not registered: $impl_name" }
    create_impl $impl_name -project $project_name -copy
} else {
    set_active_impl $impl_name -project $project_name
}

# the deployed design's constraints do not apply (its clocks and instances are gone); the S0 set replaces them
foreach f {./../constraints/ace_constraints.sdc ./../constraints/ace_placements.pdc} {
    if {[catch {disable_project_source_file -pnr_constraint -project $project_name -impl $impl_name $f} msg]} {
        puts "INFO: deployed constraint already inactive: $f ($msg)"
    } else { puts "INFO: disabled deployed constraint: $f" }
}
foreach f {./../constraints/synplify_constraints.sdc ./../constraints/synplify_constraints.fdc} {
    if {[catch {disable_project_source_file -syn_constraint -project $project_name -impl $impl_name $f} msg]} {
        puts "INFO: deployed synthesis constraint already inactive: $f ($msg)"
    } else { puts "INFO: disabled deployed synthesis constraint: $f" }
}
set s0_constraints {./../constraints/pi0_s0_ace.sdc ./../constraints/pi0_s0_ace.pdc}
# optional per-node placement regions (paper/synth/gen_node_regions.py), path relative to src/ace
if {[info exists ::env(PI0_S0_REGIONS_PDC)] && $::env(PI0_S0_REGIONS_PDC) ne ""} {
    lappend s0_constraints $::env(PI0_S0_REGIONS_PDC)
    puts "PI0_S0_REGIONS_PDC=$::env(PI0_S0_REGIONS_PDC)"
}
foreach f $s0_constraints {
    if {[catch {add_project_source_files -pnr_constraint -project $project_name $f} msg]} {
        puts "INFO: S0 constraint already registered: $f ($msg)"
    }
    enable_project_source_file -pnr_constraint -project $project_name -impl $impl_name $f
    puts "INFO: S0 constraint active: $f"
}

set_impl_option -project $project_name -impl $impl_name syn_ace_driven_synthesis {0}
set_impl_option -project $project_name -impl $impl_name syn_project_override_path $synplify_project
# ACE writes its own impl synthesis options over the override project ("# ACE Option Overrides"): the deployed
# project carries retiming 1, automatic_compile_point 1 and fanout_limit 200, which silently replaced this flow's
# retiming 0 in every S0 build before 2026-09-17 20:50 (the nodes were timed and simulated without retiming).
# PI0_S0_RETIMING=1 restores ACE's setting for an A/B.
set s0_retiming 0
if {[info exists ::env(PI0_S0_RETIMING)] && $::env(PI0_S0_RETIMING) ne ""} { set s0_retiming $::env(PI0_S0_RETIMING) }
set_impl_option -project $project_name -impl $impl_name syn_retiming $s0_retiming
set_impl_option -project $project_name -impl $impl_name syn_fanout_limit {40}
set_impl_option -project $project_name -impl $impl_name syn_advanced_options [list [list {top_module tc_ref_design_top} [list retiming $s0_retiming] {resource_sharing 0} {write_verilog 1} {automatic_compile_point 0} {frequency 500} {maxfan 40}]]
puts "PI0_S0_RETIMING=$s0_retiming"

if {[info exists ::env(PI0_ACE_SEED)] && $::env(PI0_ACE_SEED) ne ""} {
    set_impl_option -project $project_name -impl $impl_name seed $::env(PI0_ACE_SEED)
    puts "PI0_ACE_SEED=$::env(PI0_ACE_SEED)"
}
# extra implementation options, e.g. PI0_S0_IMPL_OPTS="optimize_hold_graph 0 router_num_hold 2000"
if {[info exists ::env(PI0_S0_IMPL_OPTS)] && $::env(PI0_S0_IMPL_OPTS) ne ""} {
    foreach {opt val} $::env(PI0_S0_IMPL_OPTS) {
        set_impl_option -project $project_name -impl $impl_name $opt $val
        puts "PI0_S0_IMPL_OPT $opt = $val"
    }
}
# PI0_S0_FLOW_MODE=evaluation: the router's "evaluation mode" instead of timing-driven routing.  On the half array the
# timing-driven router needs ~22 min per congestion iteration (5 iterations in 1.5 h, 100+ to go: s1j, 2026-09-18),
# while the standalone chip-fit project -- whose default flow mode is evaluation -- routed the same array in 114
# iterations / 113 min and still met 725 MHz with the per-node regions (+0.009 ns).  Timing is reported either way.
set s0_flow_mode ""
if {[info exists ::env(PI0_S0_FLOW_MODE)] && $::env(PI0_S0_FLOW_MODE) ne ""} {
    set s0_flow_mode $::env(PI0_S0_FLOW_MODE)
    set_project_option -project $project_name flow_mode $s0_flow_mode
    puts "PI0_S0_FLOW_MODE=$s0_flow_mode"
}
# a bring-up bitstream regardless of slack; the slack is reported separately
set_project_option -project $project_name check_final_timing 0
puts "PI0_S0_CHECK_FINAL_TIMING=0"

disable_flow_step run_simulation_rtl
disable_flow_step run_simulation_gate
disable_flow_step run_simulation_routed
disable_flow_step run_simulation_final
enable_flow_step run_synthesis
enable_flow_step report_timing_routed
enable_flow_step report_timing_final
enable_flow_step write_reports_final
enable_flow_step write_bitstream

clear_flow
if {$s0_flow_mode eq "evaluation"} {
    # route and report in evaluation mode; if that mode refuses the bitstream step, write it from the routed
    # database in normal mode
    run -step write_reports_final
    if {[catch {run -step write_bitstream} msg]} {
        puts "PI0_S0_FLOW: write_bitstream in evaluation mode failed ($msg); retrying in normal mode"
        set_project_option -project $project_name flow_mode normal
        run -step write_bitstream
    }
} else {
    run -step write_bitstream
}

set bitstreams [glob -nocomplain "$repo/src/ace/$impl_name/pnr/output/*.hex"]
if {[llength $bitstreams] == 0} { error "no bitstream written for $impl_name" }
puts "PI0_S0_ACE_FLOW_PASS impl=$impl_name bitstreams=$bitstreams"
exit 0
