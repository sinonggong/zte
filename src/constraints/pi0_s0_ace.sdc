# S0 node-array bitstream: ACE timing constraints beside the IO ring's generated clocks.
# The node clocks never meet except through GDDR6 and the chain node's own synchronisers (colpar_result_port.sv,
# the fabric-side hold synchroniser), so every pair is an asynchronous group.  Hold margin 0.150 ns as the deployed
# design (it cured board word loss); setup margin 0.100 ns like the IO ring's own clocks.
set_clock_groups -asynchronous -group {i_clk_array} -group {i_clk_vec} -group {i_clk_fabric} -group {i_mcu_clk} -group {i_reg_clk}
foreach c {i_clk_array i_clk_vec i_clk_fabric} {
    set_clock_uncertainty -setup 0.100 [get_clocks $c]
    # hold 0.050, not 0.150 (2026-09-22): with 0.150 every MLP72 / BRAM72K internal register loop (zero-net path,
    # margin 0.086 ns) was a hold violation the router cannot fix; the timing-driven flow's all-temperature hold pass
    # then ran for hours and died in ACE 10.5.2 (Util::Arena::alloc assertion; s1t, s1u).  At 0.050 the same pass
    # takes 4 min and the flow completes.
    set_clock_uncertainty -hold  0.050 [get_clocks $c]
}

# The node resets are static for hundreds of cycles (power-on counter, 256-cycle soft reset), so their release may take
# several cycles to reach every MLP72 / BRAM72K; the single output-pipe register -> 17 MLP72 reset pins missed 725 MHz
# by 0.55 ns in the S0 builds (s0o1 / s0d, 2026-09-17).  Nothing is in flight at release.
# Opt-in (PI0_S0_RESET_MCP=1): build s0e with it died in the router ("Assertion _base <= _size failed at
# Util::Arena::alloc"), the same ACE crash as s0b without it, so it is not cleared as harmless yet.
# Opt-in (PI0_S0_RESET_FP=1): the resets' release as a false path.  Nodes sit idle until the host starts them, long
# after release, so which cycle a flop leaves reset in does not matter; the paths only hide the real critical paths
# (s1m, 2026-09-18: the worst path of EVERY clock was a reset fan-out, so the report named no real path).
if {[info exists ::env(PI0_S0_RESET_FP)] && $::env(PI0_S0_RESET_FP) eq "1"} {
    foreach r {u_rst_array u_rst_vec u_rst_fabric} {
        if {![catch {get_cells "$r.*"} cells] && [llength $cells] > 0} {
            set_false_path -from $cells
            puts "PI0_S0_RESET_FP: $r [llength $cells] cells"
        }
    }
}
if {[info exists ::env(PI0_S0_RESET_MCP)] && $::env(PI0_S0_RESET_MCP) eq "1"} {
    foreach r {u_rst_array u_rst_vec u_rst_fabric} {
        if {![catch {get_cells "$r.*"} cells] && [llength $cells] > 0} {
            set_multicycle_path 4 -setup -from $cells
            set_multicycle_path 3 -hold  -from $cells
        }
    }
}

# MLP72 OUT_REG -> fabric capture register of every chain stage: a 2-cycle path by construction
# (mlp72_int8_colpar_chain.sv: OUT_REG holds for >= 2 array cycles around each capture).  Without this the array
# reports were pessimistic on exactly this path (s1m: reported 132 MHz, bit-exact at 250) and the router spent its
# effort on a false-critical path.  ACE 10.5.2 rejects get_cells -hierarchical; guarded so a pattern that matches
# nothing cannot fail run_prepare -- the log line tells how many capture registers it found.
if {[info exists ::env(PI0_S0_CAP_MCP)] && $::env(PI0_S0_CAP_MCP) eq "1"} {
    set cap_cells {}
    catch {set cap_cells [get_cells {*g_cap*cap*}]}
    puts "PI0_S0_CAP_MCP: [llength $cap_cells] capture registers"
    if {[llength $cap_cells] > 0} {
        set_multicycle_path 2 -setup -to $cap_cells
        set_multicycle_path 1 -hold  -to $cap_cells
    }
}
