#!/usr/bin/env bash
# Re-issue a routed node-array impl's bitstream with a different ARRAY clock, without re-routing: the array clock
# lives only in the IO ring (PLL_SW_3, src/acxip/pll_array.acxip), so regenerating the IO ring and rewriting the
# bitstream from the routed database is enough.  Use it to run a bitstream whose array-clock timing missed at 725 MHz
# at the frequency the routed timing supports (Fmax from the report), or for an S0 clock sweep.
#   scripts/run_pi0_s0_reclock_bitstream.sh <impl> <array MHz> <ref> <fb> [vec ODN] [fabric ODN]   # f = 400 * fb / ref / 8
#   e.g. 500 -> ref 1 fb 10 (VCO 4000); 600 -> 1 12; 650 -> 1 13; 700 -> 1 14; 725 -> 2 29; 750 -> 1 15
#   vec ODN (PLL_SW_1, VCO 8000): 24 = 333.33 MHz (default), 28 = 285.7, 32 = 250; fabric stays ODN 32 = 250
# The impl's project must be idle (no build running in this worktree).  Output: a second hex next to the first,
# renamed tc_ref_design_top_<MHz>MHz.hex; src/acxip/pll_array.acxip and src/ace/ioring_design are left at the new clock
# (git checkout -- them to go back).  PI0_BAR1_SIZE=512M also resizes BAR1 (src/acxip/pci_express.acxip).
set -euo pipefail
IMPL=$1; MHZ=$2; REF=$3; FB=$4; VODN=${5:-24}; FODN=${6:-32}   # FODN: fabric ODN on PLL_SW_1 (32 = 250 MHz, 36 = 222.2, 44 = 181.8)
REPO=$(cd "$(dirname "$0")/.." && pwd)
ACE_ROOT=/home/sngong/ACE_10.5.2/Achronix-linux
cd "$REPO"
[ -f "src/ace/$IMPL/pnr/tc_ref_design_top_routed.acxdb" ] || { echo "no routed database for $IMPL"; exit 1; }
[ -e src/ace/tc_ref_design_top.lock ] && { echo "project lock present"; exit 1; }
python3 paper/synth/set_node_pll.py src/acxip/pll_array.acxip --target "$MHZ" --ref "$REF" --fb "$FB" --clk0 i_clk_array:8
VMHZ=$(python3 -c "print(f'{8000/$VODN:.10g}')")
# PI0_BAR1_SIZE=<256M|512M|1G>: also resize BAR1 (the GDDR6 PIO window, NoC 0 upwards) -- an IO-ring-only change
if [ -n "${PI0_BAR1_SIZE:-}" ]; then
    case "$PI0_BAR1_SIZE" in 256M) R=4096;; 512M) R=8192;; 1G) R=16384;; 2G) R=32768;; *) echo "bad PI0_BAR1_SIZE"; exit 1;; esac
    END=$(printf '%011x' $(( R * 65536 - 1 )))
    sed -i -e "s/^pf0.bar1_size=.*/pf0.bar1_size=$PI0_BAR1_SIZE/" -e "s/^pf0.bar1.region0.size=.*/pf0.bar1.region0.size=$R/" \
           -e "s/^pf0.bar1.region0.end_noc_addr=.*/pf0.bar1.region0.end_noc_addr=$END/" src/acxip/pci_express.acxip
    grep -E "^pf0.bar1_size|^pf0.bar1.region0.(size|end_noc_addr)" src/acxip/pci_express.acxip
fi
python3 paper/synth/set_node_pll.py src/acxip/pll_nap.acxip --target "$VMHZ" --ref 1 --fb 20 --clk0 i_clk_vec:$VODN --clk1 i_clk_fabric:$FODN
OUT=src/ace/$IMPL/pnr/output
mkdir -p "$OUT/keep"; cp -n "$OUT"/tc_ref_design_top.hex "$OUT/keep/tc_ref_design_top_first.hex" 2>/dev/null || true
T=$(mktemp --suffix=.tcl)
cat > "$T" <<TCL
restore_project "$REPO/src/ace/tc_ref_design_top.acxprj" -activeimpl impl_1 -no_db
set_active_impl $IMPL -project tc_ref_design_top
set_project_option -project tc_ref_design_top check_final_timing 0
run -step generate_all_ip_design_files
restore_impl "$REPO/src/ace/$IMPL/pnr/tc_ref_design_top_routed.acxdb" -impl $IMPL
disable_flow_step run_simulation_rtl
disable_flow_step run_simulation_gate
disable_flow_step run_simulation_routed
disable_flow_step run_simulation_final
run -step write_bitstream
puts "PI0_RECLOCK_PASS"
exit 0
TCL
LOG=build/s0/reclock_${IMPL}_${MHZ}MHz.log
# the SDC's opt-in constraints (reset false path, capture multicycle) are re-read here: use the build's switches, or the
# regenerated timing reports lose them (s1s 2026-09-22: array "115 MHz" of reset paths instead of 355.6)
if [ -f "build/s0/$IMPL.env" ]; then set -a; . "build/s0/$IMPL.env"; set +a
  echo "constraint switches from build/s0/$IMPL.env: RESET_FP=${PI0_S0_RESET_FP:-} CAP_MCP=${PI0_S0_CAP_MCP:-}"
else echo "WARNING: no build/s0/$IMPL.env -- export PI0_S0_RESET_FP / PI0_S0_CAP_MCP as at build time"; fi
export PI0_S0_RESET_FP PI0_S0_CAP_MCP PI0_S0_RESET_MCP
ACE_INSTALL_DIR=$ACE_ROOT RLM_LICENSE=1710@127.0.0.1 "$ACE_ROOT/ace" -batch -script_file "$T" > "$LOG" 2>&1 || true
rm -f "$T"
grep -E "create_clock -period [0-9.]+ \{i_clk_(array|vec|fabric)\}" src/ace/ioring_design/tc_ref_design_top_ioring.sdc
if grep -q PI0_RECLOCK_PASS "$LOG"; then
    TAG="${MHZ}MHz"; [ "$VODN" != 24 ] && TAG="${MHZ}MHz_v${VODN}"; [ "$FODN" != 32 ] && TAG="${TAG}_f${FODN}"; [ -n "${PI0_BAR1_SIZE:-}" ] && TAG="${TAG}_bar1_${PI0_BAR1_SIZE}"
    mv "$OUT/tc_ref_design_top.hex" "$OUT/tc_ref_design_top_${TAG}.hex"
    echo "RECLOCK OK: $OUT/tc_ref_design_top_${TAG}.hex"
else
    echo "RECLOCK FAILED, see $LOG"; grep -nE "^ERROR" "$LOG" | head -5; exit 1
fi
