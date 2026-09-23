#!/usr/bin/env bash
# Dry-run the node-array PLL plan through ACE 10.5.2's IO-ring generation, on a scratch copy of the
# 10.5.2 worktree's src/ (ACE rewrites the acxip files and ioring_design/ in place).  ~30 s per option.
#
# Usage: run_node_pll_dryrun.sh <out_dir> <src_dir_of_10.5.2_worktree> <tag|baseline> [set_node_pll.py args]
# Env:   ARRAY_PLL=<placement>:<target>:<ref>:<fb>  adds a second node PLL (src/acxip/pll_array.acxip, clkout0 =
#        i_clk_array at ODN 8, so VCO = 8 x target) on the same reference and registers it in the acxprj.
# Reads afterwards: <out>/ace.log, <out>/src/acxip/pll_nap.acxip (as ACE re-solved it), the generated
# ioring_design/*.sdc create_clock lines and the PLL_SW_1 .txt, and diffs of every acxip against the input.
set -euo pipefail
OUT=$1; SRC=$2; OPT=$3; shift 3
# the login shell exports ACE_INSTALL_DIR=10.3.1, which cannot read 10.5.2 acxip files: use a name it cannot leak into
export ACE_INSTALL_DIR=${NODE_ACE_INSTALL_DIR:-/home/sngong/ACE_10.5.2/Achronix-linux}
export PATH=$ACE_INSTALL_DIR:$PATH
HERE=$(cd "$(dirname "$0")" && pwd)
rm -rf "$OUT"; mkdir -p "$OUT/src"
for d in acxip ioring rtl constraints; do [ -d "$SRC/$d" ] && cp -r "$SRC/$d" "$OUT/src/$d"; done
mkdir -p "$OUT/src/ace"; cp "$SRC/ace/tc_ref_design_top.acxprj" "$OUT/src/ace/"
cp -r "$SRC/ace/ioring_design" "$OUT/src/ace/ioring_design"
cp -r "$OUT/src/acxip" "$OUT/acxip_in"; cp -r "$OUT/src/ace/ioring_design" "$OUT/ioring_in"
if [ "$OPT" != baseline ]; then
    python3 "$HERE/set_node_pll.py" "$OUT/src/acxip/pll_nap.acxip" "$@" | tee "$OUT/plan.txt"
    cp "$OUT/src/acxip/pll_nap.acxip" "$OUT/pll_nap_requested.acxip"
fi
if [ -n "${ARRAY_PLL:-}" ]; then
    IFS=: read -r APLACE ATARGET AREF AFB <<< "$ARRAY_PLL"
    AF=$OUT/src/acxip/pll_array.acxip
    cp "$OUT/acxip_in/pll_nap.acxip" "$AF"
    sed -i -e "s/^placement=.*/placement=$APLACE/" -e "s/^pll_lock.port_name=.*/pll_lock.port_name=pll_array_lock/" \
           -e "s/^reset_from_fabric.port_name=.*/reset_from_fabric.port_name=pll_array_user_rstn/" \
           -e "s/^clkout\([1-3]\).clkout.port_name=.*/clkout\1.clkout.port_name=pll_array_clkout\1/" "$AF"
    python3 "$HERE/set_node_pll.py" "$AF" --target "$ATARGET" --ref "$AREF" --fb "$AFB" --clk0 i_clk_array:8 | tee -a "$OUT/plan.txt"
    cp "$AF" "$OUT/pll_array_requested.acxip"
    echo 'add_project_source_files -ip -project tc_ref_design_top {{./../acxip/pll_array.acxip}}' >> "$OUT/src/ace/tc_ref_design_top.acxprj"
fi
cat > "$OUT/dry.tcl" <<V
load_project "$OUT/src/ace/tc_ref_design_top.acxprj"
run -step generate_all_ip_design_files
puts "PLL_DRYRUN_DONE"
V
cd "$OUT"
( time timeout 900 ace -batch -script_file "$OUT/dry.tcl" ) > "$OUT/ace.log" 2>&1 || true
if grep -q PLL_DRYRUN_DONE "$OUT/ace.log"; then echo "DRYRUN OK  $OPT"; else echo "DRYRUN FAILED  $OPT"; grep -nE "ERROR|Error|Invalid" "$OUT/ace.log" | head -8; fi
for f in pll_nap pll_array; do [ -f "$OUT/src/acxip/$f.acxip" ] || continue; echo "--- $f as ACE re-solved it:"
grep -E "^(placement|reference_divider|feedback_divider|float_target|output_port_count|clkout[0-3]\.(clkout\.port_name|int_ODN_output_divider|is_output_connected_to_core))" "$OUT/src/acxip/$f.acxip"; done
echo "--- pll.acxip (PLL_SW_0) diff vs input:"; diff "$OUT/acxip_in/pll.acxip" "$OUT/src/acxip/pll.acxip" | grep -vE "^(---|[0-9,]+c[0-9,]+)$|generated on" || true
echo "--- other acxip changed:"; for f in "$OUT"/acxip_in/*.acxip; do b=$(basename "$f"); cmp -s <(grep -v "^#" "$f") <(grep -v "^#" "$OUT/src/acxip/$b") || echo "  $b"; done
echo "--- generated clocks:"; grep -E "create_clock" "$OUT/src/ace/ioring_design/tc_ref_design_top_ioring.sdc" | grep -vE "v_acx" || true
grep -E "trunk" "$OUT/src/ace/ioring_design/tc_ref_design_top_ioring.pdc" | grep -E "clk" || true
