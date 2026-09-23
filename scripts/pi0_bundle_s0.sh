#!/usr/bin/env bash
# Copy an S0 impl's bitstream into ~/pi0_board_bundle/bitstream/<name>/ with INFO.txt and SHA256SUMS.
#   scripts/pi0_bundle_s0.sh <impl dir name> <bundle name> [note...]
set -euo pipefail
IMPL=$1; NAME=$2; shift 2
REPO=$(cd "$(dirname "$0")/.." && pwd)
SRC="$REPO/src/ace/$IMPL/pnr/output"
DST="${BUNDLE:-$HOME/pi0_board_bundle}/bitstream/$NAME"
# the impl's tc_ref_design_top.hex when present (a reclock leaves tc_ref_design_top_<MHz>....hex next to it, and
# locale collation sorts those FIRST: "ls | head -1" once bundled a 250 MHz file as s1m_400, 2026-09-18)
if [ -f "$SRC/tc_ref_design_top.hex" ]; then HEX="$SRC/tc_ref_design_top.hex"; else HEX=$(LC_ALL=C ls "$SRC"/*.hex | head -1); fi
[ -f "$HEX" ] || { echo "no bitstream in $SRC"; exit 1; }
mkdir -p "$DST"
cp "$HEX" "$DST/tc_ref_design_top.hex"
R="$REPO/src/ace/$IMPL/pnr/reports"
{
  echo "$NAME"; echo "==============="; echo
  echo "S0 node-array test bitstream, ACE 10.5.2, worktree $REPO, impl $IMPL, commit $(git -C "$REPO" rev-parse --short=12 HEAD)"
  echo "Clocks: array 725 MHz (PLL_SW_3), vector 333.33 MHz + fabric 250 MHz (PLL_SW_1), PLL_SW_0 as deployed."
  echo "Host window: BAR0 -> NOC[3][4], paper/rtl/pi0_chip_ctrl.sv; GDDR6 via the SDK bulk path."
  echo "$*"; echo
  echo "Routed timing (slow corner 0C):"
  grep -hA6 "Setup (max)" "$R"/tc_ref_design_top_timing_final_C1_0p90V_0C.txt 2>/dev/null | grep -E "sc_s" | head -6 || true
  grep -hA6 "Hold (min)" "$R"/tc_ref_design_top_timing_final_C1_0p90V_0C.txt 2>/dev/null | grep -E "sc_h" | head -6 || true
} > "$DST/INFO.txt"
(cd "$DST" && sha256sum tc_ref_design_top.hex > SHA256SUMS)
ls -la "$DST"; cat "$DST/INFO.txt"
