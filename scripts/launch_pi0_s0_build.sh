#!/usr/bin/env bash
# Launch the S0 node-array bitstream build (ACE 10.5.2) in the background, in a memory-capped user scope.
#   scripts/launch_pi0_s0_build.sh <tag> [N_CHAIN=1] [N_PV=0] [N_VNODE=1] [N_DEEP=0]
# Env: PI0_ACE_SEED, PI0_S0_ID_WORD (24-bit, default 24'h533001), PI0_S0_NAPS_PER_NODE (2), PI0_S0_MAX_OUT (8: the
#      nodes' AXI read/write outstanding limit; build variants to sweep it on silicon), MEM_HIGH (20G),
#      PI0_S0_STRIPE (12 = 4 KB GDDR6 channel striping), PI0_S0_RESET_FP (1 = reset outputs false-path),
#      PI0_S0_CAP_MCP (1 = 2-cycle multicycle on the MLP72 -> capture path, guarded, see pi0_s0_ace.sdc),
#      PI0_S0_FLOW_MODE (evaluation = fast non-timing-driven routing), PI0_S0_N_LANE (vector-node lanes, 2), PI0_S0_REGIONS_PDC (per-node placement regions, path relative to src/ace, e.g. ./../build/s0/regions.pdc)
# Logs: build/s0/<impl>.log; the flow's own logs in src/ace/<impl>/.  The licence ports must be listening.
set -euo pipefail
TAG=${1:?tag}; NC=${2:-1}; NPV=${3:-0}; NV=${4:-1}; ND=${5:-0}
REPO=$(cd "$(dirname "$0")/.." && pwd)
ACE_ROOT=/home/sngong/ACE_10.5.2/Achronix-linux      # explicit: the login shell exports 10.3.1
cd "$REPO"
[ -e src/ace/tc_ref_design_top.lock ] && { echo "refusing: ACE project lock present (src/ace/tc_ref_design_top.lock)" >&2; exit 1; }
for port in 1710 27000; do ss -ltn | grep -q ":$port " || { echo "refusing: licence port $port not listening" >&2; exit 1; }; done
COMMIT=$(git rev-parse --short=12 HEAD)
LSUF=""; [ "${PI0_S0_N_LANE:-2}" != 2 ] && LSUF="l${PI0_S0_N_LANE}"; [ "${PI0_S0_STRIPE:-0}" != 0 ] && LSUF="${LSUF}s${PI0_S0_STRIPE}"
IMPL="impl_s0_${TAG}_c${NC}p${NPV}v${NV}d${ND}o${PI0_S0_MAX_OUT:-8}${LSUF}_${COMMIT}_$(date -u +%Y%m%dT%H%M%SZ)"
TABLES="$REPO/build/s0/tables"
mkdir -p "$REPO/build/s0" "$TABLES"
python3 "$REPO/paper/sw/vector_unit_ref.py" --tables "$TABLES" > /dev/null
LOG="$REPO/build/s0/$IMPL.log"
echo "S0 build $IMPL: chains=$NC pv=$NPV vec=$NV deep=$ND naps/node=${PI0_S0_NAPS_PER_NODE:-2} max_out=${PI0_S0_MAX_OUT:-8} seed=${PI0_ACE_SEED:-default}"
echo "log: $LOG"
nohup systemd-run --user --scope -q -p MemoryHigh="${MEM_HIGH:-20G}" -p MemorySwapMax=infinity \
  env -i HOME=/home/sngong PATH="$ACE_ROOT:$ACE_ROOT/Synplify/bin:/usr/local/bin:/usr/bin:/bin" \
      ACE_INSTALL_DIR="$ACE_ROOT" SYN_HOME="$ACE_ROOT/Synplify" \
      RLM_LICENSE=1710@127.0.0.1 SNPSLMD_LICENSE_FILE=27000@127.0.0.1 LM_LICENSE_FILE=27000@127.0.0.1 \
      PI0_REPO_ROOT="$REPO" PI0_ACE_IMPL="$IMPL" PI0_S0_SYN_PRJ="$REPO/paper/synth/s0/pi0_s0_synth.prj" \
      PI0_S0_TABLES="$TABLES" PI0_S0_N_CHAIN="$NC" PI0_S0_N_PV="$NPV" PI0_S0_N_VNODE="$NV" PI0_S0_N_DEEP="$ND" PI0_S0_N_LANE="${PI0_S0_N_LANE:-2}" \
      PI0_S0_NAPS_PER_NODE="${PI0_S0_NAPS_PER_NODE:-2}" PI0_S0_MAX_OUT="${PI0_S0_MAX_OUT:-8}" \
      PI0_S0_ID_WORD="${PI0_S0_ID_WORD:-}" PI0_ACE_SEED="${PI0_ACE_SEED:-}" PI0_S0_REGIONS_PDC="${PI0_S0_REGIONS_PDC:-}" PI0_S0_RESET_MCP="${PI0_S0_RESET_MCP:-}" PI0_S0_IMPL_OPTS="${PI0_S0_IMPL_OPTS:-}" PI0_S0_RETIMING="${PI0_S0_RETIMING:-}" PI0_S0_FLOW_MODE="${PI0_S0_FLOW_MODE:-}" PI0_S0_STRIPE="${PI0_S0_STRIPE:-0}" PI0_S0_RESET_FP="${PI0_S0_RESET_FP:-}" PI0_S0_CAP_MCP="${PI0_S0_CAP_MCP:-}" \
  nice -n 5 "$ACE_ROOT/ace" -batch -print_progress -script_file "$REPO/scripts/run_pi0_s0_ace_flow.tcl" > "$LOG" 2>&1 &
echo "$IMPL" > "$REPO/build/s0/LAST_IMPL"
# the SDC switches are read from the environment at every constraint load (prepare, STA, reclock, re-route): record them
printf 'PI0_S0_RESET_FP=%q\nPI0_S0_CAP_MCP=%q\nPI0_S0_RESET_MCP=%q\nPI0_S0_STRIPE=%q\nPI0_S0_FLOW_MODE=%q\nPI0_S0_IMPL_OPTS=%q\n' \
  "${PI0_S0_RESET_FP:-}" "${PI0_S0_CAP_MCP:-}" "${PI0_S0_RESET_MCP:-}" "${PI0_S0_STRIPE:-}" "${PI0_S0_FLOW_MODE:-}" "${PI0_S0_IMPL_OPTS:-}" > "$REPO/build/s0/$IMPL.env"
