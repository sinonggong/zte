#!/usr/bin/env bash
# The expert's o_proj at its REAL width on one chain node: 51 x 2048 @ 2048 x 1024 in sixteen 64-column
# tiles, each writing a column block of the row-major result.  Bit-exact against
# paper/sw/pi0_wide_gemm_golden.py (int64 matmul on the real checkpoint weights).  See tb_pi0_wide_gemm.sv.
#
# Usage: run_pi0_wide_gemm_sim.sh [LAYER=0] [STEP=0]
#        run_pi0_wide_gemm_sim.sh all   # 3 layers, M = 3 feeder rows, 1/4/8-node splits, 4 negative controls
#        NODES=4 TILE_GROUPS=2 run_pi0_wide_gemm_sim.sh 0 0   # split by column tiles AND rows
# Env: VERILATOR, PYTHON (numpy + safetensors + torch: ~/lerobot/.venv/bin/python), BUILD,
#      NEG=blockgap|tileorder|blockbeats|stageorder  (a negative control: the run must FAIL)
#      FEEDER_ROWS=<n>  feeder rows per load; the result must NOT depend on it
#      NODES=<n> TILE_GROUPS=<g>  spread the GEMM over n chain nodes: g split the column tiles, n/g the rows
set -euo pipefail
REPO=$(cd "$(dirname "$0")/../.." && pwd)
VERILATOR=${VERILATOR:-$HOME/tools/verilator5/bin/verilator}
PYTHON=${PYTHON:-$HOME/lerobot/.venv/bin/python}
[ -x "$PYTHON" ] || PYTHON=python3
BUILD=${BUILD:-$REPO/build/paper_pi0_wide_gemm_sim}

if [ "${1:-}" = all ]; then
    rc=0
    "$0" 0 0 || rc=1
    "$0" 9 1 || rc=1
    "$0" 17 0 || rc=1
    FEEDER_ROWS=3 "$0" 0 0 || { echo "the result changed with 3 feeder rows -- it must not"; rc=1; }
    NODES=2 TILE_GROUPS=2 "$0" 0 0 || rc=1      # column tiles only, 2 nodes
    NODES=4 TILE_GROUPS=2 "$0" 0 0 || rc=1      # 2 tile groups x 2 row slices
    NODES=8 TILE_GROUPS=1 "$0" 0 0 || rc=1      # rows only: the QK^T / PV case, 1 column tile group
    for n in blockgap tileorder blockbeats stageorder; do
        if NEG=$n "$0" 0 0; then echo "NEGATIVE CONTROL $n PASSED -- it must not"; rc=1
        else echo "  negative control $n failed as intended"; fi
    done
    exit $rc
fi

L=${1:-0}; ST=${2:-0}
NEG=${NEG:-}
FEEDER_ROWS=${FEEDER_ROWS:-0}
NODES=${NODES:-1}
TILE_GROUPS=${TILE_GROUPS:-0}
C=$REPO/paper/rtl
SRCS="$C/sim_models/acx_float_behav.sv $C/sim_models/acx_bram72k_behav.sv $C/sim_models/acx_mlp72_behav.sv \
      $C/colpar_prog_fetch.sv $C/colpar_node_ctrl.sv $C/colpar_chain_node.sv $C/colpar_tile_loader.sv \
      $C/colpar_row_sequencer.sv $C/mlp72_int8_colpar_chain.sv $C/colpar_result_port.sv \
      $C/colpar_result_writer.sv $C/colpar_nap_mux.sv $C/node_sync.sv $C/tb_axi_gddr6_model.sv $C/tb_pi0_wide_gemm.sv"
mkdir -p "$BUILD"

VEC=$BUILD/vec_L${L}_s${ST}${NEG:+_neg_$NEG}$([ "$FEEDER_ROWS" = 0 ] || echo "_m$FEEDER_ROWS")$([ "$NODES" = 1 ] || echo "_n${NODES}g${TILE_GROUPS}")
"$PYTHON" "$REPO/paper/sw/pi0_wide_gemm_golden.py" --layer "$L" --step "$ST" ${NEG:+--neg "$NEG"} \
    --feeder-rows "$FEEDER_ROWS" --nodes "$NODES" --tile-groups "$TILE_GROUPS" \
    --out "$VEC" > "$VEC.log" || { echo "GOLDEN FAILED"; tail -20 "$VEC.log"; exit 1; }
sed 's/^/  /' "$VEC.log"

OBJ=$BUILD/obj_n$NODES
EXE=$OBJ/Vtb_pi0_wide_gemm
rebuild=0
[ -x "$EXE" ] || rebuild=1
for f in $SRCS; do [ "$f" -nt "$EXE" ] && rebuild=1; done
if [ $rebuild = 1 ]; then
    rm -rf "$OBJ"
    # shellcheck disable=SC2086
    "$VERILATOR" --binary --timing -j 4 -O2 -Wno-fatal -Wno-lint -Wno-style -Wno-PINMISSING \
        -Wno-MULTIDRIVEN -Wno-UNOPTFLAT --top-module tb_pi0_wide_gemm -GN_NODE="$NODES" \
        --Mdir "$OBJ" $SRCS > "$BUILD/build_n$NODES.log" 2>&1 \
        || { echo "BUILD FAILED (log $BUILD/build_n$NODES.log)"; grep -E "%Error" "$BUILD/build_n$NODES.log" | head -20; exit 1; }
fi

LOG=$BUILD/run_L${L}_s${ST}${NEG:+_neg_$NEG}$([ "$FEEDER_ROWS" = 0 ] || echo "_m$FEEDER_ROWS")$([ "$NODES" = 1 ] || echo "_n${NODES}g${TILE_GROUPS}").log
"$EXE" +vec="$VEC" +verilator+seed+1 > "$LOG" 2>&1 || true
grep -E "MISMATCH|MISSING|RESULT" "$LOG" | head -14
echo "  layer=$L step=$ST neg=${NEG:-none} feeder_rows=${FEEDER_ROWS} nodes=${NODES} tile_groups=${TILE_GROUPS} log=$LOG"
grep -q "RESULT PASS" "$LOG"
