#!/usr/bin/env bash
# One pi0 linear layer across node types (vector QUANT -> column-parallel chain GEMM -> vector DEQUANT) on
# shared GDDR6, bit-exact against paper/sw/pi0_linear_golden.py.  See paper/rtl/tb_pi0_linear.sv.
#
# Usage: run_pi0_linear_sim.sh [SEED=1] [TOKENS=10] [K=64] [PASSES=3]
#        run_pi0_linear_sim.sh all     # (seed, T, K, P) = (1,10,64,3) (2,51,32,2) (3,7,1024,8)
# Env: VERILATOR, PYTHON (numpy), BUILD (default build/paper_pi0_linear_sim)
set -euo pipefail
REPO=$(cd "$(dirname "$0")/../.." && pwd)
VERILATOR=${VERILATOR:-$HOME/tools/verilator5/bin/verilator}
PYTHON=${PYTHON:-$HOME/lerobot/.venv/bin/python}
[ -x "$PYTHON" ] || PYTHON=python3
BUILD=${BUILD:-$REPO/build/paper_pi0_linear_sim}

if [ "${1:-}" = all ]; then
    rc=0
    "$0" 1 10 64 3 || rc=1
    "$0" 2 51 32 2 || rc=1
    "$0" 3 7 1024 8 || rc=1
    exit $rc
fi

SEED=${1:-1}; T=${2:-10}; KIN=${3:-64}; P=${4:-3}
V=$REPO/paper/rtl/vector_unit
C=$REPO/paper/rtl
SRCS="$REPO/paper/rtl/sim_models/acx_float_behav.sv $REPO/paper/rtl/sim_models/acx_bram72k_behav.sv \
      $REPO/paper/rtl/sim_models/acx_mlp72_behav.sv \
      $V/vu_pkg.sv $V/vu_add.sv $V/vu_mul.sv $V/vu_round.sv $V/vu_tbl.sv $V/vu_qtab.sv $V/vu_lane.sv \
      $V/vu_slot_loader.sv $V/vu_beat_writer.sv $V/vu_node.sv \
      $C/colpar_prog_fetch.sv $C/colpar_node_ctrl.sv $C/colpar_chain_node.sv $C/node_sync.sv $C/colpar_tile_loader.sv \
      $C/colpar_row_sequencer.sv $C/mlp72_int8_colpar_chain.sv $C/colpar_result_port.sv $C/colpar_result_writer.sv \
      $C/colpar_nap_mux.sv $C/tb_axi_gddr6_model.sv $C/tb_pi0_linear.sv"
mkdir -p "$BUILD"
TBL=$BUILD/tables
"$PYTHON" "$REPO/paper/sw/vector_unit_ref.py" --tables "$TBL" > /dev/null
VEC=$BUILD/vec_s${SEED}_t${T}_k${KIN}_p${P}
"$PYTHON" "$REPO/paper/sw/pi0_linear_golden.py" --seed "$SEED" --tokens "$T" --k "$KIN" --passes "$P" --out "$VEC" > "$VEC.log"

OBJ=$BUILD/obj
EXE=$OBJ/Vtb_pi0_linear
rebuild=0
[ -x "$EXE" ] || rebuild=1
for f in $SRCS "$TBL"/*.mem; do [ "$f" -nt "$EXE" ] && rebuild=1; done
if [ $rebuild = 1 ]; then
    rm -rf "$OBJ"
    # shellcheck disable=SC2086
    "$VERILATOR" --binary --timing -j 4 -O2 -Wno-fatal -Wno-lint -Wno-style -Wno-PINMISSING -Wno-MULTIDRIVEN \
        -Wno-UNOPTFLAT -I"$V" --top-module tb_pi0_linear \
        -GF_GELU="\"$TBL/vu_tbl_gelu.mem\"" -GF_SIGM="\"$TBL/vu_tbl_sigm.mem\"" \
        -GF_EXP="\"$TBL/vu_tbl_exp.mem\"" -GF_RSQRT="\"$TBL/vu_tbl_rsqrt.mem\"" -GF_QUANT="\"$TBL/vu_tbl_quant.mem\"" \
        --Mdir "$OBJ" $SRCS > "$BUILD/build.log" 2>&1 \
        || { echo "BUILD FAILED (log $BUILD/build.log)"; grep -E "%Error" "$BUILD/build.log" | head -20; exit 1; }
fi
LOG=$BUILD/run_s${SEED}_t${T}_k${KIN}_p${P}.log
"$EXE" +vec="$VEC" +verilator+seed+"$SEED" > "$LOG" 2>&1 || true
grep -E "MISMATCH|MISSING|RESULT" "$LOG" | head -12
echo "  log=$LOG  $(tail -1 "$VEC.log")"
grep -q "RESULT PASS" "$LOG"
