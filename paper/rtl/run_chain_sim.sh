#!/usr/bin/env bash
# Bit-exact Verilator run of paper/rtl/mlp72_int8_chain.sv against the behavioural
# ACX_MLP72 / ACX_BRAM72K models (paper/rtl/sim_models/) and the numpy golden
# (paper/sw/mlp72_chain_golden.py).
#
# Usage: run_chain_sim.sh <N_STAGE> <WORDS_PER_ROW> [ADDR_CASCADE=0] [SEED=1] [INROW_GAPS=1]
#        run_chain_sim.sh all        # N_STAGE 1 4 16 x words 1 3 32 x ADDR_CASCADE 0 1
# Env:   MODEL_DEFS="+define+ACX_MLP72_BEHAV_FWDO_PRE_REG"  flip a model assumption
#        CHAIN_RTL=<file>   simulate another chain RTL (same ports)
#        BUILD=<dir>        default build/paper_chain_sim
# On fics run it niced next to the FPGA build, e.g.
#   systemd-run --user --scope -q -p MemoryHigh=6G nice -n 10 paper/rtl/run_chain_sim.sh all
set -euo pipefail
REPO=$(cd "$(dirname "$0")/../.." && pwd)
VERILATOR=${VERILATOR:-$HOME/tools/verilator5/bin/verilator}
BUILD=${BUILD:-$REPO/build/paper_chain_sim}
RTL=${CHAIN_RTL:-$REPO/paper/rtl/mlp72_int8_chain.sv}
DEFS=${MODEL_DEFS:-}

if [ "${1:-}" = all ]; then
    rc=0
    for C in 0 1; do for N in 1 4 16; do for W in 1 3 32; do
        "$0" "$N" "$W" "$C" "${SEED:-1}" "${INROW_GAPS:-1}" || rc=1
    done; done; done
    exit $rc
fi

N=$1; W=$2; C=${3:-0}; SEED=${4:-1}; G=${5:-1}
MODELS="$REPO/paper/rtl/sim_models/acx_float_behav.sv $REPO/paper/rtl/sim_models/acx_bram72k_behav.sv $REPO/paper/rtl/sim_models/acx_mlp72_behav.sv"
TB=$REPO/paper/rtl/tb_mlp72_int8_chain.sv
TAG=$(echo "$DEFS|$RTL" | md5sum | cut -c1-8)
OBJ=$BUILD/obj_n${N}_c${C}_$TAG
VEC=$BUILD/vec_n${N}_w${W}_s${SEED}
EXE=$OBJ/Vtb_mlp72_int8_chain
mkdir -p "$BUILD"

python3 "$REPO/paper/sw/mlp72_chain_golden.py" --n-stage "$N" --words "$W" --seed "$SEED" --out "$VEC" > "$VEC.log"

rebuild=0
[ -x "$EXE" ] || rebuild=1
for f in $MODELS $TB $RTL; do [ "$f" -nt "$EXE" ] 2>/dev/null && rebuild=1; done
if [ $rebuild = 1 ]; then
    rm -rf "$OBJ"
    # shellcheck disable=SC2086
    "$VERILATOR" --binary --timing -j 4 -O2 -Wno-fatal -Wno-lint -Wno-style -Wno-PINMISSING \
        -Wno-MULTIDRIVEN -Wno-UNOPTFLAT --top-module tb_mlp72_int8_chain \
        -GN_STAGE="$N" -GADDR_CASCADE="$C" $DEFS --Mdir "$OBJ" \
        $MODELS "$RTL" "$TB" > "$BUILD/build_n${N}_c${C}_$TAG.log" 2>&1 \
        || { echo "BUILD FAILED (log $BUILD/build_n${N}_c${C}_$TAG.log)"; tail -25 "$BUILD/build_n${N}_c${C}_$TAG.log"; exit 1; }
fi

LOG=$BUILD/run_n${N}_w${W}_c${C}_s${SEED}_$TAG.log
"$EXE" +vec="$VEC" +nrows="$(cat "$VEC/nrows.txt")" +inrow_gaps="$G" +verilator+seed+"$SEED" > "$LOG" 2>&1 || true
grep -E "MISMATCH|ERROR|RESULT" "$LOG" | head -6
echo "  words/row=$W seed=$SEED defs='${DEFS}' log=$LOG"
grep -q "RESULT PASS" "$LOG"
