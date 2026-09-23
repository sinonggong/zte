#!/usr/bin/env bash
# Tile-level GEMM check of colpar_row_sequencer.sv + mlp72_int8_colpar_chain.sv against the
# behavioural ACX_MLP72 / ACX_BRAM72K models and paper/sw/colpar_tile_golden.py (int64 matmul).
#
# Usage: run_colpar_tile_sim.sh <N_STAGE> [ADDR_CASCADE=0] [SEED=1]
#        run_colpar_tile_sim.sh all        # N_STAGE 1 4 16 x ADDR_CASCADE 0 1
# Env:   MODEL_DEFS="+define+..."  (e.g. a COLPAR_NEG_* negative control)
#        BUILD=<dir>                default build/paper_colpar_tile_sim
#        MIN_PASS=<cycles>          minimum W + G in the vectors (default 14; smaller breaks the result port rule)
#        MULT_MODE=13               build the chain with uint8 x int8 multipliers (5'h13) and uint8 activation vectors
#        VEC_UNSIGNED=0|1           override the vectors' activation sign (negative control: the other mode's vectors)
set -euo pipefail
REPO=$(cd "$(dirname "$0")/../.." && pwd)
VERILATOR=${VERILATOR:-$HOME/tools/verilator5/bin/verilator}
BUILD=${BUILD:-$REPO/build/paper_colpar_tile_sim}
DEFS=${MODEL_DEFS:-}

if [ "${1:-}" = all ]; then
    rc=0
    for C in 0 1; do for N in 1 4 16; do
        "$0" "$N" "$C" "${SEED:-1}" || rc=1
    done; done
    exit $rc
fi

N=$1; C=${2:-0}; SEED=${3:-1}
MODELS="$REPO/paper/rtl/sim_models/acx_float_behav.sv $REPO/paper/rtl/sim_models/acx_bram72k_behav.sv $REPO/paper/rtl/sim_models/acx_mlp72_behav.sv"
RTL="$REPO/paper/rtl/colpar_row_sequencer.sv $REPO/paper/rtl/mlp72_int8_colpar_chain.sv $REPO/paper/rtl/colpar_result_port.sv"
TB=$REPO/paper/rtl/tb_colpar_tile.sv
MM=${MULT_MODE:-00}
VU=${VEC_UNSIGNED:-$([ "$MM" = 13 ] && echo 1 || echo 0)}
TAG=$(echo "$DEFS mm$MM" | md5sum | cut -c1-8)
OBJ=$BUILD/obj_n${N}_c${C}_$TAG
VEC=$BUILD/vec_n${N}_s${SEED}
EXE=$OBJ/Vtb_colpar_tile
mkdir -p "$BUILD"

MINP=${MIN_PASS:-14}
[ "$MINP" != 14 ] && VEC=${VEC}_minp$MINP
UFLAG=""
[ "$VU" = 1 ] && { VEC=${VEC}_u8; UFLAG=--unsigned-act; }
python3 "$REPO/paper/sw/colpar_tile_golden.py" --n-stage "$N" --seed "$SEED" --min-pass "$MINP" $UFLAG --out "$VEC" > "$VEC.log"

rebuild=0
[ -x "$EXE" ] || rebuild=1
for f in $MODELS $TB $RTL; do [ "$f" -nt "$EXE" ] 2>/dev/null && rebuild=1; done
if [ $rebuild = 1 ]; then
    rm -rf "$OBJ"
    # shellcheck disable=SC2086
    "$VERILATOR" --binary --timing -j 4 -O2 -Wno-fatal -Wno-lint -Wno-style -Wno-PINMISSING \
        -Wno-MULTIDRIVEN -Wno-UNOPTFLAT --top-module tb_colpar_tile \
        -GN_STAGE="$N" -GADDR_CASCADE="$C" -GMULT_MODE="5'h$MM" $DEFS --Mdir "$OBJ" \
        $MODELS $RTL "$TB" > "$BUILD/build_n${N}_c${C}_$TAG.log" 2>&1 \
        || { echo "BUILD FAILED (log $BUILD/build_n${N}_c${C}_$TAG.log)"; tail -25 "$BUILD/build_n${N}_c${C}_$TAG.log"; exit 1; }
fi

LOG=$BUILD/run_n${N}_c${C}_s${SEED}_u${VU}_$TAG.log
"$EXE" +vec="$VEC" +ntiles="$(cat "$VEC/ntiles.txt")" +npulses="$(cat "$VEC/npulses.txt")" \
    +verilator+seed+"$SEED" > "$LOG" 2>&1 || true
grep -E "MISMATCH|ERROR|PROTOCOL|RESULT" "$LOG" | head -6
echo "  seed=$SEED mult_mode=5'h$MM vec_unsigned=$VU defs='${DEFS}' log=$LOG"
grep -q "RESULT PASS" "$LOG"
