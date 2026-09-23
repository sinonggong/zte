#!/usr/bin/env bash
# Program-level check of colpar_chain_node.sv (command executor + loader + sequencer + chain +
# result port + writer) running prog.memh from paper/sw/colpar_tile_golden.py against GDDR6
# AXI models; the records in GDDR6 must equal exp_prog.memh (int64 matmul).
#
# Usage: run_colpar_prog_sim.sh <N_STAGE> [RD_BURST_BEATS=16] [SEED=1] [MAX_OUT=8] [PROG_GDDR6=0]
#        run_colpar_prog_sim.sh all      # N_STAGE 1 4 16 x (16-beat x 1/8 outstanding, 256-beat x 8)
# Env: WR_PIPE, VALID_COPIES (deep chains: 4 / 4), WR_SLIM (result writer: 1 record FIFO, default; 0 the original)
set -euo pipefail
REPO=$(cd "$(dirname "$0")/../.." && pwd)
VERILATOR=${VERILATOR:-$HOME/tools/verilator5/bin/verilator}
BUILD=${BUILD:-$REPO/build/paper_colpar_prog_sim}

if [ "${1:-}" = all ]; then
    rc=0
    for N in 1 4 16; do
        "$0" "$N" 16 "${SEED:-1}" 1 || rc=1
        "$0" "$N" 16 "${SEED:-1}" 8 || rc=1
        "$0" "$N" 256 "${SEED:-1}" 8 || rc=1
        "$0" "$N" 16 "${SEED:-1}" 8 1 || rc=1
    done
    exit $rc
fi

N=$1; B=${2:-16}; SEED=${3:-1}; MO=${4:-8}; PG=${5:-0}
WP=${WR_PIPE:-0}
VC=${VALID_COPIES:-1}
SL=${WR_SLIM:-1}
DEFS=${MODEL_DEFS:-}
TAG=$(echo "$DEFS" | md5sum | cut -c1-6)
MODELS="$REPO/paper/rtl/sim_models/acx_float_behav.sv $REPO/paper/rtl/sim_models/acx_bram72k_behav.sv $REPO/paper/rtl/sim_models/acx_mlp72_behav.sv"
RTL="$REPO/paper/rtl/colpar_prog_fetch.sv $REPO/paper/rtl/colpar_node_ctrl.sv $REPO/paper/rtl/colpar_chain_node.sv $REPO/paper/rtl/node_sync.sv $REPO/paper/rtl/colpar_tile_loader.sv $REPO/paper/rtl/colpar_row_sequencer.sv $REPO/paper/rtl/mlp72_int8_colpar_chain.sv $REPO/paper/rtl/colpar_result_port.sv $REPO/paper/rtl/colpar_result_writer.sv $REPO/paper/rtl/tb_axi_gddr6_model.sv"
TB=$REPO/paper/rtl/tb_colpar_node_prog.sv
OBJ=$BUILD/obj_n${N}_b${B}_o${MO}_p${PG}_w${WP}_v${VC}_sl${SL}_$TAG
VEC=$BUILD/vec_n${N}_s${SEED}
EXE=$OBJ/Vtb_colpar_node_prog
mkdir -p "$BUILD"

python3 "$REPO/paper/sw/colpar_tile_golden.py" --n-stage "$N" --seed "$SEED" --out "$VEC" > "$VEC.log"

rebuild=0
[ -x "$EXE" ] || rebuild=1
for f in $MODELS $TB $RTL; do [ "$f" -nt "$EXE" ] 2>/dev/null && rebuild=1; done
if [ $rebuild = 1 ]; then
    rm -rf "$OBJ"
    # shellcheck disable=SC2086
    "$VERILATOR" --binary --timing -j 4 -O2 -Wno-fatal -Wno-lint -Wno-style -Wno-PINMISSING \
        -Wno-MULTIDRIVEN -Wno-UNOPTFLAT --top-module tb_colpar_node_prog \
        -GN_STAGE="$N" -GRD_BURST_BEATS="$B" -GMAX_OUT="$MO" -GPROG_GDDR6="$PG" -GWR_PIPE="$WP" -GVALID_COPIES="$VC" -GWR_SLIM="$SL" $DEFS --Mdir "$OBJ" \
        $MODELS $RTL "$TB" > "$BUILD/build_n${N}_b${B}_o${MO}_p${PG}.log" 2>&1 \
        || { echo "BUILD FAILED (log $BUILD/build_n${N}_b${B}.log)"; tail -25 "$BUILD/build_n${N}_b${B}_o${MO}_p${PG}.log"; exit 1; }
fi

LOG=$BUILD/run_n${N}_b${B}_o${MO}_p${PG}_w${WP}_v${VC}_sl${SL}_${TAG}_s${SEED}.log
"$EXE" +vec="$VEC" +npulses="$(cat "$VEC/npulses_prog.txt")" +verilator+seed+"$SEED" > "$LOG" 2>&1 || true
grep -E "MISMATCH|ERROR|RESULT" "$LOG" | head -6
echo "  seed=$SEED wr_pipe=$WP valid_copies=$VC log=$LOG"
grep -q "RESULT PASS" "$LOG"
