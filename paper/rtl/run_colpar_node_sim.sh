#!/usr/bin/env bash
# End-to-end chain node check: GDDR6 AXI model -> colpar_tile_loader -> chain BRAMs ->
# colpar_row_sequencer -> mlp72_int8_colpar_chain -> colpar_result_port -> colpar_result_writer
# -> GDDR6 AXI write model, against
# paper/sw/colpar_tile_golden.py (int64 matmul) and the behavioural MLP72/BRAM72K models.
#
# Usage: run_colpar_node_sim.sh <N_STAGE> [BEATS_PER_BURST=16] [ADDR_CASCADE=0] [SEED=1] [WR_BURST_BEATS=16] [MAX_OUT=8]
#        run_colpar_node_sim.sh all      # N_STAGE 4 16 x (16/7-beat x 1/8 outstanding, 256-beat x 8)
# Env:   BUILD=<dir>   default build/paper_colpar_node_sim
#        WR_SLIM=0|1   result writer: 1 record FIFO from the port's bank (default), 0 the original skid/serialiser
set -euo pipefail
REPO=$(cd "$(dirname "$0")/../.." && pwd)
VERILATOR=${VERILATOR:-$HOME/tools/verilator5/bin/verilator}
BUILD=${BUILD:-$REPO/build/paper_colpar_node_sim}
DEFS=${MODEL_DEFS:-}
WR_SLIM=${WR_SLIM:-1}

if [ "${1:-}" = all ]; then
    rc=0
    for N in 4 16; do
        for B in 16 7; do for MO in 1 8; do
            "$0" "$N" "$B" 0 "${SEED:-1}" 16 "$MO" || rc=1
        done; done
        "$0" "$N" 256 0 "${SEED:-1}" 16 8 || rc=1
    done
    exit $rc
fi

N=$1; B=${2:-16}; C=${3:-0}; SEED=${4:-1}; WB=${5:-16}; MO=${6:-8}
MODELS="$REPO/paper/rtl/sim_models/acx_float_behav.sv $REPO/paper/rtl/sim_models/acx_bram72k_behav.sv $REPO/paper/rtl/sim_models/acx_mlp72_behav.sv"
RTL="$REPO/paper/rtl/colpar_tile_loader.sv $REPO/paper/rtl/colpar_row_sequencer.sv $REPO/paper/rtl/mlp72_int8_colpar_chain.sv $REPO/paper/rtl/colpar_result_port.sv $REPO/paper/rtl/colpar_result_writer.sv $REPO/paper/rtl/tb_axi_gddr6_model.sv"
TB=$REPO/paper/rtl/tb_colpar_node.sv
TAG=$(echo "$DEFS" | md5sum | cut -c1-8)
OBJ=$BUILD/obj_n${N}_b${B}_c${C}_w${WB}_o${MO}_sl${WR_SLIM}_$TAG
VEC=$BUILD/vec_n${N}_s${SEED}
EXE=$OBJ/Vtb_colpar_node
mkdir -p "$BUILD"

python3 "$REPO/paper/sw/colpar_tile_golden.py" --n-stage "$N" --seed "$SEED" --out "$VEC" > "$VEC.log"

rebuild=0
[ -x "$EXE" ] || rebuild=1
for f in $MODELS $TB $RTL; do [ "$f" -nt "$EXE" ] 2>/dev/null && rebuild=1; done
if [ $rebuild = 1 ]; then
    rm -rf "$OBJ"
    # shellcheck disable=SC2086
    "$VERILATOR" --binary --timing -j 4 -O2 -Wno-fatal -Wno-lint -Wno-style -Wno-PINMISSING \
        -Wno-MULTIDRIVEN -Wno-UNOPTFLAT --top-module tb_colpar_node \
        -GN_STAGE="$N" -GADDR_CASCADE="$C" -GBEATS_PER_BURST="$B" -GWR_BURST_BEATS="$WB" -GMAX_OUT="$MO" -GWR_SLIM="$WR_SLIM" $DEFS --Mdir "$OBJ" \
        $MODELS $RTL "$TB" > "$BUILD/build_n${N}_b${B}_c${C}_$TAG.log" 2>&1 \
        || { echo "BUILD FAILED (log $BUILD/build_n${N}_b${B}_c${C}_$TAG.log)"; tail -25 "$BUILD/build_n${N}_b${B}_c${C}_$TAG.log"; exit 1; }
fi

LOG=$BUILD/run_n${N}_b${B}_c${C}_w${WB}_o${MO}_sl${WR_SLIM}_s${SEED}_$TAG.log
"$EXE" +vec="$VEC" +ntiles="$(cat "$VEC/ntiles.txt")" +npulses="$(cat "$VEC/npulses.txt")" \
    +verilator+seed+"$SEED" > "$LOG" 2>&1 || true
grep -E "MISMATCH|ERROR|RESULT" "$LOG" | head -6
echo "  seed=$SEED log=$LOG"
grep -q "RESULT PASS" "$LOG"
