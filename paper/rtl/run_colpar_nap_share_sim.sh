#!/usr/bin/env bash
# NM chain nodes share one NAP (colpar_nap_mux.sv), each running its own command program
# against shared GDDR6 read/write models; every node's records must land in its own address
# window, in order (paper/sw/colpar_tile_golden.py --addr-offset i<<32).
#
# Usage: run_colpar_nap_share_sim.sh [NM=3] [RD_BURST_BEATS=256] [SEED=1]
#        run_colpar_nap_share_sim.sh all     # NM 1 2 4 x RD_BURST_BEATS 256 16
set -euo pipefail
REPO=$(cd "$(dirname "$0")/../.." && pwd)
VERILATOR=${VERILATOR:-$HOME/tools/verilator5/bin/verilator}
BUILD=${BUILD:-$REPO/build/paper_colpar_nap_sim}

if [ "${1:-}" = all ]; then
    rc=0
    for M in 1 2 4; do for B in 256 16; do
        "$0" "$M" "$B" "${SEED:-1}" || rc=1
    done; done
    exit $rc
fi

NM=${1:-3}; B=${2:-16}; SEED=${3:-1}; N=4
MODELS="$REPO/paper/rtl/sim_models/acx_float_behav.sv $REPO/paper/rtl/sim_models/acx_bram72k_behav.sv $REPO/paper/rtl/sim_models/acx_mlp72_behav.sv"
RTL="$REPO/paper/rtl/colpar_prog_fetch.sv $REPO/paper/rtl/colpar_nap_mux.sv $REPO/paper/rtl/colpar_node_ctrl.sv $REPO/paper/rtl/colpar_chain_node.sv $REPO/paper/rtl/node_sync.sv $REPO/paper/rtl/colpar_tile_loader.sv $REPO/paper/rtl/colpar_row_sequencer.sv $REPO/paper/rtl/mlp72_int8_colpar_chain.sv $REPO/paper/rtl/colpar_result_port.sv $REPO/paper/rtl/colpar_result_writer.sv $REPO/paper/rtl/tb_axi_gddr6_model.sv"
TB=$REPO/paper/rtl/tb_colpar_nap_share.sv
OBJ=$BUILD/obj_m${NM}_b${B}
EXE=$OBJ/Vtb_colpar_nap_share
mkdir -p "$BUILD"

for ((i = 0; i < NM; i++)); do
    python3 "$REPO/paper/sw/colpar_tile_golden.py" --n-stage "$N" --seed $((SEED * 100 + i + 1)) \
        --addr-offset $((i << 32)) --out "$BUILD/vec_s${SEED}_node$i" > "$BUILD/vec_s${SEED}_node$i.log"
done

rebuild=0
[ -x "$EXE" ] || rebuild=1
for f in $MODELS $TB $RTL; do [ "$f" -nt "$EXE" ] 2>/dev/null && rebuild=1; done
if [ $rebuild = 1 ]; then
    rm -rf "$OBJ"
    # shellcheck disable=SC2086
    "$VERILATOR" --binary --timing -j 4 -O2 -Wno-fatal -Wno-lint -Wno-style -Wno-PINMISSING \
        -Wno-MULTIDRIVEN -Wno-UNOPTFLAT --top-module tb_colpar_nap_share \
        -GNM="$NM" -GN_STAGE="$N" -GRD_BURST_BEATS="$B" --Mdir "$OBJ" \
        $MODELS $RTL "$TB" > "$BUILD/build_m${NM}_b${B}.log" 2>&1 \
        || { echo "BUILD FAILED (log $BUILD/build_m${NM}_b${B}.log)"; tail -25 "$BUILD/build_m${NM}_b${B}.log"; exit 1; }
fi

LOG=$BUILD/run_m${NM}_b${B}_s${SEED}.log
"$EXE" +vecdir="$BUILD/vec_s${SEED}_node" +npulses="$(cat "$BUILD/vec_s${SEED}_node0/npulses_prog.txt")" +verilator+seed+"$SEED" > "$LOG" 2>&1 || true
grep -E "MISMATCH|ERROR|RESULT" "$LOG" | head -6
echo "  seed=$SEED log=$LOG"
grep -q "RESULT PASS" "$LOG"
