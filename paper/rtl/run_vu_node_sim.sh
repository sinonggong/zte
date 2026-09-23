#!/usr/bin/env bash
# End-to-end Verilator run of the vector node (paper/rtl/vector_unit/vu_node.sv: program fetch,
# operand slots by shape, chunked lane runs, element + summary writers; or the multi-lane
# vu_node_ml.sv) against paper/sw/vu_node_golden.py (bit-exact vector-unit reference) over a GDDR6
# AXI model.
#
# Usage: run_vu_node_sim.sh [SEED=1] [MAX_OUT=8]
#        run_vu_node_sim.sh all      # seeds 1 2 x MAX_OUT 1 8
# Env: VERILATOR, PYTHON (numpy; default ~/lerobot/.venv/bin/python), BUILD (default build/paper_vu_node_sim),
#      N_LANE (default 0 = the one-lane vu_node.sv; 1, 2, 4 = vu_node_ml.sv with that many lanes),
#      SUITE (base = the op list of vu_node_golden.py; ml = multi-lane chunking cases;
#             attn = RoPE rotate-half partner read + 255-level softmax,
#             wide = 4096-wide rows as the full-size MLP streams them; needs SLOT_BITS=12, must FAIL at 11),
#      SLOT_BITS (default 11; rows up to 2^SLOT_BITS elements per lane),
#      DEFS (e.g. +define+VU_NODE_NEG_EARLY_DRAIN, +define+VU_NODE_NEG_FIELD_SWAP (vu_node.sv) or
#            +define+VU_NODE_ML_NEG_MERGE_ORDER (vu_node_ml.sv, N_LANE >= 2): negative controls, must FAIL)
set -euo pipefail
REPO=$(cd "$(dirname "$0")/../.." && pwd)
VERILATOR=${VERILATOR:-$HOME/tools/verilator5/bin/verilator}
PYTHON=${PYTHON:-$HOME/lerobot/.venv/bin/python}
[ -x "$PYTHON" ] || PYTHON=python3
BUILD=${BUILD:-$REPO/build/paper_vu_node_sim}
N_LANE=${N_LANE:-0}
SUITE=${SUITE:-base}
SLOT_BITS=${SLOT_BITS:-11}
N_LD=${N_LD:-1}                     # vu_node_ml.sv operand loaders (parallel lane-group loading when > 1)
DTAG=""; [ "$N_LD" != 1 ] && DTAG="_d$N_LD"

if [ "${1:-}" = all ]; then
    rc=0
    for S in 1 2; do for MO in 1 8; do "$0" "$S" "$MO" || rc=1; done; done
    exit $rc
fi

SEED=${1:-1}; MO=${2:-8}
V=$REPO/paper/rtl/vector_unit
SRCS="$V/vu_pkg.sv $V/vu_add.sv $V/vu_mul.sv $V/vu_round.sv $V/vu_tbl.sv $V/vu_qtab.sv $V/vu_lane.sv \
      $V/vu_slot_loader.sv $V/vu_word_loader.sv $V/vu_rd_fanout.sv $V/vu_beat_writer.sv $V/vu_wr_merge.sv $V/vu_pdq.sv $V/vu_node.sv $V/vu_node_ml.sv $REPO/paper/rtl/node_sync.sv \
      $REPO/paper/rtl/colpar_prog_fetch.sv $REPO/paper/rtl/colpar_nap_mux.sv $REPO/paper/rtl/tb_axi_gddr6_model.sv \
      $V/tb_vu_node.sv"
mkdir -p "$BUILD"
TBL=$BUILD/tables
"$PYTHON" "$REPO/paper/sw/vector_unit_ref.py" --tables "$TBL" > /dev/null
if [ "$SUITE" = base ]; then VEC=$BUILD/vec_s$SEED; else VEC=$BUILD/vec_${SUITE}_s$SEED; fi
[ -f "$VEC/mem_exp.hex" ] && [ "$VEC/mem_exp.hex" -nt "$REPO/paper/sw/vu_node_golden.py" ] || \
    "$PYTHON" "$REPO/paper/sw/vu_node_golden.py" --seed "$SEED" --suite "$SUITE" --out "$VEC" > "$VEC.log"

DEFS=${DEFS:-}
TAG=$(echo "$DEFS" | md5sum | cut -c1-6)
OBJ=$BUILD/obj_n${N_LANE}${DTAG}_o${MO}_sb${SLOT_BITS}_$TAG
EXE=$OBJ/Vtb_vu_node
rebuild=0
[ -x "$EXE" ] || rebuild=1
for f in $SRCS "$TBL"/*.mem; do [ "$f" -nt "$EXE" ] && rebuild=1; done
if [ $rebuild = 1 ]; then
    rm -rf "$OBJ"
    # shellcheck disable=SC2086
    "$VERILATOR" --binary --timing -j 4 -O2 -Wno-fatal -Wno-lint -Wno-style -Wno-PINMISSING -Wno-MULTIDRIVEN \
        -Wno-UNOPTFLAT -I"$V" --top-module tb_vu_node -GMAX_OUT="$MO" -GN_LANE="$N_LANE" -GSLOT_BITS="$SLOT_BITS" -GN_LD="$N_LD" $DEFS \
        -GF_GELU="\"$TBL/vu_tbl_gelu.mem\"" -GF_SIGM="\"$TBL/vu_tbl_sigm.mem\"" \
        -GF_EXP="\"$TBL/vu_tbl_exp.mem\"" -GF_RSQRT="\"$TBL/vu_tbl_rsqrt.mem\"" -GF_QUANT="\"$TBL/vu_tbl_quant.mem\"" \
        --Mdir "$OBJ" $SRCS > "$BUILD/build_n${N_LANE}${DTAG}_o${MO}_sb${SLOT_BITS}_$TAG.log" 2>&1 \
        || { echo "BUILD FAILED (log $BUILD/build_n${N_LANE}${DTAG}_o${MO}_sb${SLOT_BITS}_$TAG.log)"; grep -E "%Error" "$BUILD/build_n${N_LANE}${DTAG}_o${MO}_sb${SLOT_BITS}_$TAG.log" | head -20; exit 1; }
fi
LOG=$BUILD/run_${SUITE}_s${SEED}_n${N_LANE}${DTAG}_o${MO}_sb${SLOT_BITS}_$TAG.log
"$EXE" +vec="$VEC" +verilator+seed+"$SEED" > "$LOG" 2>&1 || true
grep -E "MISMATCH|MISSING|RESULT" "$LOG" | head -12
echo "  suite=$SUITE seed=$SEED n_lane=$N_LANE n_ld=$N_LD slot_bits=$SLOT_BITS log=$LOG  golden: $(tail -1 "$VEC.log")"
grep -q "RESULT PASS" "$LOG"
