#!/usr/bin/env bash
# Bit-exact Verilator run of the pi0 vector-unit lane (paper/rtl/vector_unit/vu_lane.sv)
# against the numpy reference (paper/sw/vector_unit_ref.py), on random and captured vectors
# (paper/sw/vector_unit_vectors.py), with random input gaps and output back-pressure.
#
# Usage: run_vu_lane_sim.sh [pos|neg_table|neg_round|all|tp] [SEED=1] [GAP=25] [STALL=20]
#   pos        the real tables and RTL: must PASS
#   neg_table  negative control: GELU table V+1 LSB in every entry: must FAIL (GELU/GEGLU cases)
#   neg_round  negative control: bf16 ties rounded half-up (+define+VU_NEG_ROUND_HALF_UP): must FAIL
#   all        the three; exit 0 only if pos passes and both controls fail
#   tp         throughput: 1 row vs 8 rows of 256 per op, no gaps, no back-pressure (log run_tp.log)
# Env: VERILATOR (default ~/tools/verilator5/bin/verilator), PYTHON (numpy; default ~/lerobot/.venv/bin/python),
#      NPZ (captured activations; default build/paper_vector_unit/acts_demo1_ep20_f02.npz, skipped if absent),
#      BUILD (default build/paper_vu_sim)
# On fics next to an FPGA build:
#   systemd-run --user --scope -q -p MemoryHigh=6G nice -n 10 paper/rtl/run_vu_lane_sim.sh all
set -euo pipefail
REPO=$(cd "$(dirname "$0")/../.." && pwd)
VERILATOR=${VERILATOR:-$HOME/tools/verilator5/bin/verilator}
PYTHON=${PYTHON:-$HOME/lerobot/.venv/bin/python}
[ -x "$PYTHON" ] || PYTHON=python3
BUILD=${BUILD:-$REPO/build/paper_vu_sim}
NPZ=${NPZ:-$REPO/build/paper_vector_unit/acts_demo1_ep20_f02.npz}
MODE=${1:-all}; SEED=${2:-1}; GAP=${3:-25}; STALL=${4:-20}
RTL=$REPO/paper/rtl/vector_unit
SRCS="$RTL/vu_pkg.sv $RTL/vu_add.sv $RTL/vu_mul.sv $RTL/vu_round.sv $RTL/vu_tbl.sv $RTL/vu_qtab.sv $RTL/vu_lane.sv $RTL/tb_vu_lane.sv"
mkdir -p "$BUILD"

# tables (and the perturbed GELU table of the negative control)
"$PYTHON" "$REPO/paper/sw/vector_unit_ref.py" --tables "$BUILD/tables" > /dev/null
mkdir -p "$BUILD/tables_neg"
cp "$BUILD"/tables/*.mem "$BUILD/tables_neg/"
"$PYTHON" - "$BUILD/tables/vu_tbl_gelu.mem" "$BUILD/tables_neg/vu_tbl_gelu.mem" <<'PY'
import sys
w = [int(l, 16) for l in open(sys.argv[1])]
open(sys.argv[2], "w").write("".join(f"{(x + (1 << 15)) & ((1 << 36) - 1):09x}\n" for x in w))   # V + 1
PY

VEC=$BUILD/vec_s$SEED
if [ ! -f "$VEC/cases.txt" ] || [ "$REPO/paper/sw/vector_unit_vectors.py" -nt "$VEC/cases.txt" ] \
   || [ "$REPO/paper/sw/vector_unit_ref.py" -nt "$VEC/cases.txt" ]; then
    NPZARG=""
    [ -f "$NPZ" ] && NPZARG="--npz $NPZ"
    # shellcheck disable=SC2086
    "$PYTHON" "$REPO/paper/sw/vector_unit_vectors.py" --out "$VEC" --seed "$SEED" $NPZARG
fi

build() {  # name tabledir defines
    local name=$1 tbl=$2 defs=$3 obj=$BUILD/obj_$1
    local exe=$obj/Vtb_vu_lane
    local rebuild=0
    [ -x "$exe" ] || rebuild=1
    for f in $SRCS "$tbl"/*.mem; do [ "$f" -nt "$exe" ] && rebuild=1; done
    if [ $rebuild = 1 ]; then
        rm -rf "$obj"
        # shellcheck disable=SC2086
        "$VERILATOR" --binary --timing -j 4 -O2 -Wno-fatal -Wno-lint -Wno-style -I"$RTL" \
            --top-module tb_vu_lane $defs \
            -GF_GELU="\"$tbl/vu_tbl_gelu.mem\"" -GF_SIGM="\"$tbl/vu_tbl_sigm.mem\"" \
            -GF_EXP="\"$tbl/vu_tbl_exp.mem\"" -GF_RSQRT="\"$tbl/vu_tbl_rsqrt.mem\"" -GF_QUANT="\"$tbl/vu_tbl_quant.mem\"" \
            --Mdir "$obj" $SRCS > "$BUILD/build_$name.log" 2>&1 \
            || { echo "BUILD FAILED $name (log $BUILD/build_$name.log)"; tail -30 "$BUILD/build_$name.log"; exit 2; }
    fi
}

run() {  # name -> prints RESULT, returns 0 on PASS
    local name=$1 log=$BUILD/run_${1}_s${SEED}.log
    "$BUILD/obj_$name/Vtb_vu_lane" +vec="$VEC" +gap="$GAP" +stall="$STALL" +verilator+seed+"$SEED" > "$log" 2>&1 || true
    grep -E "^CASE .*FAIL|MISMATCH" "$log" | head -8 || true
    grep -E "^RESULT" "$log" || echo "RESULT MISSING"
    echo "  log=$log"
    grep -q "RESULT PASS" "$log"
}

if [ "$MODE" = tp ]; then
    "$PYTHON" "$REPO/paper/sw/vector_unit_vectors.py" --out "$BUILD/vec_tp" --seed "$SEED" --throughput 256 8
    build pos "$BUILD/tables" ""
    "$BUILD/obj_pos/Vtb_vu_lane" +vec="$BUILD/vec_tp" +gap=0 +stall=0 > "$BUILD/run_tp.log" 2>&1 || true
    grep -E "^CASE|^RESULT" "$BUILD/run_tp.log"
    grep -q "RESULT PASS" "$BUILD/run_tp.log"
    exit $?
fi

rc=0
case "$MODE" in
    pos|all)
        build pos "$BUILD/tables" ""
        echo "== pos (must PASS)"
        run pos || rc=1 ;;
esac
case "$MODE" in
    neg_table|all)
        build neg_table "$BUILD/tables_neg" ""
        echo "== neg_table (must FAIL)"
        if run neg_table; then echo "NEGATIVE CONTROL DID NOT FAIL"; rc=1; fi ;;
esac
case "$MODE" in
    neg_round|all)
        build neg_round "$BUILD/tables" "+define+VU_NEG_ROUND_HALF_UP"
        echo "== neg_round (must FAIL)"
        if run neg_round; then echo "NEGATIVE CONTROL DID NOT FAIL"; rc=1; fi ;;
esac
exit $rc
