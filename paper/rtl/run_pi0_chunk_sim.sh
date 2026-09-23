#!/usr/bin/env bash
# The whole pi0 chunk (or a cut-down one) on a reduced node array in Verilator, host-free, bit-exact against
# paper/sw/pi0_chunk_program.py.  See paper/rtl/tb_pi0_chunk.sv.
#
# Usage: run_pi0_chunk_sim.sh <vec_dir_name> [generator args...]
#   e.g. run_pi0_chunk_sim.sh tiny --siglip-layers 1 --prefix-layers 1 --expert-layers 1 --steps 1
# Env: N_VN / N_CH / N_PV (default 2 / 2 / 1: must match --n-vec / --n-chain / --n-pv, which the runner passes), N_DEEP (0),
#      STRIPE (12: the chip's GDDR6 channel striping in front of every node; PER_NODE_MEM=1 only),
#      DEFS (extra Verilator options, e.g. +define+NODE_DRAIN_ONLY), PER_NODE_MEM (0: one shared NAP; 1: a memory port per node, as on the chip),
#      N_LANE (2), N_LD (2), SLOT_BITS (12), N_STAGE (16), MAX_OUT (8), SIM_ARGS (+nostall), TIMEOUT_PS, BUILD,
#      GEN=0 to reuse an existing vector directory, VERILATOR, PYTHON
set -euo pipefail
REPO=$(cd "$(dirname "$0")/../.." && pwd)
VERILATOR=${VERILATOR:-$HOME/tools/verilator5/bin/verilator}
PYTHON=${PYTHON:-$HOME/lerobot/.venv/bin/python}
[ -x "$PYTHON" ] || PYTHON=python3
BUILD=${BUILD:-$REPO/build/paper_pi0_chunk}
N_VN=${N_VN:-2}; N_CH=${N_CH:-2}; N_PV=${N_PV:-1}
N_LANE=${N_LANE:-2}; N_LD=${N_LD:-2}; SLOT_BITS=${SLOT_BITS:-12}; N_STAGE=${N_STAGE:-16}; MAX_OUT=${MAX_OUT:-8}
N_DEEP=${N_DEEP:-0}      # the first N_DEEP int8 chains are 32-stage (pi0_chip_top's mixed array)
NAME=${1:?vector directory name}; shift
V=$REPO/paper/rtl/vector_unit
C=$REPO/paper/rtl
SRCS="$C/sim_models/acx_float_behav.sv $C/sim_models/acx_bram72k_behav.sv $C/sim_models/acx_mlp72_behav.sv \
      $V/vu_pkg.sv $V/vu_add.sv $V/vu_mul.sv $V/vu_round.sv $V/vu_tbl.sv $V/vu_qtab.sv $V/vu_lane.sv \
      $V/vu_slot_loader.sv $V/vu_word_loader.sv $V/vu_rd_fanout.sv $V/vu_beat_writer.sv $V/vu_wr_merge.sv $V/vu_pdq.sv $V/vu_node.sv $V/vu_node_ml.sv \
      $C/colpar_prog_fetch.sv $C/colpar_node_ctrl.sv $C/colpar_chain_node.sv $C/colpar_tile_loader.sv \
      $C/colpar_row_sequencer.sv $C/mlp72_int8_colpar_chain.sv $C/colpar_result_port.sv $C/colpar_result_writer.sv \
      $C/colpar_nap_mux.sv $C/axi_stripe.sv $C/axi_id_reorder.sv $C/node_sync.sv $C/tb_axi_gddr6_model.sv $C/tb_pi0_chunk.sv"
mkdir -p "$BUILD"
TBL=$BUILD/tables
# regenerate the ROM tables into a scratch dir and install only the ones that changed, so a run does not force a
# Verilator rebuild (and two concurrent runs do not both rm -rf the shared object dir)
TBLNEW=$(mktemp -d)
"$PYTHON" "$REPO/paper/sw/vector_unit_ref.py" --tables "$TBLNEW" > /dev/null
mkdir -p "$TBL"
for f in "$TBLNEW"/*.mem; do cmp -s "$f" "$TBL/$(basename "$f")" || cp "$f" "$TBL/"; done
rm -rf "$TBLNEW"
VEC=$BUILD/$NAME
if [ "${GEN:-1}" = 1 ]; then
    # shellcheck disable=SC2068
    "$PYTHON" "$REPO/paper/sw/pi0_chunk_program.py" --out "$VEC" --n-vec "$N_VN" --n-chain "$N_CH" --n-pv "$N_PV" \
        --n-stage "$N_STAGE" --n-deep "$N_DEEP" --slot-bits "$SLOT_BITS" --hex $@ > "$VEC.log" 2>&1 || { echo "GENERATOR FAILED (log $VEC.log)"; tail -5 "$VEC.log"; exit 1; }
    tail -1 "$VEC.log"
fi
KSUF=""; [ "$N_DEEP" != 0 ] && KSUF=_k$N_DEEP
[ "${PER_NODE_MEM:-0}" != 0 ] && KSUF=${KSUF}_pn
[ "${STRIPE:-0}" != 0 ] && KSUF=${KSUF}_st${STRIPE}
[ "${VN_RD_FIFO_LOG2:-7}" != 7 ] && KSUF=${KSUF}_vf${VN_RD_FIFO_LOG2}
[ -n "${DEFS:-}" ] && KSUF=${KSUF}_$(echo "$DEFS" | md5sum | cut -c1-6)
OBJ=$BUILD/obj_v${N_VN}c${N_CH}p${N_PV}_n${N_LANE}_d${N_LD}_sb${SLOT_BITS}_s${N_STAGE}_o${MAX_OUT}${KSUF}
EXE=$OBJ/Vtb_pi0_chunk
rebuild=0
[ -x "$EXE" ] || rebuild=1
for f in $SRCS "$TBL"/*.mem; do [ "$f" -nt "$EXE" ] && rebuild=1; done
if [ $rebuild = 1 ]; then
    rm -rf "$OBJ"
    "$VERILATOR" --binary --timing -j 4 -O2 -Wno-fatal -Wno-lint -Wno-style -Wno-PINMISSING -Wno-MULTIDRIVEN \
        -Wno-UNOPTFLAT -I"$V" --top-module tb_pi0_chunk -GN_VN="$N_VN" -GN_CH="$N_CH" -GN_PV="$N_PV" \
        -GN_LANE="$N_LANE" -GSLOT_BITS="$SLOT_BITS" -GN_LD="$N_LD" -GN_STAGE="$N_STAGE" -GN_DEEP="$N_DEEP" -GMAX_OUT="$MAX_OUT" -GPER_NODE_MEM="${PER_NODE_MEM:-0}" -GSTRIPE="${STRIPE:-0}" -GVN_RD_FIFO_LOG2="${VN_RD_FIFO_LOG2:-7}" \
        -GF_GELU="\"$TBL/vu_tbl_gelu.mem\"" -GF_SIGM="\"$TBL/vu_tbl_sigm.mem\"" \
        -GF_EXP="\"$TBL/vu_tbl_exp.mem\"" -GF_RSQRT="\"$TBL/vu_tbl_rsqrt.mem\"" -GF_QUANT="\"$TBL/vu_tbl_quant.mem\"" \
        --Mdir "$OBJ" ${DEFS:-} $SRCS > "$OBJ.build.log" 2>&1 \
        || { echo "BUILD FAILED (log $OBJ.build.log)"; grep -E "%Error" "$OBJ.build.log" | head -20; exit 1; }
fi
LOG=$VEC.run${SIM_ARGS:+_$(echo "$SIM_ARGS" | tr -dc 'a-z0-9')}.log
# shellcheck disable=SC2086
"$EXE" +vec="$VEC" +verilator+seed+1 ${TIMEOUT_PS:++timeout_ps=$TIMEOUT_PS} ${SIM_ARGS:-} > "$LOG" 2>&1 || true
grep -E "MISMATCH|MISSING|ALL HALTED|RESULT" "$LOG" | head -16
echo "  vec=$VEC log=$LOG"
grep -q "RESULT PASS" "$LOG"
