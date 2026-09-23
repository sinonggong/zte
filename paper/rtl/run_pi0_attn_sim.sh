#!/usr/bin/env bash
# One expert attention block of the real checkpoint across a 4-lane vector node, an int8 chain node and a
# uint8 PV chain node on shared GDDR6, bit-exact against paper/sw/pi0_attn_golden.py.  See paper/rtl/tb_pi0_attn.sv.
#
# Usage: run_pi0_attn_sim.sh [LAYER=0] [STEP=0] [TOKENS=8] [DIM=256]
#        run_pi0_attn_sim.sh all      # (layer, step) = (0,0) (9,1) (17,0)
#        FULL=1 run_pi0_attn_sim.sh [LAYER] [STEP]   # the layer at its real size (pi0_attn_golden.py --full):
#                                     # SLOT_BITS 12, column-tiled GEMMs
#        PART=lm [LM_TOKENS=N] run_pi0_attn_sim.sh [LAYER]   # a PaliGemma prefix layer at real width (hidden
#                                     # 2048, FF 16384 in 4096 sub-blocks, down K split); N = first N tokens
#        PART=siglip [IMAGE=0|1] run_pi0_attn_sim.sh [LAYER]   # a SigLIP encoder layer at real size
#                                     # (paper/sw/pi0_siglip_golden.py); LAYER 0, 13 or 26
#        PART=head run_pi0_attn_sim.sh [STEP]          # the action head + Euler update (pi0_action_head_golden.py), STEP 0 or 9
#        PART=vision [IMAGE=0|1] run_pi0_attn_sim.sh   # patch embedding; post-LN + projector (pi0_vision_ends_golden.py)
# Env: VERILATOR, PYTHON (numpy + safetensors + torch: ~/lerobot/.venv/bin/python), BUILD, N_LANE (default 4),
#      DEFS (e.g. +define+VU_NODE_NEG_NO_ROT or +define+COLPAR_LD_NEG_SEG_STEP: negative controls, must FAIL),
#      SLOT_BITS (vector node; default 11, 12 with FULL=1), TIMEOUT_PS (simulated time limit),
#      SIM_ARGS (extra plusargs, e.g. +nostall: the GDDR6 model without random ready gaps),
#      N_LD (vector node operand loaders, default 1), N_STAGE (chain node stages, 16 or 32; default 16)
set -euo pipefail
REPO=$(cd "$(dirname "$0")/../.." && pwd)
VERILATOR=${VERILATOR:-$HOME/tools/verilator5/bin/verilator}
PYTHON=${PYTHON:-$HOME/lerobot/.venv/bin/python}
[ -x "$PYTHON" ] || PYTHON=python3
BUILD=${BUILD:-$REPO/build/paper_pi0_attn_sim}
N_LANE=${N_LANE:-4}
N_LD=${N_LD:-1}
DTAG=""; [ "$N_LD" != 1 ] && DTAG="_d$N_LD"
N_STAGE=${N_STAGE:-16}
[ "$N_STAGE" != 16 ] && DTAG="${DTAG}_s$N_STAGE"
export COLPAR_N_STAGE=$N_STAGE
VTAG=""; [ "$N_STAGE" != 16 ] && VTAG="_s$N_STAGE"
FULL=${FULL:-}
PART=${PART:-exp}
[ "$PART" = lm ] && FULL=1
[ "$PART" = siglip ] && FULL=1
[ "$PART" = head ] && FULL=1
[ "$PART" = vision ] && FULL=1
SLOT_BITS=${SLOT_BITS:-${FULL:+12}}
SLOT_BITS=${SLOT_BITS:-11}

if [ "${1:-}" = all ]; then
    rc=0
    "$0" 0 0 || rc=1
    "$0" 9 1 || rc=1
    "$0" 17 0 || rc=1
    exit $rc
fi

L=${1:-0}; ST=${2:-0}; T=${3:-8}; D=${4:-256}
V=$REPO/paper/rtl/vector_unit
C=$REPO/paper/rtl
SRCS="$C/sim_models/acx_float_behav.sv $C/sim_models/acx_bram72k_behav.sv $C/sim_models/acx_mlp72_behav.sv \
      $V/vu_pkg.sv $V/vu_add.sv $V/vu_mul.sv $V/vu_round.sv $V/vu_tbl.sv $V/vu_qtab.sv $V/vu_lane.sv \
      $V/vu_slot_loader.sv $V/vu_word_loader.sv $V/vu_rd_fanout.sv $V/vu_beat_writer.sv $V/vu_wr_merge.sv $V/vu_pdq.sv $V/vu_node.sv $V/vu_node_ml.sv \
      $C/colpar_prog_fetch.sv $C/colpar_node_ctrl.sv $C/colpar_chain_node.sv $C/colpar_tile_loader.sv \
      $C/colpar_row_sequencer.sv $C/mlp72_int8_colpar_chain.sv $C/colpar_result_port.sv $C/colpar_result_writer.sv \
      $C/colpar_nap_mux.sv $C/node_sync.sv $C/tb_axi_gddr6_model.sv $C/tb_pi0_attn.sv"
mkdir -p "$BUILD"
TBL=$BUILD/tables
"$PYTHON" "$REPO/paper/sw/vector_unit_ref.py" --tables "$TBL" > /dev/null
if [ "$PART" = lm ]; then
    VEC=$BUILD/vec_lm_L${L}_t${LM_TOKENS:-all}${VTAG}
    SIZE=lm_t${LM_TOKENS:-all}
    GARGS="--part lm --lm-tokens ${LM_TOKENS:-0}"
elif [ -n "$FULL" ]; then
    VEC=$BUILD/vec_L${L}_s${ST}_full${VTAG}
    SIZE=full
    GARGS="--full"
else
    VEC=$BUILD/vec_L${L}_s${ST}_t${T}_d${D}${VTAG}
    SIZE=t${T}_d${D}
    GARGS="--tokens $T --dim $D"
fi
if [ "$PART" = head ]; then
    VEC=$BUILD/vec_head_s${L}${VTAG}
    SIZE=head
    "$PYTHON" "$REPO/paper/sw/pi0_action_head_golden.py" --step "$L" --out "$VEC" > "$VEC.log"
elif [ "$PART" = vision ]; then
    VEC=$BUILD/vec_vision_i${IMAGE:-0}${VTAG}
    SIZE=vision_i${IMAGE:-0}
    "$PYTHON" "$REPO/paper/sw/pi0_vision_ends_golden.py" --image "${IMAGE:-0}" --out "$VEC" > "$VEC.log"
elif [ "$PART" = siglip ]; then
    VEC=$BUILD/vec_vis_L${L}_i${IMAGE:-0}_${O_PROJ:-head}${VTAG}
    SIZE=vis_i${IMAGE:-0}_${O_PROJ:-head}
    "$PYTHON" "$REPO/paper/sw/pi0_siglip_golden.py" --layer "$L" --image "${IMAGE:-0}" --slot-bits "$SLOT_BITS" --o-proj "${O_PROJ:-head}" \
        --out "$VEC" > "$VEC.log"
else
    # shellcheck disable=SC2086
    "$PYTHON" "$REPO/paper/sw/pi0_attn_golden.py" --layer "$L" --step "$ST" $GARGS --slot-bits "$SLOT_BITS" \
        --tiles ${TILES:-1} ${BARRIER:+--barrier} --out "$VEC" > "$VEC.log"
fi

DEFS=${DEFS:-}
TAG=$(echo "$DEFS" | md5sum | cut -c1-6)
OBJ=$BUILD/obj_n${N_LANE}${DTAG}_sb${SLOT_BITS}_$TAG
EXE=$OBJ/Vtb_pi0_attn
rebuild=0
[ -x "$EXE" ] || rebuild=1
for f in $SRCS "$TBL"/*.mem; do [ "$f" -nt "$EXE" ] && rebuild=1; done
if [ $rebuild = 1 ]; then
    rm -rf "$OBJ"
    # shellcheck disable=SC2086
    "$VERILATOR" --binary --timing -j 4 -O2 -Wno-fatal -Wno-lint -Wno-style -Wno-PINMISSING -Wno-MULTIDRIVEN \
        -Wno-UNOPTFLAT -I"$V" --top-module tb_pi0_attn -GN_LANE="$N_LANE" -GSLOT_BITS="$SLOT_BITS" -GN_LD="$N_LD" -GN_STAGE="$N_STAGE" $DEFS \
        -GF_GELU="\"$TBL/vu_tbl_gelu.mem\"" -GF_SIGM="\"$TBL/vu_tbl_sigm.mem\"" \
        -GF_EXP="\"$TBL/vu_tbl_exp.mem\"" -GF_RSQRT="\"$TBL/vu_tbl_rsqrt.mem\"" -GF_QUANT="\"$TBL/vu_tbl_quant.mem\"" \
        --Mdir "$OBJ" $SRCS > "$BUILD/build_n${N_LANE}${DTAG}_sb${SLOT_BITS}_$TAG.log" 2>&1 \
        || { echo "BUILD FAILED (log $BUILD/build_n${N_LANE}${DTAG}_sb${SLOT_BITS}_$TAG.log)"; grep -E "%Error" "$BUILD/build_n${N_LANE}${DTAG}_sb${SLOT_BITS}_$TAG.log" | head -20; exit 1; }
fi
LOG=$BUILD/run_L${L}_s${ST}_${SIZE}_n${N_LANE}${DTAG}_sb${SLOT_BITS}_$TAG${SIM_ARGS:+_$(echo "$SIM_ARGS" | tr -dc 'a-z0-9')}.log
# shellcheck disable=SC2086
"$EXE" +vec="$VEC" +verilator+seed+1 ${TIMEOUT_PS:++timeout_ps=$TIMEOUT_PS} ${SIM_ARGS:-} > "$LOG" 2>&1 || true
grep -E "MISMATCH|MISSING|NODE ERROR|RESULT" "$LOG" | head -14
echo "  layer=$L step=$ST log=$LOG"
grep -E "chip vs fp32" "$VEC.log" || true
grep -q "RESULT PASS" "$LOG"
