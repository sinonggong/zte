#!/usr/bin/env bash
# Every gate of the node array at pi0's real size, with the outcome each must have (negative controls must FAIL).
# Prints one line per gate and a summary; exit status 0 only if every gate has its intended outcome.
#
# Usage: run_pi0_full_size_gates.sh [quick|full]
#   quick  vector-node suites, chain-node suites, cut-down layers, expert layer at full size, action head, vision ends,
#          a SigLIP-layer + action-head chunk on a reduced array (host-free),
#          SigLIP layer (token-major), one negative control per node kind                 (~25 min, < 3 GB)
#   full   quick + prefix layer at 525 tokens, host-free full-size layers, both vector-node sizes (4-lane / 2-lane),
#          a chunk with one layer of every kind and one Euler step on the reduced array,
#          SigLIP per-head form, more negative controls                                    (~2.5 h, ~6 GB per prefix run)
# Env: BUILD_ROOT (default build/paper_gates), N_LD / N_LANE are set per gate.
# Beside an ACE route, run inside a capped scope:
#   nohup systemd-run --user --scope -p MemoryMax=7G nice -n 10 paper/rtl/run_pi0_full_size_gates.sh full > log 2>&1 &
set -u
REPO=$(cd "$(dirname "$0")/../.." && pwd)
cd "$REPO"
MODE=${1:-quick}
ROOT=${BUILD_ROOT:-$REPO/build/paper_gates}
mkdir -p "$ROOT"
n_ok=0; n_bad=0; report=""

gate() {   # gate <name> <PASS|FAIL> <command...>
    local name=$1 want=$2; shift 2
    local t0 got
    t0=$(date +%s)
    if "$@" > "$ROOT/$name.log" 2>&1; then got=PASS; else got=FAIL; fi
    local line
    line=$(printf "%-44s want %-4s got %-4s %5ss" "$name" "$want" "$got" "$(( $(date +%s) - t0 ))")
    if [ "$got" = "$want" ]; then n_ok=$((n_ok + 1)); line="ok   $line"; else n_bad=$((n_bad + 1)); line="BAD  $line"; fi
    echo "$line"
    report="$report$line"$'\n'
}
attn() { env BUILD="$ROOT/attn_$1" "${@:2}"; }

# ---- vector node (4 lanes / 4 loaders is the chip's; 2 / 2 the most tile-efficient) ----
for cfg in "4 4" "2 2" "1 1"; do set -- $cfg
    for su in base ml attn wide; do
        gate "vu_${su}_l$1_d$2" PASS env BUILD="$ROOT/vu" SUITE=$su SLOT_BITS=12 N_LANE=$1 N_LD=$2 ./paper/rtl/run_vu_node_sim.sh 1 8
    done
done
gate vu_wide_sb11_rejects                     FAIL env BUILD="$ROOT/vu" SUITE=wide SLOT_BITS=11 N_LANE=4 N_LD=4 ./paper/rtl/run_vu_node_sim.sh 1 8
gate vu_neg_merge_order                       FAIL env BUILD="$ROOT/vu" SUITE=ml SLOT_BITS=12 N_LANE=4 N_LD=4 DEFS=+define+VU_NODE_ML_NEG_MERGE_ORDER ./paper/rtl/run_vu_node_sim.sh 1 8
gate vu_neg_delay_ram                         FAIL env BUILD="$ROOT/vu" SUITE=base SLOT_BITS=12 N_LANE=4 N_LD=4 DEFS=+define+VU_LANE_NEG_DELAY_RAM ./paper/rtl/run_vu_node_sim.sh 1 8

# ---- chain node ----
gate chain_prog_all    PASS ./paper/rtl/run_colpar_prog_sim.sh all
gate chain_node_all    PASS ./paper/rtl/run_colpar_node_sim.sh all
gate chain_nap_share   PASS ./paper/rtl/run_colpar_nap_share_sim.sh all
gate chain_node_all_orig_writer PASS env WR_SLIM=0 ./paper/rtl/run_colpar_node_sim.sh all
gate chain_prog_32     PASS env WR_PIPE=4 VALID_COPIES=4 ./paper/rtl/run_colpar_prog_sim.sh 32 16 1 8
gate chain_prog_32_gddr6 PASS env WR_PIPE=4 VALID_COPIES=4 ./paper/rtl/run_colpar_prog_sim.sh 32 16 1 8 1

# ---- layers ----
gate layer_cutdown_all                        PASS attn cut ./paper/rtl/run_pi0_attn_sim.sh all
gate layer_cutdown_barrier_tiles2             PASS attn cut env TILES=2 BARRIER=1 N_LD=4 SLOT_BITS=12 ./paper/rtl/run_pi0_attn_sim.sh 9 1
gate layer_cutdown_s32                        PASS attn cut env N_STAGE=32 ./paper/rtl/run_pi0_attn_sim.sh 0 0
gate layer_cutdown_neg_wr_beat_order          FAIL attn cut env DEFS=+define+COLPAR_WR_NEG_BEAT_ORDER ./paper/rtl/run_pi0_attn_sim.sh 0 0
gate expert_full_l4d4                         PASS attn full env FULL=1 N_LD=4 SIM_ARGS=+nostall ./paper/rtl/run_pi0_attn_sim.sh 0 0
gate expert_full_neg_seg_step                 FAIL attn full env FULL=1 N_LD=4 DEFS=+define+COLPAR_LD_NEG_SEG_STEP ./paper/rtl/run_pi0_attn_sim.sh 0 0
gate expert_full_s32_l2d2                     PASS attn s32 env N_STAGE=32 FULL=1 N_LANE=2 N_LD=2 SIM_ARGS=+nostall ./paper/rtl/run_pi0_attn_sim.sh 0 0
gate expert_full_s32_neg_target5              FAIL attn s32 env N_STAGE=32 FULL=1 N_LD=4 DEFS=+define+COLPAR_LD_NEG_TARGET5 ./paper/rtl/run_pi0_attn_sim.sh 0 0
gate siglip_token_l4d4                        PASS attn full env PART=siglip O_PROJ=token N_LD=4 SIM_ARGS=+nostall ./paper/rtl/run_pi0_attn_sim.sh 0
gate action_head_s0                           PASS attn full env PART=head N_LD=4 ./paper/rtl/run_pi0_attn_sim.sh 0
gate vision_ends_cam1                         PASS attn full env PART=vision IMAGE=1 N_LD=4 ./paper/rtl/run_pi0_attn_sim.sh 0

# ---- the chunk, host-free, on a reduced array (paper/sw/pi0_chunk_program.py, tb_pi0_chunk.sv): stages sequenced
# by GDDR6 flags, chain stages split by column tile and vector stages by rows over the nodes ----
gate chunk_smoke_vis_head                     PASS env BUILD="$ROOT/chunk" SIM_ARGS=+nostall ./paper/rtl/run_pi0_chunk_sim.sh smoke --siglip-layers 1 --prefix-layers 0 --expert-layers 0 --steps 1

if [ "$MODE" = full ]; then
    gate chunk_tiny_all_kinds                 PASS env BUILD="$ROOT/chunk" SIM_ARGS=+nostall TIMEOUT_PS=4000000000000000 ./paper/rtl/run_pi0_chunk_sim.sh tiny --siglip-layers 1 --prefix-layers 1 --expert-layers 1 --steps 1
    gate expert_full_l9s1                     PASS attn full env FULL=1 N_LD=4 SIM_ARGS=+nostall ./paper/rtl/run_pi0_attn_sim.sh 9 1
    gate expert_full_neg_shared_addr          FAIL attn full env FULL=1 N_LD=4 DEFS=+define+COLPAR_NEG_SHARED_ADDR ./paper/rtl/run_pi0_attn_sim.sh 0 0
    gate expert_full_barrier                  PASS attn barrier env FULL=1 N_LD=4 BARRIER=1 ./paper/rtl/run_pi0_attn_sim.sh 0 0
    gate expert_full_l2d2                     PASS attn l2 env FULL=1 N_LANE=2 N_LD=2 SIM_ARGS=+nostall ./paper/rtl/run_pi0_attn_sim.sh 0 0
    gate siglip_head_l26_cam1                 PASS attn full env PART=siglip O_PROJ=head IMAGE=1 N_LD=4 ./paper/rtl/run_pi0_attn_sim.sh 26
    gate siglip_neg_merge_order               FAIL attn full env PART=siglip O_PROJ=token N_LD=4 DEFS=+define+VU_NODE_ML_NEG_MERGE_ORDER ./paper/rtl/run_pi0_attn_sim.sh 0
    gate action_head_s9_l2d2                  PASS attn l2 env PART=head N_LANE=2 N_LD=2 ./paper/rtl/run_pi0_attn_sim.sh 9
    gate prefix_t160_barrier                  PASS attn barrier env PART=lm LM_TOKENS=160 N_LD=4 BARRIER=1 ./paper/rtl/run_pi0_attn_sim.sh 0
    gate prefix_t525_l4d4                     PASS attn full env PART=lm N_LD=4 SIM_ARGS=+nostall TIMEOUT_PS=2000000000000000 ./paper/rtl/run_pi0_attn_sim.sh 0
    gate prefix_t525_l17_l2d2                 PASS attn l2 env PART=lm N_LANE=2 N_LD=2 SIM_ARGS=+nostall TIMEOUT_PS=2000000000000000 ./paper/rtl/run_pi0_attn_sim.sh 17
    gate prefix_t525_s32_l2d2                 PASS attn s32 env N_STAGE=32 PART=lm N_LANE=2 N_LD=2 SIM_ARGS=+nostall TIMEOUT_PS=2000000000000000 ./paper/rtl/run_pi0_attn_sim.sh 0
    gate siglip_token_s32_l2d2                PASS attn s32 env N_STAGE=32 PART=siglip O_PROJ=token N_LANE=2 N_LD=2 SIM_ARGS=+nostall ./paper/rtl/run_pi0_attn_sim.sh 0
fi

echo
echo "gates: $n_ok with the intended outcome, $n_bad not ($MODE; logs in $ROOT)"
printf "%s" "$report" > "$ROOT/summary_$MODE.txt"
[ "$n_bad" = 0 ]
