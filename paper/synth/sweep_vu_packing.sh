#!/usr/bin/env bash
# Does the 4-lane vector node have to be 3,309 RLB tiles?
#
# It holds 21,631 LUTs and 32,190 DFFs, and an RLB tile takes 12 LUTs and 24 DFFs, so 3,309 tiles are
# only 54 % full of LUTs and 40 % full of DFFs.  The same 1.83x spread shows up on the whole chip
# (27.7 % of LUT sites occupying 50.85 % of tiles), and RLB occupancy -- not LUT count -- is what the
# 55 % routability line is about.  So the question is whether the spread is the placer's choice, the
# netlist's size, or the synthesis options, which were fixed for the FIRST single-lane experiment and
# never rechecked: -maxfan 40, -resource_sharing 0, -retiming 0.
#
# Each run is synthesis + place + route of one node, a few minutes.  Results land in
# build/paper_vu_synth/<name>/rev_1/pnr/reports.
#
# Usage: sweep_vu_packing.sh [N_LANE=4] [PERIOD_NS=4.0]
set -euo pipefail
NL=${1:-4}; PER=${2:-4.0}
REPO=$(cd "$(dirname "$0")/../.." && pwd)
cd "$REPO"

run() {   # name, then the env assignments
    local name=$1; shift
    if [ -d "build/paper_vu_synth/$name" ] && \
       [ -f "build/paper_vu_synth/$name/rev_1/pnr/reports/vn_top_utilization_routed.txt" ]; then
        echo "[skip] $name already built"; return
    fi
    echo "[$(date +%T)] $name: $*"
    env "$@" nice -n 12 paper/synth/run_vu_node_synth.sh "$name" "$PER" 11 6 "$NL" \
        > "build/paper_vu_synth/$name.run.log" 2>&1 || echo "  FAILED (see $name.run.log)"
}

mkdir -p build/paper_vu_synth
run pk_base    SYN_MAXFAN=40  SYN_RSHARE=0 SYN_RETIME=0
run pk_rshare  SYN_MAXFAN=40  SYN_RSHARE=1 SYN_RETIME=0
run pk_retime  SYN_MAXFAN=40  SYN_RSHARE=0 SYN_RETIME=1
run pk_maxfan  SYN_MAXFAN=200 SYN_RSHARE=0 SYN_RETIME=0
run pk_all     SYN_MAXFAN=200 SYN_RSHARE=1 SYN_RETIME=1
run pk_seed7   SYN_MAXFAN=40  SYN_RSHARE=0 SYN_RETIME=0 ACE_SEED=7
run pk_seed11  SYN_MAXFAN=40  SYN_RSHARE=0 SYN_RETIME=0 ACE_SEED=11

echo
printf "%-12s %8s %8s %8s %8s %10s %s\n" name RLB LUT DFF BRAM "LUT/tile" "worst setup slack"
for n in pk_base pk_rshare pk_retime pk_maxfan pk_all pk_seed7 pk_seed11; do
    R=build/paper_vu_synth/$n/rev_1/pnr/reports
    [ -f "$R/vn_top_utilization_routed.txt" ] || { printf "%-12s %8s\n" "$n" "-"; continue; }
    rlb=$(grep -hE "^   RLB Tiles \(Occupied\)" "$R/vn_top_utilization_routed.txt" | awk '{print $4}')
    lut=$(grep -hE "^   LUT Total" "$R/vn_top_utilization_routed.txt" | awk '{print $5}')
    dff=$(grep -hE "^   DFF Total" "$R/vn_top_utilization_routed.txt" | awk '{print $5}')
    bram=$(grep -hE "^   BRAM Total" "$R/vn_top_utilization_routed.txt" | awk '{print $5}')
    slack=$(grep -hA3 "Clock / Group" "$R"/vn_top_timing_routed_C1_0p90V_0C.txt 2>/dev/null \
            | awk '/i_clk/{print $NF; exit}')
    printf "%-12s %8s %8s %8s %8s %10.1f %s\n" "$n" "$rlb" "$lut" "$dff" "$bram" \
        "$(python3 -c "print($lut/max(1,$rlb))")" "${slack:-?}"
done
