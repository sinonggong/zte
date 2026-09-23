#!/usr/bin/env bash
# The 3-node complete-function array (chip nodes: 0 = int8 16-stage chain, 1 = uint8 PV chain, 2 = 2-lane vector node)
# on a programmed board: bridge selftest, the compact smoke and tiny chunks with EVERY expected beat compared, then
# the WHOLE pi0 chunk (2.8 GB image over all 16 GDDR6 channels; flags + actions compared with the generator).
#   scripts/pi0_s0t_silicon_session.sh <chunk root> <prefix>     e.g. build/paper_pi0_chunk s0tq   (fused, QTAIL hardware)
#                                                                     ../pi0-chunk-gen/build/paper_pi0_chunk s0t (unfused)
# Log: build/s0_vec/session_<prefix>_<time>.log.  Stop only with SIGINT.
set -uo pipefail
ROOT=${1:?chunk root}; PFX=${2:?prefix}
REPO=$(cd "$(dirname "$0")/.." && pwd)
RUN=$REPO/build/host/pi0_chunk_run
MAP=vector:2,int8:0,uint8:1
export PI0_HOST_BRIDGE=1 PI0_FPGA_DBI_ROUTE=${PI0_FPGA_DBI_ROUTE:-comp}
mkdir -p "$REPO/build/s0_vec"
LOG=$REPO/build/s0_vec/session_${PFX}_$(date +%Y%m%dT%H%M%S).log
{
    echo "== selftest $(date +%T)"; "$RUN" selftest || exit 1
    for c in smoke_c tiny_c; do
        echo "== ${PFX}_$c $(date +%T)"
        "$RUN" run "$ROOT/${PFX}_$c" --map $MAP --timeout-s 120 --show 6; echo "== rc=$?"
    done
    echo "== ${PFX}_full $(date +%T)"
    "$RUN" run "$ROOT/${PFX}_full" --map $MAP --timeout-s 180 --repeat ${REPEAT:-2}; echo "== rc=$? $(date +%T)"
} 2>&1 | tee "$LOG"
echo "log: $LOG"
