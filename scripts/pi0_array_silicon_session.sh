#!/usr/bin/env bash
# The node-array silicon session on a programmed bridge bitstream (s1j half array, s2e full array): bridge selftest,
# then chunks from paper/sw/pi0_chunk_program.py in order of size, each checked on the board.
#   scripts/pi0_array_silicon_session.sh <half|full> <chunk dir>... -> build/s0_vec/session_<tag>_<time>.log
# half: chip nodes 0-5 32-stage int8, 6-11 16-stage int8, 12-13 PV, 14-17 vector
# full: chip nodes 0-11 32-stage int8, 12-23 16-stage int8, 24-26 PV, 27-34 vector
# A chunk generated uniform 16-stage (no node_depth.txt 32 entries) maps int8 onto the 16-stage chains; a mixed one
# (--n-deep) maps int8 from chip node 0.  Env RUN_ARGS: extra pi0_chunk_run options (e.g. "--repeat 5").
# Stop only with SIGINT (Ctrl+C): the runner holds /dev/ac7t15xx0.
set -uo pipefail
ARR=${1:?half|full}; shift
REPO=$(cd "$(dirname "$0")/.." && pwd)
RUN=$REPO/build/host/pi0_chunk_run
export PI0_HOST_BRIDGE=1 PI0_FPGA_DBI_ROUTE=${PI0_FPGA_DBI_ROUTE:-comp}
mkdir -p "$REPO/build/s0_vec"
LOG=$REPO/build/s0_vec/session_${ARR}_$(date +%Y%m%dT%H%M%S).log
case $ARR in
    half) VEC=14; PV=12; DEEP0=0; SHALLOW0=6 ;;
    full) VEC=27; PV=24; DEEP0=0; SHALLOW0=12 ;;
    *) echo "half|full" >&2; exit 2 ;;
esac
{
    echo "== selftest $(date +%T)"
    "$RUN" selftest
    for d in "$@"; do
        if grep -q " int8 32$" "$d/node_depth.txt" 2>/dev/null; then map="vector:$VEC,int8:$DEEP0,uint8:$PV"
        else map="vector:$VEC,int8:$SHALLOW0,uint8:$PV"; fi
        echo "== $(basename "$d") map $map $(date +%T)"
        # shellcheck disable=SC2086
        "$RUN" run "$d" --map "$map" --timeout-s 120 ${RUN_ARGS:-}
        echo "== $(basename "$d") rc=$? $(date +%T)"
    done
} 2>&1 | tee "$LOG"
echo "log: $LOG"
