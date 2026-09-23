#!/usr/bin/env bash
# The S0 acceptance matrix on the programmed board: identify, then every vector set on its node, 3 runs each.
#   scripts/pi0_s0_silicon_matrix.sh <tag> [repeat=3] [prefix=reloc_]   -> build/s0_vec/silicon_<tag>.log
# Chain sets go to node 0 (the int8 chain), vector sets to node 1 (the 2-lane vector node).  The default sets are
# the relocated ones (paper/sw/s0_vectors.py --relocate 0x10000: everything below 2 MB) moved over the PIO window
# (PI0_FPGA_BULK=pio, BAR1 -> NoC 0), because on the S0 bitstream the SDK's DMA engine init through the compressed
# DBI gateway wedges the gateway (docs/PI0_S0_BRINGUP_20260917.md §4.1).  Stop only with SIGINT.
set -uo pipefail
TAG=${1:?tag}; REP=${2:-3}; PFX=${3:-reloc_}
REPO=$(cd "$(dirname "$0")/.." && pwd)
R=$REPO/build/host/pi0_s0_replay
V=$REPO/build/s0_vec
LOG=$V/silicon_$TAG.log
export PI0_FPGA_DBI_ROUTE=${PI0_FPGA_DBI_ROUTE:-comp}
export PI0_FPGA_BULK=${PI0_FPGA_BULK:-pio} PI0_FPGA_PIO_BAR=${PI0_FPGA_PIO_BAR:-1} PI0_FPGA_PIO_NOC_BASE=${PI0_FPGA_PIO_NOC_BASE:-0}
{
  echo "== S0 silicon matrix $TAG $(date) dbi=$PI0_FPGA_DBI_ROUTE bulk=${PI0_FPGA_BULK:-dma}"
  timeout -s INT 120 "$R" identify || { echo "IDENTIFY FAILED"; exit 1; }
  for s in ${PFX}chain16_s1; do [ -d "$V/$s" ] || continue; echo "== $s node 0"; timeout -s INT 600 "$R" run "$V/$s" --node 0 --repeat "$REP" --timeout-ms 20000; done
  for s in ${PFX}vu_base_s1 ${PFX}vu_attn_s1 ${PFX}vu_ml_s1 ${PFX}vu_wide_s1; do
    [ -d "$V/$s" ] || continue
    echo "== $s node 1"; timeout -s INT 900 "$R" run "$V/$s" --node 1 --repeat "$REP" --timeout-ms 30000
  done
  echo "== status"; timeout -s INT 60 "$R" status --node 0; timeout -s INT 60 "$R" status --node 1
} 2>&1 | tee "$LOG"
grep -E "^run [0-9]+: (PASS|FAIL)|TIMEOUT|IDENTIFY" "$LOG" | sort | uniq -c
