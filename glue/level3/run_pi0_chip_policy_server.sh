#!/usr/bin/env bash
# Desktop: serve pi0 with the WHOLE model on the node array (PI0_ACTION_EXPERT=chip) to the Jetson's pi0_remote
# policy, through the same RPC server as the other backends.  The chunk image must already be on the board:
#   PI0_HOST_BRIDGE=1 PI0_FPGA_DBI_ROUTE=comp build/host/pi0_chunk_run run <chunk> --map <map>   (loads + checks)
#
#   PI0_CHIP_CHUNK=<generated chunk dir> PI0_CHIP_MAP=vector:14,int8:0,uint8:12 \
#   ASYNC_POLICY_PATH=~/pi0_ckpt/ur7e-demo-2-pi0-fsdp/010000/pretrained_model ./run_pi0_chip_policy_server.sh
# Env: PI0_CHUNK_RUN (the host runner binary), PI0_RPC_PORT (8081).  The policy is memory-mapped (only the embedding
# tables, the patch convolution and state_proj are touched), so it runs beside ACE builds.  Stop with Ctrl+C.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
: "${PI0_CHIP_CHUNK:?set PI0_CHIP_CHUNK to the generated chunk dir loaded on the board}"
: "${PI0_CHIP_MAP:?set PI0_CHIP_MAP (half array: vector:14,int8:0,uint8:12; full: vector:27,int8:0,uint8:24)}"
export PI0_ACTION_EXPERT=chip PI0_CHIP_CHUNK PI0_CHIP_MAP
export PI0_CHUNK_RUN="${PI0_CHUNK_RUN:-$HERE/../../build/host/pi0_chunk_run}"
export PI0_HOST_BRIDGE=1 PI0_FPGA_DBI_ROUTE="${PI0_FPGA_DBI_ROUTE:-comp}" OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
[ -x "$PI0_CHUNK_RUN" ] || { echo "no runner at $PI0_CHUNK_RUN" >&2; exit 2; }
exec "$HERE/run_pi0_policy_rpc_server.sh" "$@"
