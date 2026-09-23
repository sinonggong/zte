#!/usr/bin/env bash
# Desktop: serve pi0 predict_action_chunk to the Jetson's pi0_remote policy (used by the
# unchanged run_rollout.sh on the Jetson).  Same env knobs as run_pi0_fpga_policy_server.sh.
#
#   ASYNC_POLICY_PATH=~/pi0_ckpt/ur7e-demo-2-pi0-fsdp/010000/pretrained_model ./run_pi0_policy_rpc_server.sh
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LEROBOT_ROOT="${LEROBOT_ROOT:-$HOME/lerobot}"
export PATH="$LEROBOT_ROOT/.venv/bin:$PATH"
: "${ASYNC_POLICY_PATH:?set ASYNC_POLICY_PATH to the pi0 checkpoint dir}"
export ASYNC_POLICY_PATH PI0_RPC_HOST="${PI0_RPC_HOST:-0.0.0.0}" PI0_RPC_PORT="${PI0_RPC_PORT:-8081}"
export PI0_ACTION_EXPERT="${PI0_ACTION_EXPERT:-torch}" PI0_FPGA_BACKEND="${PI0_FPGA_BACKEND:-mock}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-$(nproc)}" TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}" PYTHONUNBUFFERED=1
if [ "$PI0_FPGA_BACKEND" = "real" ] && [ "${PI0_I_CHECKED_THE_BITSTREAM:-0}" != "1" ]; then
    echo "PI0_FPGA_BACKEND=real: check which bitstream is loaded, then set PI0_I_CHECKED_THE_BITSTREAM=1." >&2; exit 2
fi
if systemctl --user list-units 'pi0_*' --no-pager 2>/dev/null | grep -q running && [ "${ALLOW_ACE_BUILD:-0}" != "1" ]; then
    echo "an ACE build is running; use swapfile2 + a MemoryHigh scope and set ALLOW_ACE_BUILD=1." >&2; exit 2
fi
mkdir -p "$HOME/pi0_glue/logs"; LOG="$HOME/pi0_glue/logs/pi0_policy_rpc_$(date +%Y%m%d_%H%M%S).log"
echo "+ pi0 policy RPC server: expert=$PI0_ACTION_EXPERT backend=$PI0_FPGA_BACKEND policy=$ASYNC_POLICY_PATH port=$PI0_RPC_PORT log=$LOG" >&2
exec python "$HERE/pi0_policy_rpc_server.py" --host "$PI0_RPC_HOST" --port "$PI0_RPC_PORT" "$@" 2>&1 | tee "$LOG"
