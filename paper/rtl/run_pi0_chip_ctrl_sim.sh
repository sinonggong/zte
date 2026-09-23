#!/usr/bin/env bash
# Host-window protocol / register check: tb_pi0_chip_ctrl.sv on pi0_chip_ctrl.sv (Verilator 5, --timing).
set -euo pipefail
REPO=$(cd "$(dirname "$0")/../.." && pwd)
VERILATOR=${VERILATOR:-$HOME/tools/verilator5/bin/verilator}
W=${BUILD:-$REPO/build/paper_chip_ctrl_sim}
mkdir -p "$W"; cd "$W"
"$VERILATOR" --binary --timing -j 4 -O2 -Wno-fatal -Wno-lint -Wno-style -Wno-WIDTH --top-module tb_pi0_chip_ctrl \
    "$REPO/paper/rtl/pi0_chip_ctrl.sv" "$REPO/paper/rtl/tb_pi0_chip_ctrl.sv" -o Vtb > build.log 2>&1
./obj_dir/Vtb 2>&1 | grep -E "RESULT|FAIL"
