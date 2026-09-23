#!/bin/bash
# Finish board bring-up after a WARM host reboot, with the FPGA already
# JTAG-programmed before the reboot.  Does NOT reprogram.
#
# Why this exists: if fics BOOTS with a dead card (flash boot is broken, so every
# cold boot does), the kernel marks root port 0000:00:01.0 "broken device,
# retraining non-functional downstream link at 2.5GT/s", and a design
# JTAG-loaded afterwards never enumerates -- retrain, secondary bus reset and
# remove/rescan all fail (memory pcie-two-stage-programming, 2026-09-10; seen
# again 2026-09-13 with p150noc4).  AN027's fix, and the bundle README's "first
# load after power-on": program over JTAG FIRST, then warm-reboot the host with
# scripts/pi0_board_prepare_warm_reboot.sh -- NOT `systemctl --force --force
# reboot`, which hung fics on 2026-09-13 (reasons in that script).  A warm reboot
# only asserts PERST#; findings 15 says the card keeps auxiliary power and does
# not reload from flash, so BIOS enumerates the already-live endpoint at POST --
# UNVERIFIED on this host, and a 2026-09-10 note says the card reverted.  AN027
# says it can take two tries.
#
#   SUDO_ASKPASS=<helper that echoes the sudo password> scripts/pi0_board_after_warm_reboot.sh
#
# Then: scripts/pi0_board_verify_matrix.sh pio
set -uo pipefail
BUNDLE="${BUNDLE:-$HOME/pi0_board_bundle}"
RP="${PI0_ROOT_PORT:-0000:00:01.0}"
: "${SUDO_ASKPASS:?set SUDO_ASKPASS to a helper that echoes the sudo password}"

echo "== 1. is the endpoint enumerated?"
if ! lspci -nnd 1b59: | grep -q .; then
  echo "NOT ENUMERATED."
  sudo -A dmesg | grep -E "0000:00:01.0|retrain" | tail -5
  echo "AN027: JTAG-program again, then scripts/pi0_board_prepare_warm_reboot.sh again."
  echo "It can take two tries.  Do NOT power-cycle: that drops the JTAG load."
  exit 1
fi
lspci -nnd 1b59:

echo "== 2. force Gen4/16 GT/s on $RP (GDDR6 is unreliable at 2.5 GT/s)"
lc2=$(sudo -A setpci -s "$RP" CAP_EXP+0x30.w)
sudo -A setpci -s "$RP" CAP_EXP+0x30.w="$(printf '%04x' $(( (0x$lc2 & 0xfff0) | 0x4 )))"
lc=$(sudo -A setpci -s "$RP" CAP_EXP+0x10.w)
sudo -A setpci -s "$RP" CAP_EXP+0x10.w="$(printf '%04x' $(( 0x$lc | 0x20 )))"
sleep 3
sudo -A lspci -vv -s "$RP" 2>/dev/null | grep -E "LnkSta:"

echo "== 3. driver"
sudo -A rmmod acxpcie 2>/dev/null
sudo -A insmod "$BUNDLE/sdk/driver/acxpcie.ko" && sleep 1
if [ -e /dev/ac7t15xx0 ]; then
  sudo -A chmod 666 /dev/ac7t15xx0 && ls -la /dev/ac7t15xx0
else
  echo "driver loaded but /dev/ac7t15xx0 is missing"; exit 1
fi

echo "== ready only if LnkSta above says 16GT/s.  Then: scripts/pi0_board_verify_matrix.sh pio"
