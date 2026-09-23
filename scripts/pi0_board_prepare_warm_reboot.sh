#!/bin/bash
# The first half of AN027's "first load after power-on", made safe on fics.
# Run AFTER the card has been JTAG-programmed.  It ENDS IN A HOST REBOOT.
#
# Why not the bundle README's `sudo systemctl --force --force reboot`: on
# 2026-09-13 it hung fics.  The integrated GPU switched off (screen went grey)
# but no reboot happened -- the journal shows NetworkManager and wpa_supplicant
# still logging and three "Power key pressed" events after that, until the host
# was hard powered off, which also dropped the JTAG load.  The kernel had
# entered its shutdown path and stalled.
#
# What the three shutdowns of that night say (journal + sudo audit log):
#   hung   --force --force reboot  acxpcie LOADED      root port present (rescanned)
#   clean  plain `sudo reboot`     acxpcie not loaded  root port present (stuck, cold boot)
#   clean  orderly power-off       acxpcie not loaded  root port absent
# So the stuck root port by itself does NOT hang a reboot (an earlier version of
# this comment blamed it; the second row disproves that).  The hang is either
# --force --force skipping the orderly teardown, or acxpcie being loaded, and
# these samples cannot separate the two.  This script avoids both: it unloads
# acxpcie and reboots in order.  Removing the port too costs nothing -- BIOS
# re-enumerates the slot at POST -- and is kept as belt and braces.
#
#   SUDO_ASKPASS=<helper that echoes the sudo password> scripts/pi0_board_prepare_warm_reboot.sh
#
# Then wait for the login screen (POST with the card in the slot can take a few
# minutes with a blank screen -- give it at least 3 before assuming a hang), and
# run scripts/pi0_board_after_warm_reboot.sh.
set -uo pipefail
cd "$(dirname "$0")/.."
RP="${PI0_ROOT_PORT:-0000:00:01.0}"
: "${SUDO_ASKPASS:?set SUDO_ASKPASS to a helper that echoes the sudo password}"

if systemctl --user list-units 'pi0_*' --no-pager 2>/dev/null | grep -q running; then
  echo "a pi0 job (ACE build / policy server) is running; not rebooting."; exit 2
fi

LOG=$(ls -t build/jtag_*.log build/bringup_*.log 2>/dev/null | head -1)
if [ -n "$LOG" ] && grep -q "all interfaces ready" "$LOG"; then
  echo "latest programming log $LOG: all interfaces ready"
else
  echo "WARNING: no recent programming log reports 'all interfaces ready' ($LOG)."
  echo "         Rebooting an unprogrammed card only reproduces the dead-card boot."
fi

echo "== 1. unload acxpcie"
sudo -A rmmod acxpcie 2>/dev/null && echo "rmmod acxpcie: ok" || echo "acxpcie: not loaded"

echo "== 2. remove $RP from the kernel device tree"
[ -e "/sys/bus/pci/devices/$RP" ] && { echo 1 | sudo -A tee "/sys/bus/pci/devices/$RP/remove" >/dev/null; sleep 1; }
if [ -e "/sys/bus/pci/devices/$RP" ]; then echo "$RP is still present; refusing to reboot"; exit 1; fi
echo "$RP is out of the device tree"

echo "== 3. orderly reboot in 10 s (Ctrl-C to abort).  Do not power off during POST."
sync
sleep 10
sudo -A systemctl reboot
