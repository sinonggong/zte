#!/bin/bash
# Full board bring-up from fics (JTAG on a fics USB-A port; no laptop).
#   pi0_board_bringup_from_fics.sh <bitstream_hex>
# Steps: JTAG-program via ACE lab mode -> host remove/rescan -> force 16 GT/s
# -> insmod driver.  Then run scripts/pi0_board_verify_matrix.sh.
set -uo pipefail
HEX="${1:?usage: pi0_board_bringup_from_fics.sh <path to tc_ref_design_top.hex, relative to bundle>}"
BUNDLE="${BUNDLE:-$HOME/pi0_board_bundle}"
ACE="${ACE_INSTALL_DIR:-$HOME/ACE_10.3.1/Achronix-linux}"
RP="${PI0_ROOT_PORT:-0000:00:01.0}"
: "${SUDO_ASKPASS:?set SUDO_ASKPASS to a helper that echoes the sudo password}"
cd "$BUNDLE" || exit 1
echo "== 1. JTAG program $HEX"
T=$(mktemp); cat > "$T" <<TCL
set id [lindex [jtag::get_connected_devices] 0]
jtag::open \$id
jtag::initialize_scan_chain \$id 0 0 0 -single_device
jtag::ac7t1500_initialize_fcu \$id -reset
use_acx_device_manager -enable
jtag::ac7t1500_program_bitstream \$id $HEX
mcu_status -wait
jtag::ac7t1500_exit_fcu \$id
jtag::close \$id
puts PROG_DONE
TCL
ACE_INSTALL_DIR="$ACE" timeout 360 "$ACE/ace" -lab_mode -batch -script_file "$T" 2>&1 | grep -iE 'status =|all interfaces|PROG_DONE|error|fail'; rm -f "$T"
echo "== 2. host remove + rescan"
sudo -A rmmod acxpcie 2>/dev/null
echo 1 | sudo -A tee /sys/bus/pci/devices/$RP/remove >/dev/null; sleep 2
echo 1 | sudo -A tee /sys/bus/pci/rescan >/dev/null; sleep 6
echo "== 3. force the PCIe link to Gen4/16GT/s (GDDR6 is unreliable at 2.5 GT/s)"
lc2=$(sudo -A setpci -s "$RP" CAP_EXP+0x30.w); sudo -A setpci -s "$RP" CAP_EXP+0x30.w=$(printf '%04x' $(( (0x$lc2 & 0xfff0) | 0x4 )))
lc=$(sudo -A setpci -s "$RP" CAP_EXP+0x10.w); sudo -A setpci -s "$RP" CAP_EXP+0x10.w=$(printf '%04x' $((0x$lc | 0x20))); sleep 2
sudo -A lspci -vv -s "$RP" 2>/dev/null | grep 'LnkSta:'
echo "== 4. driver"
sudo -A insmod "$BUNDLE/sdk/driver/acxpcie.ko" && sleep 1 && sudo -A chmod 666 /dev/ac7t15xx0
lspci -nn -d 1b59:
echo "Ready.  Verify:  scripts/pi0_board_verify_matrix.sh pio"
