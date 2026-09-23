#!/bin/bash
# Program an S0 / node-array bitstream on the VP815 from fics and bring the PCIe endpoint back, then identify the
# host window.  Same sequence as scripts/pi0_board_recover.sh (JTAG program with the driver unloaded, retrain to
# DLActive+, remove + rescan, 16 GT/s, insmod) but it re-programs a live board on purpose and ends with
# pi0_s0_replay identify instead of TC_MEM_MODE.  Refuses while a pi0 server holds the device.
#
#   scripts/pi0_board_program_s0.sh <bundle-relative hex>      e.g. bitstream/s0_s0b/tc_ref_design_top.hex
#   env SUDO_ASKPASS for non-interactive sudo; PI0_ALLOW_ACE_BUILD=1 to run beside an ACE build (a PCIe reset
#   has taken fics down before: memory pi0-fpga-policy-server-on-silicon; the caller accepts the risk)
set -uo pipefail
BUNDLE="${BUNDLE:-$HOME/pi0_board_bundle}"
ACE="${PI0_JTAG_ACE:-$HOME/ACE_10.3.1/Achronix-linux}"
RP="${PI0_ROOT_PORT:-0000:00:01.0}"
REPLAY="${PI0_S0_REPLAY:-$(cd "$(dirname "$0")/.." && pwd)/build/host/pi0_s0_replay}"
SUDO=(sudo); [ -n "${SUDO_ASKPASS:-}" ] && SUDO=(sudo -A)
HEX="${1:?bundle-relative hex path}"
link() { "${SUDO[@]}" lspci -vv -s "$RP" 2>/dev/null | grep -A1 "LnkSta:" | tr -s '\t ' ' ' | tr '\n' ' '; }
retrain() { local lc; lc=$("${SUDO[@]}" setpci -s "$RP" CAP_EXP+0x10.w); "${SUDO[@]}" setpci -s "$RP" CAP_EXP+0x10.w="$(printf '%04x' $(( 0x$lc | 0x20 )))"; }
allow_16gt() { local lc2; lc2=$("${SUDO[@]}" setpci -s "$RP" CAP_EXP+0x30.w); "${SUDO[@]}" setpci -s "$RP" CAP_EXP+0x30.w="$(printf '%04x' $(( (0x$lc2 & 0xfff0) | 0x4 )))"; }

"${SUDO[@]}" -v || exit 1
# exact process names / interpreter-anchored command lines only: a plain pgrep -f also matches any shell whose
# command line merely mentions these names (e.g. the caller's own)
if pgrep -x pi0_s0_replay >/dev/null || pgrep -x pi0_chunk_run >/dev/null || \
   pgrep -f "^[^ ]*python[0-9.]* [^ ]*pi0_(expert|policy)_rpc_server" >/dev/null; then
    echo "A process holds the device (pi0 server / replay / chunk run); stop it with SIGINT first."; exit 2
fi
if pgrep -f "[a]cx -batch" >/dev/null && [ "${PI0_ALLOW_ACE_BUILD:-0}" != 1 ]; then
    echo "An ACE build is running; set PI0_ALLOW_ACE_BUILD=1 to accept the PCIe-reset risk."; exit 2
fi
[ -f "$BUNDLE/$HEX" ] || { echo "no such bitstream: $BUNDLE/$HEX"; exit 1; }
sums=$(cd "$BUNDLE/$(dirname "$HEX")" && sha256sum -c --ignore-missing SHA256SUMS 2>/dev/null)
[[ "$sums" == *"$(basename "$HEX"): OK"* ]] || { echo "Bitstream checksum failed for $HEX"; echo "$sums"; exit 1; }
[ -x "$REPLAY" ] || { echo "replay tool missing: $REPLAY"; exit 1; }

echo "== 1. unload the driver, wake the root port, allow 16 GT/s ($(date +%T))"
"${SUDO[@]}" rmmod acxpcie 2>/dev/null
"${SUDO[@]}" sh -c "echo on > /sys/bus/pci/devices/$RP/power/control"
allow_16gt

echo "== 2. JTAG-program $HEX (about 1 minute)"
T=$(mktemp)
cat > "$T" <<TCL
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
out=$(cd "$BUNDLE" && ACE_INSTALL_DIR="$ACE" timeout 400 "$ACE/ace" -lab_mode -batch -script_file "$T" 2>&1)
rm -f "$T"
echo "$out" | grep -iE 'status =|all interfaces|PROG_DONE|error|fail' | tail -6
[[ "$out" == *PROG_DONE* ]] || { echo "JTAG programming failed (is the JTAG cable in a fics USB port?)"; exit 1; }

echo "== 3. retrain until DLActive+"
up=0
for attempt in 1 2 3; do
    retrain
    for _ in $(seq 10); do sleep 2; if [[ "$(link)" == *"DLActive+"* ]]; then up=1; break 2; fi; done
done
link; echo
[ $up = 1 ] || { echo "LINK DID NOT COME UP.  Not rescanning (see scripts/pi0_board_recover.sh)."; exit 1; }

echo "== 4. remove + rescan"
"${SUDO[@]}" sh -c "echo 1 > /sys/bus/pci/devices/$RP/remove"; sleep 2
"${SUDO[@]}" sh -c "echo 1 > /sys/bus/pci/rescan"; sleep 6
lspci -nnd 1b59: | grep . || { echo "No 1b59 device after the rescan."; exit 1; }

echo "== 5. force 16 GT/s"
allow_16gt; retrain; sleep 3; link; echo

echo "== 6. driver"
"${SUDO[@]}" insmod "$BUNDLE/sdk/driver/acxpcie.ko"; sleep 1
"${SUDO[@]}" chmod 666 /dev/ac7t15xx0 || { echo "Driver loaded but /dev/ac7t15xx0 is missing."; exit 1; }

echo "== 7. identify the node-array window"
# PIO, never the DMA engine: its init through the node-array bitstreams' compressed DBI gateway wedges the gateway
PI0_FPGA_DBI_ROUTE=${PI0_FPGA_DBI_ROUTE:-comp} PI0_FPGA_BULK=pio PI0_FPGA_PIO_BAR=1 PI0_FPGA_PIO_NOC_BASE=0 \
    timeout -s INT 60 "$REPLAY" identify
