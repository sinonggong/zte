#!/bin/bash
# Check the VP815, or bring it back WITHOUT rebooting fics.
#
#   scripts/pi0_board_recover.sh --check        report only: device, link, driver node, TC_MEM_MODE
#   scripts/pi0_board_recover.sh [hex]          recover; hex is relative to ~/pi0_board_bundle (default p150noc4)
#
# Recovery = wake root port 0000:00:01.0 and allow 16 GT/s -> JTAG-program -> retrain until DLActive+ -> only then
# remove + rescan -> force 16 GT/s -> insmod -> TC_MEM_MODE=1 (BAR0+0x244, needed on every fresh load of a
# NoC-weights bitstream).  Done by hand on 2026-09-13 21:25 PDT after fics had booted with a dead card and the kernel
# had marked the root port "broken device" (memory pi0-board-bringup-from-fics).  If the link never comes up this
# script stops before the rescan, because a rescan onto a dead link is what marks the port broken; the fallback is
# the warm reboot path (scripts/pi0_board_prepare_warm_reboot.sh).  Refuses while an ACE build or a pi0 server runs.
# Asks for the sudo password once in a terminal (or set SUDO_ASKPASS).
set -uo pipefail
BUNDLE="${BUNDLE:-$HOME/pi0_board_bundle}"
ACE="${ACE_INSTALL_DIR:-$HOME/ACE_10.3.1/Achronix-linux}"
RP="${PI0_ROOT_PORT:-0000:00:01.0}"
SUDO=(sudo); [ -n "${SUDO_ASKPASS:-}" ] && SUDO=(sudo -A)
PEEK=("$BUNDLE/sdk/tool.sh" acx_pcie_peek_poke -i 0)

link() { "${SUDO[@]}" lspci -vv -s "$RP" 2>/dev/null | grep -A1 "LnkSta:" | tr -s '\t ' ' ' | tr '\n' ' '; }
retrain() {
    local lc; lc=$("${SUDO[@]}" setpci -s "$RP" CAP_EXP+0x10.w)
    "${SUDO[@]}" setpci -s "$RP" CAP_EXP+0x10.w="$(printf '%04x' $(( 0x$lc | 0x20 )))"
}
allow_16gt() {
    local lc2; lc2=$("${SUDO[@]}" setpci -s "$RP" CAP_EXP+0x30.w)
    "${SUDO[@]}" setpci -s "$RP" CAP_EXP+0x30.w="$(printf '%04x' $(( (0x$lc2 & 0xfff0) | 0x4 )))"
}
report() {
    local dev lnk node mode="-" ready=1
    dev=$(lspci -nnd 1b59: | head -1); lnk=$(link)
    node=$(ls -la /dev/ac7t15xx0 2>/dev/null)
    [ -e /dev/ac7t15xx0 ] && mode=$("${PEEK[@]}" read bar0 0x244 2>&1 | tail -1)
    echo "device      : ${dev:-NONE (no 1b59 device on the bus)}"
    echo "link        : ${lnk:-unknown}"
    echo "driver node : ${node:-NONE (acxpcie not loaded)}"
    echo "TC_MEM_MODE : $mode"
    [ -n "$dev" ] || ready=0
    [[ "$lnk" == *"16GT/s"*"DLActive+"* ]] || ready=0
    [ -n "$node" ] || ready=0
    [ "$mode" = "0x00000001" ] || ready=0
    if [ $ready = 1 ]; then echo "BOARD READY"; else echo "BOARD NOT READY"; fi
    [ $ready = 1 ]
}

if [ "${1:-}" = "--check" ]; then report; exit $?; fi
HEX="${1:-bitstream/ours_p150noc4_150MHz_reg64_nocwgt/tc_ref_design_top.hex}"

"${SUDO[@]}" -v || exit 1
if report >/dev/null 2>&1; then
    report; echo "The board is already up; nothing to do (re-programming would kill a running server)."; exit 0
fi
if pgrep -f "[a]cx -batch|[m]_generic" >/dev/null; then
    echo "An ACE build is running; refusing (PCIe changes can reset fics)."; exit 2
fi
if pgrep -f "[p]i0_expert_rpc_server|[p]i0_policy_rpc_server" >/dev/null; then
    echo "A pi0 server is running; stop it first (Ctrl+C in the \"pi0 FPGA server\" window)."; exit 2
fi
# capture first: under pipefail, `sha256sum | grep -q` fails when grep closes the pipe early or another listed file differs
sums=$(cd "$BUNDLE/$(dirname "$HEX")" && sha256sum -c --ignore-missing SHA256SUMS 2>/dev/null)
[[ "$sums" == *"$(basename "$HEX"): OK"* ]] || { echo "Bitstream checksum failed for $HEX"; echo "$sums"; exit 1; }

echo "== 1. wake the root port, allow 16 GT/s ($(date +%T))"
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
    for _ in $(seq 10); do
        sleep 2
        if [[ "$(link)" == *"DLActive+"* ]]; then up=1; break 2; fi
    done
done
link; echo
if [ $up != 1 ]; then
    echo "LINK DID NOT COME UP.  Not rescanning.  Next: scripts/pi0_board_prepare_warm_reboot.sh (memory pi0-board-bringup-from-fics)."
    exit 1
fi

echo "== 4. remove + rescan"
"${SUDO[@]}" sh -c "echo 1 > /sys/bus/pci/devices/$RP/remove"; sleep 2
"${SUDO[@]}" sh -c "echo 1 > /sys/bus/pci/rescan"; sleep 6
lspci -nnd 1b59: | grep . || { echo "No 1b59 device after the rescan."; exit 1; }

echo "== 5. force 16 GT/s"
allow_16gt; retrain; sleep 3; link; echo

echo "== 6. driver"
"${SUDO[@]}" insmod "$BUNDLE/sdk/driver/acxpcie.ko"; sleep 1
"${SUDO[@]}" chmod 666 /dev/ac7t15xx0 || { echo "Driver loaded but /dev/ac7t15xx0 is missing."; exit 1; }

echo "== 7. TC_MEM_MODE=1"
"${PEEK[@]}" write bar0 0x244 0x1 >/dev/null
report
