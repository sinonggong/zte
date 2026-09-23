#!/usr/bin/env bash
# FIT build of the whole node array: paper/rtl/pi0_chip_top.sv with N_CHAIN chain nodes and N_VNODE
# vector nodes, each on its own NAP, plus the host register window.  Synthesise, then ACE prepare and
# PLACE, and report placed utilisation -- placement is the fit test; routing is a separate, much longer
# question and is only run with ROUTE=1.
#
# Usage: run_chip_fit_synth.sh <name> [N_CHAIN=32] [N_PV=8] [N_VNODE=4] [N_LANE=4] [ARRAY_NS=1.333] [FABRIC_NS=4.0]
# Env:   VEC_NS=<ns> vector-node clock period (default 3.0 = 333 MHz)
#        EXTRA_PDC=<file> an extra placement constraint file (e.g. per-node regions from gen_node_regions.py)
# Env:   N_STAGE=<n> WR_PIPE=<n> VALID_COPIES=<n> NAPS_PER_NODE=1|2
#        N_DEEP=<n> DEEP_STAGE=<n> (default 0 / 32): the first N_DEEP int8 chains DEEP_STAGE deep (mixed-depth array)
#        A 32-stage chain needs WR_PIPE and VALID_COPIES to route and hold 750 MHz (design doc 3.9).
# Env:   ROUTE=1      also run_route + report_timing_routed (hours)
#        SYN_ONLY=1   stop after Synplify (resource estimate only, minutes)
#        ACE_INSTALL_DIR
#
# The licence services must be up and swap on:
#   systemctl --user restart ace-license.service synplify-license.service
#   sudo swapon /swapfile2; sudo sysctl vm.swappiness=140 vm.watermark_scale_factor=200
# Run it inside a capped scope, the way the full-chip synthesis has to be run:
#   nohup systemd-run --user --scope -q -p MemoryHigh=20G nice -n 10 paper/synth/run_chip_fit_synth.sh c32v4 > log 2>&1 &
set -euo pipefail
NAME=$1; NC=${2:-32}; NPV=${3:-8}; NV=${4:-4}; NL=${5:-4}; ANS=${6:-1.333}; FNS=${7:-4.0}
REPO=$(cd "$(dirname "$0")/../.." && pwd)
export ACE_INSTALL_DIR=${ACE_INSTALL_DIR:-/home/sngong/ACE_10.5.2/Achronix-linux}
export PATH=$ACE_INSTALL_DIR:$ACE_INSTALL_DIR/Synplify/bin:$PATH
C=$REPO/paper/rtl
V=$C/vector_unit
W=$REPO/build/paper_chip_fit/$NAME
rm -rf "$W"; mkdir -p "$W"; cd "$W"
python3 "$REPO/paper/sw/vector_unit_ref.py" --tables "$W/tables" > /dev/null

cat > top.sv <<V
module chip_top (
    input wire i_clk_array, input wire i_clk_fabric, input wire i_clk_vec,
    input wire i_rstn_array, input wire i_rstn_fabric, input wire i_rstn_vec,
    output wire o_all_halt, output wire o_any_error, output wire o_soft_rst);
  pi0_chip_top #(.F_GELU("$W/tables/vu_tbl_gelu.mem"), .F_SIGM("$W/tables/vu_tbl_sigm.mem"),
                 .F_EXP("$W/tables/vu_tbl_exp.mem"), .F_RSQRT("$W/tables/vu_tbl_rsqrt.mem"),
                 .F_QUANT("$W/tables/vu_tbl_quant.mem"),
                 .N_CHAIN($NC), .N_PV($NPV), .N_VNODE($NV), .N_LANE($NL),
                 .N_STAGE(${N_STAGE:-16}), .WR_PIPE_EVERY(${WR_PIPE:-0}),
                 .VALID_COPIES(${VALID_COPIES:-1}), .NAPS_PER_NODE(${NAPS_PER_NODE:-1}),
                 .N_DEEP(${N_DEEP:-0}), .DEEP_STAGE(${DEEP_STAGE:-32}),
                 .SLOT_BITS(${SLOT_BITS:-11}), .N_LD(${N_LD:-1})) u (
    .i_clk_array(i_clk_array), .i_clk_fabric(i_clk_fabric), .i_clk_vec(i_clk_vec),
    .i_rstn_array(i_rstn_array), .i_rstn_fabric(i_rstn_fabric), .i_rstn_vec(i_rstn_vec),
    .o_all_halt(o_all_halt), .o_any_error(o_any_error), .o_soft_rst(o_soft_rst));
endmodule
V

cat > syn.sdc <<V
create_clock -name i_clk_array  [get_ports i_clk_array]  -period $ANS
create_clock -name i_clk_fabric [get_ports i_clk_fabric] -period $FNS
create_clock -name i_clk_vec    [get_ports i_clk_vec]    -period ${VEC_NS:-3.0}
set_clock_groups -asynchronous -group {i_clk_array} -group {i_clk_fabric} -group {i_clk_vec}
V
cp syn.sdc pnr.sdc

RTL_FILES="
$V/vu_pkg.sv
$V/vu_add.sv
$V/vu_mul.sv
$V/vu_round.sv
$V/vu_tbl.sv
$V/vu_qtab.sv
$V/vu_lane.sv
$V/vu_slot_loader.sv
$V/vu_word_loader.sv
$V/vu_rd_fanout.sv
$V/vu_beat_writer.sv
$V/vu_wr_merge.sv
$V/vu_pdq.sv
$C/colpar_nap_mux.sv
$V/vu_node.sv
$V/vu_node_ml.sv
$C/colpar_row_sequencer.sv
$C/colpar_result_port.sv
$C/colpar_tile_loader.sv
$C/colpar_result_writer.sv
$C/colpar_node_ctrl.sv
$C/colpar_prog_fetch.sv
$C/node_sync.sv
$C/mlp72_int8_colpar_chain.sv
$C/colpar_chain_node.sv
$C/nap_axi_ports.sv
$C/pi0_chip_ctrl.sv
$C/pi0_host_gddr_bridge.sv
$C/axi_stripe.sv
$C/axi_id_reorder.sv
$C/pi0_chip_top.sv
$W/top.sv
"
{
  echo "add_file -verilog -vlog_std sysv \"$ACE_INSTALL_DIR/libraries/device_models/AC7t1500_synplify.sv\""
  for f in $RTL_FILES; do echo "add_file -verilog -vlog_std sysv \"$f\""; done
  echo "add_file -constraint \"$ACE_INSTALL_DIR/libraries/device_models/AC7t1500_synplify.fdc\""
  echo "add_file -constraint \"$W/syn.sdc\""
  cat <<V
impl -add rev_1 -type fpga
set_option -include_path "$ACE_INSTALL_DIR/libraries;$V"
set_option -vlog_std sysv
set_option -hdl_define -set ACX_DEVICE_AC7t1500
set_option -technology AchronixSpeedster7t
set_option -part AC7t1500
set_option -package F53
set_option -speed_grade C1
set_option -top_module chip_top
set_option -frequency $(python3 -c "print(round(1000/$FNS))")
set_option -maxfan 40
set_option -resource_sharing 0
set_option -retiming 0
set_option -pipe 1
set_option -infer_mlp72 1
set_option -disable_io_insertion 1
set_option -write_verilog 1
set_option -multi_file_compilation_unit 1
project -result_file "$W/rev_1/chip_top.vm"
project -run
V
} > syn.prj

echo "[$(date +%T)] synplify $NAME: ${NC} chain x ${N_STAGE:-16} stages (${NPV} uint8 PV) + ${NV} x ${NL}-lane vector"
echo "        wr_pipe=${WR_PIPE:-0} valid_copies=${VALID_COPIES:-1} naps/node=${NAPS_PER_NODE:-1} deep=${N_DEEP:-0}x${DEEP_STAGE:-32}"
timeout 21600 synplify_pro -batch syn.prj > syn.log 2>&1 || {
    echo "SYNPLIFY FAILED $NAME"; grep -E "^@E:" rev_1/chip_top.srr 2>/dev/null | head -12; exit 1; }
echo "[$(date +%T)] synplify done"
sed -n '/Resource Usage Report for chip_top/,/^LUT6/p' rev_1/chip_top.srr 2>/dev/null | grep -E "uses|use$" | head -24 || true
if [ "${SYN_ONLY:-0}" = 1 ]; then exit 0; fi

cat > ace.tcl <<V
create_project chip_top.acxprj -impl rev_1
set_project_option -project chip_top -- partname {AC7t1500}
set_project_option -project chip_top -- package {F53}
set_project_option -project chip_top -- speed_grade {C1}
set_project_option -project chip_top -- core_voltage {0.90}
set_project_option -project chip_top -- junction_temperature {0}
add_project_source_files -project chip_top -pnr_netlist rev_1/chip_top.vm
add_project_source_files -project chip_top -pnr_constraint pnr.sdc
${EXTRA_PDC:+add_project_source_files -project chip_top -pnr_constraint $EXTRA_PDC}
set_impl_option -project chip_top -impl rev_1 check_final_timing {0}
run_prepare
puts "ACE_PREPARE_DONE"
run_place
puts "ACE_PLACE_DONE"
V
if [ "${ROUTE:-0}" = 1 ]; then
    cat >> ace.tcl <<V
run_route
run -step report_timing_routed
puts "ACE_ROUTE_DONE"
V
fi
echo "[$(date +%T)] ace $NAME (place${ROUTE:+ + route})"
timeout 86400 ace -batch -print_progress -script_file ace.tcl > ace.log 2>&1 || true
R=rev_1/pnr/reports
if ! grep -q ACE_PREPARE_DONE ace.log; then
    echo "ACE PREPARE FAILED $NAME"; grep -nE "ERROR" ace.log | head -12; exit 1
fi
if ! grep -q ACE_PLACE_DONE ace.log; then
    echo "ACE PLACE FAILED $NAME -- the design does not fit, or placement gave up"
    grep -nE "ERROR|Utilization|exceeds" ace.log | head -20; exit 1
fi
echo "[$(date +%T)] placed"
sed -n '/Utilization Summary/,/Utilization Details/p' $R/chip_top_utilization_placed.txt 2>/dev/null | head -20 || true
grep -hE "^   (ALU8i|BRAM Total|LRAM Total|MLP Total|MLP72|LUT|DFF)" $R/chip_top_utilization_placed.txt 2>/dev/null | head -12 || true
echo "[$(date +%T)] done $NAME"
