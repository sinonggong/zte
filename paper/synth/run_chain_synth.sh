#!/usr/bin/env bash
# Synthesise + place + route the one-chain hard-block INT8 GEMM
# (paper/rtl/mlp72_int8_chain.sv) on AC7t1500 and record resources and the
# routed timing.  Phase B of docs/PI0_FULL_CHIP_ARCHITECTURE_20260910.md.
# Usage: run_chain_synth.sh <name> <N_STAGE> <period_ns> [ADDR_CASCADE=0]
set -euo pipefail
NAME=$1; NST=$2; PERIOD=$3; CASC=${4:-0}
REPO=$(cd "$(dirname "$0")/../.." && pwd)
export ACE_INSTALL_DIR=${ACE_INSTALL_DIR:-/home/sngong/ACE_10.5.2/Achronix-linux}
export PATH=$ACE_INSTALL_DIR:$ACE_INSTALL_DIR/Synplify/bin:$PATH
W=$REPO/build/paper_chain_synth/$NAME
rm -rf "$W"; mkdir -p "$W"; cd "$W"
cat > top.sv <<V
module chain_top (
    input  wire i_clk, input wire i_rstn, input wire i_wclk,
    input  wire [143:0] i_wdata, input wire [8:0] i_waddr, input wire [$NST:0] i_wen,
    input  wire i_valid, input wire [8:0] i_raddr, input wire i_first, input wire i_last,
    output wire [47:0] o_sum, output wire o_sum_valid);
  mlp72_int8_chain #(.N_STAGE($NST), .ADDR_BITS(9), .ADDR_CASCADE($CASC)) u (
    .i_clk(i_clk), .i_rstn(i_rstn), .i_wclk(i_wclk), .i_wdata(i_wdata), .i_waddr(i_waddr),
    .i_wen(i_wen), .i_valid(i_valid), .i_raddr(i_raddr), .i_first(i_first), .i_last(i_last),
    .o_sum(o_sum), .o_sum_valid(o_sum_valid));
endmodule
V
cat > syn.prj <<V
add_file -verilog -vlog_std sysv "$ACE_INSTALL_DIR/libraries/device_models/AC7t1500_synplify.sv"
add_file -verilog -vlog_std sysv "$REPO/paper/rtl/mlp72_int8_chain.sv"
add_file -verilog -vlog_std sysv "$W/top.sv"
add_file -constraint "$W/syn.sdc"
impl -add rev_1 -type fpga
set_option -include_path "$ACE_INSTALL_DIR/libraries"
set_option -vlog_std sysv
set_option -technology AchronixSpeedster7t
set_option -part AC7t1500
set_option -package F53
set_option -speed_grade C1
set_option -top_module chain_top
set_option -frequency $(python3 -c "print(round(1000/$PERIOD))")
set_option -maxfan 40
set_option -resource_sharing 0
set_option -retiming 0
set_option -pipe 1
set_option -disable_io_insertion 1
set_option -write_verilog 1
set_option -multi_file_compilation_unit 1
project -result_file "$W/rev_1/chain_top.vm"
project -run
V
cat > syn.sdc <<V
create_clock -name i_clk [get_ports i_clk] -period $PERIOD
create_clock -name i_wclk [get_ports i_wclk] -period 20.0
set_clock_groups -asynchronous -group {i_clk} -group {i_wclk}
V
cp syn.sdc pnr.sdc
echo "[$(date +%T)] synplify $NAME"
timeout 3600 synplify_pro -batch syn.prj > syn.log 2>&1 || { echo "SYNPLIFY FAILED $NAME"; grep -E "^@E" rev_1/chain_top.srr | head -8; exit 1; }
cat > ace.tcl <<V
create_project chain_top.acxprj -impl rev_1
set_project_option -project chain_top -- partname {AC7t1500}
set_project_option -project chain_top -- package {F53}
set_project_option -project chain_top -- speed_grade {C1}
set_project_option -project chain_top -- core_voltage {0.90}
set_project_option -project chain_top -- junction_temperature {0}
add_project_source_files -project chain_top -pnr_netlist rev_1/chain_top.vm
add_project_source_files -project chain_top -pnr_constraint pnr.sdc
set_impl_option -project chain_top -impl rev_1 check_final_timing {0}
run_prepare
run_place
run_route
run -step report_timing_routed
puts "ACE_CHAIN_DONE"
V
echo "[$(date +%T)] ace $NAME"
timeout 7200 ace -batch -script_file ace.tcl > ace.log 2>&1 || true
grep -q ACE_CHAIN_DONE ace.log || { echo "ACE FAILED $NAME"; grep -nE "ERROR" ace.log | head -8; exit 1; }
echo "[$(date +%T)] done $NAME"
grep -hE "RLB Tiles|LUT Sites|DFF Sites|MLP|BRAM" rev_1/pnr/reports/chain_top_utilization_routed.txt 2>/dev/null | head -8 || true
grep -hiE "upper limit" rev_1/pnr/reports/chain_top_timing_routed_C1_0p90V_0C.txt 2>/dev/null | head -4 || true
