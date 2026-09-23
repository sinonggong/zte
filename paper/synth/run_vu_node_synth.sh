#!/usr/bin/env bash
# Synthesise + place + route one vector node (paper/rtl/vector_unit/vu_node.sv: program fetch, operand
# slot loader and slot memories, one lane, element + summary writers, write mux) on AC7t1500 and record
# resources (RLB tiles above all: the full-chip fit is RLB-bound) and routed timing.
# Modelled on paper/synth/run_vu_lane_synth.sh.
#
# Usage: run_vu_node_synth.sh <name> <period_ns> [SLOT_BITS=11] [WR_FIFO_LOG2=6] [N_LANE=0] [N_LD=1]
# Env:   QTAIL=0|1 (vu_node_ml only: the fused QUANT tail in every lane, default 1)
#        SYN_MAXFAN / SYN_RSHARE / SYN_RETIME (default 40 / 0 / 0) -- the values were chosen for
#        the single-lane experiments and have never been rechecked at node or chip scale.
#        ACE_SEED=<n> re-rolls placement, which sets RLB tile OCCUPANCY (54% LUT fill today).
#   period_ns 4.0 = 250 MHz; N_LANE 0 = the one-lane vu_node.sv, 1/2/4 = vu_node_ml.sv with that many lanes
# Beside a full-chip build, run inside a capped scope:
#   nohup systemd-run --user --scope -q -p MemoryHigh=6G nice -n 10 paper/synth/run_vu_node_synth.sh vn250 4.0 > log 2>&1 &
set -euo pipefail
NAME=$1; PERIOD=$2; SLOT_BITS=${3:-11}; WR_FIFO_LOG2=${4:-6}; N_LANE=${5:-0}; N_LD=${6:-1}
REPO=$(cd "$(dirname "$0")/../.." && pwd)
export ACE_INSTALL_DIR=${ACE_INSTALL_DIR:-/home/sngong/ACE_10.5.2/Achronix-linux}
export PATH=$ACE_INSTALL_DIR:$ACE_INSTALL_DIR/Synplify/bin:$PATH
RTL=$REPO/paper/rtl/vector_unit
W=$REPO/build/paper_vu_synth/$NAME
rm -rf "$W"; mkdir -p "$W"; cd "$W"
python3 "$REPO/paper/sw/vector_unit_ref.py" --tables "$W/tables" > /dev/null

cat > top.sv <<V
module vn_top (
    input  wire i_clk, input wire i_rstn, input wire i_prog_start, input wire [41:0] i_prog_base,
    output wire o_arvalid, input wire i_arready, output wire [41:0] o_araddr, output wire [7:0] o_arlen,
    output wire [2:0] o_arsize, output wire [1:0] o_arburst, output wire o_rready, input wire i_rvalid,
    input  wire [255:0] i_rdata, input wire [1:0] i_rresp, input wire i_rlast,
    output wire o_awvalid, input wire i_awready, output wire [41:0] o_awaddr, output wire [7:0] o_awlen,
    output wire [2:0] o_awsize, output wire [1:0] o_awburst, output wire o_wvalid, input wire i_wready,
    output wire [255:0] o_wdata, output wire [31:0] o_wstrb, output wire o_wlast, input wire i_bvalid,
    output wire o_bready, input wire [1:0] i_bresp, output wire o_halt, output wire o_error);
  $([ "$N_LANE" = 0 ] && echo "vu_node" || echo "vu_node_ml") #(.F_GELU("$W/tables/vu_tbl_gelu.mem"), .F_SIGM("$W/tables/vu_tbl_sigm.mem"),
            .F_EXP("$W/tables/vu_tbl_exp.mem"), .F_RSQRT("$W/tables/vu_tbl_rsqrt.mem"),
            $([ "$N_LANE" = 0 ] || echo ".N_LANE($N_LANE), .N_LD($N_LD), .QTAIL(${QTAIL:-1}),")
            .F_QUANT("$W/tables/vu_tbl_quant.mem"), .SLOT_BITS($SLOT_BITS), .WR_FIFO_LOG2($WR_FIFO_LOG2)) u (
    .i_clk(i_clk), .i_rstn(i_rstn), .i_prog_start(i_prog_start), .i_prog_base(i_prog_base),
    .o_arvalid(o_arvalid), .i_arready(i_arready), .o_araddr(o_araddr), .o_arlen(o_arlen),
    .o_arsize(o_arsize), .o_arburst(o_arburst), .o_rready(o_rready), .i_rvalid(i_rvalid),
    .i_rdata(i_rdata), .i_rresp(i_rresp), .i_rlast(i_rlast),
    .o_awvalid(o_awvalid), .i_awready(i_awready), .o_awaddr(o_awaddr), .o_awlen(o_awlen),
    .o_awsize(o_awsize), .o_awburst(o_awburst), .o_wvalid(o_wvalid), .i_wready(i_wready),
    .o_wdata(o_wdata), .o_wstrb(o_wstrb), .o_wlast(o_wlast), .i_bvalid(i_bvalid),
    .o_bready(o_bready), .i_bresp(i_bresp), .o_halt(o_halt), .o_error(o_error));
endmodule
V
cat > syn.prj <<V
add_file -verilog -vlog_std sysv "$ACE_INSTALL_DIR/libraries/device_models/AC7t1500_synplify.sv"
add_file -verilog -vlog_std sysv "$RTL/vu_pkg.sv"
add_file -verilog -vlog_std sysv "$RTL/vu_add.sv"
add_file -verilog -vlog_std sysv "$RTL/vu_mul.sv"
add_file -verilog -vlog_std sysv "$RTL/vu_round.sv"
add_file -verilog -vlog_std sysv "$RTL/vu_tbl.sv"
add_file -verilog -vlog_std sysv "$RTL/vu_qtab.sv"
add_file -verilog -vlog_std sysv "$RTL/vu_lane.sv"
add_file -verilog -vlog_std sysv "$RTL/vu_slot_loader.sv"
add_file -verilog -vlog_std sysv "$RTL/vu_word_loader.sv"
add_file -verilog -vlog_std sysv "$RTL/vu_rd_fanout.sv"
add_file -verilog -vlog_std sysv "$RTL/vu_beat_writer.sv"
add_file -verilog -vlog_std sysv "$RTL/vu_wr_merge.sv"
add_file -verilog -vlog_std sysv "$RTL/vu_pdq.sv"
add_file -verilog -vlog_std sysv "$RTL/vu_node.sv"
add_file -verilog -vlog_std sysv "$RTL/vu_node_ml.sv"
add_file -verilog -vlog_std sysv "$REPO/paper/rtl/colpar_prog_fetch.sv"
add_file -verilog -vlog_std sysv "$REPO/paper/rtl/colpar_nap_mux.sv"
add_file -verilog -vlog_std sysv "$REPO/paper/rtl/node_sync.sv"
add_file -verilog -vlog_std sysv "$W/top.sv"
add_file -constraint "$W/syn.sdc"
impl -add rev_1 -type fpga
set_option -include_path "$ACE_INSTALL_DIR/libraries;$RTL"
set_option -vlog_std sysv
set_option -technology AchronixSpeedster7t
set_option -part AC7t1500
set_option -package F53
set_option -speed_grade C1
set_option -top_module vn_top
set_option -frequency $(python3 -c "print(round(1000/$PERIOD))")
set_option -maxfan ${SYN_MAXFAN:-40}
set_option -resource_sharing ${SYN_RSHARE:-0}
set_option -retiming ${SYN_RETIME:-0}
set_option -pipe 1
set_option -disable_io_insertion 1
set_option -write_verilog 1
set_option -multi_file_compilation_unit 1
project -result_file "$W/rev_1/vn_top.vm"
project -run
V
cat > syn.sdc <<V
create_clock -name i_clk [get_ports i_clk] -period $PERIOD
V
cp syn.sdc pnr.sdc
echo "[$(date +%T)] synplify $NAME (period $PERIOD ns, SLOT_BITS $SLOT_BITS, WR_FIFO_LOG2 $WR_FIFO_LOG2)"
timeout 3600 synplify_pro -batch syn.prj > syn.log 2>&1 || { echo "SYNPLIFY FAILED $NAME"; grep -E "^@E:" rev_1/vn_top.srr | head -8; exit 1; }
cat > ace.tcl <<V
create_project vn_top.acxprj -impl rev_1
set_project_option -project vn_top -- partname {AC7t1500}
set_project_option -project vn_top -- package {F53}
set_project_option -project vn_top -- speed_grade {C1}
set_project_option -project vn_top -- core_voltage {0.90}
set_project_option -project vn_top -- junction_temperature {0}
add_project_source_files -project vn_top -pnr_netlist rev_1/vn_top.vm
add_project_source_files -project vn_top -pnr_constraint pnr.sdc
set_impl_option -project vn_top -impl rev_1 check_final_timing {0}
${ACE_SEED:+set_impl_option -project vn_top -impl rev_1 seed $ACE_SEED}
run_prepare
run_place
run_route
run -step report_timing_routed
puts "ACE_VN_DONE"
V
echo "[$(date +%T)] ace $NAME"
timeout 7200 ace -batch -script_file ace.tcl > ace.log 2>&1 || true
grep -q ACE_VN_DONE ace.log || { echo "ACE FAILED $NAME"; grep -nE "ERROR|licen" ace.log | head -8; exit 1; }
echo "[$(date +%T)] done $NAME"
R=rev_1/pnr/reports
grep -hE "RLB Tiles|LUT Total|DFF Total|MLP72  |BRAM72K  |BRAM72K_SDP|LRAM Total" $R/vn_top_utilization_routed.txt 2>/dev/null | head -8 || true
grep -hA3 "Clock / Group  setup" $R/vn_top_timing_routed_C1_0p90V_0C.txt 2>/dev/null | head -5 || true
