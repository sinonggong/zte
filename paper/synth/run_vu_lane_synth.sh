#!/usr/bin/env bash
# Synthesise + place + route ONE pi0 vector-unit lane (paper/rtl/vector_unit/vu_lane.sv) on AC7t1500
# and record resources and the routed timing.  Workstream C of
# docs/PI0_FULL_MODEL_ON_CHIP_DESIGN_20260916.md; modelled on paper/synth/run_chain_synth.sh.
#
# Usage: run_vu_lane_synth.sh <name> <period_ns> [RETIME=0] [MULSTYLE=dsp|logic] [SEED]
#   period_ns 4.0 = 250 MHz (target), 5.714 = 175 MHz (fallback)
#   RETIME    Synplify retiming (0 = the RTL's own pipeline)
#   MULSTYLE  dsp: Synplify infers the multipliers (MLP72); logic: +define+VU_MUL_LOGIC (fabric only)
#   SEED      ACE placement seed (default: ACE's)
# Output: build/paper_vu_synth/<name>/{syn.log,ace.log,rev_1/...}, result.json (parse_vu_lane_synth.py)
# Beside a full-chip build, run inside a capped scope:
#   nohup systemd-run --user --scope -q -p MemoryHigh=6G nice -n 10 paper/synth/run_vu_lane_synth.sh l250 4.0 > log 2>&1 &
set -euo pipefail
NAME=$1; PERIOD=$2; RETIME=${3:-0}; MULSTYLE=${4:-dsp}; SEED=${5:-}
REPO=$(cd "$(dirname "$0")/../.." && pwd)
export ACE_INSTALL_DIR=${ACE_INSTALL_DIR:-/home/sngong/ACE_10.5.2/Achronix-linux}
export PATH=$ACE_INSTALL_DIR:$ACE_INSTALL_DIR/Synplify/bin:$PATH
RTL=$REPO/paper/rtl/vector_unit
W=$REPO/build/paper_vu_synth/$NAME
rm -rf "$W"; mkdir -p "$W"; cd "$W"
python3 "$REPO/paper/sw/vector_unit_ref.py" --tables "$W/tables" > /dev/null
DEFINE=""
[ "$MULSTYLE" = logic ] && DEFINE='set_option -hdl_define -set "VU_MUL_LOGIC"'

cat > top.sv <<V
module vu_top (
    input  wire i_clk, input wire i_rstn,
    input  wire i_desc_we, input wire [3:0] i_op, input wire i_b_fp32, input wire i_bias_en,
    input  wire i_out_fp32, input wire [31:0] i_k, output wire o_idle,
    input  wire i_valid, output wire o_ready, input wire i_last, input wire i_mask,
    input  wire [47:0] i_x, input wire [15:0] i_rs,
    input  wire [31:0] i_b, input wire [31:0] i_c, input wire [31:0] i_d, input wire [31:0] i_e,
    output wire o_valid, input wire i_ready, output wire [1:0] o_kind, output wire [63:0] o_data);
  vu_lane #(.F_GELU("$W/tables/vu_tbl_gelu.mem"), .F_SIGM("$W/tables/vu_tbl_sigm.mem"),
            .F_EXP("$W/tables/vu_tbl_exp.mem"), .F_RSQRT("$W/tables/vu_tbl_rsqrt.mem"),
            .F_QUANT("$W/tables/vu_tbl_quant.mem")) u (
    .i_clk(i_clk), .i_rstn(i_rstn), .i_desc_we(i_desc_we), .i_op(i_op), .i_b_fp32(i_b_fp32),
    .i_bias_en(i_bias_en), .i_out_fp32(i_out_fp32), .i_k(i_k), .o_idle(o_idle),
    .i_valid(i_valid), .o_ready(o_ready), .i_last(i_last), .i_mask(i_mask), .i_x(i_x), .i_rs(i_rs),
    .i_b(i_b), .i_c(i_c), .i_d(i_d), .i_e(i_e),
    .o_valid(o_valid), .i_ready(i_ready), .o_kind(o_kind), .o_data(o_data));
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
add_file -verilog -vlog_std sysv "$W/top.sv"
add_file -constraint "$W/syn.sdc"
impl -add rev_1 -type fpga
set_option -include_path "$ACE_INSTALL_DIR/libraries;$RTL"
set_option -vlog_std sysv
set_option -technology AchronixSpeedster7t
set_option -part AC7t1500
set_option -package F53
set_option -speed_grade C1
set_option -top_module vu_top
set_option -frequency $(python3 -c "print(round(1000/$PERIOD))")
set_option -maxfan 40
set_option -resource_sharing 0
set_option -retiming $RETIME
set_option -pipe 1
set_option -disable_io_insertion 1
set_option -write_verilog 1
set_option -multi_file_compilation_unit 1
$DEFINE
project -result_file "$W/rev_1/vu_top.vm"
project -run
V
cat > syn.sdc <<V
create_clock -name i_clk [get_ports i_clk] -period $PERIOD
V
cp syn.sdc pnr.sdc
echo "[$(date +%T)] synplify $NAME (period $PERIOD ns, retime $RETIME, mult $MULSTYLE)"
timeout 3600 synplify_pro -batch syn.prj > syn.log 2>&1 || { echo "SYNPLIFY FAILED $NAME"; grep -E "^@E" rev_1/vu_top.srr | head -8; grep -iE "licen.*(fail|error|denied)" syn.log | head -4; exit 1; }
SEEDOPT=""
[ -n "$SEED" ] && SEEDOPT="set_impl_option -project vu_top -impl rev_1 seed {$SEED}"
cat > ace.tcl <<V
create_project vu_top.acxprj -impl rev_1
set_project_option -project vu_top -- partname {AC7t1500}
set_project_option -project vu_top -- package {F53}
set_project_option -project vu_top -- speed_grade {C1}
set_project_option -project vu_top -- core_voltage {0.90}
set_project_option -project vu_top -- junction_temperature {0}
add_project_source_files -project vu_top -pnr_netlist rev_1/vu_top.vm
add_project_source_files -project vu_top -pnr_constraint pnr.sdc
set_impl_option -project vu_top -impl rev_1 check_final_timing {0}
$SEEDOPT
run_prepare
run_place
run_route
run -step report_timing_routed
puts "ACE_VU_DONE"
V
echo "[$(date +%T)] ace $NAME"
timeout 7200 ace -batch -script_file ace.tcl > ace.log 2>&1 || true
grep -q ACE_VU_DONE ace.log || { echo "ACE FAILED $NAME"; grep -nE "ERROR|licen" ace.log | head -8; exit 1; }
echo "[$(date +%T)] done $NAME"
python3 "$REPO/paper/synth/parse_vu_lane_synth.py" "$W" || true
