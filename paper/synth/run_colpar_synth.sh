#!/usr/bin/env bash
# Synthesise + place + route the column-parallel INT8 GEMM chain
# (paper/rtl/mlp72_int8_colpar_chain.sv), optionally with its array-clock row sequencer
# (paper/rtl/colpar_row_sequencer.sv), on AC7t1500; record resources and routed timing.
# Usage: run_colpar_synth.sh <name> <N_STAGE> <period_ns> [ADDR_CASCADE=0] [MULTICYCLE=0] [WITH_SEQ=0]
#   WITH_SEQ=3 builds the whole colpar_chain_node: program fetch, command executor, loader, sequencer,
#   chain, result port, writer and the node_sync master.
#   MULTICYCLE=1 declares the MLP72 dout -> capture register path as 2-cycle
#   (the RTL guarantees it: OUT_REG holds for >= 2 cycles around each capture).
#   WITH_SEQ=1 drives the chain from colpar_row_sequencer (command buses are top-level ports).
#   WITH_SEQ=2 builds the whole chain node: tile loader + sequencer + chain + result port + writer,
#   with the NAP/fabric clock i_fclk (250 MHz) as the BRAM write clock.
#   Env WR_FIFO_LOG2 / WR_AFULL_MARGIN size the result writer's beat FIFO (default 8 = 256 beats / 8 records).
#   Env MULT_MODE=13 builds uint8 x int8 stage multipliers (default 00 = int8 x int8).
#   Env WR_PIPE=<n> pipelines the weight-load broadcast once per n stages (0 = flat).
#   Above ~16 stages the flat bus does not route, so a deep chain needs WR_PIPE.
#   Env VALID_COPIES=<n> splits the result-port bank into n slices with n copies of the
#   capture enable; at 32 stages one enable net drives 1,536 loads and misses 750 MHz.
#   Env WR_SLIM=0|1 (WITH_SEQ=3): result writer records straight from the port bank (1, default) or the original path.
#   Env FCLK_NS=<ns> sets the NAP/fabric clock period (default 4.0 = 250 MHz).
set -euo pipefail
NAME=$1; NST=$2; PERIOD=$3; CASC=${4:-0}; MC=${5:-0}; SEQ=${6:-0}
REPO=$(cd "$(dirname "$0")/../.." && pwd)
export ACE_INSTALL_DIR=${ACE_INSTALL_DIR:-/home/sngong/ACE_10.5.2/Achronix-linux}
export PATH=$ACE_INSTALL_DIR:$ACE_INSTALL_DIR/Synplify/bin:$PATH
W=$REPO/build/paper_chain_synth/$NAME
rm -rf "$W"; mkdir -p "$W"; cd "$W"
if [ "$SEQ" = 3 ]; then
cat > top.sv <<V
module chain_top (
    input  wire i_clk, input wire i_fclk, input wire i_rstn,
    input  wire i_prog_start, input wire [41:0] i_prog_base,
    output wire o_arvalid, input wire i_arready, output wire [41:0] o_araddr, output wire [7:0] o_arlen,
    output wire [2:0] o_arsize, output wire [1:0] o_arburst, output wire o_rready, input wire i_rvalid,
    input  wire [255:0] i_rdata, input wire [1:0] i_rresp, input wire i_rlast,
    output wire o_awvalid, input wire i_awready, output wire [41:0] o_awaddr, output wire [7:0] o_awlen,
    output wire [2:0] o_awsize, output wire [1:0] o_awburst, output wire o_wvalid, input wire i_wready,
    output wire [255:0] o_wdata, output wire [31:0] o_wstrb, output wire o_wlast, input wire i_bvalid,
    output wire o_bready, input wire [1:0] i_bresp,
    output wire o_halt, output wire o_error);
  colpar_chain_node #(.N_STAGE($NST), .ADDR_BITS(9), .ADDR_CASCADE($CASC), .PROG_FROM_GDDR6(1),
                      .WR_FIFO_LOG2(${WR_FIFO_LOG2:-8}), .WR_AFULL_MARGIN(${WR_AFULL_MARGIN:-8}),
                      .WR_PIPE_EVERY(${WR_PIPE:-0}), .VALID_COPIES(${VALID_COPIES:-1}), .WR_SLIM(${WR_SLIM:-1}),
                      .MULT_MODE(5'h${MULT_MODE:-00})) u (
    .i_clk_array(i_clk), .i_clk_fabric(i_fclk), .i_rstn(i_rstn),
    .i_prog_start(i_prog_start), .i_prog_base(i_prog_base), .i_cmd(128'd0), .i_cmd_valid(1'b0), .o_cmd_ready(),
    .o_arvalid(o_arvalid), .i_arready(i_arready), .o_araddr(o_araddr), .o_arlen(o_arlen), .o_arsize(o_arsize),
    .o_arburst(o_arburst), .o_rready(o_rready), .i_rvalid(i_rvalid), .i_rdata(i_rdata), .i_rresp(i_rresp),
    .i_rlast(i_rlast),
    .o_awvalid(o_awvalid), .i_awready(i_awready), .o_awaddr(o_awaddr), .o_awlen(o_awlen), .o_awsize(o_awsize),
    .o_awburst(o_awburst), .o_wvalid(o_wvalid), .i_wready(i_wready), .o_wdata(o_wdata), .o_wstrb(o_wstrb),
    .o_wlast(o_wlast), .i_bvalid(i_bvalid), .o_bready(o_bready), .i_bresp(i_bresp),
    .o_halt(o_halt), .o_error(o_error));
endmodule
V
elif [ "$SEQ" = 2 ]; then
cat > top.sv <<V
module chain_top (
    input  wire i_clk, input wire i_fclk, input wire i_rstn,
    input  wire i_ld_arm, input wire [41:0] i_ld_base, input wire [4:0] i_ld_first, input wire [4:0] i_ld_nreg,
    input  wire [9:0] i_ld_nwords, input wire [9:0] i_ld_nsegs, input wire [13:0] i_ld_seg_step, input wire [13:0] i_ld_tgt_step,
    output wire o_arvalid, input wire i_arready, output wire [41:0] o_araddr, output wire [7:0] o_arlen,
    output wire [2:0] o_arsize, output wire [1:0] o_arburst, output wire o_rready, input wire i_rvalid,
    input  wire [255:0] i_rdata, input wire [1:0] i_rresp, input wire i_rlast,
    input  wire i_start, input wire [8:0] i_act_base, input wire [8:0] i_wt_base,
    input  wire [9:0] i_words, input wire [9:0] i_rows, input wire [9:0] i_passes, input wire [4:0] i_gap,
    input  wire [41:0] i_out_base, input wire i_restart, input wire i_flush,
    output wire o_awvalid, input wire i_awready, output wire [41:0] o_awaddr, output wire [7:0] o_awlen,
    output wire [2:0] o_awsize, output wire [1:0] o_awburst, output wire o_wvalid, input wire i_wready,
    output wire [255:0] o_wdata, output wire [31:0] o_wstrb, output wire o_wlast, input wire i_bvalid,
    output wire o_bready, input wire [1:0] i_bresp,
    output wire o_busy, output wire o_done, output wire o_ld_busy, output wire o_ld_done, output wire o_ld_error,
    output wire o_wr_idle, output wire o_wr_error, output wire o_overrun);
  wire [143:0] wdata; wire [8:0] waddr; wire [$NST:0] wen;
  wire s_valid, s_first, s_last, hold, sums_valid, fvalid, afull; wire [8:0] s_act, s_wt;
  wire [48*$NST-1:0] sums, fsums;
  colpar_tile_loader #(.N_STAGE($NST), .ADDR_BITS(9)) u_ld (
    .i_clk(i_fclk), .i_rstn(i_rstn), .i_arm(i_ld_arm), .i_base(i_ld_base), .i_first_target(i_ld_first),
    .i_n_regions(i_ld_nreg), .i_n_words(i_ld_nwords), .i_n_segs(i_ld_nsegs), .i_seg_step(i_ld_seg_step),
    .i_tgt_step(i_ld_tgt_step), .o_arvalid(o_arvalid), .i_arready(i_arready),
    .o_araddr(o_araddr), .o_arlen(o_arlen), .o_arsize(o_arsize), .o_arburst(o_arburst), .o_rready(o_rready),
    .i_rvalid(i_rvalid), .i_rdata(i_rdata), .i_rresp(i_rresp), .i_rlast(i_rlast),
    .o_wdata(wdata), .o_waddr(waddr), .o_wen(wen), .o_busy(o_ld_busy), .o_done(o_ld_done), .o_error(o_ld_error));
  colpar_row_sequencer #(.ADDR_BITS(9)) u_seq (
    .i_clk(i_clk), .i_rstn(i_rstn), .i_start(i_start), .i_hold(hold), .i_act_base(i_act_base), .i_wt_base(i_wt_base),
    .i_words(i_words), .i_rows(i_rows), .i_passes(i_passes), .i_gap(i_gap),
    .o_valid(s_valid), .o_first(s_first), .o_last(s_last), .o_act_raddr(s_act), .o_wt_raddr(s_wt),
    .o_busy(o_busy), .o_done(o_done));
  mlp72_int8_colpar_chain #(.N_STAGE($NST), .ADDR_BITS(9), .ADDR_CASCADE($CASC), .MULT_MODE(5'h${MULT_MODE:-00})) u (
    .i_clk(i_clk), .i_rstn(i_rstn), .i_wclk(i_fclk), .i_wdata(wdata), .i_waddr(waddr),
    .i_wen(wen), .i_valid(s_valid), .i_act_raddr(s_act), .i_wt_raddr(s_wt),
    .i_first(s_first), .i_last(s_last), .o_sums(sums), .o_sums_valid(sums_valid));
  colpar_result_port #(.N_STAGE($NST)) u_port (
    .i_clk_array(i_clk), .i_rstn_array(i_rstn), .i_sums(sums), .i_sums_valid(sums_valid), .o_hold_array(hold),
    .i_clk_fabric(i_fclk), .i_rstn_fabric(i_rstn), .i_fifo_afull(afull),
    .o_sums(fsums), .o_valid(fvalid), .o_overrun(o_overrun));
  colpar_result_writer #(.N_STAGE($NST), .SUM_BITS(32), .FIFO_LOG2(${WR_FIFO_LOG2:-8}), .AFULL_MARGIN(${WR_AFULL_MARGIN:-8})) u_wr (
    .i_clk(i_fclk), .i_rstn(i_rstn), .i_out_base(i_out_base), .i_blk_beats(10'd0), .i_gap_bytes(24'd0), .i_restart(i_restart), .i_flush(i_flush),
    .i_sums(fsums), .i_sums_valid(fvalid), .o_afull(afull),
    .o_awvalid(o_awvalid), .i_awready(i_awready), .o_awaddr(o_awaddr), .o_awlen(o_awlen), .o_awsize(o_awsize),
    .o_awburst(o_awburst), .o_wvalid(o_wvalid), .i_wready(i_wready), .o_wdata(o_wdata), .o_wstrb(o_wstrb),
    .o_wlast(o_wlast), .i_bvalid(i_bvalid), .o_bready(o_bready), .i_bresp(i_bresp),
    .o_idle(o_wr_idle), .o_error(o_wr_error));
endmodule
V
elif [ "$SEQ" = 1 ]; then
cat > top.sv <<V
module chain_top (
    input  wire i_clk, input wire i_rstn, input wire i_wclk,
    input  wire [143:0] i_wdata, input wire [8:0] i_waddr, input wire [$NST:0] i_wen,
    input  wire i_fclk, input wire i_fifo_afull,
    input  wire i_start, input wire [8:0] i_act_base, input wire [8:0] i_wt_base,
    input  wire [9:0] i_words, input wire [9:0] i_rows, input wire [9:0] i_passes, input wire [4:0] i_gap,
    output wire [48*$NST-1:0] o_fsums, output wire o_fvalid, output wire o_overrun,
    output wire o_busy, output wire o_done);
  wire s_valid, s_first, s_last, hold, sums_valid; wire [8:0] s_act, s_wt; wire [48*$NST-1:0] sums;
  colpar_row_sequencer #(.ADDR_BITS(9)) u_seq (
    .i_clk(i_clk), .i_rstn(i_rstn), .i_start(i_start), .i_hold(hold), .i_act_base(i_act_base), .i_wt_base(i_wt_base),
    .i_words(i_words), .i_rows(i_rows), .i_passes(i_passes), .i_gap(i_gap),
    .o_valid(s_valid), .o_first(s_first), .o_last(s_last), .o_act_raddr(s_act), .o_wt_raddr(s_wt),
    .o_busy(o_busy), .o_done(o_done));
  mlp72_int8_colpar_chain #(.N_STAGE($NST), .ADDR_BITS(9), .ADDR_CASCADE($CASC), .MULT_MODE(5'h${MULT_MODE:-00})) u (
    .i_clk(i_clk), .i_rstn(i_rstn), .i_wclk(i_wclk), .i_wdata(i_wdata), .i_waddr(i_waddr),
    .i_wen(i_wen), .i_valid(s_valid), .i_act_raddr(s_act), .i_wt_raddr(s_wt),
    .i_first(s_first), .i_last(s_last), .o_sums(sums), .o_sums_valid(sums_valid));
  colpar_result_port #(.N_STAGE($NST)) u_port (
    .i_clk_array(i_clk), .i_rstn_array(i_rstn), .i_sums(sums), .i_sums_valid(sums_valid), .o_hold_array(hold),
    .i_clk_fabric(i_fclk), .i_rstn_fabric(i_rstn), .i_fifo_afull(i_fifo_afull),
    .o_sums(o_fsums), .o_valid(o_fvalid), .o_overrun(o_overrun));
endmodule
V
else
cat > top.sv <<V
module chain_top (
    input  wire i_clk, input wire i_rstn, input wire i_wclk,
    input  wire [143:0] i_wdata, input wire [8:0] i_waddr, input wire [$NST:0] i_wen,
    input  wire i_valid, input wire [8:0] i_act_raddr, input wire [8:0] i_wt_raddr,
    input  wire i_first, input wire i_last,
    output wire [48*$NST-1:0] o_sums, output wire o_sums_valid);
  mlp72_int8_colpar_chain #(.N_STAGE($NST), .ADDR_BITS(9), .ADDR_CASCADE($CASC), .MULT_MODE(5'h${MULT_MODE:-00})) u (
    .i_clk(i_clk), .i_rstn(i_rstn), .i_wclk(i_wclk), .i_wdata(i_wdata), .i_waddr(i_waddr),
    .i_wen(i_wen), .i_valid(i_valid), .i_act_raddr(i_act_raddr), .i_wt_raddr(i_wt_raddr),
    .i_first(i_first), .i_last(i_last), .o_sums(o_sums), .o_sums_valid(o_sums_valid));
endmodule
V
fi
SEQ_FILE=""
[ "$SEQ" = 1 ] && SEQ_FILE="add_file -verilog -vlog_std sysv \"$REPO/paper/rtl/colpar_row_sequencer.sv\"
add_file -verilog -vlog_std sysv \"$REPO/paper/rtl/colpar_result_port.sv\""
[ "$SEQ" = 3 ] && SEQ_FILE="add_file -verilog -vlog_std sysv \"$REPO/paper/rtl/colpar_row_sequencer.sv\"
add_file -verilog -vlog_std sysv \"$REPO/paper/rtl/colpar_result_port.sv\"
add_file -verilog -vlog_std sysv \"$REPO/paper/rtl/colpar_tile_loader.sv\"
add_file -verilog -vlog_std sysv \"$REPO/paper/rtl/colpar_result_writer.sv\"
add_file -verilog -vlog_std sysv \"$REPO/paper/rtl/colpar_node_ctrl.sv\"
add_file -verilog -vlog_std sysv \"$REPO/paper/rtl/colpar_prog_fetch.sv\"
add_file -verilog -vlog_std sysv \"$REPO/paper/rtl/node_sync.sv\"
add_file -verilog -vlog_std sysv \"$REPO/paper/rtl/colpar_chain_node.sv\""
[ "$SEQ" = 2 ] && SEQ_FILE="add_file -verilog -vlog_std sysv \"$REPO/paper/rtl/colpar_row_sequencer.sv\"
add_file -verilog -vlog_std sysv \"$REPO/paper/rtl/colpar_result_port.sv\"
add_file -verilog -vlog_std sysv \"$REPO/paper/rtl/colpar_tile_loader.sv\"
add_file -verilog -vlog_std sysv \"$REPO/paper/rtl/colpar_result_writer.sv\"
add_file -verilog -vlog_std sysv \"$REPO/paper/rtl/node_sync.sv\""
cat > syn.prj <<V
add_file -verilog -vlog_std sysv "$ACE_INSTALL_DIR/libraries/device_models/AC7t1500_synplify.sv"
$SEQ_FILE
add_file -verilog -vlog_std sysv "$REPO/paper/rtl/mlp72_int8_colpar_chain.sv"
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
echo "create_clock -name i_clk [get_ports i_clk] -period $PERIOD" > syn.sdc
case "$SEQ" in
  0) echo "create_clock -name i_wclk [get_ports i_wclk] -period 20.0" >> syn.sdc
     echo "set_clock_groups -asynchronous -group {i_clk} -group {i_wclk}" >> syn.sdc ;;
  1) echo "create_clock -name i_wclk [get_ports i_wclk] -period 20.0" >> syn.sdc
     echo "create_clock -name i_fclk [get_ports i_fclk] -period ${FCLK_NS:-4.0}" >> syn.sdc
     echo "set_clock_groups -asynchronous -group {i_clk} -group {i_wclk} -group {i_fclk}" >> syn.sdc ;;
  2|3) echo "create_clock -name i_fclk [get_ports i_fclk] -period ${FCLK_NS:-4.0}" >> syn.sdc
     echo "set_clock_groups -asynchronous -group {i_clk} -group {i_fclk}" >> syn.sdc ;;
esac
cp syn.sdc pnr.sdc
if [ "$MC" = 1 ]; then
    cat >> pnr.sdc <<V
set_multicycle_path 2 -setup -to [get_cells -hierarchical {*g_cap.cap*}]
set_multicycle_path 1 -hold  -to [get_cells -hierarchical {*g_cap.cap*}]
V
fi
echo "[$(date +%T)] synplify $NAME"
timeout 3600 synplify_pro -batch syn.prj > syn.log 2>&1 || { echo "SYNPLIFY FAILED $NAME"; grep -E "^@E:" rev_1/chain_top.srr | head -8; exit 1; }
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
R=rev_1/pnr/reports
grep -hE "LUT Sites|DFF Sites|MLP72  |BRAM72K  " $R/chain_top_utilization_routed.txt 2>/dev/null | head -6 || true
grep -hA3 "Clock / Group" $R/chain_top_timing_routed_C1_0p90V_0C.txt 2>/dev/null | head -8 || true
