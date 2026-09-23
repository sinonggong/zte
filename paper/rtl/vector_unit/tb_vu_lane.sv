// ------------------------------------------------------------------
// Bit-exact testbench of vu_lane.sv against paper/sw/vector_unit_ref.py
// (vectors from paper/sw/vector_unit_vectors.py).  Random input gaps
// (+gap=PCT) and output back-pressure (+stall=PCT); every output beat is
// compared exactly.  Prints one CASE line per case and a RESULT line.
// Run: paper/rtl/run_vu_lane_sim.sh
// ------------------------------------------------------------------
`timescale 1ns/1ps
`include "vu_pkg.sv"

module tb_vu_lane;
    import vu_pkg::*;

    parameter F_GELU  = "vu_tbl_gelu.mem";
    parameter F_SIGM  = "vu_tbl_sigm.mem";
    parameter F_EXP   = "vu_tbl_exp.mem";
    parameter F_RSQRT = "vu_tbl_rsqrt.mem";
    parameter F_QUANT = "vu_tbl_quant.mem";

    reg clk = 1'b0;
    always #2 clk = ~clk;
    reg rstn = 1'b0;

    reg        desc_we = 1'b0;
    reg [3:0]  d_op = 4'd0;
    reg        d_bf = 1'b0, d_be = 1'b0, d_of = 1'b0;
    reg [31:0] d_k = 32'd0;
    wire       idle;

    reg        iv = 1'b0, ilast = 1'b0, imask = 1'b0;
    reg [47:0] ix = 48'd0;
    reg [15:0] irs = 16'd0;
    reg [31:0] ib = 32'd0, ic = 32'd0, id = 32'd0, ie = 32'd0;
    wire       irdy;

    wire       ov;
    reg        ordy = 1'b0;
    wire [1:0] okind;
    wire [63:0] odata;

    vu_lane #(.F_GELU(F_GELU), .F_SIGM(F_SIGM), .F_EXP(F_EXP), .F_RSQRT(F_RSQRT), .F_QUANT(F_QUANT)) dut (
        .i_clk(clk), .i_rstn(rstn),
        .i_desc_we(desc_we), .i_op(d_op), .i_b_fp32(d_bf), .i_bias_en(d_be), .i_out_fp32(d_of), .i_k(d_k),
        .o_idle(idle),
        .i_valid(iv), .o_ready(irdy), .i_last(ilast), .i_mask(imask), .i_x(ix), .i_rs(irs),
        .i_b(ib), .i_c(ic), .i_d(id), .i_e(ie),
        .o_valid(ov), .i_ready(ordy), .o_kind(okind), .o_data(odata));

    // vectors of the current case
    logic [0:0]  a_last [];
    logic [0:0]  a_mask [];
    logic [47:0] a_x    [];
    logic [15:0] a_rs   [];
    logic [31:0] a_b [], a_c [], a_d [], a_e [];
    logic [1:0]  e_kind [];
    logic [63:0] e_data [];

    string  vec, name;
    integer gap, stall, fd, fi, rc, ncase, nbeats, nerr, nprint, case_err, nin, nout, i, j, cycles, stall_cycles, c0, c1;
    reg [3:0]  c_op;
    reg        c_bf, c_be, c_of;
    reg [31:0] c_k;
    reg [0:0]  t_last, t_mask;
    reg [47:0] t_x;
    reg [15:0] t_rs;
    reg [31:0] t_b, t_c, t_d, t_e;
    reg [1:0]  t_kind;
    reg [63:0] t_data;

    // watchdog: no output progress for 2M cycles
    always @(posedge clk) begin
        cycles <= cycles + 1;
        if (ov && ordy)
            stall_cycles <= 0;
        else
            stall_cycles <= stall_cycles + 1;
        if (stall_cycles > 2000000) begin
            $display("RESULT FAIL timeout in case %0d %s (in %0d/%0d, out %0d/%0d)", ncase, name, i, nin, j, nout);
            $finish;
        end
    end

    task automatic drive(input integer n);
        reg pend;
        begin
            i = 0;
            pend = 1'b0;
            while (i < n) begin
                @(negedge clk);
                if (pend) begin
                    i = i + 1;
                    iv = 1'b0;
                    pend = 1'b0;
                end
                if (i < n) begin
                    if (!iv && (($urandom % 100) >= gap)) begin
                        iv    = 1'b1;
                        ilast = a_last[i];
                        imask = a_mask[i];
                        ix    = a_x[i];
                        irs   = a_rs[i];
                        ib    = a_b[i];
                        ic    = a_c[i];
                        id    = a_d[i];
                        ie    = a_e[i];
                    end
                    if (iv && irdy)
                        pend = 1'b1;
                end
            end
        end
    endtask

    task automatic monitor(input integer n);
        begin
            j = 0;
            while (j < n) begin
                @(negedge clk);
                ordy = (($urandom % 100) >= stall);
                if (ov && ordy) begin
                    if (okind !== e_kind[j] || odata !== e_data[j]) begin
                        case_err = case_err + 1;
                        if (nprint < 16) begin
                            nprint = nprint + 1;
                            $display("MISMATCH case %0d %s beat %0d: got %0d %016h expected %0d %016h",
                                     ncase, name, j, okind, odata, e_kind[j], e_data[j]);
                        end
                    end
                    j = j + 1;
                end
            end
        end
    endtask

    initial begin
        cycles = 0;
        stall_cycles = 0;
        nin = 0;
        nout = 0;
        i = 0;
        j = 0;
        name = "";
        if (!$value$plusargs("vec=%s", vec)) begin
            $display("RESULT FAIL no +vec=<dir>");
            $finish;
        end
        if (!$value$plusargs("gap=%d", gap))
            gap = 25;
        if (!$value$plusargs("stall=%d", stall))
            stall = 20;
        fd = $fopen({vec, "/cases.txt"}, "r");
        if (fd == 0) begin
            $display("RESULT FAIL cannot open %s/cases.txt", vec);
            $finish;
        end
        ncase = 0;
        nbeats = 0;
        nerr = 0;
        nprint = 0;
        repeat (5) @(negedge clk);
        rstn = 1'b1;
        repeat (3) @(negedge clk);

        forever begin
            rc = $fscanf(fd, "%h %h %h %h %h %d %d %s\n", c_op, c_bf, c_be, c_of, c_k, nin, nout, name);
            if (rc != 8)
                break;
            a_last = new[nin];  a_mask = new[nin];  a_x = new[nin];  a_rs = new[nin];
            a_b = new[nin];  a_c = new[nin];  a_d = new[nin];  a_e = new[nin];
            e_kind = new[nout];  e_data = new[nout];
            fi = $fopen($sformatf("%s/in_%03d.hex", vec, ncase), "r");
            for (i = 0; i < nin; i = i + 1) begin
                rc = $fscanf(fi, "%h %h %h %h %h %h %h %h\n", t_last, t_mask, t_x, t_rs, t_b, t_c, t_d, t_e);
                a_last[i] = t_last;  a_mask[i] = t_mask;  a_x[i] = t_x;  a_rs[i] = t_rs;
                a_b[i] = t_b;  a_c[i] = t_c;  a_d[i] = t_d;  a_e[i] = t_e;
            end
            $fclose(fi);
            fi = $fopen($sformatf("%s/out_%03d.hex", vec, ncase), "r");
            for (i = 0; i < nout; i = i + 1) begin
                rc = $fscanf(fi, "%h %h\n", t_kind, t_data);
                e_kind[i] = t_kind;
                e_data[i] = t_data;
            end
            $fclose(fi);

            // descriptor
            @(negedge clk);
            while (!idle)
                @(negedge clk);
            desc_we = 1'b1;
            d_op = c_op;  d_bf = c_bf;  d_be = c_be;  d_of = c_of;  d_k = c_k;
            @(negedge clk);
            desc_we = 1'b0;

            case_err = 0;
            c0 = cycles;
            fork
                drive(nin);
                monitor(nout);
            join
            c1 = cycles;
            // no extra beats may follow
            ordy = 1'b1;
            repeat (80) begin
                @(negedge clk);
                if (ov) begin
                    case_err = case_err + 1;
                    if (nprint < 16) begin
                        nprint = nprint + 1;
                        $display("MISMATCH case %0d %s: extra beat %0d %016h", ncase, name, okind, odata);
                    end
                end
            end
            ordy = 1'b0;
            $display("CASE %0d %-40s op=%0d in=%0d out=%0d cycles=%0d %s", ncase, name, c_op, nin, nout, c1 - c0,
                     case_err ? $sformatf("FAIL (%0d mismatches)", case_err) : "PASS");
            nerr = nerr + case_err;
            nbeats = nbeats + nout;
            ncase = ncase + 1;
        end
        $display("RESULT %s cases=%0d beats=%0d mismatches=%0d cycles=%0d gap=%0d stall=%0d",
                 nerr ? "FAIL" : "PASS", ncase, nbeats, nerr, cycles, gap, stall);
        $finish;
    end

endmodule
