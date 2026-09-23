#!/usr/bin/env python3
"""Parse one run of paper/synth/run_vu_lane_synth.sh -> <workdir>/result.json and a one-line summary.

Sources (all written by the tools, nothing estimated):
  rev_1/pnr/reports/vu_top_utilization_routed.txt   placed-and-routed instance counts (ACE)
  rev_1/pnr/reports/vu_top_timing_routed_*.txt       setup/hold slack, target and upper-limit frequency (ACE)
  rev_1/vu_top.srr                                   Synplify's own resource summary and estimated frequency
"""
from __future__ import annotations

import glob
import json
import re
import sys
from pathlib import Path


def num(pat, text, cast=int, default=None):
    m = re.search(pat, text, re.M)
    return cast(m.group(1)) if m else default


def main():
    w = Path(sys.argv[1])
    res = dict(workdir=str(w))
    util = w / "rev_1/pnr/reports/vu_top_utilization_routed.txt"
    if util.exists():
        t = util.read_text()
        res["routed"] = dict(
            lut=num(r"general purpose LUTs \(a\)\s+(\d+)", t),
            dff=num(r"DFF Total \(see notes\)\s+(\d+)", t),
            alu8=num(r"ALU8i\s+(\d+)", t),
            bram72k=num(r"BRAM Total \(see notes\)\s+(\d+)", t),
            bram_kinds={k: int(v) for k, v in re.findall(r"^\s+(BRAM72K\w*)\s+(\d+)\s*$", t, re.M)},
            lram2k=num(r"general purpose LRAMs \(a\)\s+(\d+)", t),
            mlp72=num(r"general purpose MLPs \(a\)\s+(\d+)", t),
            mlp_kinds={k: int(v) for k, v in re.findall(r"^\s+(MLP72\w*)\s+(\d+)\s*$", t, re.M)},
            mlp_sites_incl_inaccessible=num(r"MLP Total \(see notes\)\s+(\d+)", t),
            lram_sites_incl_inaccessible=num(r"LRAM Total \(see notes\)\s+(\d+)", t),
            rlb_tiles=num(r"RLB Tiles \(Occupied\)\s+(\d+)\s+57600", t),
        )
    tim = sorted(glob.glob(str(w / "rev_1/pnr/reports/vu_top_timing_routed_*.txt")))
    if tim:
        t = Path(tim[0]).read_text()
        m = re.search(r"^\s+i_clk\s+(-?[\d.]+)\s+(-?[\d.]+)\s+([\d.]+)\s+([\d.]+)", t, re.M)
        if m:
            res["timing"] = dict(report=Path(tim[0]).name, setup_slack_ns=float(m.group(1)), hold_slack_ns=float(m.group(2)),
                                 target_mhz=float(m.group(3)), upper_limit_mhz=float(m.group(4)))
        # the worst setup path (first detailed path in the report)
        p = re.search(r"Path Id: sc_s0.*?(?=Path Id: sc_s1|\Z)", t, re.S)
        if p:
            s = p.group(0)
            res["timing"]["worst_path"] = {
                "startpoint": num(r"Startpoint: (\S+)", s, str), "endpoint": num(r"Endpoint: (\S+)", s, str),
                "logic_delay_ns": num(r"Logic Delay: ([\d.]+)", s, float), "net_delay_ns": num(r"Net Delay: ([\d.]+)", s, float),
                "logic_levels": num(r"Logic Levels: (\d+)", s)}
    srr = w / "rev_1/vu_top.srr"
    if srr.exists():
        t = srr.read_text(errors="replace")
        res["synplify"] = dict(
            est_mhz=num(r"^i_clk\s+[\d.]+\s+MHz\s+([\d.]+)\s+MHz", t, float),
            errors=len(re.findall(r"^@E:", t, re.M)))
    (w / "result.json").write_text(json.dumps(res, indent=1))
    r, tm = res.get("routed", {}), res.get("timing", {})
    print(f"{w.name}: LUT {r.get('lut')} DFF {r.get('dff')} ALU8 {r.get('alu8')} BRAM72K {r.get('bram72k')} "
          f"MLP72 {r.get('mlp72')} {r.get('mlp_kinds')} LRAM2K {r.get('lram2k')} RLB {r.get('rlb_tiles')} | setup {tm.get('setup_slack_ns')} ns "
          f"hold {tm.get('hold_slack_ns')} ns target {tm.get('target_mhz')} upper limit {tm.get('upper_limit_mhz')} MHz")


if __name__ == "__main__":
    main()
