#!/usr/bin/env python3
"""Rewrite src/acxip/pll_nap.acxip (PLL_SW_1) as the node array's clock source.

The deployed IO ring has two user PLLs on the same 100 MHz reference (CLKIO_SW fpga_fab_clk_7):
PLL_SW_0 (pll.acxip) carries the NoC's mandatory 200 MHz reference beside i_reg_clk and i_mcu_clk, and
ACE re-solves a PLL's VCO from clkout0's target alone, so touching it is how a build dies in 20 s
(memory ace-pll-resolve-shared-vco).  PLL_SW_1 (pll_nap.acxip) only feeds i_mlp_clk, which the node
array does not use, so every node clock comes off PLL_SW_1 and PLL_SW_0 stays byte-for-byte as deployed.

What ACE 10.5.2 does with the file (dry runs 2026-09-17, run_node_pll_dryrun.sh):
  * it re-solves reference/feedback/clkout0 ODN from float_target_out0_frequency alone, prefilled
    dividers are ignored, and it prefers the LOWEST VCO, i.e. clkout0 ODN 8: VCO = 8 x target;
  * clkout1/2 keep the ODN written here (so their frequencies follow the solved VCO);
  * clkout3's ODN is forced equal to clkout0's, with its own OSN on top: clkout3 = clkout0 / OSN3;
  * some ODN values are refused at bitstream generation ("Requested clkoutN divider unavailable").

Usage: set_node_pll.py <pll_nap.acxip> --target <MHz string> --clk0 name:odn[:nocore]
           [--clk1 name:odn] [--clk2 name:odn] [--clk3 name:osn] [--ref n --fb n] [--no-int-fb]
  clk3 takes an OSN (its ODN is clkout0's).  A name of 'none' leaves that output disconnected.
  Hardware rule (register_data.xml, PLL_DIVQn): an output divider is 2 or a multiple of 4; ACE floors
  anything else and then refuses it ("Requested clkoutN divider unavailable").
"""
from __future__ import annotations
import argparse, re, sys
from pathlib import Path


def set_key(text: str, key: str, value: str, must: bool = True) -> str:
    pat = rf"^{re.escape(key)}=.*$"
    new, n = re.subn(pat, f"{key}={value}", text, flags=re.M)
    if n != 1 and must:
        sys.exit(f"{key}: expected exactly one line, found {n}")
    return new


def parse(spec: str | None):
    if not spec:
        return None
    f = spec.split(":")
    return {"name": f[0], "div": int(f[1]), "core": not (len(f) > 2 and f[2] == "nocore")}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("acxip", type=Path)
    ap.add_argument("--target", required=True, help="float_target_out0_frequency string")
    ap.add_argument("--clk0", required=True)
    ap.add_argument("--clk1")
    ap.add_argument("--clk2")
    ap.add_argument("--clk3", help="name:osn (ODN is clkout0's)")
    ap.add_argument("--ref", type=int, default=1, help="reference_divider to prefill (VCO = 400 * fb / ref MHz)")
    ap.add_argument("--fb", type=int, default=20, help="feedback_divider to prefill")
    ap.add_argument("--no-int-fb", action="store_true", help="is_forcing_integer_feedback=No (solver probe only)")
    a = ap.parse_args()
    vco = 400.0 * a.fb / a.ref
    t = a.acxip.read_text()
    t = set_key(t, "reference_divider", str(a.ref))
    t = set_key(t, "feedback_divider", str(a.fb))
    t = set_key(t, "is_forcing_integer_feedback", "No" if a.no_int_fb else "Yes")
    t = set_key(t, "float_target_out0_frequency", a.target)
    outs = [parse(a.clk0), parse(a.clk1), parse(a.clk2), parse(a.clk3)]
    count = max(i + 1 for i, o in enumerate(outs) if o is not None)
    t = set_key(t, "output_port_count", str(count))
    odn0 = outs[0]["div"]
    for idx, o in enumerate(outs):
        if o is None:
            continue
        name = o["name"] if o["name"] != "none" else f"pll_nap_clkout{idx}"
        odn = odn0 if idx == 3 else o["div"]
        osn = o["div"] if idx == 3 else 1
        t = set_key(t, f"clkout{idx}.clkout.port_name", name)
        t = set_key(t, f"clkout{idx}.int_ODN_output_divider", str(odn))
        t = set_key(t, f"clkout{idx}.int_OSN_output_divider", str(osn))
        t = set_key(t, f"clkout{idx}.is_OSN_enabled", "Yes" if idx == 3 else "No")
        t = set_key(t, f"clkout{idx}.is_output_connected_to_core", "Yes" if (o["core"] and o["name"] != "none") else "No")
        t = set_key(t, f"clkout{idx}.int_out_dividers_combined", str(odn * osn), must=False)
        print(f"clkout{idx} {name:16s} ODN {odn:3d} OSN {osn:2d}  {vco / odn / osn:8.3f} MHz"
              f"{'' if o['core'] and o['name'] != 'none' else '  (not to core)'}   (if VCO {vco:.1f} = 400 * {a.fb} / {a.ref})")
    a.acxip.write_text(t)


if __name__ == "__main__":
    main()
