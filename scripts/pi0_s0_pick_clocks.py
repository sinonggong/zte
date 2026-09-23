#!/usr/bin/env python3
"""Pick the node-array clocks a routed impl supports, as arguments for run_pi0_s0_reclock_bitstream.sh.

  scripts/pi0_s0_pick_clocks.py <impl dir> [--margin 0.97]

Reads the routed Fmax of i_clk_array / i_clk_vec / i_clk_fabric from the final timing reports (the worse of the 0C and
125C corners) and picks the fastest legal PLL settings at or below margin x Fmax:
  array  (PLL_SW_3): f = 400 * fb / ref / 8, ref 1 or 2 (25 MHz steps), at most 725 MHz
  vector, fabric (PLL_SW_1, VCO 8000): f = 8000 / ODN, ODN 2 or a multiple of 4 (vector at most 333.3, fabric 250)
Prints the reclock command line.  (The 09-18 s1n reclock took fabric ODN 44 = 181.8 MHz against an Fmax of 132.9
because the fabric VCO was assumed to be 5500; this avoids the hand arithmetic.)
Caveats: run it on the impl's ORIGINAL reports -- a reclock rewrites them at its clocks, with different Fmax.  The
array figure is a floor, not a limit: the chain's MLP72 -> capture path is a 2-cycle path and the report's Fmax does
not always honour it (s1m reported 132 MHz and ran bit-exact at 250).  Pass --array <MHz> to override.
"""
import re
import sys
from pathlib import Path


def fmax(rep: Path) -> dict:
    out = {}
    for line in rep.read_text(errors="replace").splitlines():
        m = re.match(r"\s+(i_clk_array|i_clk_vec|i_clk_fabric)\s+(-?[\d.]+)\s+(-?[\d.]+)\s+([\d.]+)\s+([\d.]+)", line)
        if m and m.group(1) not in out:
            out[m.group(1)] = float(m.group(5))
    return out


def main():
    impl = Path(sys.argv[1])
    margin = float(sys.argv[sys.argv.index("--margin") + 1]) if "--margin" in sys.argv else 0.97
    reps = sorted((impl / "pnr" / "reports").glob("tc_ref_design_top_timing_final_C1_0p90V_*C.txt"))
    if not reps:
        sys.exit(f"no final timing reports under {impl}/pnr/reports")
    fm = {}
    for r in reps:
        for k, v in fmax(r).items():
            fm[k] = min(fm.get(k, v), v)
    print("routed Fmax (worst corner):", {k: round(v, 1) for k, v in fm.items()})
    arr = min(725.0, float(sys.argv[sys.argv.index("--array") + 1]) if "--array" in sys.argv else margin * fm["i_clk_array"])
    best = None
    for ref in (1, 2):
        for fb in range(1, 200):
            f = 400 * fb / ref / 8
            if f <= arr + 1e-9 and (best is None or f > best[0]):
                best = (f, ref, fb)
    odns = [2] + list(range(4, 257, 4))

    def pick(limit, cap):
        lim = min(cap, margin * limit)
        return min((o for o in odns if 8000 / o <= lim + 1e-9), default=None)

    v, fab = pick(fm["i_clk_vec"], 333.34), pick(fm["i_clk_fabric"], 250.0)
    print(f"array {best[0]:.1f} MHz (ref {best[1]} fb {best[2]}), vector {8000 / v:.1f} MHz (ODN {v}), "
          f"fabric {8000 / fab:.1f} MHz (ODN {fab})")
    print(f"scripts/run_pi0_s0_reclock_bitstream.sh {impl.name} {best[0]:g} {best[1]} {best[2]} {v} {fab}")


if __name__ == "__main__":
    main()
