#!/usr/bin/env python3
"""Regenerate the SmoothQuant / MSE calibration caches the layer goldens need (pi0_layer_lower.CALIB):
calib.pt (prefix_w8a8_eval.py --stages calib on demo1_ep01..08 frame 5) and calib_exp.pt (its expert calibration
on the same frames' fp32 capture KV).  The original cache lived in a session scratchpad that a reboot wiped.

Usage: ~/lerobot/.venv/bin/python paper/sw/regen_calib.py [--cache build/paper_vector_unit/calib] [--threads 6]
"""
from __future__ import annotations
import argparse, os, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=str(REPO / "build/paper_vector_unit/calib"))
    ap.add_argument("--threads", type=int, default=6)
    a = ap.parse_args()
    Path(a.cache).mkdir(parents=True, exist_ok=True)
    import prefix_w8a8_eval as E
    sys.argv = ["prefix_w8a8_eval.py", "--stages", "calib", "--cache", a.cache, "--threads", str(a.threads),
                "--out", str(Path(a.cache) / "out")]
    E.main()                                   # writes calib.pt (and logs ALL_DONE); G stays set up
    E.load_calib_exp()                         # writes calib_exp.pt from the calibration frames' capture KV
    print("REGEN_CALIB_DONE", os.listdir(a.cache), flush=True)


if __name__ == "__main__":
    main()
