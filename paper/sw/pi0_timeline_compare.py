#!/usr/bin/env python3
"""Silicon timeline vs the calibrated schedule, per chunk segment.

  pi0_timeline_compare.py <chunk dir> <timeline file from pi0_chunk_run --timeline> [pi0_chunk_time.py options...]

Every part's flag is polled while the chunk runs (first time seen, ~25 ms resolution); the model's list schedule gives
every part's end.  Printed per segment (SigLIP / vision ends / prefix / expert per step / head): the silicon and model
time the segment's last flag was set and the segment's own duration, so the stage group where silicon falls behind
the model shows directly.
"""
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def seg_of(layer: str) -> str:
    if layer.startswith("vis."):
        return "SigLIP"
    if layer.startswith("lm."):
        return "prefix"
    if layer.startswith("exp.") or layer.startswith("x_exp") or layer.startswith("head"):
        s = layer.split(".s")[-1] if ".s" in layer else "?"
        return f"expert step {s.split(':')[0]}"
    return "vision ends"


def main():
    d, tl = Path(sys.argv[1]), Path(sys.argv[2])
    extra = sys.argv[3:]
    out = Path("/tmp") / f"tl_{d.name}.json"
    subprocess.run([sys.executable, str(HERE / "pi0_chunk_time.py"), str(d), "--json", str(out), *extra], check=True,
                   stdout=subprocess.DEVNULL)
    model = json.loads(out.read_text())["sched_end"]
    si = [float(l.split()[1]) for l in tl.read_text().split("\n") if l.strip()]
    parts = json.loads((d / "parts.json").read_text())
    segs, k = [], 0
    for st in parts:
        for _ in st["parts"]:
            segs.append(seg_of(st["layer"]))
            k += 1
    assert len(segs) == len(si) == len(model), (len(segs), len(si), len(model))
    order, last_m, last_s = [], {}, {}
    for s, m, x in zip(segs, model, si):
        if s not in last_m:
            order.append(s)
        last_m[s] = max(last_m.get(s, 0.0), m)
        last_s[s] = max(last_s.get(s, 0.0), x)
    print(f"{'segment':<18}{'model end':>10}{'silicon end':>13}{'model dur':>11}{'silicon dur':>13}{'ratio':>7}")
    pm = ps = 0.0
    for s in order:
        dm, ds = last_m[s] - pm, last_s[s] - ps
        print(f"{s:<18}{last_m[s]:>10.3f}{last_s[s]:>13.3f}{dm:>11.3f}{ds:>13.3f}{(ds / dm if dm > 0 else 0):>7.2f}")
        pm, ps = last_m[s], last_s[s]


if __name__ == "__main__":
    main()
