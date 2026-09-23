#!/usr/bin/env python3
"""Whole-pi0-on-chip accuracy on recorded frames, through the policy's own chip backend.

Loads PI0FpgaPolicy with PI0_ACTION_EXPERT=chip (PI0_CHIP_CHUNK = a generated chunk already loaded on the board,
PI0_CHIP_MAP = its pi0_chunk_run --map), runs every frame's observation (the capture pipeline's preprocessing and
noise) through _predict_chip, and compares the chip's actions with the fp32 LeRobot model's captured actions:
joint errors (joints 0..5) in degrees through the action unnormaliser's std, and rel RMS over the 7 action dims.
The frame the chunk was generated from (demo1_ep20:2) must reproduce the generator's actions bit for bit.

  PI0_ACTION_EXPERT=chip PI0_CHIP_CHUNK=... PI0_CHIP_MAP=vector:14,int8:0,uint8:12 PI0_CHUNK_RUN=.../pi0_chunk_run \\
      ~/lerobot/.venv/bin/python glue/level3/eval_chip_frames.py --frames demo1_ep20:all demo1_ep40:all \\
          recov_pi0_ep00:all recov_pi0_ep10:all --json results.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
SW = HERE.parents[1] / "paper" / "sw"
sys.path.insert(0, str(SW))
sys.path.insert(0, str(HERE))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--frames", nargs="+", default=["demo1_ep20:2"])
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    os.environ.setdefault("PI0_ACTION_EXPERT", "chip")
    torch.set_num_threads(a.threads)
    import prefix_w8a8_eval as P
    t0 = time.time()
    P.G.policy, _, _ = P.load_policy(P.CKPT)
    P.G.pwe = P.G.policy.model.paligemma_with_expert
    pol = P.G.policy
    print(f"policy loaded in {time.time() - t0:.0f} s (backend {pol.expert_backend})", flush=True)
    std = P.unnorm_std()
    gen = np.load(Path(os.environ["PI0_CHIP_CHUNK"]) / "actions_chip.npy")
    recs = []
    for fr in P.parse_frames(a.frames):
        fin, cap = P.get_inputs(fr), P.get_capture(fr)
        noise = torch.from_numpy(cap["noise"])[None]
        t1 = time.time()
        x, st = pol._predict_chip(fin["images"], fin["img_masks"], fin["lang_tokens"], fin["lang_masks"], fin["state"], noise)
        wall = time.time() - t1
        act = x[0].float().numpy()
        ref = cap["actions"].astype(np.float64)
        d = (act[:, :7].astype(np.float64) - ref[:, :7])
        jdeg = np.abs(d[:, :6]) * std[:6] * 180.0 / math.pi
        rel7 = float(np.sqrt((d ** 2).mean()) / np.sqrt((ref[:, :7] ** 2).mean()))
        r = dict(frame=f"{fr[0]}:{fr[1]}", chip_s=st.get("chip_s"), wall_s=wall, host_inputs_s=st.get("host_inputs_s"),
                 joint_max_deg=float(jdeg.max()), joint_mean_deg=float(jdeg.mean()), rel_rms_7=rel7)
        if fr == ("demo1_ep20", 2):
            r["bit_exact_vs_generator"] = bool(np.array_equal(act.astype(np.float32).view(np.uint32),
                                                              gen.astype(np.float32).view(np.uint32)))
        recs.append(r)
        print(f"{r['frame']:>18}  chip {r['chip_s'] if r['chip_s'] is not None else float('nan'):.3f} s  wall {wall:.2f} s  "
              f"joint max {r['joint_max_deg']:.2f} deg mean {r['joint_mean_deg']:.3f} deg  rel7 {rel7:.4f}"
              + (f"  bit-exact vs generator: {r['bit_exact_vs_generator']}" if "bit_exact_vs_generator" in r else ""), flush=True)
    if recs:
        s = dict(frames=len(recs), joint_max_deg=max(r["joint_max_deg"] for r in recs),
                 joint_mean_deg=float(np.mean([r["joint_mean_deg"] for r in recs])),
                 rel_rms_7_mean=float(np.mean([r["rel_rms_7"] for r in recs])),
                 chip_s_median=float(np.median([r["chip_s"] for r in recs if r["chip_s"] is not None] or [float("nan")])))
        print("SUMMARY " + json.dumps(s))
        if a.json:
            Path(a.json).write_text(json.dumps(dict(summary=s, frames=recs, chunk=os.environ.get("PI0_CHIP_CHUNK")), indent=1))


if __name__ == "__main__":
    main()
