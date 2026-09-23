#!/usr/bin/env python3
"""The whole-chip chunk's host inputs from a LeRobot pi0 observation (the torch side of pi0_chip_runtime.py).

On the node array everything from SigLIP layer 0 to the last Euler step runs on the chip; the host keeps the cheap
embedding lookups whose results enter the chip:

  SigLIP layer-0 input   vision_model.embeddings(image): the 14x14 patch convolution + the position table
  language rows          the prompt's token embeddings (valid tokens only: the compact prefix), times LANG_SCALE
  state token            model.state_proj(state)
  noise                  the flow-matching start x_t (the policy's sample_noise, or given)

chip_inputs() returns them in the shapes pi0_chip_runtime.ChunkIO.encode takes.  validate (this file's main) loads
the checkpoint, rebuilds the captured frame's inputs through the policy's own preprocessing and compares them with
the activation capture the chunk generator used -- the proof that a live observation feeds the chip the same bytes.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

LANG_SCALE = None          # set by validate(): 1 or sqrt(2048), whichever the capture's layer-0 input shows


@torch.no_grad()
def chip_inputs(policy, images, img_masks, lang_tokens, lang_masks, state, noise, lang_scale: float = 1.0):
    pwe = policy.model.paligemma_with_expert
    vm = pwe.paligemma.model.vision_tower.vision_model
    vis = []
    for img, m in zip(images, img_masks, strict=True):
        if bool(m[0]):                                   # empty slots are dropped (the compact prefix)
            vis.append(vm.embeddings(img.to(torch.float32))[0].float().numpy())
    if len(vis) != 2:
        raise ValueError(f"the chip chunk takes 2 cameras, the observation has {len(vis)}")
    lang = pwe.embed_language_tokens(lang_tokens)[0, lang_masks[0].to(torch.bool)].float().numpy() * lang_scale
    st = policy.model.state_proj(state.to(torch.float32))[0].float().numpy()
    nz = noise[0].float().numpy().astype(np.float32)
    return np.stack(vis), lang, st, nz


def rel(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-30))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--frame", default="demo1_ep20:2")
    ap.add_argument("--chunk", default=None, help="a generated chunk dir with image/ (--bin): compare the encoded bytes")
    ap.add_argument("--threads", type=int, default=6)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    import prefix_w8a8_eval as P
    import pi0_layer_lower as LL
    t0 = time.time()
    P.G.policy, _, _ = P.load_policy(P.CKPT)
    P.G.pwe = P.G.policy.model.paligemma_with_expert
    print(f"policy loaded in {time.time() - t0:.0f} s")
    ep, f = a.frame.split(":")
    fr = (ep, int(f))
    fin, cap = P.get_inputs(fr), P.get_capture(fr)
    z = np.load(LL.CAPTURE)
    noise = torch.from_numpy(cap["noise"])[None]
    vis, lang1, st, nz = chip_inputs(P.G.policy, fin["images"], fin["img_masks"], fin["lang_tokens"], fin["lang_masks"],
                                     fin["state"], noise)
    ref_lang = z["lm.L0.layer_in"][512:]
    s = float(np.sqrt(2048.0))
    scale = 1.0 if rel(lang1, ref_lang) < rel(lang1 * s, ref_lang) else s
    lang = lang1 * scale
    print(f"vision layer-0 input rel {rel(vis, z['vis.L0.layer_in']):.2e}; language rows rel {rel(lang, ref_lang):.2e} "
          f"(scale {scale:.4g}, {lang.shape[0]} tokens); state token rel {rel(st, z['ex.L0.layer_in'][0][0]):.2e}; "
          f"noise equal {np.array_equal(nz, z['ex.noise'].astype(np.float32))}")
    if a.chunk:
        import pi0_chip_runtime as RT
        io = RT.ChunkIO(a.chunk)
        bad = io.check(io.encode(vis, lang, st, nz))
        print(f"encoded live inputs vs the chunk image: {'IDENTICAL' if not bad else f'{bad} regions differ'}")


if __name__ == "__main__":
    main()
