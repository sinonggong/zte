#!/usr/bin/env python
"""Capture real prefix KV caches and fp32 reference action chunks from replay episodes.

For every frame of every replay .npz (glue/level2/export_replay_frames.py) this runs
the stock LeRobot pi0 (fp32, CPU) exactly as the policy server does -- preprocessor,
SigLIP + PaliGemma prefix, 10-step torch expert -- and stores, per frame:

  kv            float32 [L, 2, T, H, D]   post-RoPE prefix K then V, the layout of
                                          pi0::PrefixKvCacheView before the bf16 cast
  prefix_valid  uint8   [T]               prefix pad mask (image slots + language tokens)
  state         float32 [32]              normalised, padded state (what the expert sees)
  noise         float32 [50, 32]          the seeded initial noise of the Euler loop
  actions_fp32  float32 [50, 32]          torch fp32 expert result x_0 on that noise
  lang_tokens   int64   [48], lang_mask uint8 [48]

The numerics work (paper/sw) reads these; nothing here is synthetic.  One frame costs
about 10 s on the i7-13700 (7.4 s prefix + 2 s expert); the 16 GB fp32 checkpoint is
loaded once.  Run it with nohup; never beside an ACE build.

    ~/lerobot/.venv/bin/python glue/level3/capture_pi0_prefix_kv.py \
        --model ~/pi0_ckpt/ur7e-demo-2-pi0-fsdp/010000/pretrained_model \
        --out ~/pi0_glue/captures ~/pi0_glue/episodes/*.npz
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pi0_fpga_policy import PI0FpgaPolicy  # noqa: E402

RENAME_MAP = {"observation.images.front": "observation.images.base_0_rgb",
              "observation.images.wrist": "observation.images.left_wrist_0_rgb"}
JOINT_KEYS = [f"joint.{i}" for i in range(6)] + ["joint.gripper_pos"]


def make_preprocessor(policy, model_path):
    from lerobot.policies.factory import make_pre_post_processors
    pre, post = make_pre_post_processors(
        policy.config, pretrained_path=model_path,
        preprocessor_overrides={"device_processor": {"device": "cpu"},
                                "rename_observations_processor": {"rename_map": RENAME_MAP}},
        postprocessor_overrides={"device_processor": {"device": "cpu"}})
    return pre, post


def build_batch(policy, pre, replay, frame: int):
    from lerobot.async_inference.helpers import raw_observation_to_observation
    from lerobot.utils.feature_utils import hw_to_dataset_features
    raw = {k: float(replay["state"][frame, i]) for i, k in enumerate(JOINT_KEYS)}
    raw["front"], raw["wrist"], raw["task"] = replay["front"][frame], replay["wrist"][frame], str(replay["task"])
    feats = hw_to_dataset_features({**dict.fromkeys(JOINT_KEYS, float), "front": (480, 640, 3), "wrist": (480, 640, 3)},
                                   "observation", use_video=False)
    img_feats = dict(policy.config.image_features)
    for rk, pk in RENAME_MAP.items():
        if pk in img_feats:
            img_feats[rk] = img_feats[pk]
    return pre(raw_observation_to_observation(raw, feats, img_feats))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, default=0, help="noise seed base; frame seed = base*100000 + episode_hash + frame")
    p.add_argument("--max-frames", type=int, default=0, help="0 = all frames of each episode")
    p.add_argument("--threads", type=int, default=0)
    p.add_argument("replays", nargs="+")
    a = p.parse_args()
    if a.threads:
        torch.set_num_threads(a.threads)
    from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS

    out_root = Path(os.path.expanduser(a.out))
    out_root.mkdir(parents=True, exist_ok=True)
    os.environ["PI0_ACTION_EXPERT"] = "torch"
    t = time.time()
    policy = PI0FpgaPolicy.from_pretrained(os.path.expanduser(a.model))
    policy.eval()
    pre, _ = make_preprocessor(policy, os.path.expanduser(a.model))
    print(f"loaded {a.model} in {time.time() - t:.0f} s", flush=True)
    cfg = policy.config
    num_steps = cfg.num_inference_steps

    for replay_path in a.replays:
        replay = np.load(os.path.expanduser(replay_path), allow_pickle=False)
        ep_name = Path(replay_path).stem
        ep_dir = out_root / ep_name
        ep_dir.mkdir(exist_ok=True)
        n = int(replay["state"].shape[0])
        if a.max_frames:
            n = min(n, a.max_frames)
        meta = {"source": str(replay["source"]), "task": str(replay["task"]), "frames": n,
                "checkpoint": os.path.expanduser(a.model), "num_inference_steps": num_steps,
                "prefix_source": "lerobot torch fp32 cpu", "rename_map": RENAME_MAP}
        (ep_dir / "META.json").write_text(json.dumps(meta, indent=1))
        for frame in range(n):
            dst = ep_dir / f"frame_{frame:02d}.npz"
            if dst.exists():
                continue
            t0 = time.time()
            with torch.no_grad():
                batch = build_batch(policy, pre, replay, frame)
                images, img_masks = policy._preprocess_images(batch)
                lang_tokens = batch[OBS_LANGUAGE_TOKENS]
                lang_masks = batch[OBS_LANGUAGE_ATTENTION_MASK]
                state = policy.prepare_state(batch)
                seed = a.seed * 100000 + (abs(hash(ep_name)) % 1000) * 100 + frame
                g = torch.Generator().manual_seed(seed)
                noise = torch.randn((1, cfg.chunk_size, cfg.max_action_dim), generator=g, dtype=torch.float32)
                t1 = time.time()
                past_key_values, prefix_pad_masks = policy._run_prefix(images, img_masks, lang_tokens, lang_masks)
                t_prefix = time.time() - t1
                t1 = time.time()
                x_t = policy._denoise_torch(state, prefix_pad_masks, past_key_values, noise.clone(), num_steps)
                t_expert = time.time() - t1
                per_layer = []
                for entry in past_key_values:
                    k, v = entry[0], entry[1]                       # [1, H, T, D]
                    per_layer.append(torch.stack([k[0], v[0]], 0).permute(0, 2, 1, 3))  # [2, T, H, D]
                kv = torch.stack(per_layer, 0).float().numpy()      # [L, 2, T, H, D]
            np.savez(dst,
                     kv=np.ascontiguousarray(kv, dtype=np.float32),
                     prefix_valid=prefix_pad_masks[0].to(torch.uint8).numpy(),
                     state=state[0].float().numpy().astype(np.float32),
                     noise=noise[0].numpy().astype(np.float32),
                     actions_fp32=x_t[0].float().numpy().astype(np.float32),
                     lang_tokens=lang_tokens[0].numpy(),
                     lang_mask=lang_masks[0].to(torch.uint8).numpy(),
                     seed=np.int64(seed),
                     raw_state=replay["state"][frame].astype(np.float32))
            print(f"{ep_name} frame {frame:02d}: prefix {t_prefix:.1f} s, expert {t_expert:.1f} s, "
                  f"valid={int(prefix_pad_masks.sum())}/{prefix_pad_masks.shape[1]}, total {time.time() - t0:.1f} s",
                  flush=True)
    print("CAPTURE_DONE", flush=True)


if __name__ == "__main__":
    main()
