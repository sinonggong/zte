#!/usr/bin/env python
"""Robot-side round trip of the pi0_remote policy on one recorded frame, no arm:
preprocess (Jetson) -> RPC chunk (Desktop) -> postprocess (Jetson).

    python test_pi0_remote_roundtrip.py --policy ~/pi0_glue/policies/pi0_remote_ur7e-demo-2-pi0-010000 \
        --replay ~/pi0_glue/replay_ur7e_demo1_ep0.npz
"""

import argparse
import time

import numpy as np
import torch

from lerobot.async_inference.helpers import raw_observation_to_observation
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.factory import get_policy_class, load_pretrained_policy_for_inference, make_pre_post_processors
from lerobot.utils.feature_utils import hw_to_dataset_features
from lerobot.utils.import_utils import register_third_party_plugins

RENAME = {"observation.images.front": "observation.images.base_0_rgb",
          "observation.images.wrist": "observation.images.left_wrist_0_rgb"}
JOINTS = [f"joint.{i}" for i in range(6)] + ["joint.gripper_pos"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--policy", required=True)
    p.add_argument("--replay", required=True)
    p.add_argument("--frame", type=int, default=0)
    p.add_argument("--rounds", type=int, default=2)
    a = p.parse_args()
    register_third_party_plugins()
    cfg = PreTrainedConfig.from_pretrained(a.policy)
    pol = load_pretrained_policy_for_inference(a.policy, config=cfg, policy_cls=get_policy_class(cfg.type)).to("cpu").eval()
    pre, post = make_pre_post_processors(
        cfg, pretrained_path=a.policy,
        preprocessor_overrides={"device_processor": {"device": "cpu"}, "rename_observations_processor": {"rename_map": RENAME}},
        postprocessor_overrides={"device_processor": {"device": "cpu"}})
    data = np.load(a.replay)
    raw = {k: float(data["state"][a.frame, i]) for i, k in enumerate(JOINTS)}
    raw["front"], raw["wrist"], raw["task"] = data["front"][a.frame], data["wrist"][a.frame], str(data["task"])
    feats = hw_to_dataset_features({**dict.fromkeys(JOINTS, float), "front": (480, 640, 3), "wrist": (480, 640, 3)},
                                   "observation", use_video=False)
    img_feats = dict(cfg.image_features)
    for rk, pk in RENAME.items():
        if pk in img_feats:
            img_feats[rk] = img_feats[pk]
    batch = pre(raw_observation_to_observation(raw, feats, img_feats))
    print("policy:", type(pol).__name__, "->", cfg.server_url, "| task:", raw["task"])
    print("observed state:", [round(float(x), 3) for x in data["state"][a.frame]])
    for n in range(a.rounds):
        pol.reset()
        t = time.time()
        act = post(pol.select_action(batch))
        keys = ("expert_backend", "from_hardware", "prefix_s", "denoise_s", "total_s", "rpc_round_trip_s")
        print(f"round {n + 1}: {time.time() - t:.2f}s", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in pol.last_stats.items() if k in keys})
        print("  first action :", [round(float(x), 3) for x in act.flatten()])
    print("queue after one pop:", len(pol._action_queue))
    t = time.time()
    pol.select_action(batch)
    print(f"next pop from queue: {time.time() - t:.4f}s (no RPC)")
    assert torch.isfinite(act).all()
    print("PI0_REMOTE_ROUNDTRIP_OK")


if __name__ == "__main__":
    main()
