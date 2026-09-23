#!/usr/bin/env python
"""Build a `pi0_remote` policy directory from a real pi0 checkpoint directory.

Copies the checkpoint's pre/post-processor files (normaliser stats, tokenizer
config) so the robot side processes observations exactly as it does with the
local checkpoint, writes a config.json of type pi0_remote mirroring the pi0
layout, and a one-buffer model.safetensors.  No weights leave the Desktop.

    python make_pi0_remote_policy_dir.py \
        --checkpoint ~/pi0_ckpt/ur7e-demo-2-pi0-fsdp/010000/pretrained_model \
        --server-url http://192.168.10.1:8081 --out ~/pi0_glue/policies/pi0_remote_ur7e-demo-2-pi0-010000
"""

import argparse
import json
import shutil
from pathlib import Path

import torch
from safetensors.torch import save_file


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--server-url", default="http://192.168.10.1:8081")
    p.add_argument("--out", required=True)
    a = p.parse_args()
    src = Path(a.checkpoint).expanduser()
    out = Path(a.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    cfg = json.loads((src / "config.json").read_text())
    if cfg.get("type") != "pi0":
        raise SystemExit(f"{src} is a {cfg.get('type')} checkpoint; pi0_remote mirrors pi0 only")
    remote = {
        "type": "pi0_remote",
        "server_url": a.server_url,
        "request_timeout_s": 180.0,
        "remote_policy_type": "pi0",
        "n_obs_steps": cfg["n_obs_steps"],
        "input_features": cfg["input_features"],
        "output_features": cfg["output_features"],
        "device": "cpu",
        "use_amp": False,
        "chunk_size": cfg["chunk_size"],
        "n_action_steps": cfg["n_action_steps"],
        "max_state_dim": cfg["max_state_dim"],
        "max_action_dim": cfg["max_action_dim"],
        "tokenizer_max_length": cfg["tokenizer_max_length"],
        "image_resolution": cfg["image_resolution"],
        "normalization_mapping": cfg["normalization_mapping"],
        "action_feature_names": cfg.get("action_feature_names"),
        "rtc_config": None,
        "source_checkpoint": str(src),
    }
    (out / "config.json").write_text(json.dumps(remote, indent=4))
    copied = []
    for f in sorted(src.iterdir()):
        if f.name.startswith("policy_pre") or f.name.startswith("policy_post"):
            shutil.copy2(f, out / f.name)
            copied.append(f.name)
    save_file({"_device_anchor": torch.zeros(1)}, str(out / "model.safetensors"))
    print(f"wrote {out}: config.json (pi0_remote -> {a.server_url}), model.safetensors (anchor only), copied {copied}")


if __name__ == "__main__":
    main()
