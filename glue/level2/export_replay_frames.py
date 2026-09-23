#!/usr/bin/env python
"""Export a handful of recorded UR7e frames into a self-contained .npz replay file.

Runs on the machine that holds the LeRobot dataset (the Jetson, conda env `lerobot`).
The output needs only numpy to read, so the replay client does not depend on the
dataset/torchcodec stack.

Example (Jetson):
    python export_replay_frames.py \
        --root ~/lerobot/outputs/recordings/local/ur7e-demo-1_20260721_161742 \
        --repo-id local/ur7e-demo-1 --episode 0 --stride 30 --max-frames 30 \
        --out ~/pi0_glue/replay_ur7e_demo1_ep0.npz
"""

import argparse
import time

import numpy as np
import torch

from lerobot.datasets.lerobot_dataset import LeRobotDataset

JOINT_KEYS = [f"joint.{i}" for i in range(6)] + ["joint.gripper_pos"]


def to_uint8_hwc(img: torch.Tensor) -> np.ndarray:
    # dataset gives float32 (C, H, W) in [0, 1]; the robot gives uint8 (H, W, C)
    if img.dtype == torch.uint8:
        arr = img
    else:
        arr = (img.clamp(0, 1) * 255.0).round().to(torch.uint8)
    return arr.permute(1, 2, 0).contiguous().numpy()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    p.add_argument("--repo-id", default="local/ur7e-demo-1")
    p.add_argument("--episode", type=int, default=0)
    p.add_argument("--stride", type=int, default=30, help="take every k-th frame of the episode")
    p.add_argument("--max-frames", type=int, default=30)
    p.add_argument("--front-key", default="observation.images.front")
    p.add_argument("--wrist-key", default="observation.images.wrist")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    ds = LeRobotDataset(args.repo_id, root=args.root)
    fps = float(ds.meta.fps)
    ep_from = int(ds.meta.episodes[args.episode]["dataset_from_index"])
    ep_to = int(ds.meta.episodes[args.episode]["dataset_to_index"])
    idxs = list(range(ep_from, ep_to, args.stride))[: args.max_frames]
    print(f"dataset {args.repo_id}: {len(ds)} frames, fps={fps}; episode {args.episode} = [{ep_from}, {ep_to})")
    print(f"exporting {len(idxs)} frames, stride {args.stride} -> {args.stride / fps:.3f} s between frames")

    states, actions, fronts, wrists, ts = [], [], [], [], []
    t0 = time.time()
    task = None
    for i in idxs:
        x = ds[i]
        states.append(x["observation.state"].numpy().astype(np.float32))
        actions.append(x["action"].numpy().astype(np.float32))
        fronts.append(to_uint8_hwc(x[args.front_key]))
        wrists.append(to_uint8_hwc(x[args.wrist_key]))
        ts.append(float(x["timestamp"]))
        task = x.get("task", task)
    print(f"decoded in {time.time() - t0:.1f} s; task = {task!r}")

    np.savez(
        args.out,
        state=np.stack(states),
        action=np.stack(actions),
        front=np.stack(fronts),
        wrist=np.stack(wrists),
        timestamp=np.asarray(ts, dtype=np.float64),
        frame_dt=np.float64(args.stride / fps),
        source_fps=np.float64(fps),
        joint_keys=np.asarray(JOINT_KEYS),
        task=np.asarray(task if task is not None else ""),
        source=np.asarray(f"{args.repo_id} episode {args.episode} stride {args.stride} from {args.root}"),
    )
    print(f"wrote {args.out}: state {np.stack(states).shape}, front {np.stack(fronts).shape}, wrist {np.stack(wrists).shape}")


if __name__ == "__main__":
    main()
