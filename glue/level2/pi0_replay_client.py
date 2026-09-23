#!/usr/bin/env python
"""Level-2 stub client: replay recorded UR7e observations against a LeRobot
`async_inference` policy server.  No arm, no cameras, no FPGA.

It drives the *stock* `RobotClient` (same threads, same queue logic, same gRPC
service) with a duck-typed replay robot injected in place of the UR7e, and
records what the handoff asks for: per-chunk wall time, round trip, observed
throughput and the action-queue depth over time.

Round trip is clock-sync free: the server stamps the first action of a chunk
with the *client's own* observation timestamp, so `receive_time - t0` is the
full send + inference + return latency on one clock.

Example (Jetson, conda env `lerobot`):
    python pi0_replay_client.py --server 10.162.174.116:8080 \
        --replay ~/pi0_glue/replay_ur7e_demo1_ep0.npz --fps 1 \
        --chunk-size-threshold 0.6 --duration 300 --out /tmp/level2_run
"""

from __future__ import annotations

import argparse
import json
import logging
import pickle  # nosec
import platform
import shutil
import socket
import statistics
import subprocess
import threading
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch

from lerobot.async_inference.configs import RobotClientConfig
from lerobot.async_inference.helpers import TimedAction, TimedObservation
from lerobot.async_inference.robot_client import RobotClient
from lerobot.robots.ur_follower_joint.config_ur_follower_joint import URJointFollowerConfig

JOINT_KEYS = [f"joint.{i}" for i in range(6)] + ["joint.gripper_pos"]


class ReplayRobot:
    """Duck-typed stand-in for URJointFollower: same observation/action features,
    observations come from a recorded .npz, actions are logged and discarded."""

    name = "replay_ur7e"

    def __init__(self, npz_path: str, state_source: str = "recorded", speed: float = 1.0):
        d = np.load(npz_path, allow_pickle=False)
        self.state = d["state"].astype(np.float32)
        self.front = d["front"]
        self.wrist = d["wrist"]
        self.frame_dt = float(d["frame_dt"]) / max(speed, 1e-6)
        self.task = str(d["task"])
        self.source = str(d["source"])
        self.n = int(self.state.shape[0])
        self.state_source = state_source
        self._connected = False
        self._t_start: float | None = None
        self._last_action: dict[str, float] | None = None
        self._lock = threading.Lock()
        self.actions_sent: list[tuple[float, list[float]]] = []
        self.frames_served: list[tuple[float, int]] = []

    # --- Robot interface used by RobotClient -------------------------------------------------
    @property
    def observation_features(self) -> dict:
        h, w, c = self.front.shape[1:]
        return {**dict.fromkeys(JOINT_KEYS, float), "front": (h, w, c), "wrist": (h, w, c)}

    @property
    def action_features(self) -> dict:
        return dict.fromkeys(JOINT_KEYS, float)

    @property
    def is_connected(self) -> bool:
        return self._connected

    def connect(self, calibrate: bool = True) -> None:
        self._connected = True
        self._t_start = time.monotonic()

    def disconnect(self) -> None:
        self._connected = False

    def get_observation(self) -> dict:
        now = time.monotonic()
        idx = int((now - self._t_start) / self.frame_dt) % self.n
        with self._lock:
            self.frames_served.append((time.time(), idx))
            last = self._last_action
        obs: dict = {}
        if self.state_source == "follow_action" and last is not None:
            for k in JOINT_KEYS:
                obs[k] = float(last[k])
        else:
            for i, k in enumerate(JOINT_KEYS):
                obs[k] = float(self.state[idx, i])
        obs["front"] = self.front[idx]
        obs["wrist"] = self.wrist[idx]
        return obs

    def send_action(self, action: dict) -> dict:
        with self._lock:
            self._last_action = dict(action)
            self.actions_sent.append((time.time(), [float(action[k]) for k in JOINT_KEYS]))
        return action


class ReplayClient(RobotClient):
    """RobotClient with timing hooks; no behaviour change."""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.obs_log: list[dict] = []
        self.chunk_log: list[dict] = []
        self._obs_bytes: int | None = None

    def send_observation(self, obs: TimedObservation) -> bool:
        if self._obs_bytes is None:
            self._obs_bytes = len(pickle.dumps(obs))
        with self.action_queue_lock:
            qsize = self.action_queue.qsize()
        t0 = time.perf_counter()
        ok = super().send_observation(obs)
        dt = time.perf_counter() - t0
        self.obs_log.append(
            {
                "wall": obs.get_timestamp(),
                "timestep": obs.get_timestep(),
                "must_go": bool(obs.must_go),
                "queue_size_at_send": qsize,
                "send_s": dt,
                "ok": ok,
            }
        )
        return ok

    def _aggregate_action_queues(
        self, incoming_actions: list[TimedAction], aggregate_fn: Callable | None = None
    ):
        now = time.time()
        with self.action_queue_lock:
            before = self.action_queue.qsize()
        if incoming_actions:
            t0 = incoming_actions[0].get_timestamp()
            self.chunk_log.append(
                {
                    "wall": now,
                    "n_actions": len(incoming_actions),
                    "first_timestep": incoming_actions[0].get_timestep(),
                    "last_timestep": incoming_actions[-1].get_timestep(),
                    "obs_wall": t0,
                    "round_trip_s": now - t0,
                    "queue_before": before,
                }
            )
        super()._aggregate_action_queues(incoming_actions, aggregate_fn)
        if incoming_actions:
            with self.action_queue_lock:
                self.chunk_log[-1]["queue_after"] = self.action_queue.qsize()


def ping_host(host: str, count: int = 5) -> dict | None:
    if shutil.which("ping") is None:
        return None
    try:
        out = subprocess.run(
            ["ping", "-c", str(count), "-W", "2", host], capture_output=True, text=True, timeout=30
        ).stdout
    except Exception:
        return None
    rtts = []
    for line in out.splitlines():
        if "time=" in line:
            try:
                rtts.append(float(line.split("time=")[1].split()[0]))
            except ValueError:
                pass
    if not rtts:
        return {"raw": out.strip()[-200:]}
    return {"min_ms": min(rtts), "avg_ms": statistics.mean(rtts), "max_ms": max(rtts), "n": len(rtts)}


def stats(xs: list[float]) -> dict | None:
    if not xs:
        return None
    xs = sorted(xs)
    return {
        "n": len(xs),
        "min": xs[0],
        "median": statistics.median(xs),
        "mean": statistics.mean(xs),
        "p90": xs[min(len(xs) - 1, int(0.9 * (len(xs) - 1)))],
        "max": xs[-1],
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--server", required=True, help="host:port of the policy server")
    p.add_argument("--replay", required=True, help=".npz from export_replay_frames.py")
    p.add_argument("--fps", type=float, default=1.0, help="client control-loop rate (ASYNC_FPS)")
    p.add_argument("--chunk-size-threshold", type=float, default=0.6)
    p.add_argument("--aggregate-fn", default="weighted_average")
    p.add_argument("--action-window", type=int, default=None)
    p.add_argument("--task", default=None, help="override the task string stored in the replay file")
    p.add_argument("--state-source", choices=["recorded", "follow_action"], default="recorded",
                   help="recorded: joint state from the dataset; follow_action: echo the last commanded action")
    p.add_argument("--replay-speed", type=float, default=1.0, help="1.0 = real time of the recording")
    p.add_argument("--duration", type=float, default=300.0, help="seconds to run before stopping")
    p.add_argument("--queue-sample-hz", type=float, default=10.0)
    p.add_argument("--out", required=True, help="output directory for report.json / queue.png")
    p.add_argument("--label", default="", help="free-text label stored in the report (e.g. policy name)")
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    logging.getLogger().setLevel(logging.DEBUG if args.verbose else logging.INFO)

    robot = ReplayRobot(args.replay, state_source=args.state_source, speed=args.replay_speed)
    task = args.task if args.task is not None else robot.task

    # RobotClientConfig needs a RobotConfig instance; it is never used to connect anything
    # because the robot is injected below (owns_robot=False).
    cfg = RobotClientConfig(
        robot=URJointFollowerConfig(robot_ip="0.0.0.0", id="replay_ur7e", use_gripper=True, cameras={}),
        task=task,
        server_address=args.server,
        client_device="cpu",
        chunk_size_threshold=args.chunk_size_threshold,
        action_window=args.action_window,
        fps=int(args.fps) if float(args.fps).is_integer() else args.fps,
        aggregate_fn_name=args.aggregate_fn,
        debug_visualize_queue_size=False,
    )

    host = args.server.rsplit(":", 1)[0]
    link_ping = ping_host(host)

    robot.connect()
    client = ReplayClient(cfg, robot=robot, owns_robot=False)

    t_handshake = time.perf_counter()
    if not client.start():
        raise SystemExit("could not connect to the policy server")
    handshake_s = time.perf_counter() - t_handshake
    client.logger.info(f"handshake + policy instructions took {handshake_s * 1000:.1f} ms")

    queue_series: list[tuple[float, int]] = []
    stop_sampler = threading.Event()

    def sampler():
        period = 1.0 / args.queue_sample_hz
        while not stop_sampler.is_set():
            with client.action_queue_lock:
                q = client.action_queue.qsize()
            queue_series.append((time.time(), q))
            stop_sampler.wait(period)

    t_run0 = time.time()
    recv_thread = threading.Thread(target=client.receive_actions, kwargs={"verbose": args.verbose}, daemon=True)
    ctrl_thread = threading.Thread(target=client.control_loop, kwargs={"task": task, "verbose": args.verbose}, daemon=True)
    sampler_thread = threading.Thread(target=sampler, daemon=True)
    recv_thread.start()
    ctrl_thread.start()
    sampler_thread.start()

    try:
        deadline = time.time() + args.duration
        while time.time() < deadline and ctrl_thread.is_alive():
            time.sleep(1.0)
            if int(time.time() - t_run0) % 30 == 0:
                client.logger.info(
                    f"[{time.time() - t_run0:6.0f}s] obs sent={len(client.obs_log)} chunks={len(client.chunk_log)} "
                    f"actions executed={len(robot.actions_sent)} queue={queue_series[-1][1] if queue_series else '?'}"
                )
    except KeyboardInterrupt:
        client.logger.info("interrupted")
    finally:
        stop_sampler.set()
        client.stop()
        ctrl_thread.join(timeout=5)
        recv_thread.join(timeout=10)
        robot.disconnect()
    t_run1 = time.time()

    # ---- report -----------------------------------------------------------------------------
    chunk_rt = [c["round_trip_s"] for c in client.chunk_log]
    inter_arrival = [b["wall"] - a["wall"] for a, b in zip(client.chunk_log, client.chunk_log[1:])]
    send_s = [o["send_s"] for o in client.obs_log if o["ok"]]
    drained = 0
    prev_q = None
    for _, q in queue_series:
        if prev_q is not None and prev_q > 0 and q == 0:
            drained += 1
        prev_q = q
    # time with an empty queue after the first chunk arrived (= "safe pause" time)
    empty_s = 0.0
    first_chunk = client.chunk_log[0]["wall"] if client.chunk_log else None
    for (ta, qa), (tb, _) in zip(queue_series, queue_series[1:]):
        if first_chunk is not None and ta >= first_chunk and qa == 0:
            empty_s += tb - ta

    report = {
        "label": args.label,
        "client_host": socket.gethostname(),
        "client_platform": platform.platform(),
        "server": args.server,
        "replay": {"path": args.replay, "source": robot.source, "frames": robot.n, "frame_dt_s": robot.frame_dt,
                   "task": task, "state_source": args.state_source},
        "client_config": {k: v for k, v in cfg.to_dict().items() if k != "robot"},
        "observation_bytes": client._obs_bytes,
        "link_ping": link_ping,
        "handshake_s": handshake_s,
        "run_s": t_run1 - t_run0,
        "observations_sent": len(client.obs_log),
        "observations_must_go": sum(1 for o in client.obs_log if o["must_go"]),
        "observation_send_s": stats(send_s),
        "observation_send_MBps": (client._obs_bytes / 1e6) / statistics.median(send_s) if send_s and client._obs_bytes else None,
        "chunks_received": len(client.chunk_log),
        "chunk_round_trip_s": stats(chunk_rt),
        "chunk_inter_arrival_s": stats(inter_arrival),
        "chunk_sizes": sorted({c["n_actions"] for c in client.chunk_log}),
        "actions_executed": len(robot.actions_sent),
        "actions_per_s": len(robot.actions_sent) / (t_run1 - t_run0) if t_run1 > t_run0 else None,
        "queue_drain_events": drained,
        "queue_empty_s_after_first_chunk": empty_s,
        "queue_max": max((q for _, q in queue_series), default=0),
        "obs_log": client.obs_log,
        "chunk_log": client.chunk_log,
        "queue_series": queue_series,
        "actions_sent": robot.actions_sent,
        "frames_served": robot.frames_served,
    }
    (out / "report.json").write_text(json.dumps(report, indent=1))

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 4))
        if queue_series:
            t = [x[0] - t_run0 for x in queue_series]
            ax.step(t, [x[1] for x in queue_series], where="post", label="action queue size")
        for c in client.chunk_log:
            ax.axvline(c["wall"] - t_run0, color="tab:green", alpha=0.3, lw=0.8)
        for o in client.obs_log:
            ax.axvline(o["wall"] - t_run0, color="tab:red" if o["must_go"] else "tab:orange", alpha=0.25, lw=0.6, ls=":")
        ax.set_xlabel("s since start")
        ax.set_ylabel("queue size")
        ax.set_title(f"level-2 replay {args.label} @ {args.server}  (green=chunk arrived, red dotted=must_go obs)")
        ax.grid(alpha=0.3)
        ax.legend(loc="upper right")
        fig.tight_layout()
        fig.savefig(out / "queue.png", dpi=120)
    except Exception as e:  # matplotlib is optional on the client
        client.logger.warning(f"queue.png not written: {e}")

    def fmt(s):
        return "n/a" if not s else f"n={s['n']} min={s['min']:.3f} med={s['median']:.3f} p90={s['p90']:.3f} max={s['max']:.3f}"

    print("\n==== level-2 replay summary ====")
    print(f"server            : {args.server}   ping: {link_ping}")
    print(f"run               : {report['run_s']:.1f} s   handshake {handshake_s * 1000:.1f} ms   obs pickle {client._obs_bytes} B")
    print(f"observations sent : {report['observations_sent']} (must_go {report['observations_must_go']})  send s: {fmt(report['observation_send_s'])}")
    print(f"chunks received   : {report['chunks_received']} sizes {report['chunk_sizes']}  round trip s: {fmt(report['chunk_round_trip_s'])}")
    print(f"chunk inter-arrival s: {fmt(report['chunk_inter_arrival_s'])}")
    print(f"actions executed  : {report['actions_executed']}  ({report['actions_per_s'] or 0:.2f} /s at fps={args.fps})")
    print(f"queue             : max {report['queue_max']}  drain events {drained}  empty {empty_s:.1f} s after first chunk")
    print(f"report            : {out / 'report.json'}")


if __name__ == "__main__":
    main()
