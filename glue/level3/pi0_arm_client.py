#!/usr/bin/env python
"""Jetson-side arm client for the pi0 policy server: the stock LeRobot `RobotClient`
with the real UR7e, plus the safety wrapping the handoff §4.4 asks for.

From the arm's point of view there is only this process on the Jetson: it owns
the RTDE connection, the servoJ keepalive, and the stop path.  The Desktop is
just where action chunks come from.

Adds to the stock client (no LeRobot source edits):
  * chunk watchdog: if no action chunk has arrived for --watchdog-s seconds the
    client stops itself -> robot.disconnect() -> servoStop + stopScript (arm holds
    where it is; the UR's own e-stop and protective stop remain independent);
  * explicit stop: SIGINT/SIGTERM, or touch --stop-file, does the same;
  * --max-run-s hard cap;
  * the same report.json / queue.png as the replay client (obs send time, chunk
    round trip, queue depth) so arm runs and replay runs are directly comparable.

Usage (Jetson, conda env lerobot, after sourcing lerobot_site.env -- see run_pi0_arm_client.sh):
    python pi0_arm_client.py --robot.type=ur_follower_joint --robot.robot_ip=$ROBOT_IP ... \
        --server_address=192.168.10.1:8080 --fps=1 --chunk_size_threshold=0.6 \
        --task="..." --watchdog_s=60 --out=~/pi0_glue/runs/arm_<name>
"""

from __future__ import annotations

import json
import logging
import os
import signal
import statistics
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import draccus

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "level2"))  # pi0_replay_client (same dir on the Jetson)

from lerobot.async_inference.configs import RobotClientConfig  # noqa: E402
from lerobot.async_inference.robot_client import RobotClient  # noqa: E402
from lerobot.utils.import_utils import register_third_party_plugins  # noqa: E402

from pi0_replay_client import ReplayClient, ping_host, stats  # noqa: E402  (instrumentation only)


@dataclass
class ArmClientConfig(RobotClientConfig):
    watchdog_s: float = field(default=60.0, metadata={"help": "stop if no chunk arrives for this long (0 = off)"})
    max_run_s: float = field(default=0.0, metadata={"help": "hard cap on the run (0 = none)"})
    stop_file: str = field(default="/tmp/pi0_arm_stop", metadata={"help": "touching this file stops the client"})
    out: str = field(default="", metadata={"help": "report directory (report.json, queue.png)"})
    label: str = field(default="arm", metadata={"help": "free-text label in the report"})


class ArmClient(ReplayClient):
    """RobotClient + timing hooks, driving the real robot (owns it)."""

    def __init__(self, config: ArmClientConfig):
        RobotClient.__init__(self, config)  # connects the real robot
        self.obs_log, self.chunk_log, self._obs_bytes = [], [], None


def main() -> None:
    register_third_party_plugins()

    # draccus.wrap() infers the config class from the annotation of its wrapped
    # function's first parameter.  `from __future__ import annotations` (line 24)
    # makes that annotation the *string* "ArmClientConfig", and draccus 0.8.0
    # hands the string straight to dataclasses.fields(), which raises
    # "must be called with a dataclass type or instance".  This is why the arm
    # client had never started: it fails before it connects to anything.
    # draccus.parse() takes the class explicitly, so it is immune to PEP 563.
    cfg = draccus.parse(config_class=ArmClientConfig, args=sys.argv[1:])
    logging.getLogger().setLevel(logging.INFO)
    out = Path(cfg.out or f"~/pi0_glue/runs/arm_{int(time.time())}").expanduser()
    out.mkdir(parents=True, exist_ok=True)
    stop_file = Path(cfg.stop_file)
    if stop_file.exists():
        stop_file.unlink()

    host = cfg.server_address.rsplit(":", 1)[0]
    link_ping = ping_host(host)
    client = ArmClient(cfg)
    log = client.logger
    log.info(f"arm client: robot={client.robot} server={cfg.server_address} fps={cfg.fps} watchdog={cfg.watchdog_s}s "
             f"stop_file={stop_file} ping={link_ping}")

    stop_reason = {"reason": None}

    def request_stop(reason: str):
        if stop_reason["reason"] is None:
            stop_reason["reason"] = reason
            log.warning(f"STOP requested: {reason}")
            client.shutdown_event.set()

    signal.signal(signal.SIGINT, lambda *_: request_stop("SIGINT"))
    signal.signal(signal.SIGTERM, lambda *_: request_stop("SIGTERM"))

    t_hs = time.perf_counter()
    if not client.start():
        client.robot.disconnect()
        raise SystemExit("could not connect to the policy server")
    handshake_s = time.perf_counter() - t_hs

    queue_series: list[tuple[float, int]] = []
    t_run0 = time.time()
    recv = threading.Thread(target=client.receive_actions, daemon=True)
    ctrl = threading.Thread(target=client.control_loop, kwargs={"task": cfg.task}, daemon=True)
    recv.start()
    ctrl.start()
    last_chunk_wall = time.time()
    try:
        while client.running and ctrl.is_alive():
            time.sleep(0.1)
            with client.action_queue_lock:
                queue_series.append((time.time(), client.action_queue.qsize()))
            if client.chunk_log:
                last_chunk_wall = max(last_chunk_wall, client.chunk_log[-1]["wall"])
            if cfg.watchdog_s > 0 and time.time() - last_chunk_wall > cfg.watchdog_s:
                request_stop(f"no action chunk for {cfg.watchdog_s:.0f}s")
            if cfg.max_run_s > 0 and time.time() - t_run0 > cfg.max_run_s:
                request_stop("max_run_s reached")
            if stop_file.exists():
                request_stop(f"stop file {stop_file}")
    finally:
        client.stop()          # -> robot.disconnect(): keepalive off, servoStop, stopScript
        ctrl.join(timeout=5)
        recv.join(timeout=10)
    t_run1 = time.time()

    chunk_rt = [c["round_trip_s"] for c in client.chunk_log]
    send_s = [o["send_s"] for o in client.obs_log if o["ok"]]
    inter = [b["wall"] - a["wall"] for a, b in zip(client.chunk_log, client.chunk_log[1:])]
    report = {
        "label": cfg.label, "mode": "arm", "stop_reason": stop_reason["reason"] or "control loop ended",
        "server": cfg.server_address, "link_ping": link_ping, "handshake_s": handshake_s,
        "client_config": {k: v for k, v in cfg.to_dict().items()},
        "watchdog_s": cfg.watchdog_s, "run_s": t_run1 - t_run0,
        "observation_bytes": client._obs_bytes,
        "observations_sent": len(client.obs_log), "observation_send_s": stats(send_s),
        "chunks_received": len(client.chunk_log), "chunk_round_trip_s": stats(chunk_rt),
        "chunk_inter_arrival_s": stats(inter),
        "queue_max": max((q for _, q in queue_series), default=0),
        "obs_log": client.obs_log, "chunk_log": client.chunk_log, "queue_series": queue_series,
    }
    (out / "report.json").write_text(json.dumps(report, indent=1))
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(10, 4))
        if queue_series:
            ax.step([t - t_run0 for t, _ in queue_series], [q for _, q in queue_series], where="post")
        for c in client.chunk_log:
            ax.axvline(c["wall"] - t_run0, color="tab:green", alpha=0.3, lw=0.8)
        ax.set_xlabel("s since start"); ax.set_ylabel("queue size"); ax.grid(alpha=0.3)
        ax.set_title(f"arm run {cfg.label} @ {cfg.server_address} (stop: {report['stop_reason']})")
        fig.tight_layout(); fig.savefig(out / "queue.png", dpi=120)
    except Exception as e:
        log.warning(f"queue.png not written: {e}")
    print(f"\n==== arm run summary ====\nstop reason: {report['stop_reason']}\nrun {report['run_s']:.1f}s  "
          f"obs {report['observations_sent']}  chunks {report['chunks_received']}  "
          f"round trip {stats(chunk_rt)}\nreport: {out / 'report.json'}")


if __name__ == "__main__":
    main()
