#!/usr/bin/env python
"""Desktop side of the split pi0: serve `predict_action_chunk` over HTTP.

The Jetson's `pi0_remote` policy posts the *preprocessed* batch (images
224x224 float, normalised state, language tokens) plus the RTC kwargs; this
server runs the real pi0 through PI0FpgaPolicy (expert torch | fpga | fpga_torch
from PI0_ACTION_EXPERT) and returns the normalised action chunk.  Post-processing
stays on the Jetson.  One request at a time; per-chunk PI0_STAGE timing lines as
in the async server.

    ASYNC_POLICY_PATH=<pi0 checkpoint> PI0_ACTION_EXPERT=torch python pi0_policy_rpc_server.py --port 8081
    curl http://127.0.0.1:8081/health
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import os
import pickle  # nosec
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pi0_fpga_policy import PI0FpgaPolicy  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(asctime)s %(name)s: %(message)s")
logger = logging.getLogger("pi0_policy_rpc")


class State:
    policy: PI0FpgaPolicy | None = None
    lock = threading.Lock()
    chunks = 0
    started = time.time()
    rtc_applied: dict | None = None


def apply_rtc(policy: PI0FpgaPolicy, rtc: dict | None) -> None:
    """Mirror lerobot_rollout: policy.config.rtc_config = RTCConfig(...); init_rtc_processor()."""
    if rtc == State.rtc_applied:
        return
    from lerobot.policies.rtc.configuration_rtc import RTCAttentionSchedule, RTCConfig

    if rtc is None:
        policy.config.rtc_config = None
        policy.rtc_processor = None
    else:
        fields = {f.name for f in dataclasses.fields(RTCConfig)}
        kw = {k: v for k, v in rtc.items() if k in fields}
        if isinstance(kw.get("prefix_attention_schedule"), str):
            kw["prefix_attention_schedule"] = RTCAttentionSchedule[kw["prefix_attention_schedule"]]
        policy.config.rtc_config = RTCConfig(**kw)
        policy.init_rtc_processor()
    State.rtc_applied = rtc
    logger.info("RTC config now: %s", rtc)


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):  # quieter than the default
        logger.debug(fmt, *args)

    def _send(self, code: int, body: bytes, ctype: str = "application/octet-stream"):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):  # noqa: N802
        if self.path.startswith("/health"):
            p = State.policy
            info = {
                "ok": p is not None,
                "policy_type": "pi0_fpga",
                "expert_backend": getattr(p, "expert_backend", None),
                "from_hardware": bool(p.bridge.is_hardware) if p is not None and getattr(p, "bridge", None) is not None and hasattr(p.bridge, "is_hardware") else False,
                "chunks_served": State.chunks,
                "uptime_s": time.time() - State.started,
                "last_stats": getattr(p, "last_stats", {}),
                "rtc": State.rtc_applied,
            }
            self._send(200, json.dumps(info, default=str).encode(), "application/json")
        else:
            self._send(404, b"not found", "text/plain")

    def do_POST(self):  # noqa: N802
        if not self.path.startswith("/predict"):
            self._send(404, b"not found", "text/plain")
            return
        n = int(self.headers.get("Content-Length", "0"))
        payload = pickle.loads(self.rfile.read(n))  # nosec
        t0 = time.perf_counter()
        try:
            with State.lock:
                policy = State.policy
                apply_rtc(policy, payload.get("rtc_config"))
                batch = payload["batch"]
                kwargs = payload.get("kwargs", {})
                rtc_wanted = payload.get("rtc_config") is not None and any(
                    k in kwargs for k in ("inference_delay", "prev_chunk_left_over", "execution_horizon"))
                if rtc_wanted:
                    if policy.expert_backend != "torch":
                        raise RuntimeError("RTC is only available with PI0_ACTION_EXPERT=torch on this server")
                    from lerobot.policies.pi0.modeling_pi0 import PI0Policy

                    actions = PI0Policy.predict_action_chunk(policy, batch, **kwargs)  # stock RTC path, same weights
                    stats = {"expert_backend": "torch", "from_hardware": False, "rtc": True,
                             "total_s": time.perf_counter() - t0}
                else:
                    actions = policy.predict_action_chunk(batch, **kwargs)
                    stats = dict(policy.last_stats)
                State.chunks += 1
            stats["server_wall_s"] = time.perf_counter() - t0
            body = pickle.dumps({"actions": actions.detach().cpu(), "stats": stats})
            self._send(200, body)
            logger.info("served chunk %d in %.2fs (expert=%s from_hardware=%s)", State.chunks,
                        stats["server_wall_s"], stats.get("expert_backend"), stats.get("from_hardware"))
        except Exception as e:  # report to the client instead of dropping the connection
            logger.exception("predict failed")
            self._send(500, f"{type(e).__name__}: {e}".encode(), "text/plain")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--host", default=os.environ.get("PI0_RPC_HOST", "0.0.0.0"))
    p.add_argument("--port", type=int, default=int(os.environ.get("PI0_RPC_PORT", "8081")))
    p.add_argument("--policy", default=os.environ.get("ASYNC_POLICY_PATH"))
    p.add_argument("--device", default=os.environ.get("ASYNC_POLICY_DEVICE", "cpu"))
    a = p.parse_args()
    if not a.policy:
        raise SystemExit("set ASYNC_POLICY_PATH or --policy")
    t = time.time()
    if os.environ.get("PI0_ACTION_EXPERT", "").strip().lower() == "chip":
        # the whole model runs on the node array: the host only touches the embedding tables, the SigLIP patch
        # convolution and state_proj, so map the checkpoint copy-on-write instead of loading 16 GB (runs beside
        # ACE builds); paper/sw/prefix_w8a8_eval.load_policy builds the policy on meta and assigns the mmap tensors
        sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "paper" / "sw"))
        import prefix_w8a8_eval as P  # noqa: E402
        policy, _, _ = P.load_policy(a.policy)
    else:
        policy = PI0FpgaPolicy.from_pretrained(a.policy)
        policy.to(a.device).eval()
    State.policy = policy
    logger.info("loaded %s in %.0fs; expert=%s; serving on %s:%d", a.policy, time.time() - t,
                policy.expert_backend, a.host, a.port)
    server = ThreadingHTTPServer((a.host, a.port), Handler)
    server.request_queue_size = 4
    logger.info("PI0_RPC_SERVER_READY")
    server.serve_forever()


if __name__ == "__main__":
    main()
