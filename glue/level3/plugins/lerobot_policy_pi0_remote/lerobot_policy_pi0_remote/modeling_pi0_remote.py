"""Pi0RemotePolicy: the robot-side half of a split pi0.

The rollout on the Jetson keeps everything it does today -- cameras, RTDE, the
pre-processor (resize, normalise, tokenise) and the post-processor
(unnormalise) -- and only `predict_action_chunk` leaves the machine: the
preprocessed batch is pickled to the Desktop's `pi0_policy_rpc_server.py`,
which runs the real pi0 (torch expert, or the FPGA path) and returns the
normalised action chunk.  From the arm's point of view there is still one
process on the Jetson.
"""

from __future__ import annotations

import dataclasses
import logging
import pickle  # nosec: trusted link between our two machines, same as LeRobot's own transport
import time
from collections import deque

import requests
import torch
from torch import Tensor

from lerobot.policies.pretrained import PreTrainedPolicy

from .configuration_pi0_remote import Pi0RemoteConfig

logger = logging.getLogger(__name__)


def _to_cpu(obj):
    if isinstance(obj, torch.Tensor):
        return obj.detach().to("cpu")
    if isinstance(obj, dict):
        return {k: _to_cpu(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(_to_cpu(v) for v in obj)
    return obj


class Pi0RemotePolicy(PreTrainedPolicy):
    config_class = Pi0RemoteConfig
    name = "pi0_remote"

    def __init__(self, config: Pi0RemoteConfig, **kwargs):
        super().__init__(config)
        self.config = config
        # one buffer so .to(device) / dtype casts / safetensors loading have something to act on
        self.register_buffer("_device_anchor", torch.zeros(1))
        self._action_queue: deque[Tensor] = deque()
        self.last_stats: dict = {}
        self._session = requests.Session()
        self._chunks = 0
        logger.info("pi0_remote: chunks come from %s (%s)", config.server_url, config.remote_policy_type)

    # ---- PreTrainedPolicy interface ------------------------------------------------------
    def get_optim_params(self) -> dict:
        return {}

    def reset(self):
        self._action_queue.clear()

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict | None]:
        raise NotImplementedError("pi0_remote is inference-only")

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor], **kwargs) -> Tensor:
        device = self._device_anchor.device
        payload = {
            "batch": _to_cpu(batch),
            "kwargs": _to_cpu({k: v for k, v in kwargs.items() if v is not None}),
            "rtc_config": dataclasses.asdict(self.config.rtc_config) if self.config.rtc_config is not None else None,
            "chunk": self._chunks,
        }
        # enums inside rtc_config are not JSON; keep them as names
        if payload["rtc_config"] is not None:
            payload["rtc_config"] = {k: (v.name if hasattr(v, "name") else v) for k, v in payload["rtc_config"].items()}
        t0 = time.perf_counter()
        resp = self._session.post(
            self.config.server_url.rstrip("/") + "/predict",
            data=pickle.dumps(payload), timeout=self.config.request_timeout_s,
            headers={"Content-Type": "application/octet-stream"},
        )
        if resp.status_code != 200:
            raise RuntimeError(f"pi0_remote server error {resp.status_code}: {resp.text[:500]}")
        out = pickle.loads(resp.content)  # nosec
        actions = out["actions"].to(device)
        self.last_stats = dict(out.get("stats", {}))
        self.last_stats["rpc_round_trip_s"] = time.perf_counter() - t0
        self._chunks += 1
        logger.info(
            "pi0_remote chunk %d: rpc %.2fs server_total %.2fs expert=%s from_hardware=%s",
            self._chunks, self.last_stats["rpc_round_trip_s"], self.last_stats.get("total_s", float("nan")),
            self.last_stats.get("expert_backend"), self.last_stats.get("from_hardware"),
        )
        return actions

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor], **kwargs) -> Tensor:
        """Same queueing as PI0Policy.select_action: one remote chunk, then n_action_steps pops."""
        if len(self._action_queue) == 0:
            actions = self.predict_action_chunk(batch, **kwargs)[:, : self.config.n_action_steps]
            self._action_queue.extend(actions.transpose(0, 1))
        return self._action_queue.popleft()
