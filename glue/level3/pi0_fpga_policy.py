"""PI0FpgaPolicy: LeRobot's pi0 with a swappable action expert.

The prefix (SigLIP + PaliGemma over images, language and the pi0 state token
layout) is always computed by the stock PyTorch model on the Desktop.  The
10-step action-expert denoise loop is then run by one of two backends:

  PI0_ACTION_EXPERT=torch   (default) the stock PyTorch expert -- identical to PI0Policy
  PI0_ACTION_EXPERT=fpga    the host runtime's Int8ActionExpertDenoiseSession through
                            the pybind bridge; PI0_FPGA_BACKEND=mock|real picks the device
  PI0_ACTION_EXPERT=fpga_torch  the same packages driven from Python with the vector
                            operators in torch (pi0_torch_expert_session.py): same int8
                            bytes over PCIe, ~5x less host time than the C++ loops

Every chunk carries a `last_stats` dict with per-stage seconds and, for the
fpga backend, `from_hardware` taken from the backend itself -- never assumed.

The seam to the FPGA is `export_prefix_kv()` + `ActionExpertBridge.sample()`;
nothing else in the policy knows about the accelerator.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any

import numpy as np
import torch
from torch import Tensor

from lerobot.policies.pi0.modeling_pi0 import PI0Policy, make_att_2d_masks
from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS

logger = logging.getLogger("pi0_fpga_policy")

POLICY_TYPE = "pi0_fpga"


def load_prefix_v_scale(path: str | None, layers: int, head_dim: int) -> np.ndarray | None:
    """Per-layer, per-head_dim factor applied to the prefix V cache before export.

    SmoothQuant folds 1/s of the attention-output projection's input into the
    value projection's weight columns (scripts/export_pi0_lerobot_int8_session.py);
    the prefix V cache never passes through that GEMM, so the same 1/s must be
    applied here.  File: row-major float32 [layers][head_dim] written by the
    exporter as prefix_v_scale.f32.bin next to the session plan.
    """
    if not path:
        return None
    arr = np.fromfile(os.path.expanduser(path), dtype="<f4")
    if arr.size != layers * head_dim:
        raise ValueError(f"prefix V scale has {arr.size} values, expected {layers}x{head_dim}")
    return arr.reshape(layers, head_dim)


def export_prefix_kv(past_key_values, prefix_pad_masks: Tensor, layers: int, prefix_tokens: int,
                     kv_heads: int, head_dim: int, v_scale: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Turn the HF DynamicCache of the prefix pass into the host runtime's handoff blob.

    Layout required by pi0::PrefixKvCacheView: [layer][K/V][token][kv_head][head_dim]
    bf16, K before V (host/pi0_prefix_kv_cache.h).  Keys are post-RoPE, exactly as
    the cache stores them and as OpenPI hands them to the expert.  Batch must be 1.
    v_scale [L, D] (optional) multiplies V per layer and head_dim (see
    load_prefix_v_scale) before the bf16 cast.
    Returns (uint16 bf16 bits [L,2,T,H,D], uint8 prefix_valid [T]).
    """
    per_layer = []
    for layer_index, entry in enumerate(past_key_values):
        keys, values = entry[0], entry[1]  # [B, H, T, D]
        if keys.shape[0] != 1:
            raise ValueError("prefix export is batch-one only")
        if v_scale is not None:
            values = values * torch.as_tensor(v_scale[layer_index], dtype=values.dtype, device=values.device)
        kv = torch.stack([keys[0], values[0]], dim=0)          # [2, H, T, D]
        kv = kv.permute(0, 2, 1, 3).contiguous()               # [2, T, H, D]
        per_layer.append(kv)
    blob = torch.stack(per_layer, dim=0)                       # [L, 2, T, H, D]
    got = tuple(blob.shape)
    want = (layers, 2, prefix_tokens, kv_heads, head_dim)
    if got != want:
        raise ValueError(f"prefix KV shape {got} does not match the session's {want}")
    bits = blob.to(torch.bfloat16).view(torch.int16).cpu().numpy().view(np.uint16)
    valid = prefix_pad_masks[0].to(torch.uint8).cpu().numpy().copy()
    return np.ascontiguousarray(bits), valid


class PI0FpgaPolicy(PI0Policy):
    """PI0Policy whose action expert can be served by the FPGA host runtime."""

    name = POLICY_TYPE

    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        self.expert_backend = os.environ.get("PI0_ACTION_EXPERT", "torch").strip().lower()
        if self.expert_backend not in ("torch", "fpga", "fpga_torch", "chip"):
            raise ValueError(f"PI0_ACTION_EXPERT must be torch, fpga, fpga_torch or chip, got {self.expert_backend!r}")
        self.bridge = None
        self.prefix_v_scale = None
        self.last_stats: dict[str, Any] = {}
        self._chunk_counter = 0
        if self.expert_backend == "fpga":
            self.bridge = self._open_bridge()
            self.prefix_v_scale = self._load_prefix_v_scale(self.bridge.config())
        elif self.expert_backend == "fpga_torch":
            self.bridge = self._open_torch_session()
            self.prefix_v_scale = self._load_prefix_v_scale(self.bridge.config())
        self._chip = None
        if self.expert_backend == "chip":
            self._chip = self._open_chip()

    # ---- the FPGA seam -------------------------------------------------------------------
    @staticmethod
    def _open_bridge():
        import pi0_fpga_bridge  # built by glue/level3/pi0_fpga_bridge/build.sh

        runtime_plan = os.environ.get("PI0_RUNTIME_PLAN")
        session_plan = os.environ.get("PI0_SESSION_PLAN")
        if not runtime_plan or not session_plan:
            raise RuntimeError("PI0_ACTION_EXPERT=fpga needs PI0_RUNTIME_PLAN and PI0_SESSION_PLAN")
        bridge = pi0_fpga_bridge.ActionExpertBridge(
            runtime_plan_path=runtime_plan,
            session_plan_path=session_plan,
            payload_root=os.environ.get("PI0_PAYLOAD_ROOT", ""),
            allow_debug_calibration=os.environ.get("PI0_ALLOW_DEBUG_CALIBRATION", "0") == "1",
            denoise_steps=int(os.environ.get("PI0_DENOISE_STEPS", "10")),
            workers=int(os.environ.get("PI0_HOST_WORKERS", "4")),
            device_index=int(os.environ.get("PI0_DEVICE_INDEX", "0")),
            bar_index=int(os.environ.get("PI0_BAR_INDEX", "0")),
            timeout_ms=int(os.environ.get("PI0_REQUEST_TIMEOUT_MS", "30000")),
        )
        cfg = bridge.config()
        logger.info(
            "action expert = fpga bridge | %s | layers=%d prefix_tokens=%d hardware=%s "
            "production_calibration=%s", cfg["engine"], cfg["layers"], cfg["prefix_tokens"],
            cfg["is_hardware"], cfg["production_calibration"])
        if not cfg["is_hardware"]:
            logger.warning("mock backend: action chunks from this policy are NOT meaningful")
        return bridge

    @staticmethod
    def _open_torch_session():
        import sys
        import pi0_fpga_bridge

        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from pi0_torch_expert_session import TorchExpertSession

        runtime_plan = os.environ.get("PI0_RUNTIME_PLAN")
        session_plan = os.environ.get("PI0_SESSION_PLAN")
        if not runtime_plan or not session_plan:
            raise RuntimeError("PI0_ACTION_EXPERT=fpga_torch needs PI0_RUNTIME_PLAN and PI0_SESSION_PLAN")
        engine = pi0_fpga_bridge.EngineBridge(
            runtime_plan_path=runtime_plan,
            payload_root=os.environ.get("PI0_PAYLOAD_ROOT", ""),
            device_index=int(os.environ.get("PI0_DEVICE_INDEX", "0")),
            bar_index=int(os.environ.get("PI0_BAR_INDEX", "0")),
            timeout_ms=int(os.environ.get("PI0_REQUEST_TIMEOUT_MS", "30000")),
        )
        host_slots = set()
        for item in os.environ.get("PI0_HOST_SLOTS", "").split(","):
            item = item.strip()
            if item:  # "L3.q,L7.down"
                layer, slot = item.split(".")
                host_slots.add((int(layer.lstrip("L")), slot))
        golden = None
        if os.environ.get("PI0_GOLDEN_CHECK", "0") == "1":
            from pi0_session_golden import SessionGolden

            manifest = os.environ.get("PI0_DEPLOYMENT_MANIFEST") or os.path.join(os.path.dirname(session_plan), "manifest.json")
            golden = SessionGolden(manifest)
            logger.info("G5 golden check armed from %s (%d packages)", manifest, len(golden.packages))
        session = TorchExpertSession(
            engine, session_plan, payload_root=os.environ.get("PI0_PAYLOAD_ROOT", ""), golden=golden,
            allow_debug_calibration=os.environ.get("PI0_ALLOW_DEBUG_CALIBRATION", "0") == "1",
            denoise_steps=int(os.environ.get("PI0_DENOISE_STEPS", "10")),
            threads=int(os.environ["PI0_HOST_WORKERS"]) if os.environ.get("PI0_HOST_WORKERS") else None,
            host_slots=host_slots or None, numerics_npz=os.environ.get("PI0_NUMERICS"),
            checkpoint=os.environ.get("ASYNC_POLICY_PATH"))
        cfg = session.config()
        logger.info("action expert = fpga_torch | %s | layers=%d hardware=%s production_calibration=%s host_slots=%s",
                    cfg["engine"], cfg["layers"], cfg["is_hardware"], cfg["production_calibration"], sorted(host_slots))
        if not cfg["is_hardware"]:
            logger.warning("mock backend: action chunks from this policy are NOT meaningful")
        return session

    @staticmethod
    def _load_prefix_v_scale(cfg: dict) -> np.ndarray | None:
        """PI0_PREFIX_V_SCALE, else prefix_v_scale.f32.bin beside the session plan, else identity."""
        v_scale_path = os.environ.get("PI0_PREFIX_V_SCALE")
        if not v_scale_path:
            candidate = os.path.join(os.path.dirname(os.environ.get("PI0_SESSION_PLAN", "")), "prefix_v_scale.f32.bin")
            v_scale_path = candidate if os.path.exists(candidate) else None
        logger.info("prefix V scale: %s", v_scale_path or "none (identity)")
        return load_prefix_v_scale(v_scale_path, cfg["layers"], cfg["head_dim"])

    # ---- prefix, shared by both backends -------------------------------------------------
    def _run_prefix(self, images, img_masks, lang_tokens, lang_masks):
        model = self.model
        prefix_embs, prefix_pad_masks, prefix_att_masks = model.embed_prefix(
            images, img_masks, lang_tokens, lang_masks)
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
        prefix_att_2d_masks_4d = model._prepare_attention_masks_4d(prefix_att_2d_masks)
        model.paligemma_with_expert.paligemma.model.language_model.config._attn_implementation = "eager"
        _, past_key_values = model.paligemma_with_expert.forward(
            attention_mask=prefix_att_2d_masks_4d,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=True,
        )
        return past_key_values, prefix_pad_masks

    def _run_prefix_compact(self, images, img_masks, lang_tokens, lang_masks):
        """The prefix pass over the valid tokens only, for the FPGA backends.

        The stock pass embeds every camera slot (SigLIP on a constant -1 image for
        an empty slot) and runs PaliGemma over all 816 positions although the
        padded ones are masked out of every attention row.  Dropping them first
        gives the same KV for the valid tokens (positions are cumsum(valid)-1
        either way, and the attention sees the same key set) at 2/3 of the
        vision cost and ~64 % of the language-model cost with one empty slot.
        Returns (kv float32 [L, 2, T, H, D] with zeros at padded positions,
        prefix_pad_masks bool [1, T]) for T = the full 816-token layout.
        """
        model = self.model
        pwe = model.paligemma_with_expert
        embs, valid_parts = [], []
        for img, img_mask in zip(images, img_masks, strict=True):
            if bool(img_mask[0]):
                emb = pwe.embed_image(img)                       # [1, 256, W]
                embs.append(emb)
                valid_parts.append(torch.ones(emb.shape[1], dtype=torch.bool, device=emb.device))
            else:
                valid_parts.append(torch.zeros(self._image_tokens(), dtype=torch.bool, device=lang_tokens.device))
        lang_emb = pwe.embed_language_tokens(lang_tokens)        # [1, 48, W]
        lang_valid = lang_masks[0].to(torch.bool)
        embs.append(lang_emb[:, lang_valid])
        valid_parts.append(lang_valid)
        prefix_valid = torch.cat(valid_parts)                    # [T] in the stock layout
        compact = torch.cat(embs, dim=1)                         # [1, n_valid, W]
        n_valid = compact.shape[1]
        pad = torch.ones(1, n_valid, dtype=torch.bool, device=compact.device)
        att = torch.zeros(1, n_valid, dtype=torch.bool, device=compact.device)
        att_2d = make_att_2d_masks(pad, att)
        position_ids = torch.arange(n_valid, device=compact.device)[None]
        pwe.paligemma.model.language_model.config._attn_implementation = "eager"
        _, past_key_values = pwe.forward(
            attention_mask=model._prepare_attention_masks_4d(att_2d), position_ids=position_ids,
            past_key_values=None, inputs_embeds=[compact, None], use_cache=True)
        entries = list(past_key_values)                          # DynamicCache iterates as (K, V[, ...]) per layer
        layers = len(entries)
        T = int(prefix_valid.shape[0])
        first = entries[0][0]                                    # [1, H, n_valid, D]
        kv = torch.zeros(layers, 2, T, first.shape[1], first.shape[3], dtype=torch.float32)
        idx = torch.nonzero(prefix_valid, as_tuple=False)[:, 0].cpu()
        for li, entry in enumerate(entries):
            kv[li, 0, idx] = entry[0][0].permute(1, 0, 2).float().cpu()   # [n_valid, H, D]
            kv[li, 1, idx] = entry[1][0].permute(1, 0, 2).float().cpu()
        return kv, prefix_valid[None].cpu()

    def _image_tokens(self) -> int:
        cfg = self.model.paligemma_with_expert.paligemma.config.vision_config
        return (cfg.image_size // cfg.patch_size) ** 2

    def _denoise_torch(self, state, prefix_pad_masks, past_key_values, noise, num_steps):
        model = self.model
        bsize, device = state.shape[0], state.device
        dt = -1.0 / num_steps
        x_t = noise
        for step in range(num_steps):
            t = 1.0 + step * dt
            time_tensor = torch.tensor(t, dtype=torch.float32, device=device).expand(bsize)
            v_t = model.denoise_step(state=state, prefix_pad_masks=prefix_pad_masks,
                                     past_key_values=past_key_values, x_t=x_t, timestep=time_tensor)
            x_t = x_t + dt * v_t
        return x_t

    def _denoise_fpga(self, state, prefix_pad_masks, past_key_values, noise, kv_f32=None):
        cfg = self.bridge.config()
        t0 = time.perf_counter()
        if kv_f32 is not None:
            # compact prefix path: the KV already has the full layout with zeros at padding
            if tuple(kv_f32.shape) != (cfg["layers"], 2, cfg["prefix_tokens"], cfg["kv_heads"], cfg["head_dim"]):
                raise ValueError(f"prefix KV shape {tuple(kv_f32.shape)} does not match the session")
            if self.prefix_v_scale is not None:
                kv_f32[:, 1] = kv_f32[:, 1] * torch.from_numpy(self.prefix_v_scale)[:, None, None, :]
            kv_bits = kv_f32.to(torch.bfloat16).view(torch.int16).numpy().view(np.uint16)
            valid = prefix_pad_masks[0].to(torch.uint8).numpy().copy()
        else:
            kv_bits, valid = export_prefix_kv(
                past_key_values, prefix_pad_masks, cfg["layers"], cfg["prefix_tokens"],
                cfg["kv_heads"], cfg["head_dim"], v_scale=self.prefix_v_scale)
        export_s = time.perf_counter() - t0
        state_np = state[0].detach().float().cpu().numpy().astype(np.float32)
        noise_np = noise[0].detach().float().cpu().numpy().astype(np.float32)
        actions_np, stats = self.bridge.sample(kv_bits, valid, state_np, noise_np)
        stats = dict(stats)
        stats["prefix_export_s"] = export_s
        stats["prefix_kv_bytes"] = int(kv_bits.nbytes)
        return torch.from_numpy(np.ascontiguousarray(actions_np))[None].to(noise.dtype), stats

    # ---- the whole model on the node array (PI0_ACTION_EXPERT=chip) -------------------------
    @staticmethod
    def _open_chip():
        """PI0_CHIP_CHUNK: a generated chunk dir (paper/sw/pi0_chunk_program.py --bin) whose image is already in
        GDDR6 (host/pi0_chunk_run run DIR ... loads and checks it once); PI0_CHIP_MAP: pi0_chunk_run --map;
        PI0_CHIP_SW: the paper/sw directory with pi0_chip_runtime.py / pi0_chip_host_inputs.py."""
        import sys
        sw = os.environ.get("PI0_CHIP_SW", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "paper", "sw"))
        sys.path.insert(0, os.path.abspath(sw))
        import pi0_chip_runtime as RT
        chunk = os.environ.get("PI0_CHIP_CHUNK")
        if not chunk:
            raise RuntimeError("PI0_ACTION_EXPERT=chip needs PI0_CHIP_CHUNK (a generated chunk dir loaded on the board)")
        io = RT.ChunkIO(chunk)
        logger.info("action backend = whole pi0 on the node array | chunk %s | map %s", chunk, os.environ.get("PI0_CHIP_MAP", ""))
        return dict(RT=RT, io=io, map=os.environ.get("PI0_CHIP_MAP", ""),
                    work=os.environ.get("PI0_CHIP_WORK", os.path.join(chunk, "runtime")))

    def _predict_chip(self, images, img_masks, lang_tokens, lang_masks, state, noise):
        from pi0_chip_host_inputs import chip_inputs
        c = self._chip
        t0 = time.perf_counter()
        vis, lang, st, nz = chip_inputs(self, images, img_masks, lang_tokens, lang_masks, state, noise)
        n_lang = c["io"].n_lang
        if lang.shape[0] != n_lang:
            raise ValueError(f"the chunk was generated for {n_lang} prompt tokens, the observation has {lang.shape[0]}")
        host_s = time.perf_counter() - t0
        from pathlib import Path
        act, stats = c["RT"].infer(c["io"], (vis, lang, st, nz), c["map"], Path(c["work"]))
        stats = dict(stats, host_inputs_s=host_s, from_hardware=True)
        stats.pop("log", None)
        return torch.from_numpy(act)[None].to(noise.dtype), stats

    # ---- the policy entry point the server calls -----------------------------------------
    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor], **kwargs) -> Tensor:
        self.eval()
        if self._rtc_enabled():
            raise NotImplementedError("PI0FpgaPolicy does not support RTC")
        t_start = time.perf_counter()
        images, img_masks = self._preprocess_images(batch)
        lang_tokens = batch[OBS_LANGUAGE_TOKENS]
        lang_masks = batch[OBS_LANGUAGE_ATTENTION_MASK]
        state = self.prepare_state(batch)
        bsize, device = state.shape[0], state.device
        if bsize != 1:
            raise ValueError("PI0FpgaPolicy serves batch-one observations only")
        num_steps = self.config.num_inference_steps
        noise = kwargs.get("noise")
        if noise is None:
            noise = self.model.sample_noise((bsize, self.config.chunk_size, self.config.max_action_dim), device)

        if self.expert_backend == "chip":
            t0 = time.perf_counter()
            x_t, stats = self._predict_chip(images, img_masks, lang_tokens, lang_masks, state, noise)
            self._chunk_counter += 1
            self.last_stats = {"chunk": self._chunk_counter, "expert_backend": "chip", "total_s": time.perf_counter() - t_start,
                               **stats}
            logger.info("PI0_STAGE chunk=%d backend=chip chip_s=%s host_inputs_s=%.3f total_s=%.3f", self._chunk_counter,
                        stats.get("chip_s"), stats["host_inputs_s"], self.last_stats["total_s"])
            original_action_dim = self.config.output_features["action"].shape[0]
            return x_t[:, :, :original_action_dim]

        t0 = time.perf_counter()
        compact = self.expert_backend != "torch" and os.environ.get("PI0_COMPACT_PREFIX", "1") == "1"
        if compact:
            kv_f32, prefix_pad_masks = self._run_prefix_compact(images, img_masks, lang_tokens, lang_masks)
            past_key_values = None
        else:
            kv_f32 = None
            past_key_values, prefix_pad_masks = self._run_prefix(images, img_masks, lang_tokens, lang_masks)
        prefix_s = time.perf_counter() - t0

        t0 = time.perf_counter()
        if self.expert_backend in ("fpga", "fpga_torch"):
            x_t, stats = self._denoise_fpga(state, prefix_pad_masks, past_key_values, noise, kv_f32=kv_f32)
        else:
            x_t = self._denoise_torch(state, prefix_pad_masks, past_key_values, noise, num_steps)
            stats = {"from_hardware": False, "denoise_steps": num_steps}
        denoise_s = time.perf_counter() - t0

        original_action_dim = self.config.output_features["action"].shape[0]
        actions = x_t[:, :, :original_action_dim]

        self._chunk_counter += 1
        self.last_stats = {
            "chunk": self._chunk_counter,
            "expert_backend": self.expert_backend,
            "from_hardware": bool(stats.get("from_hardware", False)),
            "prefix_s": prefix_s,
            "denoise_s": denoise_s,
            "total_s": time.perf_counter() - t_start,
            "prefix_tokens": int(prefix_pad_masks.shape[1]),
            "prefix_valid": int(prefix_pad_masks.sum().item()),
            "compact_prefix": compact,
            **{k: v for k, v in stats.items() if k not in ("from_hardware",)},
        }
        logger.info(
            "PI0_STAGE chunk=%d expert=%s from_hardware=%s prefix_s=%.3f denoise_s=%.3f total_s=%.3f%s",
            self._chunk_counter, self.expert_backend, self.last_stats["from_hardware"],
            prefix_s, denoise_s, self.last_stats["total_s"],
            "" if self.expert_backend == "torch" else
            f" bridge_denoise_s={stats.get('denoise_s', 0):.3f} prefix_export_s={stats.get('prefix_export_s', 0):.3f} input_clipped={stats.get('input_clipped', 0)}"
            + (f" device_execute_s={stats.get('device_execute_s', 0):.3f} host_vector_s={stats.get('host_vector_s', 0):.3f}"
               f" golden_mismatch_bytes={stats.get('golden_mismatch_bytes', 0)} golden_compared_bytes={stats.get('golden_compared_bytes', 0)}"
               if self.expert_backend == "fpga_torch" else ""),
        )
        return actions
