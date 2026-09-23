#!/usr/bin/env python3
"""Deployment-exact numerics model of the pi0 INT8 action expert on the AC7t1500.

Models the shipped datapath op by op, from the RTL and the host runtime rather
than from a generic fake-quant:

  device  q_in  = sat127(rne(x / in_scale))                     host, static per-tensor scale
          acc   = sum_k q_in[k] * w_q[k, n]                     exact integer accumulation
          q_out = sat_int8(rne((acc + bias48[n]) * mult[n] / 2^shift[n]))
                                                                pi0_int8_requant.sv, 16-bit
                                                                multiplier, 6-bit shift
  host    y     = bf16(q_out * dequant[n])                      Int8ActionExpertLayerBackend::run_linear
          every vector op (RMSNorm, RoPE, softmax, GELU*up, residual) reads and
          writes bf16 exactly where host/pi0_action_expert_layer.cpp does; the
          suffix embedding and the action output run in fp32 with the exact
          fp32 checkpoint biases added on the host (Int8ActionLinearBackend).

On top of that datapath it implements the software-only accuracy levers of the
session brief (docs/PI0_E2E_INFERENCE_BRIEF_20260909.md section 4.2):

  * SmoothQuant (Xiao et al. 2023): per input channel s_k = amax|X_k|^a / amax|W_k|^(1-a),
    W' = diag(s) W.  1/s is folded where the host already multiplies:
      q/k/v   -> the attention RMSNorm offset  (1+g)/s - 1   (bf16, per layer)
      gate/up -> the FFW RMSNorm offset
      o       -> the value projection's per-column dequant scale AND the prefix V cache
                 (one KV head: s is per head_dim, shared by the 8 query heads)
      down    -> the up projection's per-column dequant scale
    so no host C++ changes and no new hardware are needed.
  * MSE-optimal per-output-channel weight clipping (instead of amax).
  * static activation scales by MSE / percentile search on calibration captures.
  * DFQ bias correction (Nagel et al. 2019) folded into the 48-bit requant bias
    (layer GEMMs) or the host fp32 bias (projections).
  * per-output-channel output scales (the quant bank has them; the exporter
    used to collapse them to a scalar).

Calibration statistics come from real captures (glue/level3/capture_pi0_prefix_kv.py:
LeRobot torch prefix KV, real observations); nothing here is synthetic.
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "scripts"))
import pi0_ae_model as M  # noqa: E402
from export_pi0_int8_quant_bank_pages import encode_effective_scale  # noqa: E402

WIDTH, DEPTH, MLP, HEADS, HDIM, HORIZON = M.WIDTH, M.DEPTH, M.MLP, M.HEADS, M.HDIM, M.HORIZON
LAYER_SLOTS = ("q", "k", "v", "o", "gate", "up", "down")
PROJ_SLOTS = ("state", "act_in", "tm_in", "tm_out", "act_out")
SLOT_INPUT = {"q": "attn_in", "k": "attn_in", "v": "attn_in", "o": "o_in",
              "gate": "ffw_in", "up": "ffw_in", "down": "down_in"}
SMOOTH_CLASS = {"q": "qkv", "k": "qkv", "v": "qkv", "o": "o", "gate": "gateup", "up": "gateup", "down": "down"}
bf16 = M.to_bf16


def rne(x):
    return np.rint(x)


# --------------------------------------------------------------------------
# calibration statistics
# --------------------------------------------------------------------------
class Reservoir:
    """Per-channel amax over every token seen, plus a bounded random subsample
    of rows for scale searches."""

    def __init__(self, rows: int, seed: int):
        self.rows = rows
        self.rng = np.random.default_rng(seed)
        self.amax = None
        self.sum = None
        self.count = 0
        self.sample = []
        self.nsample = 0

    def add(self, x: np.ndarray):
        x = np.asarray(x, np.float32)
        a = np.abs(x).max(axis=0)
        self.amax = a if self.amax is None else np.maximum(self.amax, a)
        self.sum = x.sum(0, dtype=np.float64) if self.sum is None else self.sum + x.sum(0, dtype=np.float64)
        self.count += x.shape[0]
        # reservoir sampling by rows
        for r in x:
            self.nsample += 1
            if len(self.sample) < self.rows:
                self.sample.append(r.copy())
            else:
                j = self.rng.integers(0, self.nsample)
                if j < self.rows:
                    self.sample[j] = r.copy()

    def rows_array(self) -> np.ndarray:
        return np.stack(self.sample) if self.sample else np.zeros((0, len(self.amax)), np.float32)

    def mean(self) -> np.ndarray:
        return (self.sum / max(self.count, 1)).astype(np.float32)


@dataclass
class CalibStats:
    """inputs[(L, name)] and outputs[(L, slot)] for the 18 layers; proj_in[slot], proj_out[slot]."""
    inputs: dict = field(default_factory=dict)
    outputs: dict = field(default_factory=dict)
    proj_in: dict = field(default_factory=dict)
    proj_out: dict = field(default_factory=dict)
    tokens: int = 0
    frames: int = 0
    rows: int = 2048

    def res(self, table, key):
        if key not in table:
            table[key] = Reservoir(self.rows, seed=hash(key) & 0xFFFF)
        return table[key]


# --------------------------------------------------------------------------
# scale selection
# --------------------------------------------------------------------------
def mse_scale_per_tensor(x: np.ndarray, qmax: int = 127, grid=None) -> float:
    """Clip value c minimising E[(deq(q(x)) - x)^2] over a grid of fractions of amax."""
    amax = float(np.abs(x).max())
    if amax == 0.0:
        return 1.0
    grid = np.linspace(0.2, 1.0, 33) if grid is None else grid
    xf = x.reshape(-1).astype(np.float32)
    if xf.size > 1_000_000:
        xf = xf[np.random.default_rng(0).choice(xf.size, 1_000_000, replace=False)]
    best, best_c = None, amax
    for f in grid:
        c = amax * f
        s = c / qmax
        err = np.mean((np.clip(rne(xf / s), -qmax, qmax) * s - xf) ** 2)
        if best is None or err < best:
            best, best_c = err, c
    return float(best_c)


def mse_clip_per_column(w: np.ndarray, qmax: int = 127, grid=None) -> np.ndarray:
    """(K, N) weight -> per column clip value minimising the column's quantisation MSE."""
    amax = np.abs(w).max(axis=0)
    grid = np.linspace(0.5, 1.0, 21) if grid is None else grid
    best = np.full(w.shape[1], np.inf)
    best_c = amax.copy()
    for f in grid:
        c = np.where(amax > 0, amax * f, 1.0)
        s = c / qmax
        err = np.mean((np.clip(rne(w / s), -qmax, qmax) * s - w) ** 2, axis=0)
        better = err < best
        best = np.where(better, err, best)
        best_c = np.where(better, c, best_c)
    return best_c.astype(np.float32)


def percentile_amax(x: np.ndarray, pct: float, axis=None) -> np.ndarray:
    return np.percentile(np.abs(x), pct, axis=axis)


# --------------------------------------------------------------------------
# one GEMM as deployed
# --------------------------------------------------------------------------
class DeployLinear:
    """Static INT8 GEMM with the hardware requant and the host dequant.

    w:        (K, N) float32, already smoothed (diag(s) folded in)
    w_clip:   (N,) per-column clip value -> w_scale = clip / 127
    in_scale: float, the host's static activation scale
    out_scale:(N,) per column (or (1,) per tensor); defines the int8 output codes
    dequant:  (N,) what the host multiplies the codes by (out_scale, possibly
              divided by the next GEMM's smoothing factor)
    bias:     (N,) float32 correction in output units, folded into the 48-bit
              requant bias (layer GEMMs) or into the host fp32 bias (projections)
    """

    def __init__(self, name, w, w_clip, in_scale, out_scale, dequant=None, bias=None,
                 host_bias=None, encode=True, out_bf16=True, bias_in_hw=True):
        self.name = name
        K, N = w.shape
        # exactly the exporter's arithmetic: float32 amax / float32 127, effective in float64
        self.w_scale = (np.asarray(w_clip, np.float32) / np.float32(127)).astype(np.float32)
        self.codes = np.clip(rne(w / self.w_scale[None, :]), -127, 127).astype(np.float64)
        self.in_scale = float(in_scale)
        out_scale = np.broadcast_to(np.asarray(out_scale, np.float32), (N,)).astype(np.float32)
        self.out_scale = out_scale
        self.dequant = out_scale if dequant is None else np.asarray(dequant, np.float32)
        eff = np.float64(float(np.float32(self.in_scale))) * self.w_scale.astype(np.float64) / out_scale.astype(np.float64)
        if encode:
            dec = np.empty(N)
            self.mult = np.empty(N, np.int64)
            self.shift = np.empty(N, np.int64)
            for n in range(N):
                m, sh, d, _ = encode_effective_scale(float(eff[n]), 1e-3)
                self.mult[n], self.shift[n], dec[n] = m, sh, d
            self.eff = dec
        else:
            self.eff = eff
            self.mult = self.shift = None
        self.host_bias = None if host_bias is None else np.asarray(host_bias, np.float32)
        self.bias48 = np.zeros(N, np.float64)
        if bias is not None:
            bias = np.asarray(bias, np.float64)
            if bias_in_hw:
                self.bias48 = rne(bias / (self.in_scale * self.w_scale))
                self.bias48 = np.clip(self.bias48, -(1 << 47), (1 << 47) - 1)
            else:
                self.host_bias = (0.0 if self.host_bias is None else self.host_bias) + bias.astype(np.float32)
        self.out_bf16 = out_bf16
        self.clipped = 0
        self.saturated = 0
        self.elements = 0
        self.max_ratio = 0.0

    def __call__(self, x: np.ndarray) -> np.ndarray:
        xs = x.astype(np.float32) / np.float32(self.in_scale)
        self.max_ratio = max(self.max_ratio, float(np.abs(xs).max()) / 127.0)
        q = rne(xs)
        self.clipped += int((np.abs(q) > 127).sum())
        self.elements += q.size
        q = np.clip(q, -127, 127).astype(np.float64)
        acc = q @ self.codes                                     # exact in float64
        if self.mult is not None:
            y = rne((acc + self.bias48) * self.mult / np.exp2(self.shift))
        else:
            y = rne((acc + self.bias48) * self.eff)
        self.saturated += int(((y > 127) | (y < -128)).sum())
        y = np.clip(y, -128, 127)
        out = (y * self.dequant).astype(np.float32)
        if self.host_bias is not None:
            out = out + self.host_bias
        return bf16(out) if self.out_bf16 else out

    def stats(self):
        return {"clip_frac": self.clipped / max(self.elements, 1), "sat": self.saturated,
                "max_input_ratio": self.max_ratio}


class Fp32Linear:
    """Exact fp32 reference GEMM that can record calibration statistics."""

    def __init__(self, name, w, bias=None, out_bf16=True, stats=None, key_in=None, key_out=None):
        self.name, self.w, self.bias, self.out_bf16 = name, w, bias, out_bf16
        self.stats, self.key_in, self.key_out = stats, key_in, key_out

    def __call__(self, x):
        y = (x.astype(np.float64) @ self.w.astype(np.float64)).astype(np.float32)
        if self.stats is not None:
            if self.key_in is not None:
                self.stats.res(self.stats.inputs, self.key_in).add(x)
            if self.key_out is not None:
                self.stats.res(self.stats.outputs, self.key_out).add(y)
        if self.bias is not None:
            y = y + self.bias
        return bf16(y) if self.out_bf16 else y


# --------------------------------------------------------------------------
# the expert as the host runs it
# --------------------------------------------------------------------------
@dataclass
class HostModel:
    """Everything the forward needs: the per-(layer, slot) GEMM callables, the
    (possibly smoothing-folded) norm offsets, the prefix-V per-channel factor."""
    lin: dict                       # (L, slot) -> callable, and proj slot -> callable
    attn_gain: np.ndarray           # (L, W) bf16-valued offsets
    ffw_gain: np.ndarray
    final_gain: np.ndarray
    prefix_v_scale: np.ndarray | None = None   # (L, HDIM) multiply the prefix V cache
    host_bf16: bool = True
    proj_bias: dict = field(default_factory=dict)
    stats: CalibStats | None = None

    def r(self, x):
        return bf16(x) if self.host_bf16 else x.astype(np.float32)


def rmsnorm_host(x, gain, eps=1e-6):
    # host: fp32 math from bf16 input, bf16 gain, bf16 output (gemma_rms_norm_bf16)
    x = x.astype(np.float32)
    ms = np.mean(x.astype(np.float32) ** 2, axis=-1, keepdims=True, dtype=np.float32)
    inv = (1.0 / np.sqrt(ms + np.float32(eps))).astype(np.float32)
    return (x * inv) * (1.0 + gain.astype(np.float32))


def layer_forward(hm: HostModel, L: int, x: np.ndarray, prefix_kv, prefix_valid, pos, mask_s):
    S = x.shape[0]
    Kp, Vp = prefix_kv[L]
    if hm.prefix_v_scale is not None:
        Vp = hm.r(Vp * hm.prefix_v_scale[L][None, :])
    P = Kp.shape[0]
    xn = hm.r(rmsnorm_host(x, hm.attn_gain[L]))
    q = hm.lin[(L, "q")](xn).reshape(S, HEADS, HDIM)
    k = hm.lin[(L, "k")](xn)
    v = hm.lin[(L, "v")](xn)
    q = hm.r(M.rope(q, pos))
    q = hm.r(q * np.float32(HDIM ** -0.5))
    k = hm.r(M.rope(k[:, None, :], pos)[:, 0])
    Kall = np.concatenate([Kp, k], axis=0)
    Vall = np.concatenate([Vp, v], axis=0)
    full_mask = np.concatenate([np.broadcast_to(prefix_valid[None], (S, P)), mask_s], axis=1)
    logits = np.einsum("shd,td->hst", q.astype(np.float32), Kall.astype(np.float32))
    logits = np.where(full_mask[None], logits, -np.inf)
    logits = logits - logits.max(axis=-1, keepdims=True)
    p = np.exp(logits)
    p = p / p.sum(axis=-1, keepdims=True)
    p = hm.r(p.astype(np.float32))                                   # masked_softmax_f32_to_bf16
    ctx = hm.r(np.einsum("hst,td->shd", p.astype(np.float32), Vall.astype(np.float32)))
    attn = hm.lin[(L, "o")](ctx.reshape(S, HEADS * HDIM))
    x = hm.r(x + attn)
    xn2 = hm.r(rmsnorm_host(x, hm.ffw_gain[L]))
    g = hm.lin[(L, "gate")](xn2)
    u = hm.lin[(L, "up")](xn2)
    h = hm.r(hm.r(M.gelu_tanh(g)) * u)
    d = hm.lin[(L, "down")](h)
    return hm.r(x + d)


def embed_suffix_host(hm: HostModel, state, x_t, t):
    st = hm.lin["state"](state[None].astype(np.float32))                  # (1, W) fp32
    at = hm.lin["act_in"](x_t.astype(np.float32))                          # (50, W)
    te = np.broadcast_to(M.posemb_sincos(t, WIDTH)[None], (HORIZON, WIDTH))
    att = np.concatenate([at, te], axis=1).astype(np.float32)
    h = M.swish(hm.lin["tm_in"](att))
    h = hm.lin["tm_out"](h)
    return hm.r(np.concatenate([st, h], axis=0).astype(np.float32))       # (51, W) bf16


def denoise_host(hm: HostModel, ep: M.Episode, steps: int = M.STEPS, trace=None) -> np.ndarray:
    prefix_valid = np.asarray(ep.prefix_valid).astype(bool) if ep.prefix_valid is not None \
        else np.ones(ep.prefix_kv[0][0].shape[0], bool)
    S = HORIZON + 1
    n_valid = int(prefix_valid.sum())
    pos = np.arange(n_valid, n_valid + S)
    mask_s = M.suffix_mask(S)
    x_t = ep.noise.astype(np.float32).copy()
    dt = np.float32(-1.0 / steps)
    t = np.float32(1.0)
    for _ in range(steps):
        x = embed_suffix_host(hm, ep.state, x_t, float(t))
        for L in range(DEPTH):
            x = layer_forward(hm, L, x, ep.prefix_kv, prefix_valid, pos, mask_s)
        xf = hm.r(rmsnorm_host(x, hm.final_gain))[1:].astype(np.float32)
        v = hm.lin["act_out"](xf)                                          # fp32 (50, 32)
        x_t = (x_t + dt * v).astype(np.float32)
        t = np.float32(t + dt)
        if trace is not None:
            trace.append(x_t.copy())
    return x_t


# --------------------------------------------------------------------------
# builders
# --------------------------------------------------------------------------
def layer_matrix(w: M.Weights, L: int, slot: str) -> np.ndarray:
    if slot == "q":
        return w.q[L]
    if slot == "k":
        return w.kv[L][:, :HDIM]
    if slot == "v":
        return w.kv[L][:, HDIM:]
    if slot == "o":
        return w.o[L]
    if slot == "gate":
        return w.gate_up[L][:, :MLP]
    if slot == "up":
        return w.gate_up[L][:, MLP:]
    if slot == "down":
        return w.down[L]
    raise KeyError(slot)


def proj_matrix(w: M.Weights, slot: str):
    return {"state": (w.state_w, w.state_b), "act_in": (w.act_in_w, w.act_in_b), "tm_in": (w.tm_in_w, w.tm_in_b),
            "tm_out": (w.tm_out_w, w.tm_out_b), "act_out": (w.act_out_w, w.act_out_b)}[slot]


def fp32_host_model(w: M.Weights, stats: CalibStats | None = None, host_bf16: bool = True) -> HostModel:
    """The host datapath with exact fp32 GEMMs: isolates the bf16 host cost and
    records calibration statistics when `stats` is given."""
    lin = {}
    for L in range(DEPTH):
        for s in LAYER_SLOTS:
            lin[(L, s)] = Fp32Linear(f"L{L}.{s}", layer_matrix(w, L, s), out_bf16=host_bf16, stats=stats,
                                     key_in=(L, SLOT_INPUT[s]) if s in ("q", "o", "gate", "down") else None,
                                     key_out=(L, s))
    for s in PROJ_SLOTS:
        mat, b = proj_matrix(w, s)
        lin[s] = Fp32Linear(s, mat, bias=b, out_bf16=False, stats=stats)
        if stats is not None:
            lin[s].key_in, lin[s].key_out = ("proj", s), ("proj", s)
            lin[s].stats = stats
    return HostModel(lin=lin, attn_gain=bf16(w.pre_attn) if host_bf16 else w.pre_attn,
                     ffw_gain=bf16(w.pre_ffw) if host_bf16 else w.pre_ffw,
                     final_gain=bf16(w.final_norm) if host_bf16 else w.final_norm, host_bf16=host_bf16, stats=stats)


@dataclass
class DeployConfig:
    alpha: dict = field(default_factory=dict)      # smoothing class -> alpha; missing = no smoothing
    w_clip: str = "amax"                           # amax | mse
    in_scale: str = "amax"                         # amax | mse | pctl
    in_pctl: float = 99.99
    in_margin: float = 1.0
    out_scale: str = "per_column"                  # per_column | per_tensor
    out_method: str = "amax"                       # amax | pctl
    out_pctl: float = 99.99
    out_margin: float = 1.0
    bias_corr: bool = False
    encode: bool = True
    host_bf16: bool = True
    proj_int8: bool = True
    name: str = ""

    def label(self):
        a = ",".join(f"{k}={v}" for k, v in sorted(self.alpha.items())) or "none"
        return (f"smooth[{a}] w:{self.w_clip} in:{self.in_scale}x{self.in_margin} "
                f"out:{self.out_scale}/{self.out_method}x{self.out_margin} bias:{int(self.bias_corr)}")


def smoothing_factors(x_amax: np.ndarray, w_rowmax: np.ndarray, alpha: float, lo=1e-2, hi=1e2) -> np.ndarray:
    xa = np.maximum(x_amax, 1e-8)
    wa = np.maximum(w_rowmax, 1e-8)
    s = xa ** alpha / wa ** (1.0 - alpha)
    return np.clip(s, lo, hi).astype(np.float32)


def choose_input_scale(cfg: DeployConfig, rows: np.ndarray, amax: float) -> float:
    if cfg.in_scale == "amax":
        c = amax
    elif cfg.in_scale == "mse":
        c = mse_scale_per_tensor(rows)
    elif cfg.in_scale == "pctl":
        c = float(percentile_amax(rows, cfg.in_pctl))
    else:
        raise ValueError(cfg.in_scale)
    return float(c * cfg.in_margin) / 127.0


def choose_output_scale(cfg: DeployConfig, rows: np.ndarray, amax: np.ndarray) -> np.ndarray:
    if cfg.out_method == "amax":
        c = amax.astype(np.float64)
    elif cfg.out_method == "pctl":
        c = percentile_amax(rows, cfg.out_pctl, axis=0)
    else:
        raise ValueError(cfg.out_method)
    c = np.maximum(c, 1e-8) * cfg.out_margin
    if cfg.out_scale == "per_tensor":
        c = np.full_like(c, c.max())
    return (c / 127.0).astype(np.float32)


def build_deploy_model(w: M.Weights, stats: CalibStats, cfg: DeployConfig) -> tuple[HostModel, dict]:
    """Fold, calibrate and quantise every GEMM from the calibration statistics.

    Every fold is a pure weight/gain change, so the export contract stays the
    exporter's (effective = in_scale * w_scale[n] / out_scale[n], host multiplies
    the codes by out_scale[n]):
      W_export = clip(diag(row_scale) @ W @ diag(col_scale), +-w_clip)
      row_scale = s of the GEMM's own input (SmoothQuant), col_scale = 1/s of
      the consumer for v (-> o) and up (-> down), else 1.
    Returns the HostModel and the export table (see export_numerics)."""
    lin, table = {}, {}
    attn_gain = w.pre_attn.astype(np.float32).copy()
    ffw_gain = w.pre_ffw.astype(np.float32).copy()
    prefix_v_scale = np.ones((DEPTH, HDIM), np.float32)
    for L in range(DEPTH):
        # ---- smoothing factors per input, shared by the GEMMs that read the same tensor
        s_in = {}
        for inp, slots in (("attn_in", ("q", "k", "v")), ("ffw_in", ("gate", "up")), ("o_in", ("o",)), ("down_in", ("down",))):
            cls = SMOOTH_CLASS[slots[0]]
            r = stats.inputs[(L, inp)]
            if cfg.alpha.get(cls) is not None:
                wcat = np.concatenate([layer_matrix(w, L, s) for s in slots], axis=1)
                wrow = np.abs(wcat).max(axis=1)
                xa = r.amax
                if inp == "o_in":
                    # one KV head: the factor must be per head_dim, shared over the 8 query heads
                    xa = xa.reshape(HEADS, HDIM).max(axis=0)
                    wrow = wrow.reshape(HEADS, HDIM).max(axis=0)
                    s = np.tile(smoothing_factors(xa, wrow, cfg.alpha[cls]), HEADS)
                else:
                    s = smoothing_factors(xa, wrow, cfg.alpha[cls])
            else:
                s = np.ones(r.amax.shape[0], np.float32)
            s_in[inp] = s
        attn_gain[L] = (1.0 + attn_gain[L]) / s_in["attn_in"] - 1.0
        ffw_gain[L] = (1.0 + ffw_gain[L]) / s_in["ffw_in"] - 1.0
        prefix_v_scale[L] = 1.0 / s_in["o_in"][:HDIM]
        for slot in LAYER_SLOTS:
            inp = SLOT_INPUT[slot]
            row_scale = s_in[inp]
            col_scale = {"v": 1.0 / s_in["o_in"][:HDIM], "up": 1.0 / s_in["down_in"]}.get(slot)
            col_scale = np.ones(layer_matrix(w, L, slot).shape[1], np.float32) if col_scale is None else col_scale.astype(np.float32)
            r = stats.inputs[(L, inp)]
            wq = layer_matrix(w, L, slot) * row_scale[:, None] * col_scale[None, :]
            rows = r.rows_array() / row_scale[None, :]
            amax = float((r.amax / row_scale).max())
            in_scale = choose_input_scale(cfg, rows, amax)
            w_clip = mse_clip_per_column(wq) if cfg.w_clip == "mse" else np.abs(wq).max(axis=0)
            w_clip = np.where(w_clip > 0, w_clip, 1.0).astype(np.float32)
            ro = stats.outputs[(L, slot)]
            out_scale = choose_output_scale(cfg, ro.rows_array() * col_scale[None, :], ro.amax * col_scale)
            bias = None
            if cfg.bias_corr:
                w_scale = w_clip / 127.0
                dw = np.clip(rne(wq / w_scale), -127, 127) * w_scale - wq
                mean_x = r.mean() / row_scale
                bias = -(mean_x.astype(np.float64) @ dw.astype(np.float64))
            lin[(L, slot)] = DeployLinear(f"L{L}.{slot}", wq, w_clip, in_scale, out_scale,
                                          bias=bias, encode=cfg.encode, out_bf16=cfg.host_bf16)
            table[(L, slot)] = {"row_scale": row_scale.astype(np.float32), "col_scale": col_scale,
                                "in_scale": np.float32(in_scale), "w_clip": w_clip, "out_scale": out_scale,
                                "bias48": lin[(L, slot)].bias48.astype(np.int64)}
    for slot in PROJ_SLOTS:
        mat, b = proj_matrix(w, slot)
        if not cfg.proj_int8:
            lin[slot] = Fp32Linear(slot, mat, bias=b, out_bf16=False)
            continue
        r = stats.inputs[("proj", slot)]
        ro = stats.outputs[("proj", slot)]
        in_scale = choose_input_scale(cfg, r.rows_array(), float(r.amax.max()))
        w_clip = mse_clip_per_column(mat) if cfg.w_clip == "mse" else np.abs(mat).max(axis=0)
        w_clip = np.where(w_clip > 0, w_clip, 1.0).astype(np.float32)
        out_scale = choose_output_scale(cfg, ro.rows_array(), ro.amax)
        bias = None
        if cfg.bias_corr:
            w_scale = w_clip / 127.0
            dw = np.clip(rne(mat / w_scale), -127, 127) * w_scale - mat
            bias = -(r.mean().astype(np.float64) @ dw.astype(np.float64))
        lin[slot] = DeployLinear(slot, mat, w_clip, in_scale, out_scale, host_bias=b, bias=bias,
                                 bias_in_hw=False, encode=cfg.encode, out_bf16=False)
        table[slot] = {"in_scale": np.float32(in_scale), "w_clip": w_clip, "out_scale": out_scale,
                       "host_bias": lin[slot].host_bias.astype(np.float32)}
    table["attn_norm"] = attn_gain.astype(np.float32)
    table["ffw_norm"] = ffw_gain.astype(np.float32)
    table["final_norm"] = w.final_norm.astype(np.float32)
    table["prefix_v_scale"] = prefix_v_scale
    hm = HostModel(lin=lin, attn_gain=bf16(attn_gain) if cfg.host_bf16 else attn_gain,
                   ffw_gain=bf16(ffw_gain) if cfg.host_bf16 else ffw_gain,
                   final_gain=bf16(w.final_norm) if cfg.host_bf16 else w.final_norm,
                   prefix_v_scale=prefix_v_scale, host_bf16=cfg.host_bf16)
    return hm, table


NUMERICS_SCHEMA = "pi0-int8-deploy-numerics-v1"


def export_numerics(table: dict, path: str, meta: dict) -> None:
    """Write the export table as one .npz the LeRobot session exporter consumes
    (scripts/export_pi0_lerobot_int8_session.py).  Keys:
      layers/{L}/attn_norm, layers/{L}/ffw_norm        (1024,) f32  folded RMSNorm offsets
      layers/{L}/{slot}/row_scale (K,) col_scale (N,) w_clip (N,) in_scale () out_scale (N,) bias48 (N,) i64
      proj/{slot}/w_clip (N,) in_scale () out_scale (N,) host_bias (N,)
      final_norm (1024,), prefix_v_scale (18, 256), meta (json)
    W_export = clip(diag(row_scale) @ W @ diag(col_scale), +-w_clip); the exporter's
    per-column amax then equals w_clip and its codes equal this model's."""
    import json
    out = {"meta": np.array(json.dumps({"schema": NUMERICS_SCHEMA, **meta})),
           "final_norm": table["final_norm"], "prefix_v_scale": table["prefix_v_scale"]}
    for L in range(DEPTH):
        out[f"layers/{L}/attn_norm"] = table["attn_norm"][L]
        out[f"layers/{L}/ffw_norm"] = table["ffw_norm"][L]
        for slot in LAYER_SLOTS:
            for k, v in table[(L, slot)].items():
                out[f"layers/{L}/{slot}/{k}"] = np.asarray(v)
    for slot in PROJ_SLOTS:
        for k, v in table[slot].items():
            out[f"proj/{slot}/{k}"] = np.asarray(v)
    np.savez(os.path.expanduser(path), **out)


def collect_stats(w: M.Weights, episodes: list, rows: int = 2048, host_bf16: bool = True) -> CalibStats:
    stats = CalibStats(rows=rows)
    hm = fp32_host_model(w, stats=stats, host_bf16=host_bf16)
    for ep in episodes:
        denoise_host(hm, ep)
        stats.frames += 1
    stats.tokens = stats.frames * M.STEPS * (HORIZON + 1)
    # the projection inputs/outputs were recorded into stats.inputs/outputs under ("proj", slot)
    return stats


# --------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------
def chunk_error(a: np.ndarray, ref: np.ndarray, dims: int = 7) -> dict:
    a, ref = np.asarray(a, np.float64), np.asarray(ref, np.float64)
    def rel(x, y):
        return float(np.linalg.norm(x - y) / np.linalg.norm(y))
    def cos(x, y):
        return float((x * y).sum() / (np.linalg.norm(x) * np.linalg.norm(y)))
    return {"rel_rms_7": rel(a[:, :dims], ref[:, :dims]), "cos_7": cos(a[:, :dims].ravel(), ref[:, :dims].ravel()),
            "rel_rms_32": rel(a, ref), "max_abs_7": float(np.abs(a[:, :dims] - ref[:, :dims]).max()),
            "per_dim_max_abs": np.abs(a[:, :dims] - ref[:, :dims]).max(axis=0).tolist()}


def summarize(errs: list[dict]) -> dict:
    keys = ("rel_rms_7", "cos_7", "rel_rms_32", "max_abs_7")
    out = {}
    for k in keys:
        v = np.array([e[k] for e in errs])
        out[k] = {"mean": float(v.mean()), "min": float(v.min()), "max": float(v.max())}
    return out
