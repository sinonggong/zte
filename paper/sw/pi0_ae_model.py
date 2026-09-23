#!/usr/bin/env python3
"""NumPy model of the pi0 action expert (Gemma-300M suffix expert) with
pluggable fake-quantisation of every GEMM operand.

Purpose: quantisation-format sensitivity studies (INT8 per-tensor / per-channel,
MXINT{8,6,4} block formats, weight-only vs weight+activation) for the paper.
Weights are the real OpenPI pi0-DROID checkpoint read through tensorstore
(OCDBT); the prefix KV cache is synthetic (see `synth_prefix_kv`), because the
prefix PaliGemma-2B MLP tensors of this checkpoint are corrupt and no JAX/OpenPI
stack is installed on this machine.

Model conventions follow openpi (pi0.py / gemma.py):
  * RMSNorm y = x * rsqrt(mean(x^2)+1e-6) * (1+scale)
  * RoPE: half-split, max_wavelength 1e4, positions = token index, prefix
    occupies 0..P-1, suffix P..P+50
  * q scaled by head_dim^-0.5 after RoPE; 8 query heads, 1 KV head, head_dim 256
  * GeGLU MLP with tanh-approximate GELU
  * suffix tokens: [state, 50 action+time tokens]; attention mask: state sees
    prefix+state, actions see prefix+state+all actions
  * flow-matching Euler loop: 10 steps, t = 1.0 .. 0.1, x += dt * v, dt = -0.1
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np

CKPT = os.environ.get("PI0_CKPT", "/home/sngong/pi0-droid/params")
WIDTH, DEPTH, MLP, HEADS, KVHEADS, HDIM = 1024, 18, 4096, 8, 1, 256
ACTION_DIM, HORIZON, PREFIX_WIDTH = 32, 50, 2048
STEPS = 10


# --------------------------------------------------------------------------
# checkpoint
# --------------------------------------------------------------------------
def read_ckpt(key: str) -> np.ndarray:
    import tensorstore as ts
    spec = {"driver": "zarr",
            "kvstore": {"driver": "ocdbt", "base": "file://" + CKPT + "/"},
            "path": key}
    return np.asarray(ts.open(spec).result().read().result(), dtype=np.float32)


@dataclass
class Weights:
    q: np.ndarray        # (18, 1024, 2048)  [D, N*H]
    kv: np.ndarray       # (18, 1024, 512)   [D, k|v]
    o: np.ndarray        # (18, 2048, 1024)  [N*H, D]
    gate_up: np.ndarray  # (18, 1024, 8192)  [D, gate|up]
    down: np.ndarray     # (18, 4096, 1024)
    pre_attn: np.ndarray  # (18, 1024)
    pre_ffw: np.ndarray   # (18, 1024)
    final_norm: np.ndarray  # (1024,)
    state_w: np.ndarray; state_b: np.ndarray
    act_in_w: np.ndarray; act_in_b: np.ndarray
    tm_in_w: np.ndarray; tm_in_b: np.ndarray
    tm_out_w: np.ndarray; tm_out_b: np.ndarray
    act_out_w: np.ndarray; act_out_b: np.ndarray
    # prefix expert pieces used to synthesise a KV cache
    prefix_kv: np.ndarray | None = None      # (18, 2048, 512)
    prefix_pre_attn: np.ndarray | None = None  # (18, 2048)


_CACHE = {}

LEROBOT_EXPERT_PREFIX = "model.paligemma_with_expert.joint_layers.{L}.expert_layer."
LEROBOT_FINAL_NORM = "model.paligemma_with_expert.gemma_expert.model.norm.weight"


def load_weights_lerobot(path: str) -> Weights:
    """Weights of a LeRobot pi0 checkpoint (HF Gemma naming, `model.safetensors`),
    in the same [K, N] layouts as `load_weights`.  HF `nn.Linear.weight` is
    [out, in], so every matrix is transposed; head order (h*256+d) and the
    RMSNorm offset convention (1 + w) are the same as OpenPI's, so nothing else
    changes.  No prefix pieces: real prefix KV comes from LeRobot torch captures."""
    from safetensors import safe_open
    path = os.path.expanduser(path)
    if os.path.isdir(path):
        path = os.path.join(path, "model.safetensors")
    with safe_open(path, "np") as f:
        def T(k):
            return np.ascontiguousarray(f.get_tensor(k).astype(np.float32).T)

        def V(k):
            return np.ascontiguousarray(f.get_tensor(k).astype(np.float32))
        q, kv, o, gu, down, pre_attn, pre_ffw = [], [], [], [], [], [], []
        for L in range(DEPTH):
            p = LEROBOT_EXPERT_PREFIX.format(L=L)
            q.append(T(p + "self_attn.q_proj.weight"))                         # (1024, 2048)
            kv.append(np.concatenate([T(p + "self_attn.k_proj.weight"),
                                      T(p + "self_attn.v_proj.weight")], axis=1))  # (1024, 512)
            o.append(T(p + "self_attn.o_proj.weight"))                         # (2048, 1024)
            gu.append(np.concatenate([T(p + "mlp.gate_proj.weight"),
                                      T(p + "mlp.up_proj.weight")], axis=1))   # (1024, 8192)
            down.append(T(p + "mlp.down_proj.weight"))                         # (4096, 1024)
            pre_attn.append(V(p + "input_layernorm.weight"))
            pre_ffw.append(V(p + "post_attention_layernorm.weight"))
        w = Weights(
            q=np.stack(q), kv=np.stack(kv), o=np.stack(o), gate_up=np.stack(gu), down=np.stack(down),
            pre_attn=np.stack(pre_attn), pre_ffw=np.stack(pre_ffw),
            final_norm=V(LEROBOT_FINAL_NORM),
            state_w=T("model.state_proj.weight"), state_b=V("model.state_proj.bias"),
            act_in_w=T("model.action_in_proj.weight"), act_in_b=V("model.action_in_proj.bias"),
            tm_in_w=T("model.action_time_mlp_in.weight"), tm_in_b=V("model.action_time_mlp_in.bias"),
            tm_out_w=T("model.action_time_mlp_out.weight"), tm_out_b=V("model.action_time_mlp_out.bias"),
            act_out_w=T("model.action_out_proj.weight"), act_out_b=V("model.action_out_proj.bias"),
        )
    for name in ("pre_attn", "pre_ffw", "final_norm"):
        assert np.abs(getattr(w, name)).max() < 0.5, f"{name} is not an RMSNorm offset (1 + w) tensor"
    return w


def load_weights(with_prefix: bool = True) -> Weights:
    if "w" in _CACHE:
        return _CACHE["w"]
    p = "params.PaliGemma.llm."
    q = read_ckpt(p + "layers.attn.q_einsum_1.w")            # (18,8,1024,256)
    q = np.ascontiguousarray(q.transpose(0, 2, 1, 3).reshape(DEPTH, WIDTH, HEADS * HDIM))
    kv = read_ckpt(p + "layers.attn.kv_einsum_1.w")          # (18,2,1,1024,256)
    kv = np.ascontiguousarray(kv[:, :, 0].transpose(0, 2, 1, 3).reshape(DEPTH, WIDTH, 2 * HDIM))
    o = read_ckpt(p + "layers.attn.attn_vec_einsum_1.w")     # (18,8,256,1024)
    o = np.ascontiguousarray(o.reshape(DEPTH, HEADS * HDIM, WIDTH))
    gu = read_ckpt(p + "layers.mlp_1.gating_einsum")         # (18,2,1024,4096)
    gu = np.ascontiguousarray(gu.transpose(0, 2, 1, 3).reshape(DEPTH, WIDTH, 2 * MLP))
    down = read_ckpt(p + "layers.mlp_1.linear")              # (18,4096,1024)
    w = Weights(
        q=q, kv=kv, o=o, gate_up=gu, down=down,
        pre_attn=read_ckpt(p + "layers.pre_attention_norm_1.scale"),
        pre_ffw=read_ckpt(p + "layers.pre_ffw_norm_1.scale"),
        final_norm=read_ckpt(p + "final_norm_1.scale"),
        state_w=read_ckpt("params.state_proj.kernel"), state_b=read_ckpt("params.state_proj.bias"),
        act_in_w=read_ckpt("params.action_in_proj.kernel"), act_in_b=read_ckpt("params.action_in_proj.bias"),
        tm_in_w=read_ckpt("params.action_time_mlp_in.kernel"), tm_in_b=read_ckpt("params.action_time_mlp_in.bias"),
        tm_out_w=read_ckpt("params.action_time_mlp_out.kernel"), tm_out_b=read_ckpt("params.action_time_mlp_out.bias"),
        act_out_w=read_ckpt("params.action_out_proj.kernel"), act_out_b=read_ckpt("params.action_out_proj.bias"),
    )
    if with_prefix:
        pkv = read_ckpt(p + "layers.attn.kv_einsum.w")       # (18,2,1,2048,256)
        w.prefix_kv = np.ascontiguousarray(pkv[:, :, 0].transpose(0, 2, 1, 3).reshape(DEPTH, PREFIX_WIDTH, 2 * HDIM))
        w.prefix_pre_attn = read_ckpt(p + "layers.pre_attention_norm.scale")
    _CACHE["w"] = w
    return w


# --------------------------------------------------------------------------
# quantisation formats (fake-quant: returns integer codes + scales)
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class QFormat:
    """kind: 'fp' (no quant), 'int' (float scale), 'mx' (power-of-two shared scale)
    bits: element bits (signed, symmetric)
    group: 'tensor' | 'channel' (per output channel of W / per token of X) | int block along K
    """
    kind: str = "fp"
    bits: int = 8
    group: object = "channel"

    def label(self) -> str:
        if self.kind == "fp":
            return "FP32"
        if self.kind == "bf16":
            return "BF16"
        g = self.group if isinstance(self.group, str) else f"b{self.group}"
        return f"{'MXINT' if self.kind == 'mx' else 'INT'}{self.bits}-{g}"


FP = QFormat()


def _qmax(bits: int) -> int:
    return (1 << (bits - 1)) - 1


def quantize_k_major(x: np.ndarray, fmt: QFormat, k_axis: int):
    """Quantise `x` where `k_axis` is the reduction (K) axis.
    Returns (codes float64, scale float64 broadcastable to x, block) with
    block=None for tensor/channel groups, else the K block size."""
    if fmt.kind == "fp":
        return None
    qmax = _qmax(fmt.bits)
    x64 = x.astype(np.float64)
    if isinstance(fmt.group, int):
        B = fmt.group
        K = x.shape[k_axis]
        assert K % B == 0, (K, B)
        xs = np.moveaxis(x64, k_axis, -1)
        shp = xs.shape[:-1] + (K // B, B)
        xs = xs.reshape(shp)
        amax = np.abs(xs).max(axis=-1, keepdims=True)
    elif fmt.group == "tensor":
        xs = x64
        amax = np.abs(xs).max(keepdims=True)
    elif fmt.group == "channel":
        xs = x64
        amax = np.abs(xs).max(axis=k_axis, keepdims=True)
    else:
        raise ValueError(fmt)
    if fmt.kind == "mx":
        # OCP MX: shared scale is a power of two (E8M0); elements INT{bits}
        # scale = 2^(floor(log2 amax) - (bits-2)) so that |q| <= 2^(bits-1)-1
        with np.errstate(divide="ignore"):
            e = np.floor(np.log2(np.where(amax > 0, amax, 1.0)))
        scale = np.exp2(e - (fmt.bits - 2))
    elif fmt.kind == "int":
        scale = np.where(amax > 0, amax, 1.0) / qmax
    else:
        raise ValueError(fmt)
    codes = np.clip(np.rint(xs / scale), -qmax, qmax)
    return codes, scale


def to_bf16(x: np.ndarray) -> np.ndarray:
    u = np.ascontiguousarray(x, dtype=np.float32).view(np.uint32)
    r = ((u >> 16) & 1) + 0x7FFF
    return ((u + r) & 0xFFFF0000).view(np.float32)


def fq(x: np.ndarray, fmt: QFormat, k_axis: int) -> np.ndarray:
    """Fake-quantise: dequantised float32 copy of x (K along k_axis)."""
    if fmt.kind == "bf16":
        return to_bf16(x)
    q = quantize_k_major(x, fmt, k_axis)
    if q is None:
        return x.astype(np.float32)
    codes, scale = q
    y = codes * scale
    if isinstance(fmt.group, int):
        y = y.reshape(y.shape[:-2] + (-1,))
        y = np.moveaxis(y, -1, k_axis)
    return np.ascontiguousarray(y, dtype=np.float32)


class QWeight:
    """A weight matrix (K,N) prepared for one format.
    fp: dequantised float32 matrix.  For block formats also codes (N,nb,B) as
    int8 and e (N,nb) = log2(shared scale) for the fixed-point accumulator."""
    __slots__ = ("fp", "codes", "e", "B")

    def __init__(self, w: np.ndarray, fmt: QFormat):
        self.B = fmt.group if isinstance(fmt.group, int) else None
        if self.B is not None:
            codes, scale = quantize_k_major(w, fmt, 0)   # (N,nb,B), (N,nb,1)
            self.codes = codes.astype(np.int8)
            self.e = np.log2(scale[..., 0]).astype(np.int16)
            self.fp = np.ascontiguousarray(
                np.moveaxis((codes * scale).reshape(w.shape[1], -1), 1, 0), dtype=np.float32)
        else:
            self.codes = self.e = None
            self.fp = fq(w, fmt, 0)


ACC_RANGE_BITS = 22   # 48-bit accumulator, 19-bit block sum, 7 bits of block count


def gemm(x: np.ndarray, w: "QWeight | np.ndarray", xf: QFormat, acc: str = "fp") -> np.ndarray:
    """y = fakequant(x) @ w.  acc='fp': exact (float64) accumulation.
    acc='fix48': per-block integer dot products aligned by shifting into a
    48-bit fixed-point accumulator whose reference exponent is the largest
    block-exponent sum of the launch minus ACC_RANGE_BITS (blocks below it
    lose low bits) -- the cheap fabric-side MX accumulator."""
    if isinstance(w, np.ndarray):
        w = QWeight(w, FP)
    if acc == "fp" or w.B is None or not isinstance(xf.group, int):
        xq = fq(x, xf, 1)
        return (xq.astype(np.float64) @ w.fp.astype(np.float64)).astype(np.float32)
    # fixed-point accumulator path (both operands block formats)
    Bx, Bw = xf.group, w.B
    B = min(Bx, Bw)
    M, K = x.shape
    N = w.fp.shape[1]
    nb = K // B
    cx, sx = quantize_k_major(x, xf, 1)                        # (M,K/Bx,Bx), (M,K/Bx,1)
    ex = np.log2(sx[..., 0]).astype(np.int32)                  # (M,K/Bx)
    cx = cx.reshape(M, nb, B)
    ex = np.repeat(ex, Bx // B, axis=1)                        # (M,nb)
    cw = w.codes.astype(np.float64).reshape(N, nb, B).transpose(1, 2, 0)   # (nb,B,N)
    ew = np.repeat(w.e.astype(np.int32), Bw // B, axis=1).T    # (nb,N)
    part = np.einsum("mkb,kbn->mkn", cx, cw)                   # (M,nb,N) exact ints
    esum = ex[:, :, None] + ew[None, :, :]                     # (M,nb,N)
    e_ref = esum.max() - ACC_RANGE_BITS
    shift = esum - e_ref
    contrib = np.where(shift >= 0, part * np.exp2(shift), np.floor(part * np.exp2(shift)))
    y = contrib.sum(axis=1) * np.exp2(e_ref)
    return y.astype(np.float32)


# --------------------------------------------------------------------------
# model pieces
# --------------------------------------------------------------------------
def rmsnorm(x, scale, eps=1e-6):
    x = x.astype(np.float32)
    var = np.mean(x.astype(np.float64) ** 2, axis=-1, keepdims=True)
    return (x * (1.0 / np.sqrt(var + eps)) * (1.0 + scale)).astype(np.float32)


def rope(x, positions, max_wavelength=10_000.0):
    # x (T, heads, H); positions (T,)
    H = x.shape[-1]
    half = H // 2
    freq_exp = (2.0 / H) * np.arange(half, dtype=np.float32)
    timescale = max_wavelength ** freq_exp
    radians = positions.astype(np.float32)[:, None] / timescale[None, :]   # (T, half)
    sin, cos = np.sin(radians), np.cos(radians)
    sin, cos = sin[:, None, :], cos[:, None, :]
    x1, x2 = x[..., :half], x[..., half:]
    return np.concatenate([x1 * cos - x2 * sin, x2 * cos + x1 * sin], axis=-1).astype(np.float32)


def gelu_tanh(x):
    x = x.astype(np.float32)
    return (0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x ** 3)))).astype(np.float32)


def swish(x):
    x = x.astype(np.float32)
    return (x / (1.0 + np.exp(-x))).astype(np.float32)


def posemb_sincos(t: float, width: int, min_period=4e-3, max_period=4.0):
    half = width // 2
    fraction = np.linspace(0.0, 1.0, half, dtype=np.float32)
    period = min_period * (max_period / min_period) ** fraction
    ang = np.float32(t) * (2.0 * np.pi / period)
    return np.concatenate([np.sin(ang), np.cos(ang)]).astype(np.float32)


@dataclass
class QConfig:
    """Which format each GEMM class uses. Slots: q,k,v,o,gate,up,down (FPGA
    GEMMs) and 'proj' (action projections; attention QK/PV stay FP32)."""
    w: dict = field(default_factory=dict)   # slot -> QFormat for weights
    x: dict = field(default_factory=dict)   # slot -> QFormat for activations
    acc: str = "fp"                         # 'fp' | 'fix48'
    name: str = ""

    def wf(self, slot):
        return self.w.get(slot, FP)

    def xf(self, slot):
        return self.x.get(slot, FP)

    def attn_quant(self):
        return any(k in self.w or k in self.x for k in ("qk", "pv"))

    @staticmethod
    def uniform(wf: QFormat, xf: QFormat = FP, slots=("q", "k", "v", "o", "gate", "up", "down"),
                acc="fp", name=""):
        return QConfig(w={s: wf for s in slots}, x={s: xf for s in slots}, acc=acc, name=name)


SLOTS = ("q", "k", "v", "o", "gate", "up", "down")


class QModel:
    """Weights prepared (fake-quantised) for one QConfig."""

    def __init__(self, w: Weights, qc: QConfig):
        self.w = w
        self.qc = qc
        self.q = [QWeight(w.q[L], qc.wf("q")) for L in range(DEPTH)]
        self.k = [QWeight(w.kv[L][:, :HDIM], qc.wf("k")) for L in range(DEPTH)]
        self.v = [QWeight(w.kv[L][:, HDIM:], qc.wf("v")) for L in range(DEPTH)]
        self.o = [QWeight(w.o[L], qc.wf("o")) for L in range(DEPTH)]
        self.gate = [QWeight(w.gate_up[L][:, :MLP], qc.wf("gate")) for L in range(DEPTH)]
        self.up = [QWeight(w.gate_up[L][:, MLP:], qc.wf("up")) for L in range(DEPTH)]
        self.down = [QWeight(w.down[L], qc.wf("down")) for L in range(DEPTH)]
        pf = qc.wf("proj")
        self.state_w = QWeight(w.state_w, pf)
        self.act_in_w = QWeight(w.act_in_w, pf)
        self.tm_in_w = QWeight(w.tm_in_w, pf)
        self.tm_out_w = QWeight(w.tm_out_w, pf)
        self.act_out_w = QWeight(w.act_out_w, pf)


def synth_prefix_kv(w: Weights, P: int, seed: int):
    """Synthetic prefix KV: random unit-RMS pre-norm hidden states pushed
    through the real prefix-expert pre-attention norm and kv_einsum, RoPE on K
    with positions 0..P-1.  Returns list of (K (P,256), V (P,256)) per layer."""
    rng = np.random.default_rng(seed)
    out = []
    pos = np.arange(P)
    for L in range(DEPTH):
        h = rng.standard_normal((P, PREFIX_WIDTH), dtype=np.float32)
        hn = rmsnorm(h, w.prefix_pre_attn[L])
        kv = hn.astype(np.float64) @ w.prefix_kv[L].astype(np.float64)
        k = rope(kv[:, :HDIM].astype(np.float32)[:, None, :], pos)[:, 0]
        v = kv[:, HDIM:].astype(np.float32)
        out.append((k, v))
    return out


def suffix_mask(S: int):
    """(S,S) boolean: state token (index 0) attends to itself; action tokens
    attend to state and all action tokens."""
    m = np.ones((S, S), dtype=bool)
    m[0, 1:] = False
    return m


def action_expert_forward(qm: QModel, tokens: np.ndarray, prefix_kv,
                          record: list | None = None, act_stats: list | None = None,
                          prefix_valid: np.ndarray | None = None) -> np.ndarray:
    """tokens (S=51, 1024) suffix embeddings -> velocity (50, 32).
    prefix_valid (P,) bool: padded prefix tokens (empty camera slot, language
    padding) are masked out of the attention and do not count towards the
    suffix RoPE positions, exactly as LeRobot's denoise_step does."""
    w, qc = qm.w, qm.qc
    S = tokens.shape[0]
    P = prefix_kv[0][0].shape[0]
    if prefix_valid is None:
        prefix_valid = np.ones(P, bool)
    prefix_valid = np.asarray(prefix_valid).astype(bool)
    n_valid = int(prefix_valid.sum())
    pos = np.arange(n_valid, n_valid + S)
    mask_s = suffix_mask(S)
    x = tokens.astype(np.float32)
    scale = HDIM ** -0.5
    for L in range(DEPTH):
        xn = rmsnorm(x, w.pre_attn[L])
        q = gemm(xn, qm.q[L], qc.xf("q"), qc.acc).reshape(S, HEADS, HDIM)
        k = gemm(xn, qm.k[L], qc.xf("k"), qc.acc)
        v = gemm(xn, qm.v[L], qc.xf("v"), qc.acc)
        q = rope(q, pos) * scale
        k = rope(k[:, None, :], pos)[:, 0]
        Kp, Vp = prefix_kv[L]
        Kall = np.concatenate([Kp, k], axis=0)           # (P+S, 256)
        Vall = np.concatenate([Vp, v], axis=0)
        full_mask = np.concatenate([np.broadcast_to(prefix_valid[None], (S, P)), mask_s], axis=1)
        T = Kall.shape[0]
        if qc.attn_quant():
            # attention as two INT8/MX GEMMs (as deployed): logits = q . K^T with
            # K along head_dim, context = P . V with K along the key axis
            # (padded to a multiple of 64 like the engine's K tiles)
            Tp = -(-T // 64) * 64
            Kpad = np.zeros((Tp, HDIM), np.float32); Kpad[:T] = Kall
            Vpad = np.zeros((Tp, HDIM), np.float32); Vpad[:T] = Vall
            qmat = q.reshape(S * HEADS, HDIM)
            logits = gemm(qmat, QWeight(np.ascontiguousarray(Kpad.T), qc.wf("qk")), qc.xf("qk"), "fp")
            logits = logits.reshape(S, HEADS, Tp).transpose(1, 0, 2).astype(np.float64)
            mpad = np.concatenate([full_mask, np.zeros((S, Tp - T), bool)], axis=1)
            logits = np.where(mpad[None], logits, -1e30)
            logits -= logits.max(axis=-1, keepdims=True)
            p = np.exp(logits)
            p /= p.sum(axis=-1, keepdims=True)
            pmat = p.transpose(1, 0, 2).reshape(S * HEADS, Tp).astype(np.float32)
            ctx = gemm(pmat, QWeight(Vpad, qc.wf("pv")), qc.xf("pv"), "fp").reshape(S, HEADS, HDIM)
        else:
            logits = np.einsum("shd,td->hst", q.astype(np.float64), Kall.astype(np.float64))
            logits = np.where(full_mask[None], logits, -1e30)
            logits -= logits.max(axis=-1, keepdims=True)
            p = np.exp(logits)
            p /= p.sum(axis=-1, keepdims=True)
            ctx = np.einsum("hst,td->shd", p, Vall.astype(np.float64)).astype(np.float32)
        attn = gemm(ctx.reshape(S, HEADS * HDIM), qm.o[L], qc.xf("o"), qc.acc)
        x = x + attn
        xn2 = rmsnorm(x, w.pre_ffw[L])
        g = gemm(xn2, qm.gate[L], qc.xf("gate"), qc.acc)
        u = gemm(xn2, qm.up[L], qc.xf("up"), qc.acc)
        h = gelu_tanh(g) * u
        d = gemm(h, qm.down[L], qc.xf("down"), qc.acc)
        x = x + d
        if record is not None:
            record.append(x.copy())
        if act_stats is not None:
            act_stats.append({"attn_in": xn, "o_in": ctx.reshape(S, HEADS * HDIM),
                              "ffw_in": xn2, "down_in": h})
    xf_ = rmsnorm(x, w.final_norm)
    v_out = gemm(xf_[1:], qm.act_out_w, qc.xf("proj"), qc.acc) + w.act_out_b
    return v_out.astype(np.float32)


def embed_suffix(qm: QModel, state: np.ndarray, x_t: np.ndarray, t: float):
    w, qc = qm.w, qm.qc
    pf = qc.xf("proj")
    st = gemm(state[None], qm.state_w, pf, qc.acc) + w.state_b                 # (1,1024)
    at = gemm(x_t, qm.act_in_w, pf, qc.acc) + w.act_in_b                       # (50,1024)
    te = np.broadcast_to(posemb_sincos(t, WIDTH)[None], (HORIZON, WIDTH))
    att = np.concatenate([at, te], axis=1)                                     # (50,2048)
    h = swish(gemm(att, qm.tm_in_w, pf, qc.acc) + w.tm_in_b)
    h = gemm(h, qm.tm_out_w, pf, qc.acc) + w.tm_out_b
    return np.concatenate([st, h], axis=0).astype(np.float32)                  # (51,1024)


def denoise(qm: QModel, state, noise, prefix_kv, steps: int = STEPS,
            trace: list | None = None, act_stats: list | None = None,
            prefix_valid: np.ndarray | None = None) -> np.ndarray:
    x_t = noise.astype(np.float32).copy()
    dt = np.float32(-1.0 / steps)
    t = np.float32(1.0)
    for s in range(steps):
        tok = embed_suffix(qm, state, x_t, float(t))
        v = action_expert_forward(qm, tok, prefix_kv, act_stats=act_stats, prefix_valid=prefix_valid)
        x_t = x_t + dt * v
        t = np.float32(t + dt)
        if trace is not None:
            trace.append(x_t.copy())
    return x_t


# --------------------------------------------------------------------------
# episodes (inputs) -- state from a unit normal (actions/state are normalised
# by norm_stats in openpi, so unit scale is the deployed scale), noise N(0,1)
# --------------------------------------------------------------------------
@dataclass
class Episode:
    state: np.ndarray
    noise: np.ndarray
    prefix_kv: list
    prefix_valid: np.ndarray | None = None
    actions_ref: np.ndarray | None = None   # torch fp32 result on the same noise, if captured
    name: str = ""


def load_capture(path: str, bf16_kv: bool = False) -> Episode:
    """A frame captured by glue/level3/capture_pi0_prefix_kv.py: real prefix KV
    (LeRobot torch fp32), real state, seeded noise and the torch fp32 action chunk.
    bf16_kv=True rounds the cache to bf16, which is what the host runtime holds."""
    z = np.load(os.path.expanduser(path))
    kv = z["kv"]                                     # (L, 2, T, H=1, D)
    kv = kv[:, :, :, 0, :]
    if bf16_kv:
        kv = to_bf16(kv)
    prefix_kv = [(np.ascontiguousarray(kv[L, 0]), np.ascontiguousarray(kv[L, 1])) for L in range(kv.shape[0])]
    return Episode(state=z["state"].astype(np.float32), noise=z["noise"].astype(np.float32),
                   prefix_kv=prefix_kv, prefix_valid=z["prefix_valid"].astype(bool),
                   actions_ref=z["actions_fp32"].astype(np.float32),
                   name=os.path.join(os.path.basename(os.path.dirname(os.path.abspath(os.path.expanduser(path)))), os.path.basename(path)))


def make_episode(w: Weights, seed: int, P: int = 816) -> Episode:
    rng = np.random.default_rng(1000 + seed)
    state = rng.standard_normal(ACTION_DIM).astype(np.float32)
    noise = rng.standard_normal((HORIZON, ACTION_DIM)).astype(np.float32)
    return Episode(state, noise, synth_prefix_kv(w, P, 5000 + seed))


if __name__ == "__main__":
    import time
    t0 = time.time()
    w = load_weights()
    print("weights loaded", round(time.time() - t0, 1), "s")
    ep = make_episode(w, 0)
    t0 = time.time()
    a = denoise(QModel(w, QConfig()), ep.state, ep.noise, ep.prefix_kv)
    print("fp32 denoise", round(time.time() - t0, 1), "s", "action rms", float(np.sqrt((a ** 2).mean())))
    for name, qc in [
        ("int8 pc/pt", QConfig.uniform(QFormat("int", 8, "channel"), QFormat("int", 8, "tensor"))),
        ("mxint8 b32", QConfig.uniform(QFormat("mx", 8, 32), QFormat("mx", 8, 32))),
        ("mxint8 b32 fix48", QConfig.uniform(QFormat("mx", 8, 32), QFormat("mx", 8, 32), acc="fix48")),
    ]:
        t0 = time.time()
        qm = QModel(w, qc)
        t1 = time.time()
        b = denoise(qm, ep.state, ep.noise, ep.prefix_kv)
        print(name, "prep", round(t1 - t0, 1), "s run", round(time.time() - t1, 1), "s rel err",
              float(np.linalg.norm(a - b) / np.linalg.norm(a)))
