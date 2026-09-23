#!/usr/bin/env python3
"""Lower one real pi0 expert layer -- and the prefix K/V it attends to -- onto the node command set, run it with
the bit-exact node references, and compare every stage with fp32.

Numerics: the chip recipe of docs/PI0_FULL_MODEL_ON_CHIP_DESIGN_20260916.md 3.1 (W8 per output channel MSE clip,
A8 per token dynamic, SmoothQuant alpha folded into the RMSNorm gains / constant scales, QK^T int8 per row x int8
per key, PV uint8 P per row x int8 V per channel).  Arithmetic: vector node ops = paper/sw/vector_unit_ref.py
(bit-exact to vu_lane.sv), chain GEMMs = exact int64 matmul (bit-exact to the column-parallel chain).

Data layouts (what the programs must realise; the gaps are listed in the report):
  keys        compact prefix (525 valid of 816 slots) padded to 528, then the 51 suffix keys padded to 64: 592 = 37 words
  K codes     one row per key, QUANT per row; the rows are the QK^T weight columns  -> stage-interleaved LOAD
  V           produced COLUMN-MAJOR (one row per head_dim channel) by a role-swapped GEMM (feeder = W_v rows, stage
              columns = the token code rows), so QUANT per row is per channel and every row is one PV weight column
  PV          two GEMMs: P[:, 0:528] x V_prefix, P[:, 528:592] x V_suffix (separate per-channel scales) -> DEQUANT, ADD
  RoPE        ROPE_A (x cos) then ROPE_B on the rotate-half partner read of the same row (sin sign folded)
  folds       q DEQUANT s_col x scaling / ln 2 (log2-domain logits); V DEQUANT s_row / s_o (o smoothing, per head_dim);
              up DEQUANT s_col / s_down; RMS gains (1 + w) / s_qkv and (1 + w) / s_gateup

Usage: pi0_layer_lower.py [--layers 0 9 17] [--steps 0 1] [--alpha 0.5] [--json out.json]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(HERE))
import vector_unit_ref as R      # noqa: E402

CKPT = os.path.expanduser("~/pi0_ckpt/ur7e-demo-2-pi0-fsdp/010000/pretrained_model/model.safetensors")
CAPTURE = REPO / "build/paper_vector_unit/acts_demo1_ep20_f02.npz"
# SmoothQuant / MSE calibration statistics (prefix_w8a8_eval.py --stages calib: calib.pt, and calib_exp.pt from its
# expert calibration), regenerated into the repo's build tree by paper/sw/regen_calib.py.  PI0_CALIB_DIR overrides.
CALIB = Path(os.environ["PI0_CALIB_DIR"]) if os.environ.get("PI0_CALIB_DIR") else REPO / "build/paper_vector_unit/calib"
W_GRID = np.linspace(0.5, 1.0, 21)
PFX = "model.paligemma_with_expert.joint_layers.{L}.{part}."
HEADS, HD = 8, 256
N_PRE_SLOTS, N_SUF_SLOTS = 528, 64
LN2 = math.log(2.0)


# ---------------------------------------------------------------- weights and smoothing
def imatmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Exact integer a @ b as int64.  numpy's integer matmul does not use BLAS (~1 GMAC/s); when every partial sum
    is below 2^53 the float64 BLAS product is exact and ~15x faster, so the chunk generator's int8 / uint8 GEMMs
    (|a|, |b| <= 255, K <= 16384: |acc| < 2^30) take that path.  Anything wider falls back to int64."""
    a = np.asarray(a); b = np.asarray(b)
    assert np.issubdtype(a.dtype, np.integer) and np.issubdtype(b.dtype, np.integer), (a.dtype, b.dtype)
    if a.size == 0 or b.size == 0:
        return (a.astype(np.int64) @ b.astype(np.int64))
    k = a.shape[-1]
    bound = float(np.abs(a).max()) * float(np.abs(b).max()) * k
    if bound < 2.0 ** 53:
        return (a.astype(np.float64) @ b.astype(np.float64)).astype(np.int64)
    return a.astype(np.int64) @ b.astype(np.int64)


def smoothing_factors(x_amax, w_rowmax, alpha, lo=1e-2, hi=1e2):
    """= pi0_deploy_model.smoothing_factors"""
    xa = np.maximum(x_amax, 1e-8)
    wa = np.maximum(w_rowmax, 1e-8)
    return np.clip(xa ** alpha / wa ** (1.0 - alpha), lo, hi).astype(np.float32)


def mse_clip_rows(w: np.ndarray) -> np.ndarray:
    """per output channel clip minimising the channel's INT8 MSE (grid and tie rule of prefix_w8a8_eval.mse_clip_rows)."""
    w = w.astype(np.float32)
    amax = np.abs(w).max(1)
    best = np.full(w.shape[0], np.inf, np.float32)
    best_c = amax.copy()
    for f in W_GRID:
        c = np.where(amax > 0, amax * np.float32(f), np.float32(1.0)).astype(np.float32)
        s = (c / np.float32(127.0))[:, None]
        err = ((np.clip(np.round(w / s), -127, 127) * s - w) ** 2).mean(1)
        better = err < best
        best = np.where(better, err, best)
        best_c = np.where(better, c, best_c)
    return np.where(best_c > 0, best_c, np.float32(1.0)).astype(np.float32)


class QW:
    """INT8 GEMM weights: codes (out, in) int64, scale (out,) float32 (fp32 constants on the chip)."""

    def __init__(self, w: np.ndarray, s_in: np.ndarray | None):
        self.w = w.astype(np.float32)                      # fp32 weight, unsmoothed (reference)
        ws = self.w * s_in[None, :] if s_in is not None else self.w
        self.scale = (mse_clip_rows(ws) / np.float32(127.0)).astype(np.float32)
        self.codes = np.clip(np.round(ws / self.scale[:, None]), -127, 127).astype(np.int64)


def load_layer(L: int, part: str, alpha: float, calib_key: str, gemms=("q", "k", "v", "o", "gate", "up", "down")):
    from safetensors import safe_open
    import torch
    t = {}
    with safe_open(CKPT, "np") as f:
        p = PFX.format(L=L, part=part)
        for n in ("input_layernorm", "post_attention_layernorm"):
            t[n] = f.get_tensor(p + n + ".weight").astype(np.float32)
        for n in ("q_proj", "k_proj", "v_proj", "o_proj"):
            t[n] = f.get_tensor(p + "self_attn." + n + ".weight").astype(np.float32)
        for n in ("gate_proj", "up_proj", "down_proj"):
            t[n] = f.get_tensor(p + "mlp." + n + ".weight").astype(np.float32)
    ch = torch.load(CALIB / ("calib_exp.pt" if part == "expert_layer" else "calib.pt"), weights_only=False)["ch_amax"]
    key = f"{calib_key}.L{L}."
    assert key + "q" in ch, f"no calibration key {key}q in {sorted(ch)[:8]}..."
    chn = lambda s: ch[key + s].numpy().astype(np.float64)                   # noqa: E731
    wr = np.maximum(np.maximum(np.abs(t["q_proj"]).max(0), np.abs(t["k_proj"]).max(0)), np.abs(t["v_proj"]).max(0))
    s_qkv = smoothing_factors(chn("q"), wr, alpha)
    wr = np.maximum(np.abs(t["gate_proj"]).max(0), np.abs(t["up_proj"]).max(0))
    s_gu = smoothing_factors(chn("gate"), wr, alpha)
    xa = chn("o").reshape(HEADS, -1).max(0)
    wr = np.abs(t["o_proj"]).max(0).reshape(HEADS, -1).max(0)
    s_o_hd = smoothing_factors(xa, wr, alpha)
    s_o = np.tile(s_o_hd, HEADS)
    s_d = smoothing_factors(chn("down"), np.abs(t["down_proj"]).max(0), alpha)
    if alpha == 0.0:
        s_qkv, s_gu, s_o, s_o_hd, s_d = (np.ones_like(v) for v in (s_qkv, s_gu, s_o, s_o_hd, s_d))
    lay = dict(g_in=t["input_layernorm"], g_post=t["post_attention_layernorm"], s_qkv=s_qkv, s_gu=s_gu, s_o=s_o,
               s_o_hd=s_o_hd, s_d=s_d)
    src = dict(q=("q_proj", s_qkv), k=("k_proj", s_qkv), v=("v_proj", s_qkv), o=("o_proj", s_o),
               gate=("gate_proj", s_gu), up=("up_proj", s_gu), down=("down_proj", s_d))
    for g in gemms:
        lay[g] = QW(t[src[g][0]], src[g][1])
    return lay


# ---------------------------------------------------------------- fp32 reference (float64 arithmetic)
def rms_fp(x, g):
    return x / np.sqrt((x * x).mean(-1, keepdims=True) + 1e-6) * (1.0 + g.astype(np.float64))


def rope_fp(x, cos, sin):
    h = x.shape[-1] // 2
    rot = np.concatenate([-x[..., h:], x[..., :h]], -1)
    return x * cos + rot * sin


def gelu_tanh(x):
    return 0.5 * x * (1.0 + np.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x ** 3)))


def softmax_masked(z, mask):
    z = np.where(mask, z, -np.inf)
    z = z - z.max(-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(-1, keepdims=True)


# ---------------------------------------------------------------- chip ops (bit-exact references)
def f32c(x) -> np.ndarray:
    return R.f32_codes(np.asarray(x, np.float32))


def bf16v(c) -> np.ndarray:
    return R.bf16_val(c).astype(np.float64)


def masked_max_code(codes, mask):
    """signed max bf16 code over the mask (the lane's tracker max; 0xFF80 when nothing is valid)."""
    v = np.where(mask, bf16v(codes), -np.inf)
    i = v.argmax(-1)
    mx = np.take_along_axis(codes, i[..., None], -1)[..., 0].astype(np.int64)
    return np.where(mask.any(-1), mx, 0xFF80)


class Chip:
    def __init__(self):
        self.ops: dict[str, int] = {}
        self.gemm_products = 0
        self.max_acc = 0

    def _n(self, name, elems):
        self.ops[name] = self.ops.get(name, 0) + int(elems)

    def rms(self, x, gain_d):
        amax = R.row_amax_code(x)
        r = R.rms_stat(x, amax, 1.0 / x.shape[-1])
        y, _ = R.rms_apply(x, r, gain_d)
        self._n("RMS_STAT", x.size)
        self._n("RMS_APPLY", x.size)
        return y

    def quant(self, x):
        q, s_row, _ = R.op_quant(x, R.row_amax_code(x))
        self._n("QUANT", x.size)
        return q.astype(np.int64), s_row

    def gemm(self, a, w):
        acc = imatmul(a, w.T)
        self.gemm_products += a.shape[0] * w.shape[0] * a.shape[1]
        self.max_acc = max(self.max_acc, int(np.abs(acc).max()))
        return acc

    def dequant(self, acc, s_row, s_col, out="bf16"):
        y, _ = R.op_dequant(acc, s_row, s_col, None, out)
        self._n("DEQUANT", acc.size)
        return y

    def add(self, a, b):
        y, _ = R.op_add(a, b)
        self._n("ADD", a.size)
        return y

    def rope(self, x, cos_f32, sin_signed_f32):
        """ROPE_A x cos, ROPE_B over the rotate-half partner (x[j + h] for j < h, x[j - h] otherwise) of the same row."""
        h = x.shape[-1] // 2
        pa, _ = R.rope_a(x, cos_f32)
        partner = np.concatenate([x[..., h:], x[..., :h]], -1)
        y, _ = R.rope_b(partner, sin_signed_f32, pa)
        self._n("ROPE_A", x.size)
        self._n("ROPE_B", x.size)
        return y

    def smax_u8(self, logits, mask):
        """SMAX_Q8 with 255 levels: uint8 P codes e_i * 255 per row, s_row = (1/S) / 255 (vector_unit_ref)."""
        s_row, q = R.smax_sum(logits, masked_max_code(logits, mask), mask, int8=True, levels=255)
        self._n("SMAX_U8", logits.size)
        return q.astype(np.int64), s_row

    def geglu(self, g, u):
        y, _ = R.op_geglu(g, u)
        self._n("GEGLU", g.size)
        return y


def rel(a, b) -> float:
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-30))


# ---------------------------------------------------------------- one layer, one Euler step
def run(L: int, step: int, alpha: float, z, lm, ex) -> dict:
    x = z[f"ex.L{L}.layer_in"][step].astype(np.float64)                      # (T, 1024)
    T = x.shape[0]
    mask867 = z[f"ex.L{L}.mask"][step]                                        # (T, 867)
    pre_idx = np.nonzero(mask867[-1, :816])[0]
    n_pre = len(pre_idx)
    assert n_pre <= N_PRE_SLOTS and T <= N_SUF_SLOTS
    lm_in = z[f"lm.L{L}.layer_in"].astype(np.float64)                         # (525, 2048)
    assert lm_in.shape[0] == n_pre
    cos_s, sin_s = z["ex.rope_cos"][step].astype(np.float64), z["ex.rope_sin"][step].astype(np.float64)
    cos_p, sin_p = z["lm.rope_cos"].astype(np.float64), z["lm.rope_sin"].astype(np.float64)
    scaling = float(z[f"ex.L{L}.scaling"])
    out: dict = {"layer": L, "step": step, "T": T, "prefix_keys": n_pre}

    # ---------------- fp32 reference
    h = rms_fp(x, ex["g_in"])
    q = (h @ ex["q"].w.T).reshape(T, HEADS, HD)
    k = h @ ex["k"].w.T
    v = h @ ex["v"].w.T
    hl = rms_fp(lm_in, lm["g_in"])
    kp = rope_fp(hl @ lm["k"].w.T, cos_p, sin_p)
    vp = hl @ lm["v"].w.T
    qr = rope_fp(q, cos_s[:, None, :], sin_s[:, None, :])
    kr = rope_fp(k, cos_s, sin_s)
    K = np.concatenate([kp, kr])                                              # (n_pre + T, 256)
    V = np.concatenate([vp, v])
    m = np.concatenate([mask867[:, pre_idx], mask867[:, 816:816 + T]], 1)     # (T, n_pre + T)
    P = softmax_masked(np.einsum("thd,kd->thk", qr, K) * scaling, m[:, None, :])
    ctx = np.einsum("thk,kd->thd", P, V).reshape(T, HEADS * HD)
    o = ctx @ ex["o"].w.T
    x2 = x + o
    h2 = rms_fp(x2, ex["g_post"])
    gate, up = h2 @ ex["gate"].w.T, h2 @ ex["up"].w.T
    mlp = (gelu_tanh(gate) * up) @ ex["down"].w.T
    y = x2 + mlp

    # validate the reference against the torch captures
    cap = {"q_pre": (h @ ex["q"].w.T, z[f"ex.L{L}.q_pre"][step]),
           "k_prefix_roped": (kp, z[f"ex.L{L}.k"][step, 0, pre_idx]),
           "k_suffix_roped": (kr, z[f"ex.L{L}.k"][step, 0, 816:816 + T]),
           "o_in": (ctx, z[f"ex.L{L}.o_in"][step]),
           "o_out": (o, z[f"ex.L{L}.o_out"][step]),
           "post_attn_in": (x2, z[f"ex.L{L}.post_attention_layernorm.in"][step]),
           "gate_out": (gate, z[f"ex.L{L}.gate_out"][step]),
           "up_out": (up, z[f"ex.L{L}.up_out"][step])}
    out["reference_vs_capture"] = {n: rel(a, b.astype(np.float64)) for n, (a, b) in cap.items()}

    # ---------------- chip
    C = Chip()
    xc = R.bf16_codes(x)
    # prefix K / V (LM layer L, computed once per chunk in the prefix pass)
    hlc = C.rms(R.bf16_codes(lm_in), f32c((1.0 + lm["g_in"]) / lm["s_qkv"]))
    hlq, s_hl = C.quant(hlc)
    kpc = C.dequant(C.gemm(hlq, lm["k"].codes), s_hl, f32c(lm["k"].scale))
    kpc = C.rope(kpc, f32c(cos_p), f32c(np.concatenate([-sin_p[:, :HD // 2], sin_p[:, HD // 2:]], 1)))
    # V prefix column-major: role-swapped GEMM, DEQUANT s_row = per channel (with 1/s_o of the EXPERT layer), s_col = token
    acc_vp = C.gemm(hlq, lm["v"].codes).T                                     # (256, n_pre)
    acc_vp = np.pad(acc_vp, ((0, 0), (0, N_PRE_SLOTS - n_pre)))
    s_hl_pad = np.pad(s_hl, (0, N_PRE_SLOTS - n_pre))
    vpc = C.dequant(acc_vp, f32c(lm["v"].scale / ex["s_o_hd"]), s_hl_pad)
    vpq, s_vp = C.quant(vpc)                                                  # per channel

    # expert layer
    hc = C.rms(xc, f32c((1.0 + ex["g_in"]) / ex["s_qkv"]))
    hq, s_h = C.quant(hc)
    qc = C.dequant(C.gemm(hq, ex["q"].codes), s_h, f32c(ex["q"].scale * (scaling / LN2)))   # log2-domain logits fold
    kc = C.dequant(C.gemm(hq, ex["k"].codes), s_h, f32c(ex["k"].scale))
    acc_vs = np.pad(C.gemm(hq, ex["v"].codes).T, ((0, 0), (0, N_SUF_SLOTS - T)))
    vsc = C.dequant(acc_vs, f32c(ex["v"].scale / ex["s_o_hd"]), np.pad(s_h, (0, N_SUF_SLOTS - T)))
    vsq, s_vs = C.quant(vsc)
    sin_signed_s = f32c(np.concatenate([-sin_s[:, :HD // 2], sin_s[:, HD // 2:]], 1))
    qrc = C.rope(qc.reshape(T * HEADS, HD), np.repeat(f32c(cos_s), HEADS, 0), np.repeat(sin_signed_s, HEADS, 0))
    krc = C.rope(kc, f32c(cos_s), sin_signed_s)
    Kc = np.zeros((N_PRE_SLOTS + N_SUF_SLOTS, HD), np.uint16)
    Kc[:n_pre], Kc[N_PRE_SLOTS:N_PRE_SLOTS + T] = kpc, krc
    Kq, s_K = C.quant(Kc)                                                     # per key; rows = QK^T weight columns
    Qq, s_Q = C.quant(qrc)                                                    # per (token, head)
    logit = C.dequant(C.gemm(Qq, Kq), s_Q, s_K)                               # (T*8, 592)
    mchip = np.zeros((T, N_PRE_SLOTS + N_SUF_SLOTS), bool)
    mchip[:, :n_pre] = mask867[:, pre_idx]
    mchip[:, N_PRE_SLOTS:N_PRE_SLOTS + T] = mask867[:, 816:816 + T]
    mrow = np.repeat(mchip, HEADS, 0)
    Pq, s_P = C.smax_u8(logit, mrow)
    ctx_p = C.dequant(C.gemm(Pq[:, :N_PRE_SLOTS], vpq), s_P, s_vp)
    ctx_s = C.dequant(C.gemm(Pq[:, N_PRE_SLOTS:], vsq), s_P, s_vs)
    ctxc = C.add(ctx_p, ctx_s).reshape(T, HEADS * HD)
    oq, s_oq = C.quant(ctxc)
    oc = C.dequant(C.gemm(oq, ex["o"].codes), s_oq, f32c(ex["o"].scale))
    x2c = C.add(xc, oc)
    h2c = C.rms(x2c, f32c((1.0 + ex["g_post"]) / ex["s_gu"]))
    h2q, s_h2 = C.quant(h2c)
    gc = C.dequant(C.gemm(h2q, ex["gate"].codes), s_h2, f32c(ex["gate"].scale))
    uc = C.dequant(C.gemm(h2q, ex["up"].codes), s_h2, f32c(ex["up"].scale / ex["s_d"]))
    mq, s_m = C.quant(C.geglu(gc, uc))
    mc = C.dequant(C.gemm(mq, ex["down"].codes), s_m, f32c(ex["down"].scale))
    yc = C.add(x2c, mc)

    # ---------------- compare
    s_o64 = ex["s_o"].astype(np.float64)
    Pchip = Pq.astype(np.float64) * np.asarray(s_P, np.uint32).view(np.float32).astype(np.float64)[:, None]
    Pref = np.zeros_like(Pchip)
    Pref[:, :n_pre] = P.reshape(T * HEADS, -1)[:, :n_pre]
    Pref[:, N_PRE_SLOTS:N_PRE_SLOTS + T] = P.reshape(T * HEADS, -1)[:, n_pre:]
    Kref = np.zeros((N_PRE_SLOTS + N_SUF_SLOTS, HD))
    Kref[:n_pre], Kref[N_PRE_SLOTS:N_PRE_SLOTS + T] = kp, kr
    stages = {
        "h (RMS)": (bf16v(hc), rms_fp(x, ex["g_in"]) / ex["s_qkv"]),
        "q_roped (x scaling/ln2)": (bf16v(qrc), qr.reshape(T * HEADS, HD) * scaling / LN2),
        "K roped": (bf16v(Kc), Kref),
        "V prefix / s_o": (bf16v(vpc)[:, :n_pre].T, vp / ex["s_o_hd"]),
        "V suffix / s_o": (bf16v(vsc)[:, :T].T, v / ex["s_o_hd"]),
        "logits (log2)": (bf16v(logit)[mrow], (np.einsum("thd,kd->thk", qr, K) * scaling / LN2).reshape(T * HEADS, -1)[
            np.concatenate([mchip[:, :n_pre], mchip[:, N_PRE_SLOTS:N_PRE_SLOTS + T]], 1).repeat(HEADS, 0)]),
        "P": (Pchip, Pref),
        "attn ctx / s_o": (bf16v(ctxc), ctx / s_o64),
        "o_out": (bf16v(oc), o),
        "x + o": (bf16v(x2c), x2),
        "gate": (bf16v(gc), gate),
        "up / s_down": (bf16v(uc), up / ex["s_d"]),
        "mlp out": (bf16v(mc), mlp),
        "layer out": (bf16v(yc), y),
    }
    out["chip_vs_fp32_rel"] = {n: rel(a, b) for n, (a, b) in stages.items()}
    out["layer_delta_rel"] = rel(bf16v(yc) - x, y - x)
    out["attn_delta_rel"] = rel(bf16v(oc), o)
    out["mlp_delta_rel"] = rel(bf16v(mc), mlp)
    out["max_abs_acc_log2"] = math.log2(max(C.max_acc, 1))
    out["vector_elems"] = C.ops
    out["gemm_products"] = C.gemm_products
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--layers", type=int, nargs="+", default=[0, 9, 17])
    ap.add_argument("--steps", type=int, nargs="+", default=[0, 1], help="capture index: 0 = Euler step 0, 1 = step 9")
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    z = np.load(CAPTURE)
    res = []
    for L in a.layers:
        t0 = time.time()
        lm = load_layer(L, "paligemma_layer", a.alpha, "lm", gemms=("k", "v"))
        ex = load_layer(L, "expert_layer", a.alpha, "exp")
        print(f"layer {L}: weights quantised in {time.time() - t0:.0f} s", flush=True)
        for st in a.steps:
            t0 = time.time()
            r = run(L, st, a.alpha, z, lm, ex)
            res.append(r)
            print(f"\nexpert L{L} step idx {st} (T={r['T']}, prefix keys {r['prefix_keys']}) in {time.time() - t0:.0f} s")
            print("  reference vs capture: " + ", ".join(f"{k} {v:.1e}" for k, v in r["reference_vs_capture"].items()))
            for k, v in r["chip_vs_fp32_rel"].items():
                print(f"  {k:26s} {100 * v:8.3f} %")
            print(f"  layer delta {100 * r['layer_delta_rel']:.3f} %  attn {100 * r['attn_delta_rel']:.3f} %  "
                  f"mlp {100 * r['mlp_delta_rel']:.3f} %  max|acc| 2^{r['max_abs_acc_log2']:.1f}")
    if a.json:
        Path(a.json).parent.mkdir(parents=True, exist_ok=True)
        Path(a.json).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
