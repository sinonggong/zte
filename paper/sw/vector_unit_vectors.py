#!/usr/bin/env python3
"""Test vectors and golden output beats for paper/rtl/vector_unit/vu_lane.sv (tb_vu_lane.sv).

Golden = paper/sw/vector_unit_ref.py; every op of the lane gets
  * random cases: row lengths 1..600, exponent spreads, zeros / -0 / subnormal inputs, saturation,
    ties, all-zero and constant rows, inconsistent row statistics where the op tolerates them
  * captured cases: rows of real pi0 activations (paper/sw/vector_unit_capture.py) at the op's real width,
    and for DEQUANT the exact integer sums of real INT8 weights x per-token INT8 activations.

Files in --out:
  cases.txt    one line per case: op b_fp32 bias_en out_fp32 k(fp32 hex) n_in n_out name
  in_NNN.hex   one line per input element:  last mask x(48 hex) rs(16) b c d e (32 hex each)
  out_NNN.hex  one line per output beat:    kind data(64 hex)
  summary.json case names, element and beat counts
Beat stream per row (vu_lane.sv header): element beats (kind 0), LN_STAT r (kind 2), summary (kind 1).
"""
from __future__ import annotations

import argparse
import json
import math
import struct
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import vector_unit_ref as R  # noqa: E402

OP = dict(ADD=0, RMS_STAT=1, RMS_APPLY=2, LN_STAT=3, LN_APPLY=4, ROPE_A=5, ROPE_B=6, GELU=7, GEGLU=8, SILU=9,
          SMAX_SUM=10, SMAX_OUT=11, SMAX_Q8=12, QUANT=13, DEQUANT=14, EULER=15)
M48 = (1 << 48) - 1
CKPT = Path.home() / "pi0_ckpt/ur7e-demo-2-pi0-fsdp/010000/pretrained_model/model.safetensors"


def f32(v) -> int:
    return int(np.array([v], np.float32).view(np.uint32)[0])


def f32val(code: int) -> float:
    return float(np.array([code], np.uint32).view(np.float32)[0])


def bf16_gt(a: int, b: int) -> bool:
    if a == b:
        return False
    if (a >> 15) != (b >> 15):
        return bool(b >> 15)
    if not a >> 15:
        return (a & 0x7FFF) > (b & 0x7FFF)
    return (a & 0x7FFF) < (b & 0x7FFF)


def tracker(codes, mask):
    """the lane's output-row amax / masked signed max (vu_lane.sv tr_amax / tr_max)."""
    c = [int(v) & 0xFFFF for v in codes]
    amax = max(v & 0x7FFF for v in c)
    mx = None
    for v, m in zip(c, mask):
        if mx is None:
            mx = v if m else 0xFF80
        elif m and bf16_gt(v, mx):
            mx = v
    return amax, mx


class Case:
    def __init__(self, name, op, k=0, b_fp32=0, bias_en=0, out_fp32=0):
        self.name, self.op, self.k = name, OP[op], k
        self.b_fp32, self.bias_en, self.out_fp32 = b_fp32, bias_en, out_fp32
        self.inp, self.out = [], []

    def row(self, x, rs=0, mask=None, b=0, c=0, d=0, e=0):
        n = len(x)
        mask = np.ones(n, bool) if mask is None else np.asarray(mask, bool)
        cols = [np.broadcast_to(np.asarray(v).astype(np.int64), (n,)) for v in (b, c, d, e)]
        xs = np.asarray(x).astype(np.int64)
        for i in range(n):
            self.inp.append((int(i == n - 1), int(mask[i]), int(xs[i]) & M48, int(rs) & 0xFFFF,
                             *[int(col[i]) & 0xFFFFFFFF for col in cols]))

    def elems(self, codes, width):
        m = {16: 0xFFFF, 32: 0xFFFFFFFF}[width]
        self.out += [(0, int(v) & m) for v in np.asarray(codes).astype(np.int64)]

    def summary(self, amax=0, mx=0, rs0=0):
        self.out.append((1, (amax << 48) | (mx << 32) | (int(rs0) & 0xFFFFFFFF)))


# ======================================================================================
# golden per op (one row at a time)
# ======================================================================================
def g_add(cs, a, b):
    codes, _ = R.op_add(a[None], b[None], "fp32" if cs.b_fp32 else "bf16")
    cs.row(a, b=b)
    cs.elems(codes[0], 16)
    cs.summary(*tracker(codes[0], np.ones(len(a), bool)))


def g_rms_stat(cs, x):
    amax = int(R.row_amax_code(x[None])[0])
    r = R.rms_stat(x[None], np.array([amax]), f32val(cs.k))[0]
    cs.row(x, rs=amax)
    cs.summary(rs0=int(r))
    return int(r)


def g_rms_apply(cs, x, r, gain):
    codes, _ = R.rms_apply(x[None], np.array([r], np.uint32), np.asarray(gain, np.uint32)[None])
    cs.row(x, c=r, d=gain)
    cs.elems(codes[0], 16)
    cs.summary(*tracker(codes[0], np.ones(len(x), bool)))
    return codes[0]


def g_ln_stat(cs, x, rs=None):
    amax = int(R.row_amax_code(x[None])[0]) if rs is None else rs
    mu, r = R.ln_stat(x[None], np.array([amax]), f32val(cs.k))
    cs.row(x, rs=amax)
    cs.out.append((2, int(r[0])))
    cs.summary(rs0=int(mu[0]))
    return int(mu[0]), int(r[0])


def g_ln_apply(cs, x, mu, r, gamma, beta):
    codes, _ = R.ln_apply(x[None], np.array([mu], np.uint32), np.array([r], np.uint32),
                          np.asarray(gamma, np.uint32)[None], np.asarray(beta, np.uint32)[None])
    cs.row(x, b=mu, c=r, d=gamma, e=beta)
    cs.elems(codes[0], 16)
    cs.summary(*tracker(codes[0], np.ones(len(x), bool)))
    return codes[0]


def g_rope_a(cs, x, cos):
    codes, _ = R.rope_a(x[None], np.asarray(cos, np.uint32)[None])
    cs.row(x, c=cos)
    cs.elems(codes[0], 32)
    cs.summary()
    return codes[0]


def g_rope_b(cs, p, sin, pa):
    codes, _ = R.rope_b(p[None], np.asarray(sin, np.uint32)[None], np.asarray(pa, np.uint32)[None])
    cs.row(p, c=sin, e=pa)
    cs.elems(codes[0], 16)
    cs.summary(*tracker(codes[0], np.ones(len(p), bool)))
    return codes[0]


def g_unary(cs, fn, x):
    codes, _ = fn(x[None])
    cs.row(x)
    cs.elems(codes[0], 16)
    cs.summary(*tracker(codes[0], np.ones(len(x), bool)))
    return codes[0]


def g_geglu(cs, g, u):
    codes, _ = R.op_geglu(g[None], u[None])
    cs.row(g, d=u)
    cs.elems(codes[0], 16)
    cs.summary(*tracker(codes[0], np.ones(len(g), bool)))
    return codes[0]


def row_max(x, mask):
    return tracker(x, mask)[1]


def g_smax_sum(cs, x, mask):
    mx = row_max(x, mask)
    inv_s, _ = R.smax_sum(x[None], np.array([mx]), mask[None])
    cs.row(x, rs=mx, mask=mask)
    cs.summary(rs0=int(inv_s[0]))
    return int(inv_s[0])


def g_smax_out(cs, x, mask, inv_s):
    mx = row_max(x, mask)
    codes, _ = R.smax_out(x[None], np.array([mx]), np.array([inv_s], np.uint32), mask[None])
    cs.row(x, rs=mx, mask=mask, c=inv_s)
    cs.elems(codes[0], 16)
    cs.summary(*tracker(codes[0], mask))


def g_smax_q8(cs, x, mask):
    mx = row_max(x, mask)
    s_row, q = R.smax_sum(x[None], np.array([mx]), mask[None], int8=True)
    cs.row(x, rs=mx, mask=mask)
    cs.elems(q[0].astype(np.int64), 32)
    cs.summary(rs0=int(s_row[0]))


def g_smax_u8(cs, x, mask):
    """SMAX_Q8 with 255 levels (Case(..., b_fp32=1)): uint8 probability codes for the PV chain."""
    mx = row_max(x, mask)
    s_row, q = R.smax_sum(x[None], np.array([mx]), mask[None], int8=True, levels=255)
    cs.row(x, rs=mx, mask=mask)
    cs.elems(q[0].astype(np.int64), 32)
    cs.summary(rs0=int(s_row[0]))


def g_quant(cs, x, rs=None):
    rs = int(R.row_amax_code(x[None])[0]) if rs is None else rs
    q, s_row, _ = R.op_quant(x[None], np.array([rs]))
    cs.row(x, rs=rs)
    cs.elems(q[0].astype(np.int64), 32)
    cs.summary(rs0=int(s_row[0]))


def g_dequant(cs, acc, s_row, s_col, bias=None, mask=None):
    n = len(acc)
    mask = np.ones(n, bool) if mask is None else mask
    codes, _ = R.op_dequant(np.asarray(acc, np.int64)[None], np.array([s_row], np.uint32),
                            np.asarray(s_col, np.uint32)[None],
                            np.asarray(bias, np.uint32)[None] if cs.bias_en else None,
                            "fp32" if cs.out_fp32 else "bf16")
    cs.row(np.asarray(acc, np.int64), c=s_row, d=s_col, e=0 if bias is None else bias, mask=mask)
    if cs.out_fp32:
        cs.elems(codes[0], 32)
        cs.summary()
    else:
        cs.elems(codes[0], 16)
        cs.summary(*tracker(codes[0], mask))


def g_euler(cs, xt, v):
    codes, _ = R.op_euler(np.asarray(xt, np.uint32)[None], np.asarray(v, np.uint32)[None], f32val(cs.k))
    cs.row(np.asarray(v, np.int64), e=xt)
    cs.elems(codes[0], 32)
    cs.summary()


# ======================================================================================
# random values
# ======================================================================================
def rbf16(rng, n, lo, hi, pz=0.03, psub=0.01):
    lo, hi = min(lo, hi), max(lo, hi)
    e8 = np.clip(rng.integers(127 + lo, 127 + hi + 1, n), 1, 254)
    code = (rng.integers(0, 2, n) << 15) | (e8 << 7) | rng.integers(0, 128, n)
    z = rng.random(n) < pz
    code = np.where(z, rng.choice(np.array([0, 0x8000]), n), code)
    sub = rng.random(n) < psub
    code = np.where(sub, (rng.integers(0, 2, n) << 15) | rng.integers(1, 128, n), code)
    return code.astype(np.int64)


def rfp32(rng, n, lo, hi, pos=False, pz=0.0):
    lo, hi = min(lo, hi), max(lo, hi)
    e8 = np.clip(rng.integers(127 + lo, 127 + hi + 1, n), 1, 254)
    code = (e8 << 23) | rng.integers(0, 1 << 23, n)
    if not pos:
        code = code | (rng.integers(0, 2, n) << 31)
    code = np.where(rng.random(n) < pz, 0, code)
    return code.astype(np.int64)


def rlen(rng):
    u = rng.random()
    if u < 0.35:
        return int(rng.integers(1, 9))
    if u < 0.8:
        return int(rng.integers(9, 129))
    return int(rng.integers(129, 601))


def random_cases(rng, rows):
    cases = []

    def spread():
        c = int(rng.integers(-24, 25))
        w = int(rng.integers(0, 10))
        return c - w, c + w

    cs = Case("rand ADD bf16", "ADD")
    for i in range(rows):
        n = rlen(rng)
        lo, hi = spread()
        a = rbf16(rng, n, lo, hi)
        b = rbf16(rng, n, lo - 2, hi + 2) if i % 3 else np.where(rng.random(n) < 0.5, a ^ 0x8000, a)  # ties / cancellation
        if i == 0:
            a = np.full(n, 0x7F7F)
            b = np.full(n, 0x7F7F)                                                      # overflow -> saturate
        g_add(cs, a, b)
    cases.append(cs)
    cs = Case("rand ADD fp32 table", "ADD", b_fp32=1)
    for i in range(rows // 2):
        n = rlen(rng)
        lo, hi = spread()
        g_add(cs, rbf16(rng, n, lo, hi), rfp32(rng, n, lo - 3, hi + 3, pz=0.05))
    cases.append(cs)

    cs = Case("rand RMS_STAT", "RMS_STAT", k=f32(1.0 / 64))
    for i in range(rows):
        n = rlen(rng)
        lo, hi = spread()
        x = rbf16(rng, n, lo, hi)
        if i == 1:
            x = np.zeros(n, np.int64)
        if i == 2:
            x = np.full(n, x[0] | 0x80)
        g_rms_stat(cs, x)
    cases.append(cs)
    cs = Case("rand RMS_APPLY", "RMS_APPLY")
    for i in range(rows):
        n = rlen(rng)
        lo, hi = spread()
        g_rms_apply(cs, rbf16(rng, n, lo, hi), int(rfp32(rng, 1, -lo - 4, -hi + 4, pos=True)[0]), rfp32(rng, n, -1, 1))
    cases.append(cs)

    cs = Case("rand LN_STAT", "LN_STAT", k=f32(1.0 / 128))
    for i in range(rows):
        n = rlen(rng)
        lo, hi = spread()
        x = rbf16(rng, n, lo, hi)
        if i == 1:
            x = np.zeros(n, np.int64)
        if i in (2, 3):
            x = np.full(n, (x[0] | 0x80) & 0x7FFF)                                     # var ~ 0: the eps / bad branch
        g_ln_stat(cs, x)
    cases.append(cs)
    cs = Case("rand LN_APPLY", "LN_APPLY")
    for i in range(rows):
        n = rlen(rng)
        lo, hi = spread()
        g_ln_apply(cs, rbf16(rng, n, lo, hi), int(rfp32(rng, 1, lo - 2, hi, pz=0.1)[0]),
                   int(rfp32(rng, 1, -hi - 2, -lo + 2, pos=True)[0]), rfp32(rng, n, -2, 2), rfp32(rng, n, -4, 1, pz=0.1))
    cases.append(cs)

    cs = Case("rand ROPE_A", "ROPE_A")
    pas = []
    for i in range(rows):
        n = rlen(rng)
        lo, hi = spread()
        x = rbf16(rng, n, lo, hi)
        pas.append((n, lo, hi, g_rope_a(cs, x, rfp32(rng, n, -8, -1, pz=0.02))))
    cases.append(cs)
    cs = Case("rand ROPE_B", "ROPE_B")
    for n, lo, hi, pa in pas:
        g_rope_b(cs, rbf16(rng, n, lo, hi), rfp32(rng, n, -8, -1, pz=0.02), pa)
    cases.append(cs)

    for op, fn, lo0, hi0 in (("GELU", R.op_gelu, -12, 6), ("SILU", R.op_silu, -12, 7)):
        cs = Case(f"rand {op}", op)
        for i in range(rows):
            n = rlen(rng)
            lo = int(rng.integers(lo0, 3))
            g_unary(cs, fn, rbf16(rng, n, lo, max(lo, hi0 if i % 4 == 0 else 3), pz=0.05))
        cases.append(cs)
    cs = Case("rand GEGLU", "GEGLU")
    for i in range(rows):
        n = rlen(rng)
        lo, hi = spread()
        g = rbf16(rng, n, -10, 5, pz=0.05)
        u = rbf16(rng, n, lo, hi) if i else np.full(n, 0x7F7F)
        g_geglu(cs, g, u)
    cases.append(cs)

    cs_sum, cs_out, cs_q8 = Case("rand SMAX_SUM", "SMAX_SUM"), Case("rand SMAX_OUT", "SMAX_OUT"), Case("rand SMAX_Q8", "SMAX_Q8")
    for i in range(rows):
        n = rlen(rng)
        hi = int(rng.integers(-4, 9)) if i % 5 else int(rng.integers(15, 22))
        x = rbf16(rng, n, hi - 12, hi, pz=0.05)
        mask = rng.random(n) < (1.0 if i % 3 == 0 else 0.7)
        mask[int(rng.integers(0, n))] = True
        inv_s = g_smax_sum(cs_sum, x, mask)
        g_smax_out(cs_out, x, mask, inv_s)
        g_smax_q8(cs_q8, x, mask)
    cases += [cs_sum, cs_out, cs_q8]

    cs = Case("rand QUANT", "QUANT")
    for i in range(rows):
        n = rlen(rng)
        lo, hi = spread()
        x = rbf16(rng, n, lo, hi)
        rs = None
        if i == 1:
            x = np.zeros(n, np.int64)
        elif i % 7 == 3:
            rs = int(max(1, (int(R.row_amax_code(x[None])[0]) - int(rng.integers(1, 300)))))   # clamp at +-127
        g_quant(cs, x, rs)
    cases.append(cs)

    for bias_en in (0, 1):
        for out_fp32 in (0, 1):
            cs = Case(f"rand DEQUANT bias{bias_en} fp32{out_fp32}", "DEQUANT", bias_en=bias_en, out_fp32=out_fp32)
            for i in range(rows // 2):
                n = rlen(rng)
                bits = rng.integers(0, 48, n)
                acc = np.array([int(rng.integers(-(1 << int(b)) + 1, 1 << int(b))) if b else int(rng.integers(-1, 2))
                                for b in bits], np.int64)
                if i == 0:
                    acc[:2] = [(1 << 47) - 1, -((1 << 47) - 1)]
                mask = rng.random(n) < 0.8
                g_dequant(cs, acc, int(rfp32(rng, 1, -20, 0, pos=True)[0]), rfp32(rng, n, -20, 0, pos=True),
                          rfp32(rng, n, -6, 4, pz=0.1), mask)
            cases.append(cs)

    cs = Case("rand EULER", "EULER", k=f32(-0.1))
    for i in range(rows):
        n = rlen(rng)
        g_euler(cs, rfp32(rng, n, -6, 6, pz=0.05), rfp32(rng, n, -6, 6, pz=0.05))
    cases.append(cs)
    return cases


# ======================================================================================
# captured activations
# ======================================================================================
def st_tensor(path: Path, key: str) -> np.ndarray:
    with open(path, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        hdr = json.loads(fh.read(n))
    v = hdr[key]
    s, _ = v["data_offsets"]
    return np.memmap(path, dtype=np.float32, mode="r", offset=8 + n + s, shape=tuple(v["shape"]))


def pick(x, nrows):
    x = np.asarray(x).reshape(-1, np.asarray(x).shape[-1])
    return x[np.linspace(0, x.shape[0] - 1, nrows).astype(int)]


def captured_cases(z, nrows, ckpt: Path | None):
    bfc, fc = R.bf16_codes, R.f32_codes
    cases = []

    # LayerNorm, SigLIP L13 (1152)
    x = bfc(pick(z["vis.L13.layer_norm1.in"], nrows))
    g, b = fc(z["vis.L13.layer_norm1.gamma"]), fc(z["vis.L13.layer_norm1.beta"])
    st, ap = Case("cap LN_STAT vis.L13.ln1", "LN_STAT", k=f32(1.0 / 1152)), Case("cap LN_APPLY vis.L13.ln1", "LN_APPLY")
    ln_out = []
    for row in x:
        mu, r = g_ln_stat(st, row)
        ln_out.append(g_ln_apply(ap, row, mu, r, g, b))
    cases += [st, ap]

    # RMSNorm, LM L9 input (2048) and expert L17 post-attention (1024)
    rms_out = {}
    for key, width in (("lm.L9.input_layernorm", 2048), ("ex.L17.post_attention_layernorm", 1024)):
        x = bfc(pick(z[f"{key}.in"], nrows))
        gain = fc(np.float32(1.0) + z[f"{key}.gain"].astype(np.float32))
        st, ap = Case(f"cap RMS_STAT {key}", "RMS_STAT", k=f32(1.0 / width)), Case(f"cap RMS_APPLY {key}", "RMS_APPLY")
        rms_out[key] = [g_rms_apply(ap, row, g_rms_stat(st, row), gain) for row in x]
        cases += [st, ap]

    # RoPE, LM L0 q (head rows of 256) with the real cos/sin
    q = z["lm.L0.q_pre"].reshape(525, 8, 256)
    cos, sin = z["lm.rope_cos"], z["lm.rope_sin"]
    ra, rb = Case("cap ROPE_A lm.L0.q", "ROPE_A"), Case("cap ROPE_B lm.L0.q", "ROPE_B")
    for t in np.linspace(1, 524, nrows).astype(int):
        xc = bfc(q[t, 3])
        pa = g_rope_a(ra, xc, fc(cos[t]))
        partner = np.concatenate([xc[128:], xc[:128]])
        ssin = np.concatenate([-sin[t, :128], sin[t, 128:]]).astype(np.float32)
        g_rope_b(rb, partner, fc(ssin), pa)
    cases += [ra, rb]

    # GELU SigLIP L13 fc1 (4304), GeGLU LM L9 (16384) and expert L0 (4096), SiLU (1024)
    cs = Case("cap GELU vis.L13.fc1", "GELU")
    for row in bfc(pick(z["vis.L13.fc1_out"], max(2, nrows // 2))):
        g_unary(cs, R.op_gelu, row)
    cases.append(cs)
    geglu_out = []
    for key, nr in (("lm.L9", 2), ("ex.L0", nrows)):
        cs = Case(f"cap GEGLU {key}", "GEGLU")
        for gr, ur in zip(bfc(pick(z[f"{key}.gate_out"], nr)), bfc(pick(z[f"{key}.up_out"], nr))):
            geglu_out.append(g_geglu(cs, gr, ur))
        cases.append(cs)
    cs = Case("cap SILU ex.action_time_mlp", "SILU")
    for row in bfc(pick(z["ex.silu_in"], nrows)):
        g_unary(cs, R.op_silu, row)
    cases.append(cs)

    # softmax on log2-domain logits: SigLIP L13 (256 keys), LM L9 (525), expert L9 (867, masked)
    for pre, li, heads in (("vis", 13, 16), ("lm", 9, 8), ("ex", 9, 8)):
        qq, kk = z[f"{pre}.L{li}.q"], z[f"{pre}.L{li}.k"]
        qq = qq.reshape(-1, *qq.shape[-3:])[0]
        kk = kk.reshape(-1, *kk.shape[-3:])[0]
        logits = np.matmul(qq, np.swapaxes(kk, -1, -2)) * np.float32(z[f"{pre}.L{li}.scaling"])
        mk = None
        if f"{pre}.L{li}.mask" in z.files:
            mk = z[f"{pre}.L{li}.mask"].reshape(-1, *z[f"{pre}.L{li}.mask"].shape[-2:])[0]
            mk = np.broadcast_to(mk, logits.shape)
        lg = pick(logits, nrows)
        mks = pick(mk, nrows) if mk is not None else np.ones(lg.shape, bool)
        lc = bfc(lg / np.float32(math.log(2.0)))
        c1, c2, c3 = (Case(f"cap SMAX_{s} {pre}.L{li}", f"SMAX_{s}") for s in ("SUM", "OUT", "Q8"))
        for row, m in zip(lc, mks):
            inv_s = g_smax_sum(c1, row, m)
            g_smax_out(c2, row, m, inv_s)
            g_smax_q8(c3, row, m)
        cases += [c1, c2, c3]

    # QUANT on real rows: LN out, RMS out, GeGLU out (down_proj input), attention output (o_proj input)
    cs = Case("cap QUANT", "QUANT")
    for row in ln_out[:2] + rms_out["lm.L9.input_layernorm"][:2] + geglu_out[:2]:
        g_quant(cs, np.asarray(row, np.int64))
    for row in bfc(pick(z["lm.L9.o_in"], 2)):
        g_quant(cs, row)
    cases.append(cs)

    # DEQUANT: exact sums of real INT8 weights x per-token INT8 activations
    if ckpt is not None and ckpt.exists():
        pj = "model.paligemma_with_expert."
        for name, wk, bk, xrows, bias_en, out_fp32 in (
                ("ex.L9.gate_proj", pj + "joint_layers.9.expert_layer.mlp.gate_proj.weight", None,
                 rms_out["ex.L17.post_attention_layernorm"][:2], 0, 0),
                ("action_out_proj", "model.action_out_proj.weight", "model.action_out_proj.bias",
                 rms_out["ex.L17.post_attention_layernorm"], 1, 1)):
            W = np.asarray(st_tensor(ckpt, wk), np.float32)
            bias = np.asarray(st_tensor(ckpt, bk), np.float32) if bk else None
            wa = np.abs(W).max(1, keepdims=True)
            sc = (np.where(wa > 0, wa, 1) / np.float32(127)).astype(np.float32)
            qw = np.clip(np.rint(W / sc), -127, 127).astype(np.int64)
            cs = Case(f"cap DEQUANT {name}", "DEQUANT", bias_en=bias_en, out_fp32=out_fp32)
            for xr in xrows:
                xv = R.bf16_val(np.asarray(xr).astype(np.uint16)).astype(np.float32)
                am = np.abs(xv).max()
                sr = np.float32(am if am > 0 else 1) / np.float32(127)
                qx = np.clip(np.rint(xv / sr), -127, 127).astype(np.int64)
                acc = qw @ qx
                g_dequant(cs, acc, f32(sr), fc(sc[:, 0]), fc(bias) if bias is not None else None)
            cases.append(cs)

    # Euler on the captured noise and first velocity
    cs = Case("cap EULER", "EULER", k=f32(-0.1))
    for xr, vr in zip(fc(z["ex.noise"]), fc(z["ex.v_t"].reshape(10, 50, 32)[0])):
        g_euler(cs, xr, vr)
    cases.append(cs)

    # residual add (SigLIP L13) and the fp32 position-table add
    cs = Case("cap ADD vis.L13 residual", "ADD")
    for a, b in zip(bfc(pick(z["vis.L13.layer_in"], nrows)), bfc(pick(z["vis.L13.attn_out"], nrows))):
        g_add(cs, a, b)
    cases.append(cs)
    cs = Case("cap ADD vis pos_emb", "ADD", b_fp32=1)
    pa = z["vis.patch_out"].reshape(-1, 256, 1152)[0]
    for t in np.linspace(0, 255, max(2, nrows // 2)).astype(int):
        g_add(cs, bfc(pa[t]), fc(z["vis.pos_table"][t]))
    cases.append(cs)
    return cases


def throughput_cases(rng, width, nrows):
    """per op: one case of 1 row and one of nrows rows of `width` random elements; with no input gaps and no
    back-pressure the lane's per-row control overhead is (cycles(nrows) - cycles(1) - (nrows-1)*width)/(nrows-1)."""
    cases = []

    def mk(name, op, fill, **kw):
        for nr in (1, nrows):
            cs = Case(f"tp {name} r{nr}", op, **kw)
            for _ in range(nr):
                fill(cs, width)
            cases.append(cs)

    def b(n):
        return rbf16(rng, n, -4, 4, pz=0.0, psub=0.0)

    def f(n, lo=-4, hi=4, pos=False):
        return rfp32(rng, n, lo, hi, pos=pos)

    ones = lambda n: np.ones(n, bool)  # noqa: E731
    mk("ADD", "ADD", lambda cs, n: g_add(cs, b(n), b(n)))
    mk("RMS_STAT", "RMS_STAT", lambda cs, n: g_rms_stat(cs, b(n)), k=f32(1.0 / width))
    mk("RMS_APPLY", "RMS_APPLY", lambda cs, n: g_rms_apply(cs, b(n), int(f(1, pos=True)[0]), f(n)))
    mk("LN_STAT", "LN_STAT", lambda cs, n: g_ln_stat(cs, b(n)), k=f32(1.0 / width))
    mk("LN_APPLY", "LN_APPLY", lambda cs, n: g_ln_apply(cs, b(n), int(f(1)[0]), int(f(1, pos=True)[0]), f(n), f(n)))
    mk("ROPE_A", "ROPE_A", lambda cs, n: g_rope_a(cs, b(n), f(n, -3, -1)))
    mk("ROPE_B", "ROPE_B", lambda cs, n: g_rope_b(cs, b(n), f(n, -3, -1), f(n)))
    mk("GELU", "GELU", lambda cs, n: g_unary(cs, R.op_gelu, b(n)))
    mk("GEGLU", "GEGLU", lambda cs, n: g_geglu(cs, b(n), b(n)))
    mk("SILU", "SILU", lambda cs, n: g_unary(cs, R.op_silu, b(n)))
    mk("SMAX_SUM", "SMAX_SUM", lambda cs, n: g_smax_sum(cs, b(n), ones(n)))
    mk("SMAX_OUT", "SMAX_OUT", lambda cs, n: g_smax_out(cs, b(n), ones(n), int(f(1, -8, -1, pos=True)[0])))
    mk("SMAX_Q8", "SMAX_Q8", lambda cs, n: g_smax_q8(cs, b(n), ones(n)))
    mk("QUANT", "QUANT", lambda cs, n: g_quant(cs, b(n)))
    mk("DEQUANT", "DEQUANT", lambda cs, n: g_dequant(cs, rng.integers(-(1 << 30), 1 << 30, n),
                                                     int(f(1, -20, -5, pos=True)[0]), f(n, -20, -5, pos=True), f(n)),
       bias_en=1)
    mk("EULER", "EULER", lambda cs, n: g_euler(cs, f(n), f(n)), k=f32(-0.1))
    return cases


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--rows", type=int, default=40, help="random rows per op")
    ap.add_argument("--npz", default=None, help="captured activations; omit for random vectors only")
    ap.add_argument("--cap-rows", type=int, default=4)
    ap.add_argument("--ckpt", default=str(CKPT))
    ap.add_argument("--throughput", nargs=2, type=int, metavar=("WIDTH", "ROWS"), default=None,
                    help="only the throughput cases (1 row and ROWS rows of WIDTH per op)")
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    cases = throughput_cases(rng, *a.throughput) if a.throughput else random_cases(rng, a.rows)
    if a.npz and not a.throughput:
        z = np.load(a.npz)
        cases += captured_cases(z, a.cap_rows, Path(a.ckpt))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    lines, summ = [], []
    for i, cs in enumerate(cases):
        with open(out / f"in_{i:03d}.hex", "w") as fh:
            fh.write("".join(f"{l:x} {m:x} {x:012x} {rs:04x} {b:08x} {c:08x} {d:08x} {e:08x}\n"
                             for l, m, x, rs, b, c, d, e in cs.inp))
        with open(out / f"out_{i:03d}.hex", "w") as fh:
            fh.write("".join(f"{k:x} {v:016x}\n" for k, v in cs.out))
        lines.append(f"{cs.op:x} {cs.b_fp32:x} {cs.bias_en:x} {cs.out_fp32:x} {cs.k & 0xFFFFFFFF:08x} "
                     f"{len(cs.inp)} {len(cs.out)} {cs.name.replace(' ', '_')}\n")
        summ.append(dict(case=i, name=cs.name, op=cs.op, elements=len(cs.inp), beats=len(cs.out)))
    (out / "cases.txt").write_text("".join(lines))
    (out / "summary.json").write_text(json.dumps(dict(seed=a.seed, cases=summ,
                                                      elements=sum(s["elements"] for s in summ),
                                                      beats=sum(s["beats"] for s in summ)), indent=1))
    print(f"{len(cases)} cases, {sum(s['elements'] for s in summ)} elements, {sum(s['beats'] for s in summ)} beats -> {out}")


if __name__ == "__main__":
    main()
