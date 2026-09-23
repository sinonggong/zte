#!/usr/bin/env python3
"""Bit-exact numpy reference of one pi0 vector-unit lane (paper/rtl/vector_unit/*.sv).

Architecture: docs/PI0_FULL_CHIP_ARCHITECTURE_20260910.md section 3.1 / 4.1; op table: paper/rtl/vector_unit/OPS.md.

Number formats
--------------
* stream I/O: bf16 codes (the top half of an IEEE fp32), fp32 codes for per-column constants and the
  Euler state, int8 codes out of QUANT, int48 GEMM sums into DEQUANT.
* inside: "xf" = sign s, signed exponent e, 24-bit mantissa M with M[23] = 1 (value = M * 2^(e-23)),
  i.e. fp32 precision.  Zero is M = 0 with the canonical exponent EZ.  Every unit TRUNCATES its
  result to 24 bits (no sticky); only the stream output rounds (round-to-nearest-even) to bf16 / int8.
  The truncation bound is 2^-23 relative per unit.  Subnormal inputs read as zero; results that
  overflow bf16 saturate to the largest finite value, results below the smallest normal read as zero.
* tables: 2048 x 36 bit BRAM72K words {V[20:0] in Q1.20, D[14:0] = V[i+1]-V[i] signed}, one table per
  function (GELU gate phi, sigmoid, 2^-f, rsqrt); linear interpolation with a 12-bit fraction:
  y = (V << 12) + D * frac  (Q1.32), then normalised to xf.
* row reductions: fixed-point accumulators relative to the row's amax exponent E (which the
  producing op tracks and hands over as a row scalar), so the accumulators never need a
  floating-point adder in a feedback loop:  sum a^2 in Q.38 of 2^(2E), sum a in Q.32 of 2^E,
  softmax sum of e_i in Q.40.

The units (every op is a fixed composition of these, in this order; RTL mirrors them 1:1):
  unpack_bf16 / unpack_fp32 / norm64 (int -> xf)   mul   add   tbl_gelu / tbl_sigm / tbl_exp / rsqrt
  acc2 / acc1 / acc_exp      to_bf16 / to_fp32 / to_int8

Modes of this file:
  --tables DIR        write the .mem table files the RTL loads (four functions + the QUANT row-constant table)
  --inventory         print the per-chunk op/element inventory (OPS.md table) and write paper/data/vector_unit/inventory.json
  --measure NPZ       per-op error of this arithmetic vs the float torch op on captured real activations
                      (paper/sw/vector_unit_capture.py) -> paper/data/vector_unit/errors.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent

MW = 24                  # internal mantissa bits
EZ = -2048               # exponent of canonical zero (12-bit signed exponent in RTL)
TBL_N = 2048
TBL_VF = 20              # V is Q1.20 (21 bits unsigned)
TBL_FR = 12              # interpolation fraction bits
F_ACC2 = 38              # sum a^2, fraction bits relative to 2^(2E)
F_ACC1 = 32              # sum a, fraction bits relative to 2^E
F_SMAX = 40              # softmax sum of e_i, fraction bits
F_EXP = 23               # softmax logit fixed point (log2 domain)
EXP_KMAX = 41            # 2^-k below 2^-41 is dropped from the sum; the element is 0 for k >= 64
EPS_NORM = 1e-6

P2 = np.array([1 << k for k in range(63)], dtype=np.int64)


def bitlen(v: np.ndarray) -> np.ndarray:
    """bit_length of non-negative int64 values (0 -> 0)."""
    return np.searchsorted(P2, np.asarray(v, np.int64), side="right").astype(np.int64)


# ======================================================================================
# xf numbers
# ======================================================================================
@dataclass
class XF:
    s: np.ndarray
    e: np.ndarray
    m: np.ndarray

    def __getitem__(self, idx):
        return XF(self.s[idx], self.e[idx], self.m[idx])

    @property
    def shape(self):
        return self.m.shape

    def value(self) -> np.ndarray:
        v = np.ldexp(self.m.astype(np.float64), (self.e - 23).astype(np.int64).clip(-1100, 1100))
        return np.where(self.s == 1, -v, v)


def xf(s, e, m) -> XF:
    s, e, m = np.broadcast_arrays(np.asarray(s, np.int64), np.asarray(e, np.int64), np.asarray(m, np.int64))
    z = m == 0
    return XF(np.where(z, 0, s), np.where(z, EZ, e), m.copy())


def const_xf(v: float, shape=()) -> XF:
    """Exact xf of an fp32 constant (what an fp32 CSR / column word decodes to)."""
    return unpack_fp32(np.full(shape, np.float32(v), np.float32).view(np.uint32))


def neg(a: XF) -> XF:
    return xf(np.where(a.m == 0, 0, 1 - a.s), a.e, a.m)


def unpack_bf16(code) -> XF:
    c = np.asarray(code).astype(np.int64) & 0xFFFF
    e8 = (c >> 7) & 0xFF
    m = ((c & 0x7F) | 0x80) << 16
    return xf((c >> 15) & 1, e8 - 127, np.where(e8 == 0, 0, m))


def unpack_fp32(code) -> XF:
    c = np.asarray(code).astype(np.int64) & 0xFFFFFFFF
    e8 = (c >> 23) & 0xFF
    m = (c & 0x7FFFFF) | 0x800000
    return xf((c >> 31) & 1, e8 - 127, np.where(e8 == 0, 0, m))


def norm64(v, frac_bits: int, e_off) -> XF:
    """signed integer v (|v| < 2^53) scaled by 2^(e_off - frac_bits) -> xf (truncate)."""
    v = np.asarray(v, np.int64)
    a = np.abs(v)
    nb = bitlen(a)
    sh = nb - MW
    m = np.where(sh >= 0, a >> np.maximum(sh, 0), a << np.maximum(-sh, 0))
    return xf((v < 0).astype(np.int64), nb - 1 + np.asarray(e_off, np.int64) - frac_bits, m)


def mul(a: XF, b: XF) -> XF:
    p = a.m * b.m                                   # < 2^48
    hi = p >> 47
    m = np.where(hi == 1, p >> 24, p >> 23)
    return xf(a.s ^ b.s, a.e + b.e + hi, m)


ADD_G = 3                 # guard bits of the adder


def add(a: XF, b: XF) -> XF:
    swap = (b.e > a.e) | ((b.e == a.e) & (b.m > a.m))
    ls, le, lm = np.where(swap, b.s, a.s), np.where(swap, b.e, a.e), np.where(swap, b.m, a.m)
    ss, se, sm = np.where(swap, a.s, b.s), np.where(swap, a.e, b.e), np.where(swap, a.m, b.m)
    d = np.minimum(le - se, MW + ADD_G)             # >= 27 shifts everything out
    A = lm << ADD_G
    B = (sm << ADD_G) >> d
    R = np.where(ls != ss, A - B, A + B)            # 0 .. 2^28
    nb = bitlen(R)
    sh = nb - MW
    m = np.where(sh >= 0, R >> np.maximum(sh, 0), R << np.maximum(-sh, 0))
    return xf(ls, le + nb - (MW + ADD_G), m)


def to_bf16(a: XF) -> np.ndarray:
    m8 = a.m >> 16
    g = (a.m >> 15) & 1
    st = (a.m & 0x7FFF) != 0
    m8 = m8 + (g & (st | (m8 & 1)))
    ov = m8 >> 8
    m8 = np.where(ov == 1, 0x80, m8)
    e8 = a.e + ov + 127
    code = (a.s << 15) | (np.clip(e8, 0, 255) << 7) | (m8 & 0x7F)
    code = np.where(e8 >= 255, (a.s << 15) | 0x7F7F, code)
    code = np.where((e8 <= 0) | (a.m == 0), 0, code)
    return code.astype(np.uint16)


def to_fp32(a: XF) -> np.ndarray:
    e8 = a.e + 127
    code = (a.s << 31) | (np.clip(e8, 0, 255) << 23) | (a.m & 0x7FFFFF)
    code = np.where(e8 >= 255, (a.s << 31) | 0x7F7FFFFF, code)
    code = np.where((e8 <= 0) | (a.m == 0), 0, code)
    return code.astype(np.uint32)


def _round_int(a: XF, top: int) -> np.ndarray:
    """round-to-nearest-even of |value| to an integer, clamped to `top` (127 int8 / 255 uint8)."""
    sh = np.clip(23 - a.e, 1, 63)                   # value = M >> sh
    q = np.where(sh <= 24, a.m >> np.minimum(sh, 24), 0)
    g = np.where(sh <= 25, (a.m >> np.minimum(sh - 1, 24)) & 1, 0)
    st = (a.m & ((np.int64(1) << np.minimum(sh - 1, 40)) - 1)) != 0
    q = q + (g & (st | (q & 1)))
    q = np.where(a.e >= (7 if top == 127 else 8), top, np.minimum(q, top))
    return np.where(a.m == 0, 0, q)


def to_int8(a: XF) -> np.ndarray:
    """round-to-nearest-even of the xf value to an integer, clamped to [-127, 127]."""
    q = _round_int(a, 127)
    return np.where(a.s == 1, -q, q).astype(np.int8)


def to_uint8(a: XF) -> np.ndarray:
    """round-to-nearest-even to an unsigned integer, clamped to [0, 255] (SMAX_Q8 with 255 levels;
    the softmax probabilities are never negative)."""
    return np.where(a.s == 1, 0, _round_int(a, 255)).astype(np.uint8)


# ======================================================================================
# tables
# ======================================================================================
def _phi_gelu(x):
    return 0.5 * (1.0 + np.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x ** 3)))


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _table(f_of_i) -> tuple[np.ndarray, np.ndarray]:
    i = np.arange(TBL_N + 1, dtype=np.float64)
    v = np.rint(f_of_i(i) * (1 << TBL_VF)).astype(np.int64)
    v = np.clip(v, 0, (1 << (TBL_VF + 1)) - 1)
    d = v[1:] - v[:-1]
    assert d.min() >= -(1 << 14) and d.max() < (1 << 14), (d.min(), d.max())
    return v[:-1], d


TABLES = {
    # GELU gate phi(g) on g in [-8, 8), h = 2^-7
    "gelu": _table(lambda i: _phi_gelu(-8.0 + i * 2.0 ** -7)),
    # sigmoid on x in [-16, 16), h = 2^-6
    "sigm": _table(lambda i: _sigmoid(-16.0 + i * 2.0 ** -6)),
    # 2^-f on f in [0, 1), h = 2^-11
    "exp": _table(lambda i: 2.0 ** (-i / 2048.0)),
    # rsqrt(w): w in [1,2) h = 2^-10 (i < 1024), w in [2,4) h = 2^-9 (i >= 1024)
    "rsqrt": _table(lambda i: np.where(i < 1024, 1.0 + i / 1024.0, 2.0 + (i - 1024) / 512.0) ** -0.5),
}


def write_tables(d: Path) -> None:
    d.mkdir(parents=True, exist_ok=True)
    for name, (v, dd) in TABLES.items():
        words = (v << 15) | (dd & 0x7FFF)
        (d / f"vu_tbl_{name}.mem").write_text("\n".join(f"{int(w):09x}" for w in words) + "\n")
    (d / "vu_tbl_quant.mem").write_text("\n".join(f"{w:016x}" for w in quant_table()) + "\n")


def quant_table() -> list:
    """QUANT row constants without microcode: quant_setup's R = rsqrt(A)^2 * 127 and s_row = A * fp32(1/127)
    depend only on the amax code's 7-bit mantissa k and exponent parity odd, up to an exponent shift:
        R.e = gR - (e - odd),   s_row.e = gS + (e - odd)      (e = amax exponent)
    256 words {gR[7:0], MR[23:0], gS[7:0], MS[23:0]} indexed {odd, k}; checked here against quant_setup at
    several exponents, and bit-exact in RTL simulation against op_quant."""
    words = []
    for odd in (0, 1):
        for k in range(128):
            es = [odd, odd - 6, odd + 8, odd - 30, odd + 40]
            R, S = quant_setup(np.array([((127 + e) << 7) | k for e in es]))
            gR, gS = int(R.e[0]), int(S.e[0])
            for i, e in enumerate(es):
                assert R.m[i] == R.m[0] and R.e[i] == gR - (e - odd) and R.s[i] == 0, (odd, k, e)
                assert S.m[i] == S.m[0] and S.e[i] == gS + (e - odd) and S.s[i] == 0, (odd, k, e)
            words.append(((gR & 0xFF) << 56) | (int(R.m[0]) << 32) | ((gS & 0xFF) << 24) | int(S.m[0]))
    return words


def tbl_interp(name: str, idx, frac) -> np.ndarray:
    v, d = TABLES[name]
    return (v[idx] << TBL_FR) + d[idx] * frac          # Q1.32, >= 0


def raw_to_xf(y, e_off) -> XF:
    """non-negative Q1.32 table output -> xf with an extra exponent offset."""
    nb = bitlen(y)
    sh = nb - MW
    m = np.where(sh >= 0, y >> np.maximum(sh, 0), y << np.maximum(-sh, 0))
    return xf(0, nb - 33 + np.asarray(e_off, np.int64), m)


def tbl_gate(name: str, a: XF) -> XF:
    """phi_gelu(a) (F = 19, domain [-8, 8)) or sigmoid(a) (F = 18, domain [-16, 16))."""
    ftop = 4 if name == "gelu" else 5                  # z = M >> (ftop - e) ; saturate for e >= ftop - 1
    sat = a.e >= ftop - 1
    rs = np.clip(ftop - a.e, 0, 24)
    zm = np.where(a.m == 0, 0, a.m >> rs)
    u = np.where(a.s == 1, (1 << 22) - zm, (1 << 22) + zm)
    u = np.clip(u, 0, (1 << 23) - 1)
    y = tbl_interp(name, u >> 12, u & 0xFFF)
    t = raw_to_xf(y, 0)
    one = xf(0, 0, 1 << 23)
    satv = XF(np.where(a.s == 1, 0, one.s), np.where(a.s == 1, EZ, one.e), np.where(a.s == 1, 0, one.m))
    return XF(np.where(sat, satv.s, t.s), np.where(sat, satv.e, t.e), np.where(sat, satv.m, t.m))


FIX_EXP_W = 42            # signed fixed width of the log2-domain logits (Q18.23)


def to_fix_exp(a: XF) -> np.ndarray:
    mag = np.where(a.e >= 0, a.m << np.clip(a.e, 0, 17), a.m >> np.clip(-a.e, 0, 24))
    mag = np.where(a.e > 17, (1 << (FIX_EXP_W - 1)) - 1, mag)
    mag = np.where(a.m == 0, 0, mag)
    return np.where(a.s == 1, -mag, mag)


def tbl_exp(a: XF, mx: XF, mask=None):
    """e_i = 2^-(max - a) in the log2 domain.  Returns (y Q1.32, k, valid) -- the raw form the
    accumulator uses -- and the normalised xf."""
    t = to_fix_exp(mx) - to_fix_exp(a)
    t = np.maximum(t, 0)
    k = t >> F_EXP
    f = t & ((1 << F_EXP) - 1)
    valid = k < 64
    if mask is not None:
        valid = valid & mask
    y = np.where(valid, tbl_interp("exp", f >> 12, f & 0xFFF), 0)
    k = np.minimum(k, 63)
    return y, k, valid, raw_to_xf(y, -k)


def rsqrt(v: XF) -> XF:
    odd = v.e & 1
    idx = odd * 1024 + ((v.m - (1 << 23)) >> 13)
    frac = (v.m >> 1) & 0xFFF
    y = tbl_interp("rsqrt", np.clip(idx, 0, TBL_N - 1), frac)
    return raw_to_xf(y, -((v.e - odd) >> 1))


# ======================================================================================
# accumulators (relative to the row amax exponent E)
# ======================================================================================
def acc2_terms(a: XF, E) -> np.ndarray:
    m8 = a.m >> 16
    sh = np.minimum(2 * (a.e - E) + (F_ACC2 - 14), F_ACC2 - 14)
    t = np.where(sh >= 0, (m8 * m8) << np.maximum(sh, 0), (m8 * m8) >> np.minimum(-sh, 63))
    return np.where((a.m == 0) | (sh <= -17), 0, t)


def acc1_terms(a: XF, E) -> np.ndarray:
    m8 = a.m >> 16
    sh = np.minimum((a.e - E) + (F_ACC1 - 7), F_ACC1 - 7)
    t = np.where(sh >= 0, m8 << np.maximum(sh, 0), m8 >> np.minimum(-sh, 63))
    t = np.where((a.m == 0) | (sh <= -9), 0, t)
    return np.where(a.s == 1, -t, t)


def accexp_terms(y, k) -> np.ndarray:
    sh = (F_SMAX - 32) - k
    t = np.where(sh >= 0, y << np.maximum(sh, 0), y >> np.minimum(-sh, 63))
    return np.where(k >= EXP_KMAX, 0, t)


# ======================================================================================
# ops (one row at a time; rows along axis -1 are vectorised)
# ======================================================================================
def row_amax_code(codes) -> np.ndarray:
    return (np.asarray(codes).astype(np.int64) & 0x7FFF).max(axis=-1)


def bf16_val(codes) -> np.ndarray:
    return (np.asarray(codes).astype(np.uint32) << 16).view(np.float32)


def row_max_code(codes) -> np.ndarray:
    c = np.asarray(codes)
    i = bf16_val(c).argmax(axis=-1)
    return np.take_along_axis(c, i[..., None], axis=-1)[..., 0]


def _rows(x: XF):   # broadcast helper for per-row scalars
    return XF(x.s[..., None], x.e[..., None], x.m[..., None])


def op_add(a_code, b, b_fmt="bf16"):
    A = unpack_bf16(a_code)
    B = unpack_bf16(b) if b_fmt == "bf16" else unpack_fp32(b)
    y = add(A, B)
    return to_bf16(y), y


def rms_stat(a_code, amax_code, inv_n: float):
    """-> r = rsqrt(mean(a^2) + eps) as an fp32 code per row (the row scalar the lane emits)."""
    A = unpack_bf16(a_code)
    E = unpack_bf16(amax_code).e[..., None]
    acc2 = acc2_terms(A, E).sum(axis=-1)
    ms = mul(norm64(acc2, F_ACC2, 2 * E[..., 0]), const_xf(inv_n, acc2.shape))
    return to_fp32(rsqrt(add(ms, const_xf(EPS_NORM, acc2.shape))))


def rms_apply(a_code, r_fp32, c_fp32):
    y = mul(mul(unpack_bf16(a_code), _rows(unpack_fp32(r_fp32))), unpack_fp32(c_fp32))
    return to_bf16(y), y


def ln_stat(a_code, amax_code, inv_n: float):
    A = unpack_bf16(a_code)
    E = unpack_bf16(amax_code).e[..., None]
    acc1 = acc1_terms(A, E).sum(axis=-1)
    acc2 = acc2_terms(A, E).sum(axis=-1)
    inv = const_xf(inv_n, acc1.shape)
    mu = mul(norm64(acc1, F_ACC1, E[..., 0]), inv)
    ms = mul(norm64(acc2, F_ACC2, 2 * E[..., 0]), inv)
    var = add(ms, neg(mul(mu, mu)))
    eps = const_xf(EPS_NORM, acc1.shape)
    ve = add(var, eps)
    bad = (ve.s == 1) | (ve.m == 0)
    ve = XF(np.where(bad, eps.s, ve.s), np.where(bad, eps.e, ve.e), np.where(bad, eps.m, ve.m))
    return to_fp32(mu), to_fp32(rsqrt(ve))


def ln_apply(a_code, mu_fp32, r_fp32, c_fp32, d_fp32):
    mu, r = unpack_fp32(mu_fp32), unpack_fp32(r_fp32)
    y = add(mul(mul(add(unpack_bf16(a_code), neg(_rows(mu))), _rows(r)), unpack_fp32(c_fp32)), unpack_fp32(d_fp32))
    return to_bf16(y), y


def rope_a(a_code, cos_fp32):
    """RoPE pass A: x_i * cos, kept exact as an fp32 word (the lane's MUL-A/MUL-B/ADD-2 order has
    one product per element on the multiply chain, so the two RoPE products take two passes)."""
    y = mul(unpack_bf16(a_code), unpack_fp32(cos_fp32))
    return to_fp32(y), y


def rope_b(partner_code, sin_signed_fp32, pa_fp32):
    """RoPE pass B: partner * (+-sin) + pass-A word -> bf16."""
    y = add(mul(unpack_bf16(partner_code), unpack_fp32(sin_signed_fp32)), unpack_fp32(pa_fp32))
    return to_bf16(y), y


def op_rope(a_code, b_code, cos_fp32, sin_signed_fp32):
    pa, _ = rope_a(a_code, cos_fp32)
    return rope_b(b_code, sin_signed_fp32, pa)


def op_gelu(a_code):
    A = unpack_bf16(a_code)
    y = mul(A, tbl_gate("gelu", A))
    return to_bf16(y), y


def op_geglu(a_code, b_code):
    A = unpack_bf16(a_code)
    y = mul(mul(A, tbl_gate("gelu", A)), unpack_bf16(b_code))
    return to_bf16(y), y


def op_silu(a_code):
    A = unpack_bf16(a_code)
    y = mul(A, tbl_gate("sigm", A))
    return to_bf16(y), y


C127 = None


def smax_sum(a_code, max_code, mask=None, int8=False, levels: int = 127):
    """pass 1: e_i and S; returns (invS or s_row8, codes8 or None).  levels = 127 (int8 codes) or
    255 (uint8 codes, the PV multiplier's unsigned operand: MLP72 multmode 5'h13)."""
    A = unpack_bf16(a_code)
    MX = _rows(unpack_bf16(max_code))
    y, k, valid, e = tbl_exp(A, MX, mask)
    S = accexp_terms(y, k).sum(axis=-1)
    sx = norm64(S, F_SMAX, np.zeros_like(S))
    r = rsqrt(sx)
    inv_s = mul(r, r)
    if not int8:
        return to_fp32(inv_s), None
    assert levels in (127, 255)
    p = mul(e, const_xf(float(levels), e.shape))
    q = to_int8(p) if levels == 127 else to_uint8(p)
    s_row = mul(inv_s, const_xf(1.0 / levels, inv_s.shape))
    return to_fp32(s_row), q


def smax_out(a_code, max_code, inv_s_fp32, mask=None):
    A = unpack_bf16(a_code)
    _, _, _, e = tbl_exp(A, _rows(unpack_bf16(max_code)), mask)
    y = mul(e, _rows(unpack_fp32(inv_s_fp32)))
    return to_bf16(y), y


def quant_setup(amax_code):
    A = unpack_bf16(amax_code)
    r = rsqrt(A)
    R = mul(mul(r, r), const_xf(127.0, A.shape))
    R = XF(R.s, np.where(A.m == 0, EZ, R.e), np.where(A.m == 0, 0, R.m))
    s_row = mul(A, const_xf(1.0 / 127.0, A.shape))
    return R, s_row


def op_quant(a_code, amax_code):
    R, s_row = quant_setup(amax_code)
    y = mul(unpack_bf16(a_code), _rows(R))
    return to_int8(y), to_fp32(s_row), y


def op_dequant(acc, s_row_fp32, s_col_fp32, bias_fp32=None, out="bf16"):
    y = mul(mul(norm64(acc, 0, 0), _rows(unpack_fp32(s_row_fp32))), unpack_fp32(s_col_fp32))
    if bias_fp32 is not None:
        y = add(y, unpack_fp32(bias_fp32))
    return (to_bf16(y) if out == "bf16" else to_fp32(y)), y


def op_euler(x_fp32, v_fp32, dt: float):
    y = add(unpack_fp32(x_fp32), mul(unpack_fp32(v_fp32), const_xf(dt, np.shape(v_fp32))))
    return to_fp32(y), y


# ======================================================================================
# float helpers
# ======================================================================================
def f32_codes(x) -> np.ndarray:
    return np.ascontiguousarray(np.asarray(x, np.float32)).view(np.uint32)


def bf16_codes(x) -> np.ndarray:
    u = f32_codes(x).astype(np.uint64)
    r = ((u >> 16) & 1) + 0x7FFF
    return (((u + r) >> 16) & 0xFFFF).astype(np.uint16)


# ======================================================================================
# inventory
# ======================================================================================
def inventory(int8_attention: bool = True) -> dict:
    """Vector-unit element visits per 50-action chunk, row by row.  A row is the lane's row: the unit its
    row statistic, microcode and summary beat apply to.  Compact prefix: 2 cameras x 256 + 13 language
    = 525 tokens; expert: 51 tokens (state + 50 actions), 867 keys (816 prefix slots, 291 masked, + 51),
    10 Euler steps.  chain_side = the dequant of a GEMM result (fabric side of the chain in design v2
    section 3.1; the same lane op, counted separately).
    int8_attention (design v2 section 3.2): QK^T and PV on the chains in INT8 -> QUANT of q, k, v per head
    row and a one-pass SMAX_Q8; False -> bf16 probabilities (SMAX_SUM + SMAX_OUT), no attention QUANT."""
    rows = []

    def r(stage, what, ops, width, n, chain=False):
        e = int(width) * int(n)
        rows.append(dict(stage=stage, op="+".join(ops), ops=list(ops), where=what, row=int(width), rows=int(n),
                         elements=e, passes=len(ops), visits=e * len(ops), chain_side=chain))

    smax = ["SMAX_Q8"] if int8_attention else ["SMAX_SUM", "SMAX_OUT"]
    # SigLIP So400m/14: 27 layers, D 1152, MLP 4304, 16 heads x 72; 2 images x 256 tokens (attention per image)
    L, T, D, M, H, HD = 27, 512, 1152, 4304, 16, 72
    r("SigLIP", "patch rows (14x14x3 taps) in", ["QUANT"], 588, T)
    r("SigLIP", "patch embedding out (+bias)", ["DEQUANT"], D, T, chain=True)
    r("SigLIP", "+ position table (fp32)", ["ADD"], D, T)
    r("SigLIP", "layer_norm1/2", ["LN_STAT", "LN_APPLY"], D, 2 * T * L)
    r("SigLIP", "q/k/v in (shared), out_proj in, fc1 in", ["QUANT"], D, 3 * T * L)
    r("SigLIP", "q/k/v/out_proj out (+bias)", ["DEQUANT"], D, 4 * T * L, chain=True)
    if int8_attention:
        r("SigLIP", "q, k, v per head row (INT8 QK^T, PV)", ["QUANT"], HD, 3 * H * T * L)
    r("SigLIP", "QK^T logits, 256 keys", ["DEQUANT"], 256, H * T * L, chain=True)
    r("SigLIP", "softmax", smax, 256, H * T * L)
    r("SigLIP", "PV per head", ["DEQUANT"], HD, H * T * L, chain=True)
    r("SigLIP", "residual adds", ["ADD"], D, 2 * T * L)
    r("SigLIP", "fc1 out (+bias)", ["DEQUANT"], M, T * L, chain=True)
    r("SigLIP", "GELU (gelu_pytorch_tanh)", ["GELU"], M, T * L)
    r("SigLIP", "fc2 in", ["QUANT"], M, T * L)
    r("SigLIP", "fc2 out (+bias)", ["DEQUANT"], D, T * L, chain=True)
    r("SigLIP", "post_layernorm", ["LN_STAT", "LN_APPLY"], D, T)
    r("SigLIP", "projector in", ["QUANT"], D, T)
    r("SigLIP", "projector out (+bias)", ["DEQUANT"], 2048, T, chain=True)
    # PaliGemma prefix (Gemma 2B): 18 layers, D 2048, MLP 16384, 8 heads x 256, 1 KV head, 525 tokens
    L, T, D, M, H, HD = 18, 525, 2048, 16384, 8, 256
    r("LM", "input/post_attention RMSNorm", ["RMS_STAT", "RMS_APPLY"], D, 2 * T * L)
    r("LM", "q/k/v in (shared), o in, gate/up in (shared)", ["QUANT"], D, 3 * T * L)
    r("LM", "q, o, down out", ["DEQUANT"], D, 3 * T * L, chain=True)
    r("LM", "k, v out", ["DEQUANT"], HD, 2 * T * L, chain=True)
    r("LM", "RoPE q (8 heads) and k", ["ROPE_A", "ROPE_B"], HD, (H + 1) * T * L)
    if int8_attention:
        r("LM", "q per head, k, v per token (INT8 QK^T, PV)", ["QUANT"], HD, (H + 2) * T * L)
    r("LM", "QK^T logits, 525 keys", ["DEQUANT"], T, H * T * L, chain=True)
    r("LM", "softmax", smax, T, H * T * L)
    r("LM", "PV per head", ["DEQUANT"], HD, H * T * L, chain=True)
    r("LM", "residual adds", ["ADD"], D, 2 * T * L)
    r("LM", "gate, up out", ["DEQUANT"], M, 2 * T * L, chain=True)
    r("LM", "GeGLU gelu_tanh(gate) * up", ["GEGLU"], M, T * L)
    r("LM", "down in", ["QUANT"], M, T * L)
    # action expert (Gemma 300M): 18 layers, D 1024, MLP 4096, 8 heads x 256, 1 KV head, 51 tokens, 10 steps
    S, L, T, K, D, M, H, HD, A = 10, 18, 51, 867, 1024, 4096, 8, 256, 50
    r("expert", "state_proj / action_in_proj in", ["QUANT"], 32, S * (1 + A))
    r("expert", "state_proj / action_in_proj out (+bias)", ["DEQUANT"], D, S * (1 + A), chain=True)
    r("expert", "action_time_mlp_in in (action ++ time)", ["QUANT"], 2 * D, S * A)
    r("expert", "action_time_mlp_in out (+bias)", ["DEQUANT"], D, S * A, chain=True)
    r("expert", "SiLU", ["SILU"], D, S * A)
    r("expert", "action_time_mlp_out in", ["QUANT"], D, S * A)
    r("expert", "action_time_mlp_out out (+bias)", ["DEQUANT"], D, S * A, chain=True)
    r("expert", "input/post_attention RMSNorm", ["RMS_STAT", "RMS_APPLY"], D, S * 2 * T * L)
    r("expert", "q/k/v in (shared), gate/up in (shared)", ["QUANT"], D, S * 2 * T * L)
    r("expert", "o in", ["QUANT"], H * HD, S * T * L)
    r("expert", "q out", ["DEQUANT"], H * HD, S * T * L, chain=True)
    r("expert", "k, v out", ["DEQUANT"], HD, S * 2 * T * L, chain=True)
    r("expert", "RoPE q (8 heads) and k", ["ROPE_A", "ROPE_B"], HD, S * (H + 1) * T * L)
    if int8_attention:
        r("expert", "q per head, k, v per token (INT8 QK^T, PV)", ["QUANT"], HD, S * (H + 2) * T * L)
    r("expert", "QK^T logits, 867 keys (291 masked)", ["DEQUANT"], K, S * H * T * L, chain=True)
    r("expert", "softmax (masked)", smax, K, S * H * T * L)
    r("expert", "PV per head", ["DEQUANT"], HD, S * H * T * L, chain=True)
    r("expert", "o, down out", ["DEQUANT"], D, S * 2 * T * L, chain=True)
    r("expert", "residual adds", ["ADD"], D, S * 2 * T * L)
    r("expert", "gate, up out", ["DEQUANT"], M, S * 2 * T * L, chain=True)
    r("expert", "GeGLU gelu_tanh(gate) * up", ["GEGLU"], M, S * T * L)
    r("expert", "down in", ["QUANT"], M, S * T * L)
    r("expert", "final norm (50 action rows)", ["RMS_STAT", "RMS_APPLY"], D, S * A)
    r("expert", "action_out_proj in", ["QUANT"], D, S * A)
    r("expert", "action_out_proj out (+bias, fp32)", ["DEQUANT"], 32, S * A, chain=True)
    r("expert", "Euler x_t + dt v_t (fp32)", ["EULER"], 32, S * A)

    def tot(pred):
        return int(sum(x["visits"] for x in rows if pred(x)))

    summary = dict(visits_all=tot(lambda x: True), visits_vector_unit=tot(lambda x: not x["chain_side"]),
                   visits_chain_side_dequant=tot(lambda x: x["chain_side"]))
    for st in ("SigLIP", "LM", "expert"):
        summary[f"visits_{st}_vector_unit"] = tot(lambda x, st=st: x["stage"] == st and not x["chain_side"])
        summary[f"visits_{st}_chain_side"] = tot(lambda x, st=st: x["stage"] == st and x["chain_side"])
    per_op = {}
    for x in rows:
        for o in x["ops"]:
            d = per_op.setdefault(o, dict(elements=0, rows=0))
            d["elements"] += x["elements"]
            d["rows"] += x["rows"]
    summary["per_lane_op"] = per_op
    return dict(int8_attention=int8_attention, rows=rows, summary=summary)


# ======================================================================================
# measurement on captured activations
# ======================================================================================
def _stats(y_hw: XF, codes_hw, ref: np.ndarray, floor_rel=1e-3) -> dict:
    ref = ref.astype(np.float64)
    v = y_hw.value()
    out_v = bf16_val(codes_hw).astype(np.float64)
    ref_rowmax = np.abs(ref).max(axis=-1, keepdims=True)
    big = np.abs(ref) >= floor_rel * ref_rowmax
    big &= ref_rowmax > 0
    rel = np.abs(v - ref)[big] / np.abs(ref)[big]
    ref_bf16 = bf16_val(bf16_codes(ref.astype(np.float32))).astype(np.float64)

    def rowrel(x):
        n = np.sqrt(((x - ref) ** 2).sum(axis=-1)) / np.maximum(np.sqrt((ref ** 2).sum(axis=-1)), 1e-300)
        return n

    rr, rf = rowrel(out_v), rowrel(ref_bf16)
    ideal = bf16_codes(ref.astype(np.float32))
    dev = np.abs(out_v - ref_bf16) / np.maximum(ref_rowmax, 1e-300)     # lane output vs the ideal bf16 rounding
    return dict(elements=int(ref.size),
                big_elements=int(big.sum()),
                big_code_match=float((np.asarray(codes_hw)[big] == ideal[big]).mean()) if big.any() else 1.0,
                out_dev_vs_bf16_ideal_over_rowmax_max=float(dev.max()),
                out_dev_vs_bf16_ideal_row_rel_rms_max=float((np.sqrt(((out_v - ref_bf16) ** 2).sum(-1))
                                                            / np.maximum(np.sqrt((ref ** 2).sum(-1)), 1e-300)).max()),
                arith_rel_max=float(rel.max()) if rel.size else 0.0,
                arith_rel_rms=float(np.sqrt((rel ** 2).mean())) if rel.size else 0.0,
                arith_abs_max_over_rowmax=float((np.abs(v - ref) / np.maximum(ref_rowmax, 1e-300)).max()),
                out_row_rel_rms_max=float(rr.max()), out_row_rel_rms_mean=float(rr.mean()),
                bf16_floor_row_rel_rms_max=float(rf.max()), bf16_floor_row_rel_rms_mean=float(rf.mean()),
                bf16_code_match=float((np.asarray(codes_hw) == bf16_codes(ref.astype(np.float32))).mean()))


def measure(npz_path: str, out_json: str, max_rows: int = 256) -> dict:
    import torch
    import torch.nn.functional as F

    z = np.load(npz_path)
    res = {}
    T = torch.from_numpy

    def as_bf16_f32(x):
        return bf16_val(bf16_codes(x))

    def rec(name, d):
        res[name] = d
        print(f"{name:34s} n={d['elements']:>9d} rel max {d['arith_rel_max']:.2e} rms {d['arith_rel_rms']:.2e} "
              f"| out row rms max {d['out_row_rel_rms_max']:.2e} (bf16 floor {d['bf16_floor_row_rel_rms_max']:.2e}) "
              f"| match {d['bf16_code_match']:.4f} big {d.get('big_code_match', 1):.4f} "
              f"dev/rowmax {d.get('out_dev_vs_bf16_ideal_over_rowmax_max', 0):.1e}", flush=True)

    def flat_rows(x, n=max_rows):
        x = x.reshape(-1, x.shape[-1])
        if x.shape[0] > n:
            x = x[np.linspace(0, x.shape[0] - 1, n).astype(int)]
        return x

    quant_inputs = []
    # ---- LayerNorm (SigLIP)
    for li in (0, 13, 26):
        for nm in ("layer_norm1", "layer_norm2"):
            x = flat_rows(z[f"vis.L{li}.{nm}.in"])
            g, b = z[f"vis.L{li}.{nm}.gamma"], z[f"vis.L{li}.{nm}.beta"]
            xc = bf16_codes(x)
            ref = F.layer_norm(T(bf16_val(xc)), (x.shape[-1],), T(g), T(b), 1e-6).numpy()
            mu, r = ln_stat(xc, row_amax_code(xc), 1.0 / x.shape[-1])
            codes, y = ln_apply(xc, mu, r, f32_codes(np.broadcast_to(g, x.shape)), f32_codes(np.broadcast_to(b, x.shape)))
            rec(f"LN vis.L{li}.{nm}", _stats(y, codes, ref))
            quant_inputs.append((f"vis.L{li}.{nm}.out", ref))
    x = flat_rows(z["vis.post_ln.in"])
    xc = bf16_codes(x)
    ref = F.layer_norm(T(bf16_val(xc)), (1152,), T(z["vis.post_ln.gamma"]), T(z["vis.post_ln.beta"]), 1e-6).numpy()
    mu, r = ln_stat(xc, row_amax_code(xc), 1.0 / 1152)
    codes, y = ln_apply(xc, mu, r, f32_codes(np.broadcast_to(z["vis.post_ln.gamma"], x.shape)),
                        f32_codes(np.broadcast_to(z["vis.post_ln.beta"], x.shape)))
    rec("LN vis.post_layernorm", _stats(y, codes, ref))

    # ---- RMSNorm (Gemma 1+w)
    def rms_ref(xv, w):
        xt = T(xv)
        var = torch.mean(torch.square(xt), dim=-1, keepdim=True)
        return (xt * torch.rsqrt(var + 1e-6) * (1.0 + T(w))).numpy()

    for pre in ("lm", "ex"):
        for li in (0, 9, 17):
            for nm in ("input_layernorm", "post_attention_layernorm"):
                x = flat_rows(z[f"{pre}.L{li}.{nm}.in"])
                w = z[f"{pre}.L{li}.{nm}.gain"]
                xc = bf16_codes(x)
                ref = rms_ref(bf16_val(xc), w)
                r = rms_stat(xc, row_amax_code(xc), 1.0 / x.shape[-1])
                cst = f32_codes(np.broadcast_to(np.float32(1.0) + w.astype(np.float32), x.shape))
                codes, y = rms_apply(xc, r, cst)
                rec(f"RMS {pre}.L{li}.{nm}", _stats(y, codes, ref))
                quant_inputs.append((f"{pre}.L{li}.{nm}.out", ref))
    x = flat_rows(z["ex.final_norm.in"])
    w = z["ex.final_norm.gain"]
    xc = bf16_codes(x)
    ref = rms_ref(bf16_val(xc), w)
    codes, y = rms_apply(xc, rms_stat(xc, row_amax_code(xc), 1.0 / 1024),
                         f32_codes(np.broadcast_to(np.float32(1.0) + w, x.shape)))
    rec("RMS ex.final_norm", _stats(y, codes, ref))

    # ---- RoPE (half split, Gemma rotate_half)
    from transformers.models.gemma.modeling_gemma import apply_rotary_pos_emb
    for pre in ("lm", "ex"):
        cos_all, sin_all = z[f"{pre}.rope_cos"], z[f"{pre}.rope_sin"]
        cos_all = cos_all.reshape(-1, cos_all.shape[-2], cos_all.shape[-1])[0]
        sin_all = sin_all.reshape(-1, sin_all.shape[-2], sin_all.shape[-1])[0]
        for li in (0, 9, 17):
            for kk, nh in (("q_pre", 8), ("k_pre", 1)):
                x = z[f"{pre}.L{li}.{kk}"]
                x = x.reshape(-1, x.shape[-2], x.shape[-1])[0]                 # (T, nh*256)
                Tn = x.shape[0]
                xh = x.reshape(Tn, nh, 256).transpose(1, 0, 2)                 # (nh, T, 256)
                xc = bf16_codes(xh)
                xv = bf16_val(xc)
                cos, sin = cos_all[:Tn], sin_all[:Tn]
                qr, _ = apply_rotary_pos_emb(T(xv)[None], T(xv)[None], T(cos)[None], T(sin)[None], unsqueeze_dim=1)
                ref = qr[0].numpy()
                half = 128
                partner = np.concatenate([xc[..., half:], xc[..., :half]], axis=-1)
                sgn_sin = np.concatenate([-sin[:, :half], sin[:, half:]], axis=-1).astype(np.float32)
                codes, y = op_rope(xc, partner, f32_codes(np.broadcast_to(cos, xc.shape)),
                                   f32_codes(np.broadcast_to(sgn_sin, xc.shape)))
                rec(f"ROPE {pre}.L{li}.{kk}", _stats(y, codes, ref))

    # ---- softmax (log2-domain bf16 logits in; masked keys e = 0)
    for pre, layers in (("vis", (0, 13, 26)), ("lm", (0, 9, 17)), ("ex", (0, 9, 17))):
        for li in layers:
            q, k = z[f"{pre}.L{li}.q"], z[f"{pre}.L{li}.k"]
            sc = float(z[f"{pre}.L{li}.scaling"])
            q = q.reshape(-1, *q.shape[-3:])[0]
            k = k.reshape(-1, *k.shape[-3:])[0]
            logits = torch.matmul(T(q), T(k).transpose(-1, -2)).numpy() * np.float32(sc)   # (heads, Tq, Tk)
            mask = None
            if f"{pre}.L{li}.mask" in z.files:
                mask = z[f"{pre}.L{li}.mask"].reshape(-1, *z[f"{pre}.L{li}.mask"].shape[-2:])[0]
                mask = np.broadcast_to(mask, logits.shape)
            lg = flat_rows(logits)
            mk = flat_rows(mask) if mask is not None else None
            l2 = lg / np.float32(math.log(2.0))
            lc = bf16_codes(l2)
            lin = bf16_val(lc).astype(np.float64) * math.log(2.0)          # the logits the op actually sees
            lt = T(lin)
            if mk is not None:
                lt = lt.masked_fill(~T(mk.copy()), float("-inf"))
            ref = torch.softmax(lt, dim=-1).numpy()
            mx = row_max_code(np.where(mk, lc, 0xFF80) if mk is not None else lc)      # max over valid keys
            inv_s, _ = smax_sum(lc, mx, mk)
            codes, y = smax_out(lc, mx, inv_s, mk)
            st = _stats(y, codes, ref)
            reff = torch.softmax(T(lg).masked_fill(~T(mk.copy()), float("-inf")) if mk is not None else T(lg), dim=-1).numpy()
            st["out_row_rel_rms_max_vs_fp32_logits"] = float((np.sqrt(((bf16_val(codes) - reff) ** 2).sum(-1))
                                                             / np.sqrt((reff ** 2).sum(-1))).max())
            s8, q8 = smax_sum(lc, mx, mk, int8=True)
            pm = ref.max(axis=-1, keepdims=True)
            q_ref = np.clip(np.rint(ref / (pm / 127.0)), -127, 127)
            st["int8_code_match"] = float((q8 == q_ref).mean())
            st["int8_srow_rel_err_max"] = float(np.abs(s8.view(np.float32) / (pm[:, 0] / 127.0) - 1).max())
            st["logit_abs_max"] = float(np.abs(lg[np.isfinite(lg)]).max())
            rec(f"SMAX {pre}.L{li}", st)

    # ---- GELU / GeGLU / SiLU
    for li in (0, 13, 26):
        x = flat_rows(z[f"vis.L{li}.fc1_out"])
        xc = bf16_codes(x)
        ref = F.gelu(T(bf16_val(xc)), approximate="tanh").numpy()
        codes, y = op_gelu(xc)
        rec(f"GELU vis.L{li}", _stats(y, codes, ref))
        quant_inputs.append((f"vis.L{li}.fc2_in", ref))
    for pre in ("lm", "ex"):
        for li in (0, 9, 17):
            g, u = flat_rows(z[f"{pre}.L{li}.gate_out"]), flat_rows(z[f"{pre}.L{li}.up_out"])
            gc, uc = bf16_codes(g), bf16_codes(u)
            ref = (F.gelu(T(bf16_val(gc)), approximate="tanh") * T(bf16_val(uc))).numpy()
            codes, y = op_geglu(gc, uc)
            rec(f"GEGLU {pre}.L{li}", _stats(y, codes, ref))
            quant_inputs.append((f"{pre}.L{li}.down_in", ref))
    x = flat_rows(z["ex.silu_in"])
    xc = bf16_codes(x)
    ref = F.silu(T(bf16_val(xc))).numpy()
    codes, y = op_silu(xc)
    rec("SILU ex.action_time_mlp", _stats(y, codes, ref))

    # ---- residual add / pos-emb add
    for pre, layers, attn in (("vis", (0, 13, 26), "attn_out"), ("lm", (0, 9, 17), "o_out"), ("ex", (0, 9, 17), "o_out")):
        for li in layers:
            a, b = flat_rows(z[f"{pre}.L{li}.layer_in"]), flat_rows(z[f"{pre}.L{li}.{attn}"])
            ac, bc = bf16_codes(a), bf16_codes(b)
            ref = (T(bf16_val(ac)) + T(bf16_val(bc))).numpy()
            codes, y = op_add(ac, bc)
            rec(f"ADD {pre}.L{li}.residual", _stats(y, codes, ref))
            if pre != "vis":
                quant_inputs.append((f"{pre}.L{li}.o_in", flat_rows(z[f"{pre}.L{li}.o_in"])))
    pa = flat_rows(z["vis.patch_out"])
    pos = z["vis.pos_table"]
    posr = flat_rows(np.broadcast_to(pos, z["vis.patch_out"].shape))
    ac = bf16_codes(pa)
    ref = (T(bf16_val(ac)) + T(posr.copy())).numpy()
    codes, y = op_add(ac, f32_codes(posr), "fp32")
    rec("ADD vis.pos_emb (fp32 table)", _stats(y, codes, ref))

    # ---- per-token INT8 quantisation
    for name, x in quant_inputs:
        xc = bf16_codes(flat_rows(x))
        xv = bf16_val(xc)
        am = np.abs(xv).max(-1, keepdims=True)
        s = np.where(am > 0, am, 1.0).astype(np.float32) / np.float32(127.0)
        qref = np.clip(np.rint(xv / s), -127, 127)
        xv64, am64 = xv.astype(np.float64), am.astype(np.float64)
        qref64 = np.clip(np.rint(127.0 * xv64 / np.where(am64 > 0, am64, 1.0)), -127, 127)   # exact ties
        q, srow, yq = op_quant(xc, row_amax_code(xc))
        srow_v = srow.view(np.float32)
        d = dict(elements=int(xc.size), code_match=float((q == qref).mean()),
                 code_mismatch=int((q != qref).sum()), code_absdiff_max=int(np.abs(q.astype(int) - qref).max()),
                 code_mismatch_vs_exact_f64=int((q != qref64).sum()),
                 torch_fp32_ref_mismatch_vs_exact_f64=int((qref != qref64).sum()),
                 scale_rel_err_max=float(np.abs(srow_v / s[:, 0] - 1).max()),
                 arith_rel_max=float(np.abs(yq.value()[xv != 0] / (xv / s)[xv != 0] - 1).max()))
        res[f"QUANT {name}"] = d
        print(f"QUANT {name:28s} n={d['elements']:>9d} code match {d['code_match']:.6f} (mismatch {d['code_mismatch']}, "
              f"max |dq| {d['code_absdiff_max']}; vs exact f64 {d['code_mismatch_vs_exact_f64']}, torch vs exact "
              f"{d['torch_fp32_ref_mismatch_vs_exact_f64']}) scale rel {d['scale_rel_err_max']:.1e} arith {d['arith_rel_max']:.1e}")

    # ---- dequant: exact int sums of real INT8 weights x per-token INT8 activations
    from safetensors import safe_open
    ck = os.path.expanduser("~/pi0_ckpt/ur7e-demo-2-pi0-fsdp/010000/pretrained_model/model.safetensors")
    pj = "model.paligemma_with_expert."
    specs = [("lm.L0.q_proj", pj + "joint_layers.0.paligemma_layer.self_attn.q_proj.weight", None,
              "lm.L0.input_layernorm.out", "bf16"),
             ("ex.L9.gate_proj", pj + "joint_layers.9.expert_layer.mlp.gate_proj.weight", None,
              "ex.L9.post_attention_layernorm.out", "bf16"),
             ("vis.L13.fc1 (+bias)", pj + "paligemma.model.vision_tower.vision_model.encoder.layers.13.mlp.fc1.weight",
              pj + "paligemma.model.vision_tower.vision_model.encoder.layers.13.mlp.fc1.bias", "vis.L13.layer_norm2.out", "bf16"),
             ("action_out_proj (+bias, fp32 out)", "model.action_out_proj.weight", "model.action_out_proj.bias",
              None, "fp32")]
    qi = dict(quant_inputs)
    with safe_open(ck, "np") as fh:
        for name, wk, bk, xin, outfmt in specs:
            W = fh.get_tensor(wk).astype(np.float32)
            bias = fh.get_tensor(bk).astype(np.float32) if bk else None
            if xin is None:
                x = rms_ref(bf16_val(bf16_codes(z["ex.final_norm.in"][..., 1:, :].reshape(-1, 1024)[:64])),
                            z["ex.final_norm.gain"])
            else:
                x = flat_rows(qi[xin], 64)
            xv = bf16_val(bf16_codes(x))
            am = np.abs(xv).max(-1, keepdims=True)
            sr = (np.where(am > 0, am, 1) / np.float32(127)).astype(np.float32)
            qx = np.clip(np.rint(xv / sr), -127, 127).astype(np.int64)
            wa = np.abs(W).max(1, keepdims=True)
            sc = (np.where(wa > 0, wa, 1) / np.float32(127)).astype(np.float32)
            qw = np.clip(np.rint(W / sc), -127, 127).astype(np.int64)
            acc = qx @ qw.T                                             # (rows, out)
            ref = acc.astype(np.float64) * sr.astype(np.float64) * sc[:, 0].astype(np.float64)[None, :]
            if bias is not None:
                ref = ref + bias.astype(np.float64)[None, :]
            codes, y = op_dequant(acc, f32_codes(sr[:, 0]),
                                  f32_codes(np.broadcast_to(sc[:, 0], acc.shape)),
                                  None if bias is None else f32_codes(np.broadcast_to(bias, acc.shape)), outfmt)
            if outfmt == "bf16":
                rec(f"DEQUANT {name}", _stats(y, codes, ref))
            else:
                v = codes.view(np.float32).astype(np.float64)
                big = np.abs(ref) > 1e-3 * np.abs(ref).max(-1, keepdims=True)
                d = dict(elements=int(ref.size), acc_abs_max=int(np.abs(acc).max()),
                         arith_rel_max=float((np.abs(v - ref)[big] / np.abs(ref)[big]).max()),
                         fp32_ulp_rel=float(2.0 ** -23))
                res[f"DEQUANT {name}"] = d
                print(f"DEQUANT {name}: fp32 out rel max {d['arith_rel_max']:.2e} (1 fp32 ulp = 1.2e-7)")

    # ---- Euler, the 10 steps of the captured frame in fp32
    x = z["ex.noise"].astype(np.float32)
    vts = z["ex.v_t"].reshape(10, 50, 32)
    xh = f32_codes(x)
    errs = []
    for s in range(10):
        xt_ref = (T(x) + (-1.0 / 10) * T(vts[s])).numpy()
        xh, y = op_euler(xh, f32_codes(vts[s]), -1.0 / 10)
        x = xt_ref
        errs.append(float(np.abs(xh.view(np.float32).astype(np.float64) - xt_ref).max() / np.abs(xt_ref).max()))
    res["EULER ex (10 steps chained, fp32)"] = dict(elements=500 * 32, abs_err_max_over_rowmax=max(errs),
                                                    final_vs_capture_actions_max_abs=float(np.abs(
                                                        xh.view(np.float32) - z["ex.actions_fp32"]).max()))
    print("EULER", res["EULER ex (10 steps chained, fp32)"])
    Path(out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(out_json).write_text(json.dumps(res, indent=1))
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tables", default=None)
    ap.add_argument("--inventory", action="store_true")
    ap.add_argument("--measure", default=None)
    ap.add_argument("--out", default=str(REPO / "paper" / "data" / "vector_unit" / "errors.json"))
    a = ap.parse_args()
    if a.tables:
        write_tables(Path(a.tables))
        print(f"tables -> {a.tables}")
    if a.inventory:
        inv = dict(int8_attention=inventory(True), bf16_attention=inventory(False))
        for key, v in inv.items():
            print(f"== {key}")
            for r in v["rows"]:
                print(f"{r['stage']:7s} {r['op']:20s} row {r['row']:>5d} x {r['rows']:>9,d} rows -> {r['visits']:>13,d}"
                      f"{' (chain side)' if r['chain_side'] else ''}  {r['where']}")
            sm = v["summary"]
            print(f"visits per chunk: all {sm['visits_all']:,d}, vector unit {sm['visits_vector_unit']:,d}, "
                  f"chain-side dequant {sm['visits_chain_side_dequant']:,d}")
        p = REPO / "paper" / "data" / "vector_unit" / "inventory.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(inv, indent=1))
    if a.measure:
        measure(a.measure, a.out)


if __name__ == "__main__":
    main()
