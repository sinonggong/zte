#!/usr/bin/env python3
"""GDDR6 image, op program and expected output beats for tb_vu_node.sv (paper/rtl/vector_unit/vu_node.sv).

Every op of the program is built with the lane golden functions of vector_unit_vectors.py (which call the
bit-exact reference vector_unit_ref.py) over R rows of L elements.  Each operand is laid out in GDDR6 in one
of the shapes the node handles -- E (element, R x L), R (one per row), C (one per column, same every row),
K (constant in the op record) -- and one of the formats bf16, fp32, int32, or 64-bit summary records
{amax, max, rs0} with a field select.  Rows are long enough that several ops run in more than one chunk.
Writes into --out:
  mem_in.hex     beat index (address >> 5) and 256-bit beat: operand tensors and the program
  mem_exp.hex    every beat the node must write (element and summary regions of each op)
  prog_base.txt  program address
  ops.txt        one line per op
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import vector_unit_ref as R      # noqa: E402
import vector_unit_vectors as V  # noqa: E402

SH_E, SH_R, SH_C, SH_K = 0, 1, 2, 3
F_BF16, F_FP32, F_INT32, F_SUM = 0, 1, 2, 3
FIELD_RS0, FIELD_MAX, FIELD_AMAX = 0, 1, 2
WIDTH = {F_BF16: 16, F_FP32: 32, F_INT32: 32, F_SUM: 64}
PROG_BASE = 0x0800_0000
IN_BASE = 0x1000_0000
OUT_BASE = 0x4000_0000


class Mem:
    def __init__(self):
        self.beats: dict[int, int] = {}

    def put(self, addr: int, values, width: int) -> None:
        assert addr % 32 == 0
        values = [int(v) for v in values]
        b0 = addr >> 5
        mask = (1 << width) - 1
        for i, v in enumerate(values):
            bit = i * width
            b = b0 + bit // 256
            self.beats[b] = self.beats.get(b, 0) | ((v & mask) << (bit % 256))
        for b in range(b0, b0 + (len(values) * width + 255) // 256):
            self.beats.setdefault(b, 0)

    def write(self, path: Path) -> None:
        with open(path, "w") as f:
            for b in sorted(self.beats):
                f.write(f"{b:x} {self.beats[b]:064x}\n")


class Alloc:
    def __init__(self, base: int):
        self.p = base

    def __call__(self, nbits: int) -> int:
        a = self.p
        self.p += ((nbits + 255) // 256 + 4) * 32
        return a


def desc(base: int, shape: int, fmt: int = 0, field: int = 0) -> int:
    return base | (shape << 42) | (fmt << 44) | (field << 46)


def K(value: int) -> int:
    return (int(value) & 0xFFFFFFFF) | (SH_K << 42)


QTAIL_ALL = False
SMALL_PDQ = False


class Builder:
    def __init__(self, rng):
        self.rng = rng
        self.mem, self.exp = Mem(), Mem()
        self.ain, self.aout = Alloc(IN_BASE), Alloc(OUT_BASE)
        self.words: list[int] = []
        self.ops: list[str] = []
        self.qtail = QTAIL_ALL    # --suite *_q: every bf16-output op carries the fused QUANT flag (w0[118])

    def operand(self, values, shape: int, fmt: int, field: int = 0) -> int:
        """lay out one operand; `values` holds what the lane must see (flattened in row order)."""
        vals = [int(v) for v in np.asarray(values).reshape(-1)]
        if fmt == F_SUM:  # summary records: the used field, random bits elsewhere
            recs = []
            for v in vals:
                r = int(self.rng.integers(0, 1 << 62)) | (int(self.rng.integers(0, 4)) << 62)
                if field == FIELD_RS0:
                    r = (r & ~0xFFFFFFFF) | (v & 0xFFFFFFFF)
                elif field == FIELD_MAX:
                    r = (r & ~(0xFFFF << 32)) | ((v & 0xFFFF) << 32)
                else:
                    r = (r & ~(0xFFFF << 48)) | ((v & 0xFFFF) << 48)
                recs.append(r)
            vals = recs
        base = self.ain(len(vals) * WIDTH[fmt])
        self.mem.put(base, vals, WIDTH[fmt])
        return desc(base, shape, fmt, field)

    def op(self, cs, rows: int, length: int, ew: int, x, b=None, c=None, d=None, e=None, rs=None, mask=None,
           x_rot: bool = False, pdq: bool = False):
        """cs: a filled vector_unit_vectors.Case; ew: output element width (0 none)."""
        fused = self.qtail and ew == 16 and not any(k == 2 for k, _ in cs.out)
        if fused:
            # the lane must emit what an OP_QUANT record over this op's bf16 output would: per row, int8 codes of
            # the row quantised with its own amax, and the summary {0, 0, s_row}
            cq = V.Case(cs.name + " +Q", "QUANT")
            row = []
            for k, v in cs.out:
                if k == 0:
                    row.append(v)
                else:
                    codes = np.array(row, np.uint16)
                    V.g_quant(cq, codes, int(R.row_amax_code(codes[None])[0]))
                    row = []
            assert not row
            out, ew = cq.out, 8
        else:
            out = cs.out
        elems = [v for k, v in out if k in (0, 2)]
        sums = [v for k, v in out if k == 1]
        assert len(sums) == rows, (cs.name, len(sums), rows)
        width = 32 if any(k == 2 for k, _ in out) else ew
        out_base = self.aout(max(1, len(elems) * max(width, 8)))
        sum_base = self.aout(max(1, len(sums) * 64))
        if elems:
            self.exp.put(out_base, elems, width)
        self.exp.put(sum_base, sums, 64)
        unused = K(0)
        w0 = ((0xA << 124) | (int(pdq) << 119) | (int(fused) << 118) | (out_base << 76) | (length << 60) | (rows << 40)
              | ((cs.k & 0xFFFFFFFF) << 8)
              | (int(x_rot) << 7) | (cs.out_fp32 << 6) | (cs.bias_en << 5) | (cs.b_fp32 << 4) | cs.op)
        w1 = sum_base | (x << 42)
        w2 = (b if b is not None else unused) | ((c if c is not None else unused) << 48)
        w3 = (d if d is not None else unused) | ((e if e is not None else unused) << 48)
        w4 = (rs if rs is not None else unused) | ((mask if mask is not None else K(1)) << 48)
        self.words += [w0, w1, w2, w3, w4, 0]
        self.ops.append(f"{cs.name}{' +PDQ' if pdq else ''}{' +QTAIL' if fused else ''} R={rows} L={length} out={out_base:#x} sum={sum_base:#x} "
                        f"elements={len(elems)}x{width} summaries={len(sums)}")


def build(seed: int) -> Builder:
    rng = np.random.default_rng(seed)
    B = Builder(rng)
    bf = lambda n, lo=-6, hi=6: V.rbf16(rng, n, lo, hi)                       # noqa: E731
    fp = lambda n, lo=-4, hi=4, pos=False: V.rfp32(rng, n, lo, hi, pos=pos)  # noqa: E731

    # ADD, bf16 b (E, E)
    Rr, L = 5, 37
    xs, bs = bf(Rr * L).reshape(Rr, L), bf(Rr * L, -8, 8).reshape(Rr, L)
    cs = V.Case("ADD bf16", "ADD")
    for r in range(Rr):
        V.g_add(cs, xs[r], bs[r])
    B.op(cs, Rr, L, 16, B.operand(xs, SH_E, F_BF16), b=B.operand(bs, SH_E, F_BF16))

    # ADD, fp32 position table (E, C)
    Rr, L = 4, 256
    xs, tab = bf(Rr * L).reshape(Rr, L), fp(L, -6, 2)
    cs = V.Case("ADD fp32 C", "ADD", b_fp32=1)
    for r in range(Rr):
        V.g_add(cs, xs[r], tab)
    B.op(cs, Rr, L, 16, B.operand(xs, SH_E, F_BF16), b=B.operand(tab, SH_C, F_FP32))

    # RMS_STAT (rs = amax from summary records) then RMS_APPLY (c = r from summary records, d = C gains)
    Rr, L = 7, 300
    xs = bf(Rr * L, -8, 8).reshape(Rr, L)
    cs = V.Case("RMS_STAT", "RMS_STAT", k=V.f32(1.0 / L))
    amax = [int(R.row_amax_code(xs[r][None])[0]) for r in range(Rr)]
    rv = [V.g_rms_stat(cs, xs[r]) for r in range(Rr)]
    B.op(cs, Rr, L, 0, B.operand(xs, SH_E, F_BF16), rs=B.operand(amax, SH_R, F_SUM, FIELD_AMAX))
    gain = fp(L, -2, 1, pos=True)
    cs = V.Case("RMS_APPLY", "RMS_APPLY")
    for r in range(Rr):
        V.g_rms_apply(cs, xs[r], rv[r], gain)
    B.op(cs, Rr, L, 16, B.operand(xs, SH_E, F_BF16), c=B.operand(rv, SH_R, F_SUM, FIELD_RS0),
         d=B.operand(gain, SH_C, F_FP32))

    # LN_STAT (rows longer than half a chunk: one row per chunk) then LN_APPLY
    Rr, L = 3, 1152
    xs = bf(Rr * L, -5, 5).reshape(Rr, L)
    amax = [int(R.row_amax_code(xs[r][None])[0]) for r in range(Rr)]
    cs = V.Case("LN_STAT", "LN_STAT", k=V.f32(1.0 / L))
    mur = [V.g_ln_stat(cs, xs[r], amax[r]) for r in range(Rr)]
    B.op(cs, Rr, L, 0, B.operand(xs, SH_E, F_BF16), rs=B.operand(amax, SH_R, F_BF16))
    gamma, beta = fp(L, -2, 1, pos=True), fp(L, -6, 0)
    cs = V.Case("LN_APPLY", "LN_APPLY")
    for r in range(Rr):
        V.g_ln_apply(cs, xs[r], mur[r][0], mur[r][1], gamma, beta)
    B.op(cs, Rr, L, 16, B.operand(xs, SH_E, F_BF16), b=B.operand([m for m, _ in mur], SH_R, F_FP32),
         c=B.operand([q for _, q in mur], SH_R, F_FP32), d=B.operand(gamma, SH_C, F_FP32),
         e=B.operand(beta, SH_C, F_FP32))

    # GELU (5 chunks), GEGLU (E, E), SILU
    Rr, L = 9, 700
    xs = bf(Rr * L, -4, 4).reshape(Rr, L)
    cs = V.Case("GELU", "GELU")
    for r in range(Rr):
        V.g_unary(cs, R.op_gelu, xs[r])
    B.op(cs, Rr, L, 16, B.operand(xs, SH_E, F_BF16))
    Rr, L = 4, 256
    gs, us = bf(Rr * L, -4, 4).reshape(Rr, L), bf(Rr * L, -6, 6).reshape(Rr, L)
    cs = V.Case("GEGLU", "GEGLU")
    for r in range(Rr):
        V.g_geglu(cs, gs[r], us[r])
    B.op(cs, Rr, L, 16, B.operand(gs, SH_E, F_BF16), d=B.operand(us, SH_E, F_BF16))
    Rr, L = 6, 32
    xs = bf(Rr * L, -5, 5).reshape(Rr, L)
    cs = V.Case("SILU", "SILU")
    for r in range(Rr):
        V.g_unary(cs, R.op_silu, xs[r])
    B.op(cs, Rr, L, 16, B.operand(xs, SH_E, F_BF16))

    # QUANT (rs = amax from summary records)
    Rr, L = 6, 588
    xs = bf(Rr * L, -7, 7).reshape(Rr, L)
    amax = [int(R.row_amax_code(xs[r][None])[0]) for r in range(Rr)]
    cs = V.Case("QUANT", "QUANT")
    for r in range(Rr):
        V.g_quant(cs, xs[r], amax[r])
    B.op(cs, Rr, L, 8, B.operand(xs, SH_E, F_BF16), rs=B.operand(amax, SH_R, F_SUM, FIELD_AMAX))

    # DEQUANT: int32 sums, s_row (R), s_col and bias (C), bf16 out; and fp32 out without bias
    Rr, L = 5, 100
    acc = rng.integers(-(1 << 27), 1 << 27, (Rr, L)).astype(np.int64)
    s_row, s_col, bias = fp(Rr, -14, -6, pos=True), fp(L, -8, -2, pos=True), fp(L, -6, 1)
    cs = V.Case("DEQUANT bias bf16", "DEQUANT", bias_en=1)
    for r in range(Rr):
        V.g_dequant(cs, acc[r], int(s_row[r]), s_col, bias)
    B.op(cs, Rr, L, 16, B.operand(acc, SH_E, F_INT32), c=B.operand(s_row, SH_R, F_FP32),
         d=B.operand(s_col, SH_C, F_FP32), e=B.operand(bias, SH_C, F_FP32))
    Rr, L = 3, 7
    acc = rng.integers(-(1 << 27), 1 << 27, (Rr, L)).astype(np.int64)
    s_row, s_col = fp(Rr, -14, -6, pos=True), fp(L, -8, -2, pos=True)
    cs = V.Case("DEQUANT fp32", "DEQUANT", out_fp32=1)
    for r in range(Rr):
        V.g_dequant(cs, acc[r], int(s_row[r]), s_col)
    B.op(cs, Rr, L, 32, B.operand(acc, SH_E, F_INT32), c=B.operand(s_row, SH_R, F_SUM, FIELD_RS0),
         d=B.operand(s_col, SH_C, F_FP32))

    # softmax family: mask per key (C), row max (R)
    Rr, L = 8, 67
    xs = bf(Rr * L, -3, 4).reshape(Rr, L)
    mask = rng.random(L) < 0.8
    mask[0] = True
    mx = [int(V.row_max(xs[r], mask)) for r in range(Rr)]
    cs = V.Case("SMAX_SUM", "SMAX_SUM")
    inv = [V.g_smax_sum(cs, xs[r], mask) for r in range(Rr)]
    B.op(cs, Rr, L, 0, B.operand(xs, SH_E, F_BF16), rs=B.operand(mx, SH_R, F_SUM, FIELD_MAX),
         mask=B.operand(mask.astype(np.int64), SH_C, F_BF16))
    cs = V.Case("SMAX_OUT", "SMAX_OUT")
    for r in range(Rr):
        V.g_smax_out(cs, xs[r], mask, inv[r])
    B.op(cs, Rr, L, 16, B.operand(xs, SH_E, F_BF16), rs=B.operand(mx, SH_R, F_BF16),
         mask=B.operand(mask.astype(np.int64), SH_C, F_BF16), c=B.operand(inv, SH_R, F_FP32))
    cs = V.Case("SMAX_Q8", "SMAX_Q8")
    for r in range(Rr):
        V.g_smax_q8(cs, xs[r], mask)
    B.op(cs, Rr, L, 8, B.operand(xs, SH_E, F_BF16), rs=B.operand(mx, SH_R, F_BF16),
         mask=B.operand(mask.astype(np.int64), SH_C, F_BF16))

    # RoPE passes (all operands per element)
    Rr, L = 4, 64
    xs, cos = bf(Rr * L, -4, 4).reshape(Rr, L), fp(Rr * L, -3, 0).reshape(Rr, L)
    cs = V.Case("ROPE_A", "ROPE_A")
    for r in range(Rr):
        V.g_rope_a(cs, xs[r], cos[r])
    B.op(cs, Rr, L, 32, B.operand(xs, SH_E, F_BF16), c=B.operand(cos, SH_E, F_FP32))
    ps, sin, pa = bf(Rr * L, -4, 4).reshape(Rr, L), fp(Rr * L, -3, 0).reshape(Rr, L), fp(Rr * L, -4, 4).reshape(Rr, L)
    cs = V.Case("ROPE_B", "ROPE_B")
    for r in range(Rr):
        V.g_rope_b(cs, ps[r], sin[r], pa[r])
    B.op(cs, Rr, L, 16, B.operand(ps, SH_E, F_BF16), c=B.operand(sin, SH_E, F_FP32),
         e=B.operand(pa, SH_E, F_FP32))

    # EULER: x = v (fp32), e = x_t (fp32), k = dt
    Rr, L = 10, 32
    v, xt = fp(Rr * L, -4, 3).reshape(Rr, L), fp(Rr * L, -2, 2).reshape(Rr, L)
    cs = V.Case("EULER", "EULER", k=V.f32(-0.1))
    for r in range(Rr):
        V.g_euler(cs, xt[r], v[r])
    B.op(cs, Rr, L, 32, B.operand(v, SH_E, F_FP32), e=B.operand(xt, SH_E, F_FP32))

    B.words.append(0)                                   # END
    B.mem.put(PROG_BASE, B.words, 128)
    return B


def build_ml(seed: int) -> Builder:
    """Cases for the multi-lane node (vu_node_ml.sv): many chunks with a partial last chunk and empty trailing
    lanes, one row per lane (L = 2048), odd row lengths whose lanes start mid-word (L = 1023 / 867: one or two
    rows per lane), row reductions across lanes, every output width, and element counts that leave partial
    beats at lane boundaries.  Same program format and layout as build()."""
    rng = np.random.default_rng(seed + 1000)
    B = Builder(rng)
    bf = lambda n, lo=-6, hi=6: V.rbf16(rng, n, lo, hi)                       # noqa: E731
    fp = lambda n, lo=-4, hi=4, pos=False: V.rfp32(rng, n, lo, hi, pos=pos)  # noqa: E731

    # QUANT over many short rows: R=2000, L=16 (128 rows per lane), int8 out, rs from summary records
    Rr, L = 2000, 16
    xs = bf(Rr * L, -7, 7).reshape(Rr, L)
    amax = [int(R.row_amax_code(xs[r][None])[0]) for r in range(Rr)]
    cs = V.Case("QUANT R2000 L16", "QUANT")
    for r in range(Rr):
        V.g_quant(cs, xs[r], amax[r])
    B.op(cs, Rr, L, 8, B.operand(xs, SH_E, F_BF16), rs=B.operand(amax, SH_R, F_SUM, FIELD_AMAX))

    # ADD with one row per lane: L = 2048, R = 5 (a partial chunk with one lane)
    Rr, L = 5, 2048
    xs, bs = bf(Rr * L).reshape(Rr, L), bf(Rr * L, -8, 8).reshape(Rr, L)
    cs = V.Case("ADD L2048", "ADD")
    for r in range(Rr):
        V.g_add(cs, xs[r], bs[r])
    B.op(cs, Rr, L, 16, B.operand(xs, SH_E, F_BF16), b=B.operand(bs, SH_E, F_BF16))

    # LN_STAT / GELU with L = 1023 (1 row per lane, lanes start at every word position), rs R bf16
    Rr, L = 9, 1023
    xs = bf(Rr * L, -5, 5).reshape(Rr, L)
    amax = [int(R.row_amax_code(xs[r][None])[0]) for r in range(Rr)]
    cs = V.Case("LN_STAT L1023", "LN_STAT", k=V.f32(1.0 / L))
    for r in range(Rr):
        V.g_ln_stat(cs, xs[r], amax[r])
    B.op(cs, Rr, L, 0, B.operand(xs, SH_E, F_BF16), rs=B.operand(amax, SH_R, F_BF16))
    cs = V.Case("GELU L1023", "GELU")
    for r in range(Rr):
        V.g_unary(cs, R.op_gelu, xs[r])
    B.op(cs, Rr, L, 16, B.operand(xs, SH_E, F_BF16))

    # softmax over 867 keys (2 rows per lane, R = 13: chunks 8 + 5 with lanes 2 2 1 0), mask C, max R
    Rr, L = 13, 867
    xs = bf(Rr * L, -3, 4).reshape(Rr, L)
    mask = rng.random(L) < 0.9
    mask[0] = True
    mx = [int(V.row_max(xs[r], mask)) for r in range(Rr)]
    cs = V.Case("SMAX_SUM L867", "SMAX_SUM")
    inv = [V.g_smax_sum(cs, xs[r], mask) for r in range(Rr)]
    B.op(cs, Rr, L, 0, B.operand(xs, SH_E, F_BF16), rs=B.operand(mx, SH_R, F_SUM, FIELD_MAX),
         mask=B.operand(mask.astype(np.int64), SH_C, F_BF16))
    cs = V.Case("SMAX_OUT L867", "SMAX_OUT")
    for r in range(Rr):
        V.g_smax_out(cs, xs[r], mask, inv[r])
    B.op(cs, Rr, L, 16, B.operand(xs, SH_E, F_BF16), rs=B.operand(mx, SH_R, F_BF16),
         mask=B.operand(mask.astype(np.int64), SH_C, F_BF16), c=B.operand(inv, SH_R, F_FP32))

    # RMS_STAT across lanes (L = 51, 32 rows per lane, R = 777) then RMS_APPLY
    Rr, L = 777, 51
    xs = bf(Rr * L, -8, 8).reshape(Rr, L)
    cs = V.Case("RMS_STAT R777 L51", "RMS_STAT", k=V.f32(1.0 / L))
    amax = [int(R.row_amax_code(xs[r][None])[0]) for r in range(Rr)]
    rv = [V.g_rms_stat(cs, xs[r]) for r in range(Rr)]
    B.op(cs, Rr, L, 0, B.operand(xs, SH_E, F_BF16), rs=B.operand(amax, SH_R, F_SUM, FIELD_AMAX))
    gain = fp(L, -2, 1, pos=True)
    cs = V.Case("RMS_APPLY R777 L51", "RMS_APPLY")
    for r in range(Rr):
        V.g_rms_apply(cs, xs[r], rv[r], gain)
    B.op(cs, Rr, L, 16, B.operand(xs, SH_E, F_BF16), c=B.operand(rv, SH_R, F_SUM, FIELD_RS0),
         d=B.operand(gain, SH_C, F_FP32))

    # SILU with L = 1 (512 rows per lane, one summary per element), R = 3000
    Rr, L = 3000, 1
    xs = bf(Rr * L, -5, 5).reshape(Rr, L)
    cs = V.Case("SILU L1", "SILU")
    for r in range(Rr):
        V.g_unary(cs, R.op_silu, xs[r])
    B.op(cs, Rr, L, 16, B.operand(xs, SH_E, F_BF16))

    # 32-bit outputs: ROPE_A (L = 64, R = 100: one partial chunk), EULER (L = 32, R = 600)
    Rr, L = 100, 64
    xs, cos = bf(Rr * L, -4, 4).reshape(Rr, L), fp(Rr * L, -3, 0).reshape(Rr, L)
    cs = V.Case("ROPE_A R100 L64", "ROPE_A")
    for r in range(Rr):
        V.g_rope_a(cs, xs[r], cos[r])
    B.op(cs, Rr, L, 32, B.operand(xs, SH_E, F_BF16), c=B.operand(cos, SH_E, F_FP32))
    Rr, L = 600, 32
    v, xt = fp(Rr * L, -4, 3).reshape(Rr, L), fp(Rr * L, -2, 2).reshape(Rr, L)
    cs = V.Case("EULER R600 L32", "EULER", k=V.f32(-0.1))
    for r in range(Rr):
        V.g_euler(cs, xt[r], v[r])
    B.op(cs, Rr, L, 32, B.operand(v, SH_E, F_FP32), e=B.operand(xt, SH_E, F_FP32))

    # DEQUANT with bias, odd L = 37 and 333 rows (int32 sums sign-extended, s_row R fp32, s_col / bias C)
    Rr, L = 333, 37
    acc = rng.integers(-(1 << 27), 1 << 27, (Rr, L)).astype(np.int64)
    s_row, s_col, bias = fp(Rr, -14, -6, pos=True), fp(L, -8, -2, pos=True), fp(L, -6, 1)
    cs = V.Case("DEQUANT bias R333 L37", "DEQUANT", bias_en=1)
    for r in range(Rr):
        V.g_dequant(cs, acc[r], int(s_row[r]), s_col, bias)
    B.op(cs, Rr, L, 16, B.operand(acc, SH_E, F_INT32), c=B.operand(s_row, SH_R, F_FP32),
         d=B.operand(s_col, SH_C, F_FP32), e=B.operand(bias, SH_C, F_FP32))

    # SMAX_Q8 (int8 out) with L = 7, R = 1500 (256 rows per lane at N_LANE 2, element counts odd)
    Rr, L = 1500, 7
    xs = bf(Rr * L, -3, 4).reshape(Rr, L)
    mask = rng.random(L) < 0.8
    mask[0] = True
    mx = [int(V.row_max(xs[r], mask)) for r in range(Rr)]
    cs = V.Case("SMAX_Q8 R1500 L7", "SMAX_Q8")
    for r in range(Rr):
        V.g_smax_q8(cs, xs[r], mask)
    B.op(cs, Rr, L, 8, B.operand(xs, SH_E, F_BF16), rs=B.operand(mx, SH_R, F_BF16),
         mask=B.operand(mask.astype(np.int64), SH_C, F_BF16))

    B.words.append(0)                                   # END
    B.mem.put(PROG_BASE, B.words, 128)
    return B



def build_wide(seed: int) -> Builder:
    """Rows wider than 2048 elements, which a node needs SLOT_BITS >= 12 for: the full-size pi0 expert MLP
    streams 4096-wide rows (gate / up DEQUANT, GEGLU, the QUANT of their product, whose row amax has to cover
    the whole row and so cannot be split into column halves).  At SLOT_BITS = 12 a 4096-element row is one
    row per lane, so R = 51 is 12 full chunks of 4 lanes and a partial one; the L = 2048 ADD is 2 rows per
    lane.  At SLOT_BITS = 11 the node must reject these lengths (o_error): that is this suite's negative
    control.  Same program format and layout as build()."""
    rng = np.random.default_rng(seed + 3000)
    B = Builder(rng)
    bf = lambda n, lo=-6, hi=6: V.rbf16(rng, n, lo, hi)                       # noqa: E731
    fp = lambda n, lo=-4, hi=4, pos=False: V.rfp32(rng, n, lo, hi, pos=pos)  # noqa: E731
    Rr, L = 51, 4096

    # gate / up dequant: int32 sums, s_row per token (R, fp32), s_col per channel (C, fp32)
    acc_g = rng.integers(-(1 << 26), 1 << 26, (Rr, L)).astype(np.int64)
    acc_u = rng.integers(-(1 << 26), 1 << 26, (Rr, L)).astype(np.int64)
    s_row, s_cg, s_cu = fp(Rr, -14, -8, pos=True), fp(L, -8, -3, pos=True), fp(L, -8, -3, pos=True)
    gb, ub = [], []
    for acc, s_col, out, name in ((acc_g, s_cg, gb, "gate"), (acc_u, s_cu, ub, "up")):
        cs = V.Case(f"DEQUANT {name} R{Rr} L{L}", "DEQUANT")
        for r in range(Rr):
            V.g_dequant(cs, acc[r], int(s_row[r]), s_col)
        out.append(R.op_dequant(acc, np.asarray(s_row, np.uint32), np.asarray(s_col, np.uint32), None, "bf16")[0])
        B.op(cs, Rr, L, 16, B.operand(acc, SH_E, F_INT32), c=B.operand(s_row, SH_R, F_FP32),
             d=B.operand(s_col, SH_C, F_FP32))
    gs, us = gb[0], ub[0]

    # GEGLU over the two 4096-wide rows
    cs = V.Case(f"GEGLU R{Rr} L{L}", "GEGLU")
    for r in range(Rr):
        V.g_geglu(cs, gs[r], us[r])
    B.op(cs, Rr, L, 16, B.operand(gs, SH_E, F_BF16), d=B.operand(us, SH_E, F_BF16))

    # QUANT of a 4096-wide row: its amax must be the whole row's
    xs = bf(Rr * L, -7, 7).reshape(Rr, L)
    amax = [int(R.row_amax_code(xs[r][None])[0]) for r in range(Rr)]
    cs = V.Case(f"QUANT R{Rr} L{L}", "QUANT")
    for r in range(Rr):
        V.g_quant(cs, xs[r], amax[r])
    B.op(cs, Rr, L, 8, B.operand(xs, SH_E, F_BF16), rs=B.operand(amax, SH_R, F_SUM, FIELD_AMAX))

    # ADD at L = 4096 (1 row per lane) and at L = 2048 (2 rows per lane, R = 13: a partial chunk)
    for Rr2, L2 in ((Rr, L), (13, 2048)):
        xa, xb = bf(Rr2 * L2).reshape(Rr2, L2), bf(Rr2 * L2, -8, 8).reshape(Rr2, L2)
        cs = V.Case(f"ADD R{Rr2} L{L2}", "ADD")
        for r in range(Rr2):
            V.g_add(cs, xa[r], xb[r])
        B.op(cs, Rr2, L2, 16, B.operand(xa, SH_E, F_BF16), b=B.operand(xb, SH_E, F_BF16))

    B.words.append(0)                                   # END
    B.mem.put(PROG_BASE, B.words, 128)
    return B

def build_pdq(seed: int) -> Builder:
    """Fusion 2 (vu_pdq.sv): GEGLU records that take the gate and up int32 accumulators and dequantise them in front of
    the lane -- x = gate acc (E, int32), d = up acc (E, int32), c = s_row (R), e = s_col of the gate columns (C),
    b = s_col of the up columns (C) -- against DEQUANT, DEQUANT, GEGLU of the reference.  Odd and 4096-wide rows (SLOT_BITS
    12), with ordinary records in between so the PDQ flag must switch on and off.  pdq_q: with the fused QUANT tail."""
    rng = np.random.default_rng(seed + 5000)
    B = Builder(rng)
    bf = lambda n, lo=-6, hi=6: V.rbf16(rng, n, lo, hi)                       # noqa: E731
    fp = lambda n, lo=-4, hi=4, pos=False: V.rfp32(rng, n, lo, hi, pos=pos)  # noqa: E731

    def pdq_geglu(Rr, L, span):
        acc_g = rng.integers(-(1 << span), 1 << span, (Rr, L)).astype(np.int64)
        acc_u = rng.integers(-(1 << span), 1 << span, (Rr, L)).astype(np.int64)
        acc_g[0, :3] = 0                                   # zeros through norm64
        s_row, s_cg, s_cu = fp(Rr, -14, -8, pos=True), fp(L, -8, -3, pos=True), fp(L, -8, -3)
        gc = R.op_dequant(acc_g, np.asarray(s_row, np.uint32), np.asarray(s_cg, np.uint32), None, "bf16")[0]
        uc = R.op_dequant(acc_u, np.asarray(s_row, np.uint32), np.asarray(s_cu, np.uint32), None, "bf16")[0]
        cs = V.Case(f"GEGLU(PDQ) R{Rr} L{L}", "GEGLU")
        for r in range(Rr):
            V.g_geglu(cs, gc[r], uc[r])
        B.op(cs, Rr, L, 16, B.operand(acc_g, SH_E, F_INT32), d=B.operand(acc_u, SH_E, F_INT32),
             c=B.operand(s_row, SH_R, F_FP32), e=B.operand(s_cg, SH_C, F_FP32), b=B.operand(s_cu, SH_C, F_FP32),
             pdq=True)

    def plain_add(Rr, L):
        xa, xb = bf(Rr * L).reshape(Rr, L), bf(Rr * L, -8, 8).reshape(Rr, L)
        cs = V.Case(f"ADD R{Rr} L{L}", "ADD")
        for r in range(Rr):
            V.g_add(cs, xa[r], xb[r])
        B.op(cs, Rr, L, 16, B.operand(xa, SH_E, F_BF16), b=B.operand(xb, SH_E, F_BF16))

    if SMALL_PDQ:                                       # pdq_small: the quick debug set
        pdq_geglu(3, 8, 20)
        plain_add(2, 8)
        pdq_geglu(2, 37, 26)
    else:
        pdq_geglu(9, 37, 26)
        plain_add(5, 37)
        pdq_geglu(33, 256, 30)
        pdq_geglu(51, 4096, 26)
        plain_add(13, 2048)
        pdq_geglu(7, 5, 12)
    B.words.append(0)                                   # END
    B.mem.put(PROG_BASE, B.words, 128)
    return B


def build_attn(seed: int) -> Builder:
    """Attention cases: the rotate-half partner read of RoPE pass B (w0[7]) and the 255-level softmax output
    (SMAX_Q8 with b_fp32 = 1: uint8 P codes for the chain's unsigned multiplier operand)."""
    rng = np.random.default_rng(seed + 2000)
    B = Builder(rng)
    bf = lambda n, lo=-6, hi=6: V.rbf16(rng, n, lo, hi)                       # noqa: E731
    fp = lambda n, lo=-4, hi=4, pos=False: V.rfp32(rng, n, lo, hi, pos=pos)  # noqa: E731

    # RoPE over head rows: pass A (x cos) then pass B on the rotate-half partner of the SAME operand
    for Rr, L in ((24, 256), (100, 64)):
        h = L // 2
        xs = bf(Rr * L, -4, 4).reshape(Rr, L)
        cos, sgn = fp(Rr * L, -3, 0).reshape(Rr, L), fp(Rr * L, -3, 0).reshape(Rr, L)
        cs = V.Case(f"ROPE_A L{L}", "ROPE_A")
        pa = np.array([V.g_rope_a(cs, xs[r], cos[r]) for r in range(Rr)])
        B.op(cs, Rr, L, 32, B.operand(xs, SH_E, F_BF16), c=B.operand(cos, SH_E, F_FP32))
        part = np.concatenate([xs[:, h:], xs[:, :h]], 1)          # what the node must read for itself
        cs = V.Case(f"ROPE_B rot L{L}", "ROPE_B")
        for r in range(Rr):
            V.g_rope_b(cs, part[r], sgn[r], pa[r])
        B.op(cs, Rr, L, 16, B.operand(xs, SH_E, F_BF16), c=B.operand(sgn, SH_E, F_FP32),
             e=B.operand(pa, SH_E, F_FP32), x_rot=True)

    # 255-level softmax: 592 keys (the chip key layout) and many short rows
    for Rr, L in ((13, 592), (300, 7)):
        xs = bf(Rr * L, -3, 4).reshape(Rr, L)
        mask = rng.random(L) < 0.9
        mask[0] = True
        mx = [int(V.row_max(xs[r], mask)) for r in range(Rr)]
        cs = V.Case(f"SMAX_U8 R{Rr} L{L}", "SMAX_Q8", b_fp32=1)
        for r in range(Rr):
            V.g_smax_u8(cs, xs[r], mask)
        B.op(cs, Rr, L, 8, B.operand(xs, SH_E, F_BF16), rs=B.operand(mx, SH_R, F_BF16),
             mask=B.operand(mask.astype(np.int64), SH_C, F_BF16))

    B.words.append(0)                                   # END
    B.mem.put(PROG_BASE, B.words, 128)
    return B


def build_smaxpair(seed: int, variant: str = "out_q8") -> Builder:
    """Silicon isolation of the base suite's ops 13-14 (2026-09-17: SMAX_OUT's half-filled last element beat
    was found merged into SMAX_Q8's first beat on the S0 silicon while everything else matched).
    variant: out_q8 = SMAX_OUT then SMAX_Q8 (the failing pair); out = SMAX_OUT alone; out_add = SMAX_OUT then an
    ADD; q8_out = the pair the other way round."""
    rng = np.random.default_rng(seed)
    B = Builder(rng)
    bf = lambda n, lo=-6, hi=6: V.rbf16(rng, n, lo, hi)                       # noqa: E731
    Rr, L = 8, 67
    xs = bf(Rr * L, -3, 4).reshape(Rr, L)
    mask = rng.random(L) < 0.8
    mask[0] = True
    mx = [int(V.row_max(xs[r], mask)) for r in range(Rr)]
    cs = V.Case("SMAX_SUM", "SMAX_SUM")
    inv = [V.g_smax_sum(cs, xs[r], mask) for r in range(Rr)]      # the golden's 1/S, fed as an operand

    def smax_out():
        cs = V.Case("SMAX_OUT", "SMAX_OUT")
        for r in range(Rr):
            V.g_smax_out(cs, xs[r], mask, inv[r])
        B.op(cs, Rr, L, 16, B.operand(xs, SH_E, F_BF16), rs=B.operand(mx, SH_R, F_BF16),
             mask=B.operand(mask.astype(np.int64), SH_C, F_BF16), c=B.operand(inv, SH_R, F_FP32))

    def smax_q8():
        cs = V.Case("SMAX_Q8", "SMAX_Q8")
        for r in range(Rr):
            V.g_smax_q8(cs, xs[r], mask)
        B.op(cs, Rr, L, 8, B.operand(xs, SH_E, F_BF16), rs=B.operand(mx, SH_R, F_BF16),
             mask=B.operand(mask.astype(np.int64), SH_C, F_BF16))

    def add():
        R2, L2 = 5, 37
        a, b = bf(R2 * L2).reshape(R2, L2), bf(R2 * L2, -8, 8).reshape(R2, L2)
        cs = V.Case("ADD bf16", "ADD")
        for r in range(R2):
            V.g_add(cs, a[r], b[r])
        B.op(cs, R2, L2, 16, B.operand(a, SH_E, F_BF16), b=B.operand(b, SH_E, F_BF16))

    {"out_q8": [smax_out, smax_q8], "out": [smax_out], "out_add": [smax_out, add],
     "q8_out": [smax_q8, smax_out]}[variant]
    for f in {"out_q8": [smax_out, smax_q8], "out": [smax_out], "out_add": [smax_out, add],
              "q8_out": [smax_q8, smax_out]}[variant]:
        f()
    B.words.append(0)                                   # END
    B.mem.put(PROG_BASE, B.words, 128)
    return B


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--suite", choices=["base", "ml", "attn", "wide", "smax_out_q8", "smax_out", "smax_out_add", "smax_q8_out",
                                        "base_q", "ml_q", "attn_q", "wide_q", "pdq", "pdq_q", "pdq_small"], default="base",
                    help="*_q: the same suite with the fused QUANT tail (w0[118]) on every bf16-output op (vu_node_ml only)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    global QTAIL_ALL
    if a.suite.endswith("_q"):
        QTAIL_ALL = True
        a.suite = a.suite[:-2]
    global SMALL_PDQ
    if a.suite == "pdq_small":
        SMALL_PDQ, a.suite = True, "pdq"
    if a.suite.startswith("smax_"):
        B = build_smaxpair(a.seed, a.suite[5:])
    else:
        B = {"base": build, "ml": build_ml, "attn": build_attn, "wide": build_wide, "pdq": build_pdq}[a.suite](a.seed)
    out = Path(a.out)
    os.makedirs(out, exist_ok=True)
    B.mem.write(out / "mem_in.hex")
    B.exp.write(out / "mem_exp.hex")
    (out / "prog_base.txt").write_text(f"{PROG_BASE:x}\n")
    (out / "ops.txt").write_text("\n".join(B.ops) + "\n")
    print(f"vu_node golden: {len(B.ops)} ops, image {len(B.mem.beats)} beats, expected {len(B.exp.beats)} beats -> {out}")


if __name__ == "__main__":
    main()
