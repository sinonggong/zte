#!/usr/bin/env python3
"""GDDR6 image and expected beats for tb_pi0_linear.sv: one pi0 linear layer computed across node types.

  vector node 0  QUANT   x (T x K bf16) -> int8 codes (region QOUT) + per-row s_row (summary region QSUM)
  chain node     GEMM    weight tiles LOAD; one opcode-6 column group over QOUT (the int8 code region IS the
                         feeder layout: 16 bytes per word) -> int32 column sums, row-major (region CSUM)
  vector node 1  DEQUANT CSUM (int32, T x N) x s_row (QSUM summary field rs0) x s_col (C) + bias (C)
                         -> bf16 outputs (region DOUT) + summaries (DSUM)
Each node runs its own program from GDDR6.  Column assignment on the chain: column c = 16 p + s, so the
column-parallel node's row-major records are the matrix the vector node reads.  Expected values come from the
bit-exact vector-unit reference (vector_unit_ref.py via vector_unit_vectors.py) and an int64 matmul.
Writes into --out: mem_in.hex, mem_exp.hex (beat index, 256-bit beat), progs.txt (program bases), info.txt.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import vector_unit_ref as R          # noqa: E402
import vector_unit_vectors as V      # noqa: E402
from colpar_tile_golden import op_out, op_load, op_colgroup, op_flush   # noqa: E402
from vu_node_golden import Mem, Alloc, desc, K, SH_E, SH_R, SH_C, F_BF16, F_FP32, F_INT32, F_SUM, FIELD_RS0  # noqa: E402

N_STAGE = 16
PROG_VN0, PROG_CH, PROG_VN1 = 0x0800_0000, 0x0900_0000, 0x0A00_0000


def vu_record(op: int, rows: int, length: int, out_base: int, sum_base: int, x: int, b=None, c=None, d=None,
              e=None, rs=None, mask=None, k=0, b_fp32=0, bias_en=0, out_fp32=0) -> list[int]:
    unused = K(0)
    w0 = ((0xA << 124) | (out_base << 76) | (length << 60) | (rows << 40) | ((k & 0xFFFFFFFF) << 8)
          | (out_fp32 << 6) | (bias_en << 5) | (b_fp32 << 4) | op)
    return [w0, sum_base | (x << 42),
            (b if b is not None else unused) | ((c if c is not None else unused) << 48),
            (d if d is not None else unused) | ((e if e is not None else unused) << 48),
            (rs if rs is not None else unused) | ((mask if mask is not None else K(1)) << 48), 0]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--tokens", type=int, default=10)
    ap.add_argument("--k", type=int, default=64, help="input width (bytes per row; multiple of 16)")
    ap.add_argument("--passes", type=int, default=3, help="column passes: output width = 16 x passes")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    T, Kin, P = a.tokens, a.k, a.passes
    assert Kin % 16 == 0
    W = Kin // 16
    N = N_STAGE * P
    M = T if (T * W) % 2 == 0 and T * W <= 512 else 2 * max(1, 256 // W)
    assert (M * W) % 2 == 0 and M * W <= 512 and P * W <= 512 and T < 1024

    mem, exp = Mem(), Mem()
    ain, aout = Alloc(0x1000_0000), Alloc(0x4000_0000)

    # ---- inputs ----
    x = V.rbf16(rng, T * Kin, -6, 6).reshape(T, Kin)
    wt = rng.integers(-128, 128, size=(N, Kin)).astype(np.int64)
    s_col = V.rfp32(rng, N, -8, -2, pos=True)
    bias = V.rfp32(rng, N, -6, 1)

    # ---- vector node 0: QUANT ----
    amax = [int(R.row_amax_code(x[r][None])[0]) for r in range(T)]
    cq = V.Case("QUANT", "QUANT")
    for r in range(T):
        V.g_quant(cq, x[r], amax[r])
    q = np.array([v for kd, v in cq.out if kd == 0], np.int64) & 0xFF
    q = np.where(q >= 128, q - 256, q).reshape(T, Kin)
    qsum = [v for kd, v in cq.out if kd == 1]
    s_row = [int(r_ & 0xFFFFFFFF) for r_ in qsum]
    x_base = ain(T * Kin * 16)
    mem.put(x_base, x.reshape(-1), 16)
    rs_base = ain(T * 16)
    mem.put(rs_base, amax, 16)
    QOUT, QSUM = aout(T * Kin * 8), aout(T * 64)
    exp.put(QOUT, q.reshape(-1) & 0xFF, 8)
    exp.put(QSUM, qsum, 64)
    prog0 = vu_record(13, T, Kin, QOUT, QSUM, desc(x_base, SH_E, F_BF16), rs=desc(rs_base, SH_R, F_BF16)) + [0]
    mem.put(PROG_VN0, prog0, 128)

    # ---- chain node: weights + one column group ----
    # stage s image: P columns of W words each, column p = weight row 16 p + s; images contiguous
    words_per_stage = P * W
    wt_base = ain(N_STAGE * words_per_stage * 128 + 512)
    stage_bytes = []
    for s in range(N_STAGE):
        img = np.concatenate([wt[N_STAGE * p + s] for p in range(P)])
        if words_per_stage % 2:
            img = np.concatenate([img, np.zeros(16, np.int64)])        # pad to a whole beat
        stage_bytes.append(img)
    mem.put(wt_base, np.concatenate(stage_bytes) & 0xFF, 8)
    acc = q @ wt.T                                                      # (T, N) exact
    assert np.abs(acc).max() < (1 << 31)
    CSUM = aout(T * N * 32)
    exp.put(CSUM, acc.reshape(-1) & 0xFFFFFFFF, 32)
    G = max(1, N_STAGE - W, 14 - W)
    progc = [op_out(CSUM), op_load(wt_base, 1, N_STAGE, words_per_stage),
             op_colgroup(QOUT, T, 0, W, M, P, G), op_flush(), 0]
    mem.put(PROG_CH, progc, 128)

    # ---- vector node 1: DEQUANT ----
    cd = V.Case("DEQUANT", "DEQUANT", bias_en=1)
    for r in range(T):
        V.g_dequant(cd, acc[r], s_row[r], s_col, bias)
    dout = [v for kd, v in cd.out if kd == 0]
    dsum = [v for kd, v in cd.out if kd == 1]
    scol_base, bias_base = ain(N * 32), ain(N * 32)
    mem.put(scol_base, s_col, 32)
    mem.put(bias_base, bias, 32)
    DOUT, DSUM = aout(T * N * 16), aout(T * 64)
    exp.put(DOUT, dout, 16)
    exp.put(DSUM, dsum, 64)
    prog1 = vu_record(14, T, N, DOUT, DSUM, desc(CSUM, SH_E, F_INT32), c=desc(QSUM, SH_R, F_SUM, FIELD_RS0),
                      d=desc(scol_base, SH_C, F_FP32), e=desc(bias_base, SH_C, F_FP32), bias_en=1) + [0]
    mem.put(PROG_VN1, prog1, 128)

    out = Path(a.out)
    os.makedirs(out, exist_ok=True)
    mem.write(out / "mem_in.hex")
    exp.write(out / "mem_exp.hex")
    (out / "progs.txt").write_text(f"{PROG_VN0:x}\n{PROG_CH:x}\n{PROG_VN1:x}\n")
    (out / "info.txt").write_text(
        f"T={T} K={Kin} W={W} N={N} P={P} M={M} G={G}\nQOUT={QOUT:#x} QSUM={QSUM:#x} CSUM={CSUM:#x} "
        f"DOUT={DOUT:#x} DSUM={DSUM:#x}\n|acc| max={int(np.abs(acc).max())}\n")
    print(f"pi0 linear golden: T={T} K={Kin} N={N}; image {len(mem.beats)} beats, expected {len(exp.beats)} beats -> {out}")


if __name__ == "__main__":
    main()
