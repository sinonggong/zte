#!/usr/bin/env python3
"""Numpy golden for the hard-block INT8 GEMM chain (paper/rtl/mlp72_int8_chain.sv).

Writes the vectors read by paper/rtl/tb_mlp72_int8_chain.sv into --out:
  act0.memh, act1.memh   feeder BRAM72K content (activation words), set 0 / set 1
  wt0.memh,  wt1.memh    stage 1..N BRAM72K content (weight words): stage s at lines
                         (s-1)*2^ADDR_BITS .. s*2^ADDR_BITS-1
  rows.memh              4 lines per row: phase (2|3), start address, words, idle cycles before
  exp.memh               exact row sums, 48-bit two's complement, in row order
  nrows.txt
Set 1 equals set 0 on the lower half of the address space and is fresh random data
on the upper half (the testbench rewrites the upper half while phase-2 rows run on
the lower half; phase-3 rows run on the upper half).

A word is 16 int8 values packed as {8'h0, v15..v8, 8'h0, v7..v0} (UG086 Table 107).
For a row of K words at addresses r..r+K-1 the chain computes exactly
    sum_{k<K} sum_{s=1..N} sum_{j<16} act[r+k, j] * wt[s, r+k, j]
(|sum| <= K * N * 16 * 16384; < 2^28 for K = 32, N = 16, inside 48 bits).
The first two phase-2 rows are corner rows: all products (-128)(-128) = +16384 and
all products (-128)(127) = -16256.
"""
import argparse
import os

import numpy as np

MASK48 = (1 << 48) - 1


def pack_word(v):
    u = [int(x) & 0xFF for x in v]
    lo = 0
    hi = 0
    for k in range(8):
        lo |= u[k] << (8 * k)
        hi |= u[8 + k] << (8 * k)
    return (hi << 72) | lo


def row_sum(act, wt, start, k):
    """Exact sum with int64 numpy (no overflow possible at these sizes)."""
    return int(np.einsum("kj,skj->", act[start:start + k], wt[:, start:start + k]))


def row_sum_ref(act, wt, start, k):
    """Independent pure-python loop, used as a self-check."""
    s = 0
    for w in range(start, start + k):
        for st in range(wt.shape[0]):
            for j in range(16):
                s += int(act[w, j]) * int(wt[st, w, j])
    return s


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-stage", type=int, required=True)
    ap.add_argument("--words", type=int, required=True, help="words per row (16 activations per word)")
    ap.add_argument("--rows", type=int, default=0, help="rows per phase (default 16 for K <= 16, else 8)")
    ap.add_argument("--addr-bits", type=int, default=9)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    rng = np.random.default_rng(a.seed)
    n_words = 1 << a.addr_bits
    half = n_words // 2
    n, k = a.n_stage, a.words
    assert 1 <= k and 2 * k <= half, "need 2*words <= half the address space (corner rows)"

    act0 = rng.integers(-128, 128, size=(n_words, 16), dtype=np.int64)
    wt0 = rng.integers(-128, 128, size=(n, n_words, 16), dtype=np.int64)
    act0[0:k] = -128
    wt0[:, 0:k] = -128
    act0[k:2 * k] = -128
    wt0[:, k:2 * k] = 127
    act1 = act0.copy()
    wt1 = wt0.copy()
    act1[half:] = rng.integers(-128, 128, size=(n_words - half, 16), dtype=np.int64)
    wt1[:, half:] = rng.integers(-128, 128, size=(n, n_words - half, 16), dtype=np.int64)

    rows_per_phase = a.rows or (16 if k <= 16 else 8)
    rows = []
    for phase, base, act, wt in ((2, 0, act0, wt0), (3, half, act1, wt1)):
        for r in range(rows_per_phase):
            if phase == 2 and r < 2:
                start = r * k
            else:
                start = base + int(rng.integers(0, half - k + 1))
            gap = int(rng.integers(0, 4))
            rows.append((phase, start, k, gap, row_sum(act, wt, start, k)))

    # self-checks
    for i, (phase, start, kk, _, s) in enumerate(rows[:4] + rows[-2:]):
        act, wt = (act0, wt0) if phase == 2 else (act1, wt1)
        assert s == row_sum_ref(act, wt, start, kk), f"golden self-check failed on row {i}"
    assert rows[0][4] == k * n * 16 * 16384
    assert rows[1][4] == -k * n * 16 * 16256
    assert all(abs(r[4]) < (1 << 47) for r in rows)

    os.makedirs(a.out, exist_ok=True)

    def wmem(name, values, digits):
        with open(os.path.join(a.out, name), "w") as f:
            for v in values:
                f.write(f"{v:0{digits}x}\n")

    wmem("act0.memh", (pack_word(act0[i]) for i in range(n_words)), 36)
    wmem("act1.memh", (pack_word(act1[i]) for i in range(n_words)), 36)
    wmem("wt0.memh", (pack_word(wt0[s, i]) for s in range(n) for i in range(n_words)), 36)
    wmem("wt1.memh", (pack_word(wt1[s, i]) for s in range(n) for i in range(n_words)), 36)
    wmem("rows.memh", (x for r in rows for x in r[:4]), 8)
    wmem("exp.memh", (r[4] & MASK48 for r in rows), 12)
    with open(os.path.join(a.out, "nrows.txt"), "w") as f:
        f.write(f"{len(rows)}\n")
    sums = [r[4] for r in rows]
    print(f"golden: N_STAGE={n} words/row={k} rows={len(rows)} sum range [{min(sums)}, {max(sums)}] -> {a.out}")


if __name__ == "__main__":
    main()
