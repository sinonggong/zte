#!/usr/bin/env python3
"""Numpy golden for the column-parallel INT8 GEMM chain (paper/rtl/mlp72_int8_colpar_chain.sv).

Same vector files and phases as paper/sw/mlp72_chain_golden.py, with these differences:
  rows.memh  5 lines per row: phase (2|3), activation start address, weight start address,
             words, idle cycles before the row.  The two starts are independent (the feeder
             walks the activation row, the stages walk their weight column), except on the
             two corner rows.  Idle gaps obey the RTL's row protocol: gap >= 1 and
             gap + words >= N_STAGE.
  exp.memh   N_STAGE lines per row: the exact 48-bit column sum of stage 1..N,
             sum_{k<K} sum_{j<16} act[a+k, j] * wt[s, b+k, j]
The first two phase-2 rows are corner rows: every product +16384, then every product -16256.
"""
import argparse
import os

import numpy as np

from mlp72_chain_golden import MASK48, pack_word


def stage_sums(act, wt, astart, wstart, k):
    return [int(x) for x in np.einsum("kj,skj->s", act[astart:astart + k], wt[:, wstart:wstart + k])]


def stage_sums_ref(act, wt, astart, wstart, k):
    out = []
    for st in range(wt.shape[0]):
        s = 0
        for w in range(k):
            for j in range(16):
                s += int(act[astart + w, j]) * int(wt[st, wstart + w, j])
        out.append(s)
    return out


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
                astart = wstart = r * k
            else:
                astart = base + int(rng.integers(0, half - k + 1))
                wstart = base + int(rng.integers(0, half - k + 1))
            gap = max(int(rng.integers(0, 4)), 1, n - k)
            rows.append((phase, astart, wstart, k, gap, stage_sums(act, wt, astart, wstart, k)))

    for i, (phase, ast, wst, kk, _, s) in enumerate(rows[:4] + rows[-2:]):
        act, wt = (act0, wt0) if phase == 2 else (act1, wt1)
        assert s == stage_sums_ref(act, wt, ast, wst, kk), f"golden self-check failed on row {i}"
    assert rows[0][5] == [k * 16 * 16384] * n
    assert rows[1][5] == [-k * 16 * 16256] * n
    assert all(abs(x) < (1 << 47) for r in rows for x in r[5])
    # stages must differ and the two starts must differ, or stage-sum / shared-address bugs go unseen
    assert len(set(rows[2][5])) == n or n == 1
    assert any(r[1] != r[2] for r in rows)

    os.makedirs(a.out, exist_ok=True)

    def wmem(name, values, digits):
        with open(os.path.join(a.out, name), "w") as f:
            for v in values:
                f.write(f"{v:0{digits}x}\n")

    wmem("act0.memh", (pack_word(act0[i]) for i in range(n_words)), 36)
    wmem("act1.memh", (pack_word(act1[i]) for i in range(n_words)), 36)
    wmem("wt0.memh", (pack_word(wt0[s, i]) for s in range(n) for i in range(n_words)), 36)
    wmem("wt1.memh", (pack_word(wt1[s, i]) for s in range(n) for i in range(n_words)), 36)
    wmem("rows.memh", (x for r in rows for x in r[:5]), 8)
    wmem("exp.memh", (x & MASK48 for r in rows for x in r[5]), 12)
    with open(os.path.join(a.out, "nrows.txt"), "w") as f:
        f.write(f"{len(rows)}\n")
    flat = [x for r in rows for x in r[5]]
    print(f"colpar golden: N_STAGE={n} words/row={k} rows={len(rows)} sums {len(flat)} "
          f"range [{min(flat)}, {max(flat)}] -> {a.out}")


if __name__ == "__main__":
    main()
