#!/usr/bin/env python3
"""Tile-level golden for colpar_row_sequencer.sv + mlp72_int8_colpar_chain.sv (paper/rtl/tb_colpar_tile.sv).

A tile runs M activation rows (feeder BRAM72K, row r at act_base + r*W) against P weight columns
per stage (stage s BRAM72K, column p at wt_base + p*W); a row is K = 16 W int8 values.  Writes into --out:
  act.memh     feeder BRAM72K content (2^ADDR_BITS words)
  wt.memh      stage 1..N BRAM72K content, stage s at lines (s-1)*2^ADDR_BITS ..
  tiles.memh   7 lines per tile: act_base, wt_base, W, M, P, G, 0
  exp.memh     one line per result in sequencer order: tile, row r (outer), column p, stage s
  ntiles.txt, npulses.txt (one o_sums_valid per row pass)
  act_b.memh   a second activation image (drawn after everything above, so the files above do not change)
  prog.memh    a colpar_node_ctrl.sv command program: OUT, LOAD feeder, LOAD stages, tiles 0-3, then the
               GEMM column groups of COLGROUPS on the second activation image (one command each; the
               node reloads its feeder per row group), FLUSH, END (128-bit commands)
  npulses_prog.txt  records the program produces
  exp_prog.memh the records the program must leave in GDDR6, in order
With --unsigned-act the activations are uint8 (0..255; the chain built with MULT_MODE = 5'h13), weights stay int8.
Expected values are an int64 matmul A @ Wt_s.T per stage -- the GEMM the tile must compute --
self-checked against a direct loop.  Idle gaps obey the chain's row protocol (G >= 1, W + G >= N) and the
result port's crossing rule for the test bench's 3:1 clock ratio and 3-flop synchroniser
(W + G >= MIN_PASS_CYCLES = 14 array cycles).
"""
import argparse
import os

import numpy as np

from mlp72_chain_golden import MASK48, pack_word

MIN_PASS_CYCLES = 14   # (3 + 1) fabric cycles at 3 array cycles each, + 2
BASE_ACT, BASE_ACT2, BASE_WT, BASE_OUT = 0x1000_0000, 0x1800_0000, 0x2000_0000, 0x3000_0000  # = tb_colpar_node_prog.sv
BASE_PROG = 0x0800_0000   # programs in GDDR6 (colpar_prog_fetch.sv): two commands per 256-bit beat
SWITCH_TILE = 4        # the program reloads the feeder with act_b before this tile


def op_load(base, first, nreg, nwords):
    return (1 << 124) | (nwords << 52) | (nreg << 47) | (first << 42) | base


def op_tile(ab, wb, w, m, p, g):
    return (2 << 124) | (g << 48) | (p << 38) | (m << 28) | (w << 18) | (wb << 9) | ab


def op_out(base, blk_beats=0, gap_bytes=0):
    """blk_beats != 0: the writer jumps by gap_bytes after every block, so these records are a column
    block of a wider row-major matrix (blk_beats = records per row of this node x 32 bits / 256)."""
    return (3 << 124) | (gap_bytes << 52) | (blk_beats << 42) | base


def op_flush():
    return 4 << 124


def op_colgroup(base, T, wb, w, M, p, g, stride=0):
    return (stride << 96) | (6 << 124) | (g << 91) | (p << 81) | (M << 71) | (w << 61) | (wb << 52) | (T << 42) | base


def op_loadx(base, first, nreg, nwords, nsegs, seg_step, tgt_step):
    """segmented BRAM72K images: segment i of target first + r at GDDR6 word base + r tgt_step + i seg_step"""
    assert base % 32 == 0 and nsegs * nwords <= 512 and seg_step < (1 << 14) and tgt_step < (1 << 14)
    return ((5 << 124) | (tgt_step << 86) | (seg_step << 72) | (nsegs << 62) | (nwords << 52) | (nreg << 47)
            | (first << 42) | base)


def slice_results(A, Wimg, n):
    """records of a GEMM over feeder rows A (T, W*16) and stage images Wimg[s] (P, W*16): row r, column p, stage s"""
    prods = [A @ Wimg[s].T for s in range(n)]
    return [int(prods[s][r, p]) for r in range(A.shape[0]) for p in range(Wimg[0].shape[0]) for s in range(n)]


# GEMM column groups run by one command each in prog.memh: (W, T rows, M rows per feeder load, P, G)
COLGROUPS = [(16, 29, 8, 3, 1), (3, 150, 10, 4, 1), (64, 7, 2, 2, 1), (1, 33, 16, 5, 1)]


def tile_results(act, wt, tile, n):
    ab, wb, w, m, p, _ = tile
    A = act[ab:ab + m * w].reshape(m, w * 16)
    prods = [A @ wt[s, wb:wb + p * w].reshape(p, w * 16).T for s in range(n)]
    return [int(prods[s][r, pp]) for r in range(m) for pp in range(p) for s in range(n)]


def direct(act, wt, s, ab, wb, w, r, p):
    t = 0
    for k in range(w):
        for j in range(16):
            t += int(act[ab + r * w + k, j]) * int(wt[s, wb + p * w + k, j])
    return t


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-stage", type=int, required=True)
    ap.add_argument("--addr-bits", type=int, default=9)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-pass", type=int, default=MIN_PASS_CYCLES,
                    help="minimum W + G (array cycles); below the result port's rule for negative controls")
    ap.add_argument("--addr-offset", type=lambda x: int(x, 0), default=0,
                    help="added to every GDDR6 address in prog.memh (one address window per node when nodes share a NAP)")
    ap.add_argument("--unsigned-act", action="store_true",
                    help="uint8 activations (feeder) x int8 weights, for a chain with MULT_MODE = 5'h13")
    a = ap.parse_args()

    rng = np.random.default_rng(a.seed)
    nw = 1 << a.addr_bits
    n = a.n_stage
    alo = 0 if a.unsigned_act else -128
    act = rng.integers(alo, alo + 256, size=(nw, 16), dtype=np.int64)
    wt = rng.integers(-128, 128, size=(n, nw, 16), dtype=np.int64)
    act[nw - 8:] = 255 if a.unsigned_act else -128   # corner tile: every product -32640 (uint8) / +16384 (int8)
    wt[:, nw - 8:] = -128

    # (W, M, P, G): K = 16, 48, 256, 1024 (expert), 512, 8192 (one full BRAM), and the corner tile
    cfgs = [(1, 7, 5, 1), (3, 9, 4, 2), (16, 8, 3, 1), (64, 8, 8, 1), (32, 5, 2, 3), (nw, 1, 1, 1), (8, 1, 1, 1)]
    tiles = []
    for i, (w, m, p, g) in enumerate(cfgs):
        g = max(g, 1, n - w, a.min_pass - w)
        assert g < 32 and m * w <= nw and p * w <= nw
        if i == len(cfgs) - 1:
            ab = wb = nw - 8
        else:
            ab = int(rng.integers(0, nw - m * w + 1))
            wb = int(rng.integers(0, nw - p * w + 1))
        tiles.append((ab, wb, w, m, p, g))

    exp = []
    npulses = 0
    for ti, (ab, wb, w, m, p, g) in enumerate(tiles):
        A = act[ab:ab + m * w].reshape(m, w * 16)
        prods = [A @ wt[s, wb:wb + p * w].reshape(p, w * 16).T for s in range(n)]   # (M, P) per stage
        for r in range(m):
            for pp in range(p):
                for s in range(n):
                    exp.append(int(prods[s][r, pp]))
                npulses += 1
        # self-check: first and last result of every tile, every stage
        for (r, pp) in ((0, 0), (m - 1, p - 1)):
            for s in range(n):
                assert int(prods[s][r, pp]) == direct(act, wt, s, ab, wb, w, r, pp), f"self-check tile {ti}"
    assert exp[-n:] == [8 * 16 * (-32640 if a.unsigned_act else 16384)] * n
    assert all(abs(x) < (1 << 47) for x in exp)

    os.makedirs(a.out, exist_ok=True)

    def wmem(name, values, digits):
        with open(os.path.join(a.out, name), "w") as f:
            for v in values:
                f.write(f"{v:0{digits}x}\n")

    act_b = rng.integers(alo, alo + 256, size=(nw, 16), dtype=np.int64)
    off = a.addr_offset
    # LOAD / LOADX carry at most 16 targets (a 5-bit count): deeper chains load their stages in pieces
    prog = [op_out(BASE_OUT + off), op_load(BASE_ACT + off, 0, 1, nw)]
    prog += [op_load(BASE_WT + off + s0 * nw * 16, 1 + s0, min(16, n - s0), nw) for s0 in range(0, n, 16)]
    exp_prog = []
    for t in tiles[:SWITCH_TILE]:
        prog.append(op_tile(*t))
        exp_prog += tile_results(act, wt, t, n)
    # column groups: T rows contiguous in GDDR6 (act_b from an even word ab); the node reloads its
    # feeder min(M, rows left) rows at a time from BRAM address 0
    for (w, T, M, p, g) in COLGROUPS:
        g = max(g, 1, n - w, a.min_pass - w)
        assert (M * w) % 2 == 0 and M * w <= nw and T * w <= nw and p * w <= nw and g < 32
        ab = 2 * int(rng.integers(0, (nw - T * w) // 2 + 1))
        wb = int(rng.integers(0, nw - p * w + 1))
        prog.append(op_colgroup(BASE_ACT2 + off + 16 * ab, T, wb, w, M, p, g))
        r0 = 0
        while r0 < T:
            m = min(M, T - r0)
            exp_prog += tile_results(act_b, wt, (ab + r0 * w, wb, w, m, p, g), n)
            r0 += m
    # interleaved stage load (LOADX: stage s, column p = the Wk words at act_b word k0 + (n p + s) Wk, i.e. column
    # c = n p + s of a column-contiguous matrix) and GEMMs over feeder row slices (opcode 6 stride S: row r = the W
    # words at act word o + r S)
    Wk = 2
    Pk = min(8, nw // (n * Wk))
    k0 = 2 * int(rng.integers(0, (nw - n * Pk * Wk) // 2 + 1))
    prog += [op_loadx(BASE_ACT2 + off + 16 * (k0 + s0 * Wk), 1 + s0, min(16, n - s0), Wk, Pk, n * Wk, Wk)
             for s0 in range(0, n, 16)]
    wimg = [np.stack([act_b[k0 + (n * p + s) * Wk:k0 + (n * p + s) * Wk + Wk].reshape(-1) for p in range(Pk)])
            for s in range(n)]
    for (w, S, T, M) in ((Wk, 6, 13, 5), (3, 8, 9, 4)):
        g = max(1, n - w, a.min_pass - w)
        o = 2 * int(rng.integers(0, (nw - T * S) // 2 + 1))
        A = np.stack([act[o + r * S:o + r * S + w].reshape(-1) for r in range(T)])
        if w == Wk:
            prog.append(op_colgroup(BASE_ACT + off + 16 * o, T, 0, w, M, Pk, g, stride=S))
            exp_prog += slice_results(A, wimg, n)
        else:   # columns past the interleaved image still hold the original weights
            P2 = 4
            wb = Pk * Wk + int(rng.integers(0, nw - Pk * Wk - P2 * w + 1))
            prog.append(op_colgroup(BASE_ACT + off + 16 * o, T, wb, w, M, P2, g, stride=S))
            exp_prog += slice_results(A, [wt[s, wb:wb + P2 * w].reshape(P2, w * 16) for s in range(n)], n)
    # column-block writes (OUT with block beats / gap): two weight tiles of one GEMM, each 16 P2 columns wide,
    # write into a single row-major T x (32 P2) matrix, the way several nodes or several tiles share a row
    Wc, P2, Tc, Mc = 2, 2, 12, 8
    bpr = (n * 32 + 255) // 256                       # beats per record (one row pass = n sums)
    cols = n * P2                                     # columns one tile covers
    blk_beats = P2 * bpr                              # beats of one row block
    gap = P2 * bpr * 32                               # the other tile's half of the row (records are beat padded)
    oc = 2 * int(rng.integers(0, (nw - Tc * Wc) // 2 + 1))
    A2 = np.stack([act[oc + r * Wc:oc + r * Wc + Wc].reshape(-1) for r in range(Tc)])
    prog.append(op_flush())        # an OUT changes the write base: drain first
    exp_blocks = []
    base0 = BASE_OUT + off + bpr * 32 * (len(exp_prog) // n)   # the records continue the sequential stream
    for t in range(2):
        wb = Pk * Wk + t * P2 * Wc      # past the words the interleaved LOADX overwrote
        prog.append(op_out(base0 + t * gap, blk_beats, gap))
        prog.append(op_colgroup(BASE_ACT + off + 16 * oc, Tc, wb, Wc, Mc, P2,
                                max(1, n - Wc, a.min_pass - Wc), stride=Wc))
        prog.append(op_flush())
        exp_blocks.append(slice_results(A2, [wt[s2, wb:wb + P2 * Wc].reshape(P2, Wc * 16) for s2 in range(n)], n))
    # the two blocks interleave row by row in GDDR6
    per_row = len(exp_blocks[0]) // Tc
    for r in range(Tc):
        for t in range(2):
            exp_prog += exp_blocks[t][r * per_row:(r + 1) * per_row]
    prog += [op_flush(), 0]
    assert tile_results(act, wt, tiles[0], n) == exp[:tiles[0][3] * tiles[0][4] * n]

    wmem("act.memh", (pack_word(act[i]) for i in range(nw)), 36)
    wmem("act_b.memh", (pack_word(act_b[i]) for i in range(nw)), 36)
    wmem("prog.memh", prog, 32)
    wmem("exp_prog.memh", (x & MASK48 for x in exp_prog), 12)
    wmem("wt.memh", (pack_word(wt[s, i]) for s in range(n) for i in range(nw)), 36)
    wmem("tiles.memh", (x for t in tiles for x in (*t, 0)), 8)
    wmem("exp.memh", (x & MASK48 for x in exp), 12)
    with open(os.path.join(a.out, "ntiles.txt"), "w") as f:
        f.write(f"{len(tiles)}\n")
    with open(os.path.join(a.out, "npulses.txt"), "w") as f:
        f.write(f"{npulses}\n")
    with open(os.path.join(a.out, "npulses_prog.txt"), "w") as f:
        f.write(f"{len(exp_prog) // n}\n")
    print(f"tile golden: N_STAGE={n} tiles={len(tiles)} row passes={npulses} results={len(exp)} -> {a.out}")


if __name__ == "__main__":
    main()
