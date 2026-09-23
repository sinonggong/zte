#!/usr/bin/env python3
"""GDDR6 image and one chain-node program for tb_pi0_wide_gemm.sv: the expert's o_proj at its REAL width.

`paper/rtl/run_pi0_attn_sim.sh` runs a whole expert layer, but with the dimensions cut down (hidden 256,
2 heads x 64) so every image fits one chain node.  This generator does the opposite: one GEMM, no cutting
down.  o_proj of the trained checkpoint is (1024 out, 2048 in), so

    W  = K / 16                     = 128 words per column
    P  = 512 / W                    = 4 columns per stage        (a stage BRAM72K is 512 words)
    16 P                            = 64 columns per weight image
    N / 64                          = 16 COLUMN TILES

which is the widest shape the array ever has to split, and the one the cut-down layer test can only
imitate with `TILES=2`.  Each tile writes a 64-column block of the row-major 51 x 1024 result (OUT block
beats / gap), so the sixteen tiles interleave into one matrix without the host touching anything.

With `--nodes N` the work is also split by ROWS: the column tiles are dealt out over `--tile-groups`
nodes and the token rows over the rest, one program per node, all started together.  That matters
because splitting by column tiles alone leaves a node idle whenever a GEMM has fewer tiles than nodes
-- QK^T and PV are a single tile -- and a row split needs no new command: the opcode-6 column group
already takes (act_base, T) and OUT already takes a base.  This is the test that proves it at the
real shape.

Weights are the real `o_proj` slice of model.safetensors, quantised exactly as `pi0_layer_lower.py`
quantises it (per output channel MSE clip, SmoothQuant s_o folded in); activations are the captured
`ex.L<layer>.o_in` frame, quantised per token by the vector node's QUANT reference.  The expected values
are an int64 matmul, self-checked against a direct loop on a sample of entries.

Negative controls (`--neg`), each of which must make the run FAIL:
  blockgap   drop the OUT block beats / gap, so every tile writes over tile 0's block
  tileorder  swap the weight images of two column tiles
  blockbeats halve the OUT block beats, so the writer jumps in the middle of a row's records
  stageorder swap the stage images of two stages, which permutes two columns of every tile

`--feeder-rows` is not a negative control: the column group reloads the feeder min(M, rows left) rows at
a time until every row is done, so the result must not depend on M.  Running it at 3 instead of 4 proves
that, and is checked by the `all` suite.

Writes mem_in.hex, mem_exp.hex, stages.txt and info.txt into --out.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import vector_unit_ref as R                                                   # noqa: E402
import pi0_layer_lower as LL                                                  # noqa: E402
from colpar_tile_golden import op_load, op_colgroup, op_out, op_flush         # noqa: E402
from pi0_attn_golden import N_STAGE, weight_image, gap_of                     # noqa: E402
from vu_node_golden import Mem, Alloc                                         # noqa: E402

PROG_BASE = 0x0800_0000
IN_BASE = 0x1000_0000
OUT_BASE = 0x4000_0000


def direct(a_row: np.ndarray, w_row: np.ndarray) -> int:
    """the same dot product as a plain loop, for the self-check"""
    t = 0
    for k in range(a_row.shape[0]):
        t += int(a_row[k]) * int(w_row[k])
    return t


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--layer", type=int, default=0)
    ap.add_argument("--step", type=int, default=0, help="capture index: 0 = Euler step 0, 1 = step 9")
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--tokens", type=int, default=0, help="0 = every token of the frame (51)")
    ap.add_argument("--out-cols", type=int, default=0, help="0 = the real 1024 output channels")
    ap.add_argument("--neg", choices=("blockgap", "tileorder", "blockbeats", "stageorder"), default=None)
    ap.add_argument("--feeder-rows", type=int, default=0,
                    help="feeder rows per load (0 = the largest that fits); the result must not depend on it")
    ap.add_argument("--nodes", type=int, default=1, help="chain nodes to spread the GEMM over")
    ap.add_argument("--tile-groups", type=int, default=0,
                    help="of --nodes, how many split the COLUMN tiles (the rest split the ROWS); "
                         "0 = all of them, the column-only split the compiler emits today")
    a = ap.parse_args()

    z = np.load(LL.CAPTURE)
    ex = LL.load_layer(a.layer, "expert_layer", a.alpha, "exp", gemms=("o",))
    Wo = ex["o"]
    codes = Wo.codes                                          # (1024, 2048) int8 codes
    N, Kin = codes.shape
    if a.out_cols:
        codes = codes[:a.out_cols]
        N = a.out_cols

    # ---- activations: the captured o_proj input, smoothing folded out, quantised per token ----
    x = z[f"ex.L{a.layer}.o_in"][a.step].astype(np.float64) / ex["s_o"]
    if a.tokens:
        x = x[:a.tokens]
    T = x.shape[0]
    assert x.shape[1] == Kin, (x.shape, Kin)
    C = LL.Chip()
    aq, s_row = C.quant(R.bf16_codes(x))                      # (T, 2048) int8 codes
    acc = C.gemm(aq, codes)                                   # (T, N) int64

    W_in = Kin // 16
    P = 512 // W_in
    tiles = N // (N_STAGE * P)
    per = N // tiles
    assert tiles * per == N and per == N_STAGE * P, (N, tiles, per, P)
    assert P * W_in <= 512 and W_in <= 1023 and tiles > 1

    # a column tile's records land as a column block of the row-major T x N result
    bpr = (N_STAGE * 32 + 255) // 256                         # beats per record (one row pass = 16 sums)
    blk_beats = P * bpr
    gap = (N - per) * 4
    M = a.feeder_rows or min(T, 512 // W_in)
    assert 0 < M <= min(T, 512 // W_in)
    if a.neg == "blockbeats":
        blk_beats //= 2

    # self-check a sample of the expected values against a direct loop
    rng = np.random.default_rng(0)
    for r, c in [(0, 0), (T - 1, N - 1)] + [(int(rng.integers(T)), int(rng.integers(N))) for _ in range(6)]:
        assert int(acc[r, c]) == direct(aq[r], codes[c]), f"self-check ({r}, {c})"
    assert int(np.abs(acc).max()) < (1 << 31), "a record no longer fits the 32-bit write"

    mem, exp = Mem(), Mem()
    ain, aout = Alloc(IN_BASE), Alloc(OUT_BASE)

    act_base = ain(T * Kin * 8)
    mem.put(act_base, (aq & 0xFF).reshape(-1), 8)
    assert (act_base // 16) % 2 == 0 and (M * W_in) % 2 == 0

    img_base = []
    order = list(range(tiles))
    if a.neg == "tileorder":
        order[1], order[2] = order[2], order[1]
    for t in order:
        b = ain(N_STAGE * P * W_in * 16 * 8)
        img = weight_image(codes[t * per:(t + 1) * per]).reshape(N_STAGE, P * W_in * 16)
        if a.neg == "stageorder":
            img = img[[1, 0] + list(range(2, N_STAGE))]
        mem.put(b, img.reshape(-1) & 0xFF, 8)
        img_base.append(b)

    out_base = aout(T * N * 32)
    exp.put(out_base, (acc & 0xFFFFFFFF).reshape(-1), 32)

    # ---- deal the work out over the nodes: column-tile groups x row slices ----
    n_tg = a.tile_groups or a.nodes
    assert a.nodes % n_tg == 0, "--nodes must be a multiple of --tile-groups"
    n_rs = a.nodes // n_tg
    assert n_tg <= tiles and n_rs <= T, (n_tg, tiles, n_rs, T)

    def span(total, parts, i):
        # part i of `total` cut into `parts` nearly equal pieces -> (start, count)
        base_, rem = divmod(total, parts)
        start = i * base_ + min(i, rem)
        return start, base_ + (1 if i < rem else 0)

    progs = []
    for tg in range(n_tg):
        t0, nt = span(tiles, n_tg, tg)
        for rs in range(n_rs):
            r0, nr = span(T, n_rs, rs)
            prog = []
            for t in range(t0, t0 + nt):
                # this node's records are the nr x 64 block at (row r0, column 64 t) of the result
                prog += [op_out(out_base + r0 * N * 4 + t * per * 4,
                                0 if a.neg == "blockgap" else blk_beats,
                                0 if a.neg == "blockgap" else gap),
                         op_load(img_base[t], 1, N_STAGE, P * W_in),
                         op_colgroup(act_base + r0 * W_in * 16, nr, 0, W_in, min(M, nr), P, gap_of(W_in)),
                         op_flush()]
            prog.append(0)                                     # END
            progs.append(prog)
    for i, prog in enumerate(progs):
        mem.put(PROG_BASE + i * 0x4000, prog, 128)

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    mem.write(out / "mem_in.hex")
    exp.write(out / "mem_exp.hex")
    (out / "stages.txt").write_text(
        "".join(f"{i} {PROG_BASE + i * 0x4000:x}\n" for i in range(len(progs))))
    passes = tiles * T * P
    info = (f"o_proj L{a.layer} step {a.step}: {T} x {Kin} @ {Kin} x {N}\n"
            f"  W={W_in} words  P={P} columns/stage  M={M} feeder rows  tiles={tiles}  "
            f"block beats={blk_beats} gap={gap} bytes\n"
            f"  nodes={a.nodes} = {n_tg} tile groups x {n_rs} row slices, "
            f"commands/node={[len(x) for x in progs]}\n"
            f"  row passes={passes}  records={passes}  "
            f"result beats={len(exp.beats)}  input beats={len(mem.beats)}\n"
            f"  max|acc|=2^{np.log2(max(1, int(np.abs(acc).max()))):.1f}  neg={a.neg}\n")
    (out / "info.txt").write_text(info)
    print(info, end="")


if __name__ == "__main__":
    main()
