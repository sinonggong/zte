#!/usr/bin/env python3
"""D3: the whole pi0 chunk as node programs on a configurable array (docs/PI0_E2E_HARDWARE_DELIVERY_20260917.md §4 D3).

Builds every computation of a chunk with the layer builders of pi0_chunk_layers.py -- SigLIP x 27 layers x 2 cameras
(from the captured layer-0 input or a synthetic patch embedding), post-LN + projector x 2, the PaliGemma prefix x 18
layers over the 525 compact tokens (image tokens from the projector, language rows from the host), and 10 Euler steps
of the action head + 18 expert layers -- into one GDDR6 map (pi0_chunk_layout.ChunkMem), then assigns every stage to
the nodes of the array and emits one program per node:

  * a chain stage is split at its column tiles (OUT + LOAD/LOADX + column groups + FLUSH) over the chain nodes of
    its kind (int8 or uint8 PV), longest-first onto the least-loaded node;
  * a vector stage is split by rows (multiples of 16 rows, so every operand stays beat aligned) over the vector nodes;
  * every part ends with a POST of its own flag and starts with a WAIT on each part of the stage before it, so the
    stages run in program order with no host in between (node_sync.sv); the host starts every node once.

Outputs (--out DIR): mem_in.hex / mem_exp.hex (the image and every beat the nodes must write, tb format), nodes.txt
(node index, kind, program base), stages.txt, regions.json, info.json (sizes, commands, traffic per node kind, chip vs
fp32 of the final actions), and with --host-image the binary host write list (D2).

Sizes: --siglip-layers / --prefix-layers / --expert-layers / --steps cut the chunk down for simulation (the layers
that run are the model's first ones, at their real width); --n-stage 16|32 picks the chain depth (one depth per
array in this version); --n-chain / --n-pv / --n-vec the array.

Usage: pi0_chunk_program.py --out DIR [--n-chain 4 --n-pv 1 --n-vec 2 --n-stage 16 --slot-bits 12]
                            [--siglip-layers 27 --prefix-layers 18 --expert-layers 18 --steps 10] [--vision captured]
                            [--hex] [--host-image]
"""
from __future__ import annotations

import argparse
import bisect
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import vector_unit_ref as R                                              # noqa: E402
import pi0_layer_lower as LL                                            # noqa: E402
import pi0_attn_golden as AG                                            # noqa: E402
from pi0_attn_golden import VN, CH8, CHU                                # noqa: E402
from pi0_chunk_layout import ChunkMem, BEAT, check_chain_program, check_vector_program   # noqa: E402
from pi0_chunk_tiler import tile_depth  # noqa: E402
from pi0_chunk_layers import ChunkBuild, Act, Stage, GemmaLayer, SiglipLayer, VisionEnds, ActionHead, add_pass  # noqa: E402
from vu_node_golden import WIDTH as SUM_WIDTH, F_SUM   # noqa: E402

KIND = {VN: "vector", CH8: "int8", CHU: "uint8"}
OUT_W = {1: 0, 3: 4, 5: 4, 10: 0, 12: 1, 13: 1, 15: 4}       # vector op -> output element bytes (else 2; DEQUANT by flag)
DESC_W = {0: 2, 1: 4, 2: 4, 3: 8}                            # bf16, fp32, int32, summary


# ====================================================================== splitting stages over nodes
def chain_groups(words: list[int]) -> list[list[int]]:
    """the column-tile groups of a chain stage: a group starts at an OUT that is followed by a LOAD or LOADX"""
    body = [w for w in words if w]                       # drop END
    groups, cur = [], []
    for i, c in enumerate(body):
        op = c >> 124
        nxt = (body[i + 1] >> 124) if i + 1 < len(body) else 0
        if op == 3 and nxt in (1, 5) and cur:
            groups.append(cur)
            cur = []
        cur.append(c)
    if cur:
        groups.append(cur)
    return groups


def chain_cost(group: list[int]) -> int:
    cost = 0
    d = None
    for c in group:
        op = c >> 124
        if op == 6:
            T, W, P = (c >> 42) & 1023, (c >> 61) & 1023, (c >> 81) & 1023
            if d is None:
                d = tile_depth(group) or 16
            cost += T * P * max(W + 1, d, 14) + T * W              # row passes (W + G cycles each) + feeder loads
        elif op == 1:
            cost += ((c >> 47) & 31) * ((c >> 52) & 1023)
        elif op == 5:
            cost += ((c >> 47) & 31) * ((c >> 52) & 1023) * ((c >> 62) & 1023)
    return cost


def lpt(items: list, n: int, cost) -> list[list]:
    """longest-processing-time first onto n bins; returns the bins (lists of items) in their original order"""
    order = sorted(range(len(items)), key=lambda i: -cost(items[i]))
    load = [0] * n
    bins: list[list[int]] = [[] for _ in range(n)]
    for i in order:
        b = min(range(n), key=lambda k: load[k])
        bins[b].append(i)
        load[b] += cost(items[i])
    return [[items[i] for i in sorted(b)] for b in bins]


def vec_records(words: list[int]) -> list[list[int]]:
    recs, k = [], 0
    while k + 5 < len(words) and words[k]:
        recs.append(words[k:k + 6])
        k += 6
    return recs


QT_FLAG = 1 << 118                 # vector record w0[118]: fused QUANT tail (vu_lane.sv i_qtail)
QT_OPS = {0, 2, 4, 6, 7, 8, 9, 11, 14}      # bf16-output element ops: ADD, RMS/LN_APPLY, ROPE_B, GELU, GEGLU, SILU, SMAX_OUT, DEQUANT


def rec_out_w(w0: int) -> int:
    """output element bytes of a vector record"""
    if w0 & QT_FLAG:
        return 1
    op = w0 & 0xF
    return (4 if (w0 >> 6) & 1 else 2) if op == 14 else OUT_W.get(op, 2)


def rec_rows(rec):
    return (rec[0] >> 40) & 0xFFFFF


def rec_len(rec):
    return (rec[0] >> 60) & 0xFFFF


def rec_slice(rec: list[int], r0: int, n: int) -> list[int]:
    """the record over rows [r0, r0 + n): every row-indexed base moves by r0 rows"""
    w0, w1, w2, w3, w4, w5 = rec
    op = w0 & 0xF
    L = rec_len(rec)
    out_w = rec_out_w(w0)
    blk, gap = w5 & 0x3FF, w5 >> 10
    out_base = (w0 >> 76) & ((1 << 42) - 1)
    if out_base:
        # LN_STAT writes one fp32 per row; every other op writes a row of L elements (blocked rows: block + gap)
        row_bytes = (blk * BEAT + gap) if blk else (out_w if op == 3 else L * out_w)
        out_base += r0 * row_bytes
        assert out_base % BEAT == 0, "row slice not beat aligned (out)"
    sum_base = w1 & ((1 << 42) - 1)
    sum_base += r0 * 8
    assert sum_base % BEAT == 0

    def shift(d):
        d &= (1 << 48) - 1
        shape, fmt = (d >> 42) & 3, (d >> 44) & 3
        addr = d & ((1 << 42) - 1)
        if shape == 0:
            addr += r0 * L * DESC_W[fmt]
        elif shape == 1:
            addr += r0 * DESC_W[fmt]
        else:
            return d
        assert addr % BEAT == 0, "row slice not beat aligned (operand)"
        return (d & ~((1 << 42) - 1)) | addr

    def shift_mask(d):
        d &= (1 << 48) - 1
        shape = (d >> 42) & 3
        if shape == 3:
            return d
        addr = (d & ((1 << 42) - 1)) + (r0 * L * 2 if shape == 0 else r0 * 2)
        assert addr % BEAT == 0
        return (d & ~((1 << 42) - 1)) | addr

    w0n = (w0 & ~((((1 << 42) - 1) << 76) | (0xFFFFF << 40))) | (out_base << 76) | (n << 40)
    w1n = sum_base | (shift(w1 >> 42) << 42)
    w2n = shift(w2) | (shift(w2 >> 48) << 48)
    w3n = shift(w3) | (shift(w3 >> 48) << 48)
    w4n = shift(w4) | (shift_mask(w4 >> 48) << 48)
    return [w0n, w1n, w2n, w3n, w4n, w5]


def slice_gran(rec: list[int]) -> int:
    """the smallest row granularity (4, 8 or 16) at which every row-indexed base of the record stays beat aligned:
    summaries are 8 bytes per row (4 rows a beat), fp32 row operands 4 (8 rows), bf16 row operands 2 (16 rows).
    Bases are linear in the first row, so checking one multiple checks them all."""
    for g in (4, 8, 16):
        try:
            rec_slice(rec, g, 1)
            return g
        except AssertionError:
            continue
    return 0


def split_vector(words: list[int], n: int) -> list[list[int]]:
    """n parts of a vector stage: every record split by rows over the parts, in multiples of the stage's row
    granularity for that row count (the coarsest its records need, so records over the same rows split at the
    same rows and do not create cross-node dependencies).  The expert's 51 token rows go 8 8 8 8 8 8 3 over 8
    nodes at granularity 4 (16 16 16 3 on 4 of them at 16)."""
    recs = vec_records(words)
    gran: dict[int, int] = {}
    for rec in recs:
        R_ = rec_rows(rec)
        gran[R_] = max(gran.get(R_, 0), slice_gran(rec) or (1 << 30))
    parts: list[list[int]] = [[] for _ in range(n)]
    for rec in recs:
        R_ = rec_rows(rec)
        g = gran[R_]
        if n == 1 or R_ < 2 * g:
            # whole: onto the least loaded part
            k = min(range(n), key=lambda i: sum(rec_rows(r) * rec_len(r) for r in vec_records(parts[i] + [0] * 6)))
            parts[k] += rec
            continue
        per = -(-(-(-R_ // n)) // g) * g
        r0 = 0
        for k in range(n):
            if r0 >= R_:
                break
            cnt = min(per, R_ - r0)
            parts[k] += rec_slice(rec, r0, cnt)
            r0 += cnt
    return parts


def rec_access(rec: list[int]) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    """(write ranges, read ranges) in bytes of one (sliced) vector record.  Writes are beat-rounded, because the
    writers move whole beats.  Reads: E-shaped operands cover rows x L elements, R-shaped one element per row,
    C-shaped one row of L (the same for every row); constants (K) read nothing."""
    w0, w1, w2, w3, w4, w5 = rec
    op = w0 & 0xF
    n, L = rec_rows(rec), rec_len(rec)
    out_w = rec_out_w(w0)
    blk, gap = w5 & 0x3FF, w5 >> 10
    rup = lambda x: -(-x // BEAT) * BEAT                                            # noqa: E731
    writes, reads = [], []
    out_base = (w0 >> 76) & ((1 << 42) - 1)
    if out_base:
        if blk and gap:
            # a column block of a wider matrix: each row writes blk beats, then the writer skips gap bytes
            stride = blk * BEAT + gap
            writes.extend((out_base + r * stride, out_base + r * stride + blk * BEAT) for r in range(n))
        else:
            row_bytes = blk * BEAT if blk else (out_w if op == 3 else L * out_w)
            writes.append((out_base, out_base + rup(n * row_bytes)))
    sum_base = w1 & ((1 << 42) - 1)
    writes.append((sum_base, sum_base + rup(n * 8)))

    def rd(d, width=None):
        d &= (1 << 48) - 1
        shape, fmt = (d >> 42) & 3, (d >> 44) & 3
        addr = d & ((1 << 42) - 1)
        w = width if width is not None else DESC_W[fmt]
        if shape == 0:
            reads.append((addr, addr + n * L * w))
        elif shape == 1:
            reads.append((addr, addr + n * w))
        elif shape == 2:
            reads.append((addr, addr + L * w))
    rd(w1 >> 42)
    rd(w2); rd(w2 >> 48); rd(w3); rd(w3 >> 48); rd(w4); rd(w4 >> 48, width=2)
    return writes, reads


def _overlap(a: list[tuple[int, int]], b: list[tuple[int, int]]) -> bool:
    """does any interval of a intersect any interval of b (half-open byte ranges); sort + sweep"""
    if not a or not b:
        return False
    a, b = sorted(a), sorted(b)
    i = j = 0
    while i < len(a) and j < len(b):
        if a[i][0] < b[j][1] and b[j][0] < a[i][1]:
            return True
        if a[i][1] <= b[j][1]:
            i += 1
        else:
            j += 1
    return False


def split_vector_safe(words: list[int], n: int) -> list[list[list[int]]]:
    """split_vector, made safe: the parts of one stage run concurrently on different nodes with no ordering between
    them, so a record may not read (or overwrite) bytes that ANOTHER part's earlier record of the same stage writes
    (or reads).  Where that would happen -- two records of one stage split at different row boundaries, e.g. a
    DEQUANT over T token rows feeding a RoPE over T x H head rows (2026-09-17: the expert's q RoPE read 4 tokens
    another vector node had not dequantised yet) -- the stage is cut before the dependent record, so the WAIT / POST
    barrier between the two sub-stages orders them.  Returns the sub-stages, each a list of part programs."""
    recs = vec_records(words)
    if n == 1 or len(recs) < 2:
        return [split_vector(words, n)]
    subs: list[list[list[int]]] = []
    start = 0
    while start < len(recs):
        cut = len(recs)
        for j in range(start + 1, len(recs)):
            parts = split_vector(sum(recs[start:j + 1], []) + [0], n)
            per_part = [vec_records(p + [0] * 6) for p in parts]
            # the record index inside each part: split_vector keeps record order, one slice per record per part
            hazard = False
            for pj, pr in enumerate(per_part):
                for qj, qr in enumerate(per_part):
                    if pj == qj or not pr or not qr:
                        continue
                    # the last record group of part pj (record j's slice, if it has one) against every earlier
                    # record slice of part qj
                    for ra in pr:
                        wa, rda = rec_access(ra)
                        for rb in qr:
                            wb, rdb = rec_access(rb)
                            if _overlap(rda, wb) or _overlap(wa, rdb) or _overlap(wa, wb):
                                hazard = True
                                if os.environ.get("PI0_HAZARD_DEBUG"):
                                    kind = "RAW" if _overlap(rda, wb) else "WAR" if _overlap(wa, rdb) else "WAW"
                                    print(f"hazard {kind}: op {ra[0] & 0xF} rows {rec_rows(ra)} x {rec_len(ra)} (part {pj}) vs "
                                          f"op {rb[0] & 0xF} rows {rec_rows(rb)} x {rec_len(rb)} (part {qj}); "
                                          f"wa={[(hex(a), b - a) for a, b in wa]} rda={[(hex(a), b - a) for a, b in rda]} "
                                          f"wb={[(hex(a), b - a) for a, b in wb]} rdb={[(hex(a), b - a) for a, b in rdb]}", file=sys.stderr)
                                break
                        if hazard:
                            break
                    if hazard:
                        break
                if hazard:
                    break
            if hazard:
                cut = j
                break
        subs.append(split_vector(sum(recs[start:cut], []) + [0], n))
        start = cut
    return subs


def chain_access(words: list[int]) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    """(write ranges, read ranges) in bytes of a chain part, bounding per command: LOAD reads its images, LOADX its
    segments' span, COLGROUP its activation rows, and an OUT covers the rows of the COLGROUP that follows it (a column
    block of a wider matrix: row stride blk x 32 + gap).  Bounding intervals are supersets, so a dependency found
    from them is never missed (at worst one more WAIT)."""
    writes, reads = [], []
    out = None                                            # (base, blk beats, gap) of the last OUT
    body = [c for c in words if c]
    for c in body:
        op = c >> 124
        base = c & ((1 << 42) - 1)
        if op == 1:
            nwords, nreg = (c >> 52) & 1023, (c >> 47) & 31
            reads.append((base, base + nreg * nwords * 16))
        elif op == 5:
            nwords, nreg, nsegs = (c >> 52) & 1023, (c >> 47) & 31, (c >> 62) & 1023
            seg_step, tgt_step = (c >> 72) & 0x3FFF, (c >> 86) & 0x3FFF
            reads.append((base, base + ((nreg - 1) * tgt_step + (nsegs - 1) * seg_step + nwords) * 16))
        elif op == 3:
            out = (base, (c >> 42) & 1023, (c >> 52) & ((1 << 72) - 1))
        elif op == 6:
            T, W, P = (c >> 42) & 1023, (c >> 61) & 1023, (c >> 81) & 1023
            stride = (c >> 96) & 0x3FF
            reads.append((base, base + ((T - 1) * (stride or W) + W) * 16))
            if out is not None:
                ob, blk, gap = out
                if blk:
                    writes.append((ob, ob + (T - 1) * (blk * BEAT + gap) + blk * BEAT))
                else:
                    d = max(1, tile_depth(body))
                    writes.append((ob, ob + -(-(T * d * P * 4) // BEAT) * BEAT))
    return writes, reads


class DepTracker:
    """which earlier parts a part must wait for: for every node, the latest earlier part whose writes overlap this
    part's reads or writes, or whose reads overlap this part's writes (a node runs its parts in order and POSTs after
    each, so waiting for that one covers all its earlier ones).  Accesses are bucketed by GDDR6 region."""

    def __init__(self, mem):
        self.mem = mem
        self.w: dict[int, list] = {}
        self.r: dict[int, list] = {}

    def _buckets(self, lo: int, hi: int):
        i = bisect.bisect_right(self.mem._bases, lo) - 1
        i = max(i, 0)
        while i < len(self.mem.regions) and self.mem.regions[i].base < hi:
            if self.mem.regions[i].end > lo:
                yield i
            i += 1

    def deps(self, node: int, writes, reads) -> dict[int, int]:
        out: dict[int, int] = {}

        def scan(ranges, table):
            for lo, hi in ranges:
                for b in self._buckets(lo, hi):
                    for l2, h2, n2, seq in table.get(b, ()):
                        if n2 != node and l2 < hi and lo < h2 and seq > out.get(n2, -1):
                            out[n2] = seq
        scan(reads, self.w)
        scan(writes, self.w)
        scan(writes, self.r)
        return out

    def add(self, node: int, seq: int, writes, reads) -> None:
        for ranges, table in ((writes, self.w), (reads, self.r)):
            for lo, hi in ranges:
                for b in self._buckets(lo, hi):
                    table.setdefault(b, []).append((lo, hi, node, seq))


def chain_slice_rows(words: list[int], r0: int, n: int, T: int) -> list[int] | None:
    """a chain GEMM stage over token rows [r0, r0 + n) only (every tile group: one OUT, its LOADs, one COLGROUP of T
    rows, FLUSH): the COLGROUP's activation base and row count and the OUT's base move by r0 rows.  None when the
    stage does not have that shape (LOADX columns from activations, several row slices, another row count)."""
    out = []
    for g in chain_groups(words + [0]):
        cg = [c for c in g if c >> 124 == 6]
        oc = [c for c in g if c >> 124 == 3]
        if len(cg) != 1 or len(oc) != 1 or any(c >> 124 == 5 for c in g):
            return None
        c = cg[0]
        Tg, W, P = (c >> 42) & 1023, (c >> 61) & 1023, (c >> 81) & 1023
        stride = (c >> 96) & 0x3FF
        if Tg != T:
            return None
        o = oc[0]
        blk, gap = (o >> 42) & 1023, (o >> 52) & ((1 << 72) - 1)
        row_out = (blk * BEAT + gap) if blk else tile_depth(g) * P * 4
        for c2 in g:
            op = c2 >> 124
            if op == 3:
                nb = (c2 & ((1 << 42) - 1)) + r0 * row_out
                assert nb % BEAT == 0, "row block not beat aligned (OUT)"
                c2 = (c2 & ~((1 << 42) - 1)) | nb
            elif op == 6:
                ab = (c2 & ((1 << 42) - 1)) + r0 * (stride or W) * 16
                assert ab % BEAT == 0, "row block not beat aligned (COLGROUP)"
                # keep the tiler's group size: with the double-buffered feeder (bit 106) it is <= 256 / W, and
                # recomputing 512 / W here let a row group overwrite the half being run (s1q / s1r, 09-18: the
                # prefix o_gemm row blocks wrong on silicon); without it this equals min(n, 512 // W)
                M = min(n, (c2 >> 71) & 1023)
                c2 = (c2 & ~(((1 << 42) - 1) | (1023 << 42) | (1023 << 71))) | ab | (n << 42) | (M << 71)
            out.append(c2)
    return out + [0]


def row_blocks(stages: list, names: tuple, T: int, cuts: list[int]) -> list:
    """the stages named in `names` (a row-local segment of a layer) as row blocks [cuts[k], cuts[k+1]), interleaved
    stage by stage (block 0 of stage i, block 1 of stage i, block 0 of stage i+1, ...): under --sync deps a vector
    stage of block 0 runs while the chains do block 1 of the GEMM before it.  A stage whose records or tiles are not
    all T rows stays whole, in place."""
    out, seg = [], []
    for st in stages:
        if st.name not in names:
            out += row_block_seg(seg, T, cuts)
            seg = []
            out.append(st)
        else:
            seg.append(st)
    return out + row_block_seg(seg, T, cuts)


def row_block_seg(seg: list, T: int, cuts: list[int]) -> list:
    if not seg:
        return []
    blocks = []                                         # per stage: [block words] or None (whole)
    for st in seg:
        if st.kind == VN:
            recs = vec_records(st.words)
            if recs and all(rec_rows(r) == T for r in recs):
                blocks.append([sum((rec_slice(r, c0, c1 - c0) for r in recs), []) + [0] for c0, c1 in zip(cuts, cuts[1:])])
            else:
                blocks.append(None)
        else:
            bl = [chain_slice_rows([w for w in st.words if w], c0, c1 - c0, T) for c0, c1 in zip(cuts, cuts[1:])]
            blocks.append(None if any(b is None for b in bl) else bl)
    # order: runs of consecutive same-resource stages (vector / chain), each run block by block -- the vector node
    # does block 0's dequant + norm, then block 1's, while the chains do the GEMM of block 1, then block 0's next GEMM
    out, run, nb = [], [], len(cuts) - 1
    for i, (st, bl) in enumerate(zip(seg, blocks)):
        run.append((st, bl))
        nxt = seg[i + 1] if i + 1 < len(seg) else None
        if nxt is None or (nxt.kind == VN) != (st.kind == VN):
            for k in range(nb):
                for st2, bl2 in run:
                    if bl2 is None:
                        if k == 0:
                            out.append(st2)
                    else:
                        out.append(Stage(st2.kind, bl2[k], f"{st2.name}@r{k}", st2.layer))
            run = []
    return out


def fuse_quant(stages: list, mem) -> dict:
    """Operator fusion on the vector programs: a bf16-output element op X followed, in the same stage, by the QUANT
    of exactly its output (x = X's element region, rs = X's summary amax, same rows x L) becomes ONE record -- X
    with the fused-QUANT flag, writing the QUANT's int8 region and s_row summaries (vu_lane.sv's QUANT tail: bit
    for bit what the two records wrote there).  Only when nothing else in the whole chunk reads X's bf16 output or
    its summaries, which are then no longer written (their expectations are dropped).  One lane pass and one
    16-bit write + read of the row disappear per fused pair."""
    M42 = (1 << 42) - 1
    los, his = [], []
    for st in stages:
        if st.kind == VN:
            for r in vec_records(st.words):
                for lo, hi in rec_access(r)[1]:
                    los.append(lo); his.append(hi)
        else:
            for lo, hi in chain_access(st.words)[1]:
                los.append(lo); his.append(hi)
    los.sort(); his.sort()
    readers = lambda lo, hi: bisect.bisect_left(los, hi) - bisect.bisect_right(his, lo)      # noqa: E731
    n_fused = n_quant = 0
    saved = 0
    for st in stages:
        if st.kind != VN:
            continue
        recs = vec_records(st.words)
        before = sum(rec_rows(r) * rec_len(r) for r in recs)
        drop = set()
        for qi, q in enumerate(recs):
            if (q[0] >> 124) != 0xA or (q[0] & 0xF) != 13:
                continue
            n_quant += 1
            xd, rsd = (q[1] >> 42) & ((1 << 48) - 1), q[4] & ((1 << 48) - 1)
            if (xd >> 42) & 3 != 0 or (xd >> 44) & 3 != 0 or (rsd >> 42) & 3 != 1 or (rsd >> 44) & 3 != 3 or (rsd >> 46) & 3 != 2:
                continue
            for xi in range(qi - 1, -1, -1):
                x = recs[xi]
                if xi in drop or (x[0] >> 124) != 0xA or x[0] & QT_FLAG:
                    continue
                if ((x[0] >> 76) & M42) != (xd & M42) or (x[1] & M42) != (rsd & M42):
                    continue
                op = x[0] & 0xF
                n_el = rec_rows(x) * rec_len(x)
                if (op not in QT_OPS or (op == 14 and (x[0] >> 6) & 1) or x[5] != 0 or rec_rows(x) != rec_rows(q)
                        or rec_len(x) != rec_len(q)):
                    break
                ob, sb = xd & M42, rsd & M42
                if readers(ob, ob + n_el * 2) != 1 or readers(sb, sb + rec_rows(x) * 8) != 1:
                    break
                # X takes the QUANT's output regions and block layout; the QUANT record goes
                recs[xi] = [(x[0] & ~(M42 << 76)) | (q[0] & (M42 << 76)) | QT_FLAG, (x[1] & ~M42) | (q[1] & M42),
                            x[2], x[3], x[4], q[5]]
                drop.add(qi)
                for base, nbytes in ((ob, n_el * 2), (sb, rec_rows(x) * 8)):
                    r = mem.region_of(base)
                    if r.exp_beats is not None:            # no longer written: not expected
                        r.exp_beats[(base - r.base) // BEAT: -(-(base - r.base + nbytes) // BEAT)] = False
                n_fused += 1
                saved += n_el
                break
        if drop:
            st.words = sum((r for i, r in enumerate(recs) if i not in drop), []) + [0]
            p0 = getattr(st, "passes0", None) or before
            st.passes0 = p0
            st.fuse = (before - sum(rec_rows(recs[i]) * rec_len(recs[i]) for i in drop)) / p0
    return dict(quant_records=n_quant, fused=n_fused, element_passes_saved=saved)


PDQ_FLAG = 1 << 119                # vector record w0[119]: pre-dequant x and d (vu_pdq.sv)


def fuse_pdq(stages: list, mem) -> dict:
    """Operator fusion 2: DEQUANT(gate acc), DEQUANT(up acc), GEGLU in one stage become ONE GEGLU record that takes the
    int32 accumulators and dequantises them in front of the lane (vu_pdq.sv): x = gate acc, d = up acc, c = the shared
    s_row, e = the gate's s_col, b = the up's s_col.  Only when both DEQUANTs have no bias and a bf16 output of the
    GEGLU's rows x L, share their s_row descriptor, and nothing else reads their outputs or summaries (which are then
    no longer written).  Two lane passes and two 16-bit writes + reads of the rows disappear per fused triple."""
    M42, M48 = (1 << 42) - 1, (1 << 48) - 1
    los, his = [], []
    for st in stages:
        if st.kind == VN:
            for r in vec_records(st.words):
                for lo, hi in rec_access(r)[1]:
                    los.append(lo); his.append(hi)
        else:
            for lo, hi in chain_access(st.words)[1]:
                los.append(lo); his.append(hi)
    los.sort(); his.sort()
    readers = lambda lo, hi: bisect.bisect_left(los, hi) - bisect.bisect_right(his, lo)      # noqa: E731
    n_geglu = n_fused = saved = 0
    for st in stages:
        if st.kind != VN:
            continue
        recs = vec_records(st.words)
        before = sum(rec_rows(r) * rec_len(r) for r in recs)
        drop = set()
        for gi, g in enumerate(recs):
            if (g[0] >> 124) != 0xA or (g[0] & 0xF) != 8 or g[0] & PDQ_FLAG:
                continue
            n_geglu += 1
            xd, dd = (g[1] >> 42) & M48, g[3] & M48
            if any((v >> 42) & 3 != 0 or (v >> 44) & 3 != 0 for v in (xd, dd)):    # E-shaped bf16 operands only
                continue
            n, L = rec_rows(g), rec_len(g)

            def producer(addr):
                for i in range(gi - 1, -1, -1):
                    r = recs[i]
                    if i in drop or (r[0] >> 124) != 0xA:
                        continue
                    if ((r[0] >> 76) & M42) == addr:
                        ok = ((r[0] & 0xF) == 14 and not (r[0] >> 5) & 1 and not (r[0] >> 6) & 1 and not r[0] & QT_FLAG
                              and rec_rows(r) == n and rec_len(r) == L and r[5] == 0)
                        return i if ok else None
                return None
            ix, iu = producer(xd & M42), producer(dd & M42)
            if ix is None or iu is None or ix == iu:
                continue
            DX, DU = recs[ix], recs[iu]
            if (DX[2] >> 48) & M48 != (DU[2] >> 48) & M48:            # the same s_row
                continue
            outs = [(xd & M42, n * L * 2), (dd & M42, n * L * 2), (DX[1] & M42, n * 8), (DU[1] & M42, n * 8)]
            if readers(*_span(outs[0])) != 1 or readers(*_span(outs[1])) != 1 or readers(*_span(outs[2])) or readers(*_span(outs[3])):
                continue
            recs[gi] = [g[0] | PDQ_FLAG,
                        (g[1] & M42) | (((DX[1] >> 42) & M48) << 42),
                        (DU[3] & M48) | (((DX[2] >> 48) & M48) << 48),
                        ((DU[1] >> 42) & M48) | ((DX[3] & M48) << 48),
                        g[4], g[5]]
            drop.update((ix, iu))
            for base, nbytes in outs:
                r = mem.region_of(base)
                if r.exp_beats is not None:
                    r.exp_beats[(base - r.base) // BEAT: -(-(base - r.base + nbytes) // BEAT)] = False
            n_fused += 1
            saved += 2 * n * L
        if drop:
            st.words = sum((r for i, r in enumerate(recs) if i not in drop), []) + [0]
            p0 = getattr(st, "passes0", None) or before
            st.passes0 = p0
            st.fuse = (before - sum(rec_rows(recs[i]) * rec_len(recs[i]) for i in drop)) / p0
    return dict(geglu_records=n_geglu, fused=n_fused, element_passes_saved=saved)


def _span(t):
    return t[0], t[0] + t[1]


def split_chain(words: list[int], depths: list[int]) -> list[list[int]]:
    """the column tiles of a chain stage onto the nodes of its kind (their chain depths in node order): a tile goes
    to a node of the depth it was lowered for, longest-first onto the least-loaded node of that depth"""
    groups = chain_groups(words)
    parts: list[list[int]] = [[] for _ in depths]
    for d in sorted(set(depths)):
        idx = [i for i, dd in enumerate(depths) if dd == d]
        mine = [g for g in groups if tile_depth(g) == d]
        for i, b in zip(idx, lpt(mine, len(idx), chain_cost)):
            parts[i] = sum(b, [])
    stray = [tile_depth(g) for g in groups if tile_depth(g) not in depths]
    assert not stray, f"tiles lowered for depths {sorted(set(stray))} but the nodes are {sorted(set(depths))}"
    return parts


POST_ATTN = ("o_gemm", "o_dequant_residual", "rms2_quant", "gate_up_gemm", "geglu_quant", "down_gemm", "down_dequant_residual")


# ====================================================================== the chunk
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-chain", type=int, default=4, help="int8 chain nodes")
    ap.add_argument("--n-pv", type=int, default=1, help="uint8 PV chain nodes")
    ap.add_argument("--n-vec", type=int, default=2, help="vector nodes")
    ap.add_argument("--n-stage", type=int, default=16, help="depth (16 or 32) of the int8 chains that are not deep")
    ap.add_argument("--n-deep", type=int, default=0,
                    help="of the --n-chain int8 chains, this many (the first ones, as pi0_chip_top numbers them) are "
                         "32-stage; each GEMM's columns are shared by depth x nodes (pi0_chunk_tiler)")
    ap.add_argument("--pv-stage", type=int, default=None, help="depth of the uint8 PV chains (default: --n-stage)")
    ap.add_argument("--slot-bits", type=int, default=12)
    ap.add_argument("--siglip-layers", type=int, default=27)
    ap.add_argument("--prefix-layers", type=int, default=18)
    ap.add_argument("--expert-layers", type=int, default=18)
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--vision", choices=["captured", "synthetic"], default="captured",
                    help="captured: SigLIP layer 0 input from the frame (host input); synthetic: a smooth test image "
                         "through the patch embedding on chip (the capture has no pixels)")
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--hex", action="store_true", help="write mem_in.hex / mem_exp.hex for the testbench")
    ap.add_argument("--host-image", action="store_true", help="write the binary host image (D2 write list)")
    ap.add_argument("--bin", action="store_true",
                    help="write image/data.{bin,idx} and image/exp.{bin,idx} (host/pi0_chunk_run.cpp --image-bin): the "
                         "whole-chunk silicon format; with --exp-areas the expected part stays small")
    ap.add_argument("--flag-value", type=int, default=1)
    ap.add_argument("--compact", action="store_true",
                    help="(set PI0_CHUNK_COMPACT=1 in the environment instead: the map is chosen at import) every area in "
                         "the first 512 MB of GDDR6, for silicon runs through a 512 MB BAR1 window")
    ap.add_argument("--sync", choices=["barrier", "deps"], default="barrier",
                    help="barrier: every part waits for every part of the stage before it; deps: for every other node, "
                         "the latest earlier part it has a read/write conflict with (lets independent work overlap)")
    ap.add_argument("--prefix-row-blocks", type=int, default=1,
                    help="split each prefix layer's row-local post-attention segment (o_proj .. down residual) into this "
                         "many token-row blocks, interleaved (vector / chain overlap under --sync deps)")
    ap.add_argument("--fuse-quant", action="store_true",
                    help="operator fusion: fold every QUANT into the bf16 op that feeds it (vector record flag w0[118], "
                         "needs vu_lane.sv QTAIL = 1) when nothing else reads that op's output")
    ap.add_argument("--fuse-pdq", action="store_true",
                    help="operator fusion 2: DEQUANT(gate), DEQUANT(up), GEGLU -> one GEGLU record with the pre-dequant flag "
                         "(w0[119], needs vu_node_ml.sv PDQ = 1)")
    ap.add_argument("--interleave", action="store_true",
                    help="alternate the two cameras' SigLIP stages (overlap under --sync deps)")
    ap.add_argument("--exp-areas", default="",
                    help="comma list (e.g. IO,FLAGS): keep expected bytes only for these areas -- a whole chunk's "
                         "intermediates are ~8 GB; a silicon run of it compares the actions and the flags")
    ap.add_argument("--dry", action="store_true",
                    help="keep the map, programs and beat bitmaps but store no image / expected bytes (whole-chunk sizing)")
    a = ap.parse_args()
    if a.compact and os.environ.get("PI0_CHUNK_COMPACT") != "1":
        sys.exit("--compact needs PI0_CHUNK_COMPACT=1 in the environment (the layout module reads it at import)")
    t_start = time.time()
    AG.N_STAGE = a.n_stage
    a.pv_stage = a.pv_stage or a.n_stage
    assert 0 <= a.n_deep <= a.n_chain
    chain_depth = [32] * a.n_deep + [a.n_stage] * (a.n_chain - a.n_deep)      # int8 chain nodes, in node order
    z = np.load(LL.CAPTURE)
    mem = ChunkMem(store_data=not a.dry or a.host_image, store_exp=not a.dry,
                   exp_areas={s for s in a.exp_areas.split(",") if s} or None)
    B, C = ChunkBuild(mem), LL.Chip()
    B.tiler_classes = {CH8: sorted({(d, chain_depth.count(d)) for d in chain_depth}, reverse=True),
                       CHU: [(a.pv_stage, a.n_pv)]}
    shared: dict = {}
    f32c = LL.f32c
    log = lambda s: print(f"[{time.time() - t_start:6.0f} s] {s}", flush=True)     # noqa: E731

    # ------------------------------------------------------------------ vision
    n_cam = 2
    VE = VisionEnds(B, C)
    vis_scaling = float(z["vis.L0.scaling"])
    layers_vis: dict[int, SiglipLayer] = {}
    x_cam: list[Act] = []
    for cam in range(n_cam):
        if a.vision == "captured":
            B.layer = f"vis_in.c{cam}"
            xc = R.bf16_codes(z["vis.L0.layer_in"][cam].astype(np.float64))
            base = B.put_in(xc, 16, "siglip_in", area="IO")
            amax = B.put_in(R.row_amax_code(xc), 16, "siglip_in.amax", area="IO")
            x = Act(xc, base, amax_base=amax)
        else:
            import pi0_vision_ends_golden as VEG
            x = VE.patch_embed(VEG.synthetic_patches(), cam)
        x_cam.append(x)
    # the two cameras are independent streams through the same layers: with --interleave their stages alternate
    # (camera 0 stage i, camera 1 stage i, ...), so under --sync deps one camera's vector stage runs while the
    # other's GEMM does; without it camera 0's whole tower comes first (the original order)
    streams: list[list] = [[] for _ in range(n_cam)]
    for L in range(a.siglip_layers):
        if L not in layers_vis:
            layers_vis[L] = SiglipLayer(B, C, L, vis_scaling)
            log(f"SigLIP layer {L} weights placed")
        for cam in range(n_cam):
            n0 = len(B.stages)
            x_cam[cam] = layers_vis[L].run(x_cam[cam], name=f".c{cam}")["z"]
            streams[cam] += B.stages[n0:]
            del B.stages[n0:]
    if a.interleave:
        k = 0
        while any(k < len(s) for s in streams):
            B.stages += [s[k] for s in streams if k < len(s)]
            k += 1
    else:
        for s in streams:
            B.stages += s
    log(f"vision: {len(B.stages)} stages")

    # ------------------------------------------------------------------ prefix input: image tokens + language rows
    T_LM, D_LM = 525, 2048
    B.layer = "lm_in"
    X_LM = B.alloc(T_LM * D_LM * 16, "x_lm", "SCRATCH")
    XSUM_LM = B.alloc(T_LM * 64, "x_lm.sum", "SCRATCH")
    lm_in_cap = z["lm.L0.layer_in"].astype(np.float64)
    lang = R.bf16_codes(lm_in_cap[512:])                                # (13, 2048): the prompt embeddings
    mem.put(X_LM + 512 * D_LM * 2, lang, 16)
    mem.put(XSUM_LM + 512 * 8, [int(v) << 48 for v in R.row_amax_code(lang)], 64)
    xlm_codes = np.zeros((T_LM, D_LM), np.uint16)
    xlm_codes[512:] = lang
    for cam in range(n_cam):
        if a.prefix_layers > 0:
            ie = VE.projector(x_cam[cam], cam, X_LM + cam * 256 * D_LM * 2, XSUM_LM + cam * 256 * 8)
            xlm_codes[256 * cam:256 * (cam + 1)] = ie
    x_lm = Act(xlm_codes, X_LM, XSUM_LM)

    # ------------------------------------------------------------------ expert layers (their o_proj smoothing is
    # what the prefix layers scale their expert-V by), then the prefix layers
    T_EX, NPRE = 51, int(z["ex.L0.mask"][0][-1, :816].sum())
    mask867 = z["ex.L0.mask"][0]
    pre_idx = np.nonzero(mask867[-1, :816])[0]
    assert NPRE == T_LM == len(pre_idx)
    EX: list[GemmaLayer] = []
    for L in range(a.expert_layers):
        EX.append(GemmaLayer(B, C, L, "exp", a.alpha, a.slot_bits, T_EX, NPRE))
        log(f"expert layer {L} weights placed")
    NPS, TP_EX = EX[0].NPS if EX else 544, 64
    NK_EX = NPS + TP_EX
    mchip_ex = np.zeros((T_EX, NK_EX), bool)
    mchip_ex[:, :NPRE] = mask867[:, pre_idx]
    mchip_ex[:, NPS:NPS + T_EX] = mask867[:, 816:816 + T_EX]
    for L in range(a.expert_layers):
        EX[L].prepare(float(z["ex.L0.scaling"]), z["ex.rope_cos"][0].astype(np.float64),
                      z["ex.rope_sin"][0].astype(np.float64), mchip_ex, shared)
    prefix_kv: list[dict] = []
    LMK = -(-T_LM // 32) * 32                                            # 544 key slots of a prefix layer
    mchip_lm = np.zeros((T_LM, LMK), bool)
    mchip_lm[:, :T_LM] = z["lm.L0.mask"]
    for L in range(a.prefix_layers):
        PL = GemmaLayer(B, C, L, "lm", a.alpha, a.slot_bits, T_LM, 0, ex_s_o_hd=EX[L].s_o_hd if L < len(EX) else None)
        PL.prepare(float(z["lm.L0.scaling"]), z["lm.rope_cos"].astype(np.float64), z["lm.rope_sin"].astype(np.float64),
                   mchip_lm, shared)
        n0 = len(B.stages)
        r = PL.run(x_lm, keys_region=dict(slots=NK_EX))
        if a.prefix_row_blocks > 1:
            kb = a.prefix_row_blocks
            cuts = [0] + [-(-T_LM * k // kb // 16) * 16 for k in range(1, kb)] + [T_LM]
            B.stages[n0:] = row_blocks(B.stages[n0:], POST_ATTN, T_LM, cuts)
        x_lm = r["z"]
        prefix_kv.append(r)
        log(f"prefix layer {L}: {len(B.stages)} stages so far")
    if a.prefix_layers and a.expert_layers > a.prefix_layers:
        raise SystemExit("every expert layer needs its prefix layer's K/V: --expert-layers <= --prefix-layers")

    # ------------------------------------------------------------------ the 10 Euler steps
    HEAD = ActionHead(B, C)
    B.layer = "head_in"
    x_t = z["ex.noise"].astype(np.float32)
    XT = B.put_in(x_t.view(np.uint32), 32, "noise", area="IO")
    state_codes = R.bf16_codes(z["ex.L0.layer_in"][0][0].astype(np.float64))[None, :]      # the state token (host)
    actions = None
    for step in range(a.steps):
        B.layer = f"x_exp.s{step}"
        X_EXP = B.alloc(T_EX * 1024 * 16, "x_exp", "SCRATCH")
        mem.put(X_EXP, state_codes, 16)
        TOKSUM = B.alloc(50 * 64, "tok.sum", "SCRATCH")
        tok = HEAD.input_side(x_t, XT, step, X_EXP + 1024 * 2, TOKSUM)
        x = add_pass(B, C, Act(np.concatenate([state_codes, tok]), X_EXP), T_EX, 1024, f"x_exp.s{step}")
        for L in range(a.expert_layers):
            x = EX[L].run(x, prefix=prefix_kv[L], name=f".s{step}")["z"]
        last = step == a.steps - 1
        B.layer = f"head_out.s{step}"
        XN = B.alloc(50 * 32 * 32, "actions" if last else "x_next", "IO" if last else "SCRATCH")
        x_t = HEAD.output_side(x, x_t, XT, step, XN)
        XT = XN
        log(f"step {step}: {len(B.stages)} stages so far")
    actions = x_t

    fuse_info = None
    if a.fuse_pdq:
        pdq_info = fuse_pdq(B.stages, mem)
        log(f"fuse-pdq: {pdq_info}")
    if a.fuse_quant:
        fuse_info = fuse_quant(B.stages, mem)
        log(f"fuse-quant: {fuse_info}")

    # ------------------------------------------------------------------ assignment, flags, programs
    n_nodes = a.n_vec + a.n_chain + a.n_pv
    node_kind = [VN] * a.n_vec + [CH8] * a.n_chain + [CHU] * a.n_pv
    node_ids = {VN: list(range(a.n_vec)), CH8: list(range(a.n_vec, a.n_vec + a.n_chain)),
                CHU: list(range(a.n_vec + a.n_chain, n_nodes))}
    node_depth = [0] * a.n_vec + chain_depth + [a.pv_stage] * a.n_pv
    stage_parts: list[list[tuple[int, list[int]]]] = []
    part_costs: list[list[int]] = []
    stages_out = []                    # B.stages with hazardous vector stages cut into sub-stages
    n_cuts = 0
    for st in B.stages:
        ids = node_ids[st.kind]
        if st.kind == VN:
            subs = split_vector_safe(st.words, len(ids))
            n_cuts += len(subs) - 1
            for k, parts in enumerate(subs):
                costs = [sum(rec_rows(r) * rec_len(r) for r in vec_records(p + [0] * 6)) for p in parts]
                sub_st = st if len(subs) == 1 else Stage(st.kind, [], f"{st.name}#{k}", st.layer, st.fuse)
                stages_out.append(sub_st)
                stage_parts.append([(ids[i], p) for i, p in enumerate(parts) if p])
                part_costs.append([c for c, p in zip(costs, parts) if p])
            continue
        parts = split_chain(st.words, [node_depth[i] for i in ids])
        costs = [sum(chain_cost(g) for g in chain_groups(p + [0])) for p in parts]
        stages_out.append(st)
        stage_parts.append([(ids[i], p) for i, p in enumerate(parts) if p])
        part_costs.append([c for c, p in zip(costs, parts) if p])
    log(f"vector stages cut for intra-stage cross-node dependencies: {n_cuts}")
    B.stages = stages_out
    n_parts = sum(len(p) for p in stage_parts)
    B.layer = "flags"
    FLAGS = mem.alloc("flags", n_parts * BEAT, "FLAGS").base
    wait_rec = lambda n, w, v: ([(0xB << 124) | (v << 8), w, 0, 0, 0, 0] if n == VN else [(7 << 124) | (v << 64) | w])   # noqa: E731
    post_rec = lambda n, w, v: ([(0xC << 124) | (v << 8), w, 0, 0, 0, 0] if n == VN else [(8 << 124) | (v << 64) | w])   # noqa: E731
    progs: list[list[int]] = [[] for _ in range(n_nodes)]
    part_flag = {}
    part_waits: dict[tuple[int, int], list] = {}
    seq_of: dict[int, tuple[int, int]] = {}            # part sequence number -> (stage, node)
    fi = 0
    prev_flags: list[int] = []
    stage_lines = []
    dt = DepTracker(mem) if a.sync == "deps" else None
    n_waits = 0
    prev_seqs: list[int] = []
    for si, (st, parts) in enumerate(zip(B.stages, stage_parts)):
        flags_here = []
        acc = []
        fi0 = fi
        if dt is not None:
            # every part of a stage against the state before the stage: parts of one stage never wait for each other
            for node, body in parts:
                if st.kind != VN:
                    wr, rd = chain_access(body)
                else:
                    wr, rd = [], []
                    for r in vec_records(body + [0] * 6):
                        w1, r1 = rec_access(r)
                        wr += w1
                        rd += r1
                acc.append((wr, rd, dt.deps(node, wr, rd)))
        for pi, (node, body) in enumerate(parts):
            flag = FLAGS + BEAT * fi
            if dt is None:
                waits = prev_flags
                part_waits[(si, node)] = [list(seq_of[s]) for s in prev_seqs]
            else:
                dmap = acc[pi][2]
                waits = [FLAGS + BEAT * s for n2, s in sorted(dmap.items())]
                part_waits[(si, node)] = [list(seq_of[s]) for n2, s in sorted(dmap.items())]
            seq_of[fi] = (si, node)
            fi += 1
            for f in waits:
                progs[node] += wait_rec(st.kind, f, a.flag_value)
            n_waits += len(waits)
            progs[node] += body
            progs[node] += post_rec(st.kind, flag, a.flag_value)
            mem.expect(flag, [a.flag_value], 256)
            flags_here.append(flag)
            part_flag[(si, node)] = flag
        if dt is not None:
            for pi, ((node, _), (wr, rd, _)) in enumerate(zip(parts, acc)):
                dt.add(node, fi0 + pi, wr, rd)
        prev_flags = flags_here
        prev_seqs = list(range(fi0, fi))
        stage_lines.append(f"{si} {KIND[st.kind]} {st.layer}:{st.name} parts={[n for n, _ in parts]}")
    log(f"sync {a.sync}: {n_waits} WAITs over {fi} parts")
    parts_json = [dict(stage=si, kind=KIND[st.kind], layer=st.layer, name=st.name, fuse=st.fuse,
                       parts=[dict(node=n, cost=c, words=len(p), waits=part_waits[(si, n)]) for (n, p), c in zip(parts, costs)])
                  for si, (st, parts, costs) in enumerate(zip(B.stages, stage_parts, part_costs))]
    node_lines, prog_stats = [], []
    for node in range(n_nodes):
        words = progs[node] + [0]
        kind = node_kind[node]
        nbytes = len(words) * 16
        base = mem.alloc(f"prog.node{node}", nbytes, "PROG").base
        mem.put(base, words, 128)
        chk = check_chain_program(words, node_depth[node]) if kind != VN else check_vector_program(words, a.slot_bits)
        node_lines.append(f"{node} {KIND[kind]} {base:x}")
        prog_stats.append(dict(node=node, kind=KIND[kind], base=base, words=len(words), bytes=nbytes, **chk))
    mem.check()

    # ------------------------------------------------------------------ outputs
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "nodes.txt").write_text("".join(l + "\n" for l in node_lines))
    (out / "node_depth.txt").write_text("".join(f"{n} {KIND[node_kind[n]]} {node_depth[n]}\n" for n in range(n_nodes)))
    (out / "stages.txt").write_text("".join(l + "\n" for l in stage_lines))
    (out / "parts.json").write_text(json.dumps(parts_json, indent=0))
    if os.environ.get("PI0_CHUNK_STAGE_IO") == "1":
        # every stage's byte ranges written / read (the generator's own access model): data-flow tracing of a
        # silicon run's wrong beats (paper/sw/pi0_chunk_wrong_regions.py --stage-io)
        sio = []
        for si, st in enumerate(B.stages):
            if st.kind == VN:
                w_, r_ = [], []
                for rec in vec_records(st.words):
                    ww, rr = rec_access(rec)
                    w_ += ww; r_ += rr
            else:
                w_, r_ = chain_access(st.words)
            sio.append(dict(stage=si, name=f"{st.layer}:{st.name}", kind=KIND[st.kind], writes=w_, reads=r_,
                            words=[f"{w:x}" for w in st.words]))
        (out / "stage_io.json").write_text(json.dumps(sio))
    (out / "regions.json").write_text(json.dumps([dict(name=r.name, base=r.base, size=r.size, area=r.area,
                                                        data=bool(r.data is not None), exp=bool(r.exp is not None))
                                                   for r in mem.regions], indent=0))
    traffic = traffic_of(progs, node_kind)
    ref_actions = z["ex.actions_fp32"].astype(np.float64)
    info = dict(
        array=dict(n_vec=a.n_vec, n_chain=a.n_chain, n_pv=a.n_pv, n_stage=a.n_stage, n_deep=a.n_deep, pv_stage=a.pv_stage,
                   node_depth=node_depth, slot_bits=a.slot_bits, sync=a.sync),
        chunk=dict(siglip_layers=a.siglip_layers, prefix_layers=a.prefix_layers, expert_layers=a.expert_layers, steps=a.steps,
                   vision=a.vision, stages=len(B.stages), parts=n_parts),
        memory=mem.summary(), programs=prog_stats, traffic=traffic,
        actions_vs_fp32_model=dict(rel=LL.rel(actions.astype(np.float64), ref_actions),
                                   note="the chip recipe's whole-chunk actions against the fp32 model's captured actions "
                                        "(only meaningful for the full chunk on the captured frame)"),
        gddr6_map=__import__("pi0_chunk_layout").MAP_KIND, gemm_db=__import__("pi0_chunk_tiler").GEMM_DB, fuse_quant=fuse_info, chip_ops=C.ops, max_acc_log2=float(np.log2(max(C.max_acc, 1))),
        gen_seconds=time.time() - t_start)
    (out / "info.json").write_text(json.dumps(info, indent=1, default=float))
    np.save(out / "actions_chip.npy", actions)
    if a.hex and not a.dry:
        n_in = mem.write_hex(out / "mem_in.hex", "data")
        n_exp = mem.write_hex(out / "mem_exp.hex", "exp")
        info["hex"] = dict(image_beats=n_in, expected_beats=n_exp)
        (out / "info.json").write_text(json.dumps(info, indent=1, default=float))
        log(f"hex: image {n_in} beats, expected {n_exp} beats")
    if a.host_image:
        log(f"host image: {mem.write_host_image(out / 'host_image')}")
    if a.bin and not a.dry:
        info["bin"] = dict(data=mem.write_image_bin(out / "image", "data"), exp=mem.write_image_bin(out / "image", "exp"))
        (out / "info.json").write_text(json.dumps(info, indent=1, default=float))
        log(f"binary image: {info['bin']}")
    log(f"done: {len(B.stages)} stages, {n_parts} parts, {n_nodes} node programs; traffic read {traffic['read'] / 1e6:.1f} MB "
        f"write {traffic['write'] / 1e6:.1f} MB; actions vs fp32 model rel {info['actions_vs_fp32_model']['rel']:.4f}")


def traffic_of(progs: list[list[int]], node_kind: list[int]) -> dict:
    """GDDR6 bytes every node program reads and writes (program_traffic.py's rules; WAIT / POST not counted)"""
    tot = dict(read=0, write=0)
    per = []
    for node, words in enumerate(progs):
        rd = wr = 0
        if node_kind[node] == VN:
            k = 0
            while k + 5 < len(words) and words[k]:
                w = words[k:k + 6]
                if (w[0] >> 124) == 0xA:
                    op, R_, L = w[0] & 0xF, (w[0] >> 40) & 0xFFFFF, (w[0] >> 60) & 0xFFFF
                    descs = [(w[1] >> 42), w[2], w[2] >> 48, w[3], w[3] >> 48, w[4], w[4] >> 48]
                    for si, d in enumerate(descs):
                        d &= (1 << 48) - 1
                        shape, fmt = (d >> 42) & 3, (d >> 44) & 3
                        n = {0: R_ * L, 1: R_, 2: L, 3: 0}[shape]
                        rd += n * (2 if si == 6 else DESC_W[fmt])
                    ow = rec_out_w(w[0])
                    wr += (R_ if op == 3 else R_ * L) * ow + R_ * 8
                k += 6
            rd += (k + 6) * 16
        else:
            k = 0
            for c in words:
                op = c >> 124
                if op == 0:
                    break
                if op == 1:
                    rd += ((c >> 47) & 31) * ((c >> 52) & 1023) * 16
                elif op == 5:
                    rd += ((c >> 47) & 31) * ((c >> 52) & 1023) * ((c >> 62) & 1023) * 16
                elif op == 6:
                    T, W, P = (c >> 42) & 1023, (c >> 61) & 1023, (c >> 81) & 1023
                    rd += T * W * 16
                    wr += T * 16 * P * 4
                k += 1
            rd += (k + 1) * 16
        per.append(dict(node=node, kind=KIND[node_kind[node]], read=rd, write=wr))
        tot["read"] += rd
        tot["write"] += wr
    tot["per_node"] = per
    return tot


if __name__ == "__main__":
    main()
