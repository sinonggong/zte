"""Mixed-depth GEMM tiling for the chunk generator: every column tile carries its own chain depth.

The node array mixes chain depths (pi0_chip_top: N_DEEP 32-stage int8 chains, then 16-stage int8 chains, then the
16-stage uint8 PV chains), while pi0_attn_golden.Tiler lowers every tile for one module-wide N_STAGE.  A tile of
depth d and P columns per stage covers d * P output columns; its weight image (LOAD path) is laid out for d (stage
s holds columns d p + s), its LOAD / LOADX commands fill stage registers 1..d, and its OUT writes d * P columns per
row.  The data layouts the vector nodes produce (key codes, V rows, token rows: the LOADX path) do not depend on the
depth -- a LOADX tile strides through contiguous column rows by d -- so only the tiling and the static weight
images change.

Column split: a GEMM's columns are shared among the depth classes of the node kind that runs it in proportion to
d x (nodes of that depth) -- a deep node retires d columns per row pass, the same cycles as a 16-stage node's 16 --
rounded to whole d-column groups.  Each class's share is cut into tiles of at most 512 // W columns per stage and
into a multiple of that class's node count, so the longest-first assignment (pi0_chunk_program.split_chain) can
balance them.  The program assigner reads a tile's depth back from its LOAD / LOADX commands (tile_depth).
"""
from __future__ import annotations

import os as _os

import numpy as np

import pi0_attn_golden as AG
from pi0_attn_golden import CH8, CHU, LOAD_NREG_MAX
from colpar_tile_golden import op_load, op_loadx, op_out, op_flush, op_colgroup  # noqa: F401


def weight_image_d(codes: np.ndarray, d: int) -> np.ndarray:
    """(N, K) int8 weight codes -> stage images for a depth-d chain: stage s holds columns c = d p + s"""
    N, K = codes.shape
    assert N % d == 0 and K % 16 == 0, (N, K, d)
    return np.ascontiguousarray(codes.reshape(N // d, d, K).transpose(1, 0, 2)).reshape(-1)


def tile_depth(group: list[int]) -> int:
    """the chain depth a column-tile group was lowered for: the last stage register its LOAD / LOADX fill"""
    d = 0
    for c in group:
        op = c >> 124
        if op in (1, 5):
            d = max(d, ((c >> 42) & 31) + ((c >> 47) & 31) - 1)
    return d


GEMM_DB = _os.environ.get("PI0_CHUNK_GEMM_DB", "0") == "1"   # double-buffered GEMM feeder (RTL from 2026-09-18)


class MixedTiler(AG.Tiler):
    """AG.Tiler with per-tile depth.  classes: {kind: [(depth, n_nodes), ...]} for CH8 and CHU."""

    def __init__(self, B, classes: dict):
        super().__init__(B)
        self.classes = {k: [(d, n) for d, n in v if n > 0] for k, v in classes.items()}
        for k, v in self.classes.items():
            assert v, f"no chain nodes of kind {k}"

    # ------------------------------------------------------------------ column split
    def split_cols_mixed(self, C_total: int, W: int, kind: int = CH8, min_tiles: int = 1) -> list[tuple[int, int]]:
        """[(P, d), ...] tiles covering C_total columns in order: deep classes first"""
        cls = sorted(self.classes[kind], key=lambda x: -x[0])
        P_max = 512 // W
        assert P_max >= 1, W
        # a row pass of P columns per stage takes P x (W + G) cycles with W + G = max(W + 1, d, 14): a node retires
        # d columns per max(W + 1, d, 14) cycles, so a deep node is twice as fast only when W >= 31
        rate = {d: d / max(W + 1, d, 14) for d, _ in cls}
        weight = sum(rate[d] * n for d, n in cls)
        shares, left = [], C_total
        for i, (d, n) in enumerate(cls):
            if i == len(cls) - 1:
                c = left
            else:
                c = min(left, int(C_total * rate[d] * n / weight) // d * d)
            shares.append(c)
            left -= c
        tiles = []
        for (d, n), c in zip(cls, shares):
            if c == 0:
                continue
            assert c % d == 0, f"{C_total} columns: {c} left for depth {d} (not a multiple)"
            P_total = c // d
            t = max(min_tiles, -(-P_total // P_max))
            # a multiple of the class's node count for balance; with fewer column groups than nodes, one group per
            # tile (every node that can get work gets some)
            t = min(P_total, -(-t // n) * n)
            q, r = divmod(P_total, t)
            tiles += [(q + (1 if i < r else 0), d) for i in range(t)]
        return tiles

    def split_cols(self, C_total, W, min_tiles=1):          # the base-class interface (uniform tiles): not used here
        raise NotImplementedError("MixedTiler: use split_cols_mixed")

    # ------------------------------------------------------------------ LOAD path (static weight images)
    def put_images(self, codes, W_in, min_tiles=1, blocks=1, kind=CH8):
        N = codes.shape[0]
        assert N % blocks == 0
        bw = N // blocks
        imgs = []
        for b in range(blocks):
            c0 = b * bw
            for P, d in self.split_cols_mixed(bw, W_in, kind, min_tiles):
                imgs.append((self.B.put_in(weight_image_d(codes[c0:c0 + d * P], d) & 0xFF, 8), c0, P, d))
                c0 += d * P
        return imgs, bw

    def colgroup_d(self, act_base, T_rows, W, P, d, stride=0):
        """AG.Tiler.colgroup for a depth-d tile: the chain's row protocol needs W + G >= d between row passes (the
        module-wide gap_of uses AG.N_STAGE; a 32-stage tile with W = 5, the SigLIP QK^T, needs G = 27, not 11)"""
        M, G = min(T_rows, 512 // W), max(1, d - W, 14 - W)
        assert act_base % 32 == 0 and 0 < T_rows < 1024 and W < 1024 and P < 1024 and G < 32 and stride < 1024
        db = 0
        if GEMM_DB and W <= 256 and T_rows > 256 // W:
            # double-buffered feeder (opcode 6 bit 106): groups of M <= 256 / W rows alternate between the two
            # 256-word halves; every group but the last must start on a beat (M x stride even)
            M = 256 // W
            if (M * (stride or W)) % 2:
                M -= 1
            db = 1 if M >= 1 else 0
            if not db:
                M = min(T_rows, 512 // W)
        return op_colgroup(act_base, T_rows, 0, W, M, P, G, stride=stride) | (db << 106)

    def tile_cmds_d(self, out_bases, bw, c0, P, d, load, src_base, T_rows, W, stride=0):
        b = c0 // bw
        split = d * P != bw
        out = []
        for i, (r0, n) in enumerate(self.row_slices(T_rows, W, stride)):
            out.append(op_out(out_bases[b] + r0 * bw * 4 + (c0 - b * bw) * 4,
                              d * P // 8 if split else 0, (bw - d * P) * 4 if split else 0))
            if i == 0:
                out += load
            out += [self.colgroup_d(src_base + r0 * (stride or W) * 16, n, W, P, d, stride), op_flush()]
        return out

    def cmds_load(self, src_base, T_rows, W_in, images, out_bases):
        imgs, bw = images
        out_bases = [out_bases] if isinstance(out_bases, int) else list(out_bases)
        out = []
        for img, c0, P, d in imgs:
            load = [op_load(img + s0 * P * W_in * 16, 1 + s0, min(LOAD_NREG_MAX, d - s0), P * W_in)
                    for s0 in range(0, d, LOAD_NREG_MAX)]
            out += self.tile_cmds_d(out_bases, bw, c0, P, d, load, src_base, T_rows, W_in)
        return out

    # ------------------------------------------------------------------ LOADX path (columns as contiguous rows)
    def cmds_loadx(self, img_base, C_total, W, src_base, T_rows, out_base, stride=0, kind=CH8):
        out, c0 = [], 0
        for P, d in self.split_cols_mixed(C_total, W, kind):
            out += self.tile_cmds_d([out_base], C_total, c0, P, d,
                                    [op_loadx(img_base + (c0 + s0) * W * 16, 1 + s0, min(LOAD_NREG_MAX, d - s0), W, P,
                                              d * W, W) for s0 in range(0, d, LOAD_NREG_MAX)],
                                    src_base, T_rows, W, stride)
            c0 += d * P
        return out

    def gemm_cmds(self, src_base, T_rows, W_in, codes, out_base, tiles=1, blocks=1):
        return self.cmds_load(src_base, T_rows, W_in, self.put_images(codes, W_in, tiles, blocks), out_base)
