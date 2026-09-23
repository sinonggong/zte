#!/usr/bin/env python3
"""GDDR6 image, node programs and expected beats for tb_pi0_attn.sv: one expert ATTENTION BLOCK of the real
checkpoint, lowered onto the node command set and run across three nodes.

Layouts are those of paper/sw/pi0_layer_lower.py ("Data layouts"); the dimensions are cut down (default hidden
256, 2 heads x 64, 8 tokens, 64 prefix + 32 suffix key slots) so every image fits one chain node's BRAMs and
the block runs in a behavioural simulation.  The weights are real slices of model.safetensors and the
activations real slices of a captured frame, quantised exactly as the full layer is.

Stages (the testbench starts one node per stage, in order; each stage is one node program):

  0 VN   RMS_STAT, RMS_APPLY, QUANT                       x -> h codes + s_h
  1 CH8  q and k GEMMs, and the role-swapped v GEMM       -> acc_q, acc_k, acc_vT (column-major V)
  2 VN   DEQUANT q (x scaling/ln2), k and vT; RoPE pass A + pass B with the rotate-half read (w0[7]);
         QUANT of the q rows, of the suffix key rows (into the tail of the key code region) and of the
         V suffix channel rows
  3 CH8  QK^T: LOADX of the key codes (stage s takes columns 16 p + s) + one column group
  4 VN   DEQUANT of the logits (E mask), then SMAX_U8 (255 levels, uint8 P codes)
  5 CHU  PV: prefix and suffix GEMMs over the P row slices (opcode 6 stride) against the V channel rows
  6 VN   DEQUANT of both PV parts, ADD, QUANT
  7 CH8  o GEMM
  8 VN   DEQUANT o, ADD residual

Writes mem_in.hex, mem_exp.hex, stages.txt ("<node> <program base>" per line) and info.txt into --out.

--full lowers the layer at its REAL size instead: hidden 1024, 8 heads x 256, the 51 suffix tokens of the frame
and its 525 valid prefix keys, FF 4096.  Every GEMM is then split into column tiles wherever its weight image
would not fit one stage BRAM72K (P * W <= 512 words) -- q into 16 tiles, k into 2, QK^T into 2, the PV prefix
into 2, o into 16, gate and up into 32 each, down into 32 -- and every tile writes a column block of the
row-major result.  The prefix K and V are inputs, computed exactly as paper/sw/pi0_layer_lower.py's prefix pass
does (from the LM layer's own weights), and the key mask is the frame's real one.  Key slots are padded to an
even number of words, 544 prefix + 64 suffix = 608 (38 words): the PV feeder reads P rows with a stride of
W_NK words, and a stride and the words it lands on must be even, which 528 + 64 = 592 (37 words) violates.
The padded slots are masked, so the numbers do not change.  Rows 4096 wide need a vector node built with
SLOT_BITS >= 12 (--slot-bits); every vector op is checked against it.

--part lm lowers a PREFIX layer (PaliGemma LM layer) at its real width instead: hidden 2048, FF 16384, the
frame's 525 prefix tokens (--lm-tokens takes the first N) attending to each other through the frame's own mask.
Its keys are all computed in the layer (no prefix inputs, so no PV prefix GEMM, and the context ADD becomes an
ADD b = 0 pass that gives the QUANT its per-token amax).  Two things are new at this width:

  MLP sub-blocks.  A vector row holds at most 2^SLOT_BITS elements, so the FF-wide rows are processed in
  FB = min(FF, 2^SLOT_BITS) wide sub-blocks: gate and up write their column tiles into one int32 region per
  sub-block, and DEQUANT, GEGLU and QUANT run per sub-block -- so the down input is quantised per
  (token, sub-block) rather than per token (finer, never coarser).
  down K split.  A column of W words must fit one stage BRAM72K (W <= 512), and down's K = 16384 bytes is 1024
  words, so down runs as one GEMM per sub-block (K = FB) and the partial sums are combined by DEQUANT itself:
  every sub-block but the last dequantises to fp32 with the previous one as its fp32 bias (DEQUANT [+ bias]),
  the last to bf16.  No int32 add exists on the vector node, and none is needed.

Column groups longer than the command's 10-bit row count (QK^T and PV have T x 8 = 4,200 rows) run as row
slices of whole feeder loads, each with its own OUT; stage programs are spaced by their size.
"""
from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path

import os as _os
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import vector_unit_ref as R                                              # noqa: E402
import pi0_layer_lower as LL                                            # noqa: E402
from colpar_tile_golden import op_load, op_loadx, op_colgroup, op_out, op_flush   # noqa: E402
from vu_node_golden import (Mem, Alloc, desc, K, SH_E, SH_R, SH_C, F_BF16, F_FP32, F_INT32, F_SUM,  # noqa: E402
                            FIELD_RS0, FIELD_MAX, FIELD_AMAX)

# chain node stages (16, or 32 for the deep node); run_pi0_attn_sim.sh exports COLPAR_N_STAGE
N_STAGE = int(os.environ.get("COLPAR_N_STAGE", "16"))
LOAD_NREG_MAX = 16                   # stages per LOAD / LOADX command (a 5-bit field): deep chains load in pieces
LN2 = math.log(2.0)
PROG_BASE = 0x0800_0000
IN_BASE = 0x1000_0000
OUT_BASE = 0x4000_0000
VN, CH8, CHU = 0, 1, 2
OP_RMS_STAT, OP_RMS_APPLY, OP_ROPE_A, OP_ROPE_B = 1, 2, 5, 6
OP_ADD, OP_SMAX_Q8, OP_QUANT, OP_DEQUANT = 0, 12, 13, 14
OP_GEGLU = 8


def vu_record(op, rows, length, out_base, sum_base, x, b=None, c=None, d=None, e=None, rs=None, mask=None,
              k=0, b_fp32=0, bias_en=0, out_fp32=0, x_rot=0, blk_beats=0, gap_bytes=0):
    unused = K(0)
    w0 = ((0xA << 124) | (out_base << 76) | (length << 60) | (rows << 40) | ((k & 0xFFFFFFFF) << 8)
          | (x_rot << 7) | (out_fp32 << 6) | (bias_en << 5) | (b_fp32 << 4) | op)
    return [w0, sum_base | (x << 42),
            (b if b is not None else unused) | ((c if c is not None else unused) << 48),
            (d if d is not None else unused) | ((e if e is not None else unused) << 48),
            (rs if rs is not None else unused) | ((mask if mask is not None else K(1)) << 48),
            blk_beats | (gap_bytes << 10)]       # w5: element rows written as blocks of a wider matrix (0 = contiguous)


def gap_of(W):
    return max(1, N_STAGE - W, 14 - W)


def summaries(codes, mask=None):
    """the summary records a bf16-output op emits: {amax, masked max, 0}."""
    m = np.ones(codes.shape, bool) if mask is None else mask
    amax = R.row_amax_code(codes)
    mx = LL.masked_max_code(codes, m)
    return [(int(amax[r]) << 48) | ((int(mx[r]) & 0xFFFF) << 32) for r in range(codes.shape[0])]


def weight_image(codes: np.ndarray) -> np.ndarray:
    """(N, K) int8 weight codes -> stage images: stage s holds columns c = N_STAGE p + s, W words each."""
    N, Kin = codes.shape
    assert N % N_STAGE == 0 and Kin % 16 == 0
    P = N // N_STAGE
    return np.concatenate([np.concatenate([codes[N_STAGE * p + s] for p in range(P)]) for s in range(N_STAGE)])


class FastMem(Mem):
    """vu_node_golden.Mem with a vectorised put for 8/16/32/64-bit elements (the same beats, beat by beat
    instead of element by element: a full-size prefix layer writes ~10^8 elements)."""
    DT = {8: "<u1", 16: "<u2", 32: "<u4", 64: "<u8"}

    def put(self, addr: int, values, width: int) -> None:
        if width not in self.DT:
            return super().put(addr, values, width)
        assert addr % 32 == 0
        if isinstance(values, np.ndarray) and values.dtype.kind in "iu":
            arr = values.reshape(-1)
        else:                                                            # Python ints: mask them exactly
            arr = np.array([int(v) & ((1 << width) - 1) for v in values], np.uint64)
        raw = arr.astype(self.DT[width]).tobytes()                       # integer casts wrap: two's complement
        raw += bytes(-len(raw) % 32)
        b0 = addr >> 5
        beats = self.beats
        for k in range(len(raw) // 32):
            v = int.from_bytes(raw[32 * k:32 * k + 32], "little")
            beats[b0 + k] = beats.get(b0 + k, 0) | v


class Build:
    PROG_AREA = 0x40000                        # stage programs below PROG_BASE + this, node programs above

    def __init__(self):
        self.mem, self.exp = FastMem(), FastMem()
        self.ain, self.aout = Alloc(IN_BASE), Alloc(OUT_BASE)
        self.stages: list[tuple[int, int, list[int]]] = []
        self.regions: list[tuple[str, int, int, int]] = []
        self.next_prog = PROG_BASE

    def stage(self, node: int, words: list[int]) -> None:
        # programs are 0x2000 bytes (512 commands) apart, or more when a stage is longer: a stage that ran into
        # the next one would lose its END to the next stage's first word
        base = self.next_prog
        self.next_prog += max(1, -(-len(words) * 16 // 0x2000)) * 0x2000
        assert self.next_prog <= PROG_BASE + self.PROG_AREA, "stage programs overflow their area"
        self.mem.put(base, words, 128)
        self.stages.append((node, base, list(words)))

    @staticmethod
    def _flat(values, width: int):
        if isinstance(values, np.ndarray) and values.dtype.kind in "iu" and width <= 64:
            return values.reshape(-1)
        return [int(v) for v in np.asarray(values, object).reshape(-1)]

    def put_in(self, values, width: int) -> int:
        vals = self._flat(values, width)
        base = self.ain(len(vals) * width)
        self.mem.put(base, vals, width)
        return base

    def expect(self, base: int, values, width: int, name: str = "") -> None:
        vals = self._flat(values, width)
        self.exp.put(base, vals, width)
        self.regions.append((name or f"@{base:#x}", base, len(vals) * width // 8, len(self.stages)))


class Tiler:
    """Lowers GEMMs onto chain-node commands: column tiles (P * W <= 512 words per stage image), column
    regions and row slices.  Weight images are written into the Build's GDDR6 image as they are laid out."""

    def __init__(self, B: "Build"):
        self.B = B

    def split_cols(self, C_total, W, min_tiles=1):
        """columns per stage of each column tile: as few tiles as fit P * W <= 512 words (at least min_tiles),
        as even as possible.  A tile of P columns per stage covers N_STAGE * P columns."""
        assert C_total % N_STAGE == 0, C_total
        P_total, P_max = C_total // N_STAGE, 512 // W
        tiles = max(min_tiles, -(-P_total // P_max))
        q, r = divmod(P_total, tiles)
        return [q + (1 if i < r else 0) for i in range(tiles)]

    def put_images(self, codes, W_in, min_tiles=1, blocks=1):
        """write the stage images of every column tile of a weight matrix (N out x K in) into GDDR6.  With
        blocks > 1 the columns fall into that many equal regions (the MLP sub-blocks) and no tile crosses one."""
        N = codes.shape[0]
        assert N % blocks == 0
        bw = N // blocks
        imgs = []
        for b in range(blocks):
            c0 = b * bw
            for P in self.split_cols(bw, W_in, min_tiles):
                imgs.append((self.B.put_in(weight_image(codes[c0:c0 + N_STAGE * P]) & 0xFF, 8), c0, P))
                c0 += N_STAGE * P
        return imgs, bw

    def colgroup(self, act_base, T_rows, W, P, stride=0):
        M, G = min(T_rows, 512 // W), gap_of(W)
        assert act_base % 32 == 0 and 0 < T_rows < 1024 and W < 1024 and P < 1024 and G < 32 and stride < 1024
        db = 0
        if _os.environ.get("PI0_CHUNK_GEMM_DB", "0") == "1" and W <= 256 and T_rows > 256 // W:
            # double-buffered feeder (colpar_node_ctrl opcode 6 bit 106), as pi0_chunk_tiler.MixedTiler.colgroup_d
            M = 256 // W
            if (M * (stride or W)) % 2:
                M -= 1
            db = 1 if M >= 1 else 0
            if not db:
                M = min(T_rows, 512 // W)
        return op_colgroup(act_base, T_rows, 0, W, M, P, G, stride=stride) | (db << 106)

    def row_slices(self, T_rows, W, stride=0):
        """row slices of a column group: whole feeder loads of M rows, at most 1023 rows (the command's row
        count), each starting on an even word of the feeder (act_base word even)"""
        M = min(T_rows, 512 // W)
        n = (1023 // M) * M
        while (n * (stride or W)) % 2:
            n -= M
        return [(r0, min(n, T_rows - r0)) for r0 in range(0, T_rows, n)]

    def tile_cmds(self, out_bases, bw, c0, P, load, src_base, T_rows, W, stride=0):
        """one column tile: per row slice an OUT (this tile's column block of the slice's rows, in the column
        region the tile belongs to), the tile's LOAD once, then the slice's column group and FLUSH"""
        b = c0 // bw
        split = N_STAGE * P != bw
        out = []
        for i, (r0, n) in enumerate(self.row_slices(T_rows, W, stride)):
            out.append(op_out(out_bases[b] + r0 * bw * 4 + (c0 - b * bw) * 4,
                              N_STAGE * P // 8 if split else 0, (bw - N_STAGE * P) * 4 if split else 0))
            if i == 0:
                out += load
            out += [self.colgroup(src_base + r0 * (stride or W) * 16, n, W, P, stride), op_flush()]
        return out

    def cmds_load(self, src_base, T_rows, W_in, images, out_bases):
        """a GEMM over weight images written by put_images: per tile OUT, LOAD, column group, FLUSH (per row
        slice); out_bases: one int32 region per column block"""
        imgs, bw = images
        out_bases = [out_bases] if isinstance(out_bases, int) else list(out_bases)
        out = []
        for img, c0, P in imgs:
            load = [op_load(img + s0 * P * W_in * 16, 1 + s0, min(LOAD_NREG_MAX, N_STAGE - s0), P * W_in)
                    for s0 in range(0, N_STAGE, LOAD_NREG_MAX)]
            out += self.tile_cmds(out_bases, bw, c0, P, load, src_base, T_rows, W_in)
        return out

    def cmds_loadx(self, img_base, C_total, W, src_base, T_rows, out_base, stride=0):
        """a GEMM whose weight columns are already in GDDR6 as C_total contiguous rows of W words (key codes,
        V channel rows, token code rows): per tile OUT, a segmented LOADX of that tile's columns, column group
        and FLUSH.  Column c of the tile starting at c0 is the row at word c0 * W + c * W."""
        out, c0 = [], 0
        for P in self.split_cols(C_total, W):
            out += self.tile_cmds([out_base], C_total, c0, P,
                                  [op_loadx(img_base + (c0 + s0) * W * 16, 1 + s0, min(LOAD_NREG_MAX, N_STAGE - s0), W, P,
                                            N_STAGE * W, W) for s0 in range(0, N_STAGE, LOAD_NREG_MAX)],
                                  src_base, T_rows, W, stride)
            c0 += N_STAGE * P
        return out

    def gemm_cmds(self, src_base, T_rows, W_in, codes, out_base, tiles=1, blocks=1):
        """LOAD + column group + OUT per column tile; each tile is a column block of the row-major result
        (OUT block beats / gap) whenever there is more than one, the way the wide projections have to be split.
        --tiles forces at least that many; at full size the weight image decides."""
        return self.cmds_load(src_base, T_rows, W_in, self.put_images(codes, W_in, tiles, blocks), out_base)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--layer", type=int, default=0)
    ap.add_argument("--step", type=int, default=0)
    ap.add_argument("--tokens", type=int, default=8)
    ap.add_argument("--dim", type=int, default=256)
    ap.add_argument("--heads", type=int, default=2)
    ap.add_argument("--head-dim", type=int, default=64)
    ap.add_argument("--pre-keys", type=int, default=64)
    ap.add_argument("--ff", type=int, default=512, help="MLP width slice (0 = attention only)")
    ap.add_argument("--tiles", type=int, default=1,
                    help="split the wide GEMMs (o, gate, up, down) into this many column tiles, each written\nas a column block of the row-major result -- what the full-size projections need")
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--full", action="store_true",
                    help="the layer at its real size (overrides --tokens/--dim/--heads/--head-dim/--pre-keys/--ff)")
    ap.add_argument("--part", choices=["exp", "lm"], default="exp",
                    help="exp: an action-expert layer (cut down, or --full); lm: a PaliGemma prefix layer at real width")
    ap.add_argument("--lm-tokens", type=int, default=0, help="--part lm: the first N prefix tokens (0 = all 525)")
    ap.add_argument("--slot-bits", type=int, default=11,
                    help="SLOT_BITS of the vector node the program is for: a row may hold 2^SLOT_BITS elements")
    ap.add_argument("--barrier", action="store_true",
                    help="one program per node, stages sequenced by GDDR6 flags (WAIT/POST) instead of the host")
    a = ap.parse_args()

    z = np.load(LL.CAPTURE)
    LMP = a.part == "lm"
    EXP_FULL = a.full and not LMP                  # the expert layer at its real size, with real prefix inputs
    if LMP:
        # a prefix layer: 525 compact prefix tokens, keys all computed here (no prefix inputs)
        T_all, D = z[f"lm.L{a.layer}.layer_in"].shape
        T = a.lm_tokens or T_all
        H, HD, FF = LL.HEADS, LL.HD, 16384
        NPRE = NPS = 0
        TP = -(-T // 32) * 32                                                  # key slots, 32 | TP -> even words
    elif a.full:
        # the real layer: sizes from the capture, 525 valid prefix keys picked by the frame's own mask
        T, D = z[f"ex.L{a.layer}.layer_in"].shape[1:]
        H, HD, FF = LL.HEADS, LL.HD, 4096
        mask867 = z[f"ex.L{a.layer}.mask"][a.step]                             # (T, 867)
        pre_idx = np.nonzero(mask867[-1, :816])[0]
        NPRE = len(pre_idx)                                                    # real prefix keys
        NPS = -(-NPRE // 32) * 32                                              # prefix slots, 32 | NPS -> even words
        TP = -(-T // 32) * 32                                                  # suffix slots
    else:
        T, D, H, HD, NPRE, FF = a.tokens, a.dim, a.heads, a.head_dim, a.pre_keys, a.ff
        NPS = NPRE                                                             # cut down: every slot holds a key
        TP = ((T + 15) // 16) * 16 * 2          # padded tokens = suffix key slots; 2 words per V row
    NK = NPS + TP                           # key slots, an even number of words
    QN, QR_ROWS = H * HD, T * H
    assert QN % 16 == 0 and HD % 16 == 0 and D % 16 == 0 and NPS % 16 == 0 and (NK // 16) % 2 == 0
    W_IN, W_HD, W_NK, W_PRE, W_SUF = D // 16, HD // 16, NK // 16, NPS // 16, TP // 16
    assert W_PRE % 2 == 0 and W_SUF % 2 == 0, "LOADX segments and feeder strides must land on even words"

    gem = ("q", "k", "v", "o") + (("gate", "up", "down") if FF else ())
    if LMP:
        ex = LL.load_layer(a.layer, "paligemma_layer", a.alpha, "lm", gemms=gem)
        x = z[f"lm.L{a.layer}.layer_in"][:T, :D].astype(np.float64)
        cos = z["lm.rope_cos"][:T, :HD].astype(np.float64)
        sin = z["lm.rope_sin"][:T, :HD].astype(np.float64)
        scaling = float(z[f"lm.L{a.layer}.scaling"])
    else:
        ex = LL.load_layer(a.layer, "expert_layer", a.alpha, "exp", gemms=gem)
        x = z[f"ex.L{a.layer}.layer_in"][a.step][:T, :D].astype(np.float64)
        cos = z["ex.rope_cos"][a.step][:T, :HD].astype(np.float64)
        sin = z["ex.rope_sin"][a.step][:T, :HD].astype(np.float64)
        scaling = float(z[f"ex.L{a.layer}.scaling"])
    if not a.full and not LMP:
        lm_in = z[f"lm.L{a.layer}.layer_in"][:NPRE, :D].astype(np.float64)
        kpre_f = z[f"ex.L{a.layer}.k"][a.step, 0, :NPRE, :HD].astype(np.float64)     # real post-RoPE prefix keys

    g_in, s_qkv, s_o_hd = ex["g_in"][:D], ex["s_qkv"][:D], ex["s_o_hd"][:HD]
    Wq = LL.QW(ex["q"].w[:QN, :D], s_qkv)
    Wk = LL.QW(ex["k"].w[:HD, :D], s_qkv)
    Wv = LL.QW(ex["v"].w[:HD, :D], s_qkv)
    Wo = LL.QW(ex["o"].w[:D, :QN], np.tile(s_o_hd, H))
    if FF:
        assert FF % N_STAGE == 0          # wide GEMMs are tiled, so no one-image limit on FF x D any more
        g_post, s_gu, s_d = ex["g_post"][:D], ex["s_gu"][:D], ex["s_d"][:FF]
        Wg = LL.QW(ex["gate"].w[:FF, :D], s_gu)
        Wu = LL.QW(ex["up"].w[:FF, :D], s_gu)
        Wd = LL.QW(ex["down"].w[:D, :FF], s_d)

    B, C = Build(), LL.Chip()
    TL = Tiler(B)
    put_images, cmds_load, cmds_loadx, gemm_cmds = TL.put_images, TL.cmds_load, TL.cmds_loadx, TL.gemm_cmds
    f32c, bf16v = LL.f32c, LL.bf16v
    prog: dict[int, list[int]] = {}

    # ================= stage 0: RMS + QUANT =================
    xc = R.bf16_codes(x)
    x_base = B.put_in(xc, 16)
    amax_x = R.row_amax_code(xc)
    gain_base = B.put_in(f32c((1.0 + g_in) / s_qkv), 32)
    amax_base = B.put_in(amax_x, 16)
    RSUM = B.aout(T * 64)
    r_rms = R.rms_stat(xc, amax_x, 1.0 / D)
    B.expect(RSUM, [int(v) for v in r_rms], 64)
    HB, HSUM = B.aout(T * D * 16), B.aout(T * 64)
    hc, _ = R.rms_apply(xc, r_rms, f32c((1.0 + g_in) / s_qkv))
    B.expect(HB, hc, 16)
    B.expect(HSUM, summaries(hc), 64)
    HQ, HQSUM = B.aout(TP * D * 8), B.aout(T * 64)   # TP rows: the role-swapped GEMM reads the pad tokens too
    hq, s_h = C.quant(hc)
    B.expect(HQ, hq & 0xFF, 8)
    B.expect(HQSUM, [int(v) for v in s_h], 64)
    B.stage(VN, vu_record(OP_RMS_STAT, T, D, 0, RSUM, desc(x_base, SH_E, F_BF16),
                          rs=desc(amax_base, SH_R, F_BF16), k=int(np.float32(1.0 / D).view(np.uint32)))
            + vu_record(OP_RMS_APPLY, T, D, HB, HSUM, desc(x_base, SH_E, F_BF16),
                        c=desc(RSUM, SH_R, F_SUM, FIELD_RS0), d=desc(gain_base, SH_C, F_FP32))
            + vu_record(OP_QUANT, T, D, HQ, HQSUM, desc(HB, SH_E, F_BF16),
                        rs=desc(HSUM, SH_R, F_SUM, FIELD_AMAX)) + [0])

    # ================= stage 1: q, k and the transposed v GEMM =================
    q_imgs = put_images(Wq.codes, W_IN)
    k_imgs = put_images(Wk.codes, W_IN)
    Wv_rows = B.put_in(np.concatenate([Wv.codes[c] for c in range(HD)]) & 0xFF, 8)
    accq, acck = C.gemm(hq, Wq.codes), C.gemm(hq, Wk.codes)
    hq_pad = np.zeros((TP, D), np.int64)
    hq_pad[:T] = hq
    accv = Wv.codes @ hq_pad.T                                            # (HD, TP), column-major V
    ACCQ, ACCK, ACCV = B.aout(T * QN * 32), B.aout(T * HD * 32), B.aout(HD * TP * 32)
    B.expect(ACCQ, accq & 0xFFFFFFFF, 32)
    B.expect(ACCK, acck & 0xFFFFFFFF, 32)
    B.expect(ACCV, accv & 0xFFFFFFFF, 32)
    B.stage(CH8, cmds_load(HQ, T, W_IN, q_imgs, ACCQ)
            + cmds_load(HQ, T, W_IN, k_imgs, ACCK)
            + cmds_loadx(HQ, TP, W_IN, Wv_rows, HD, ACCV) + [0])       # role-swapped: token rows are the columns

    # ================= stage 2: dequant, RoPE, quant =================
    sq_base = B.put_in(f32c(Wq.scale.astype(np.float64) * (scaling / LN2)), 32)
    sk_base = B.put_in(f32c(Wk.scale), 32)
    sv_base = B.put_in(f32c(Wv.scale / s_o_hd), 32)
    s_h_pad = np.pad(np.asarray(s_h, np.uint32), (0, TP - T))
    shpad_base = B.put_in(s_h_pad, 32)
    QB, QBSUM = B.aout(T * QN * 16), B.aout(T * 64)
    qc = C.dequant(accq, s_h, f32c(Wq.scale.astype(np.float64) * (scaling / LN2)))
    B.expect(QB, qc, 16)
    B.expect(QBSUM, summaries(qc), 64)
    KB, KBSUM = B.aout(T * HD * 16), B.aout(T * 64)
    kc = C.dequant(acck, s_h, f32c(Wk.scale))
    B.expect(KB, kc, 16)
    B.expect(KBSUM, summaries(kc), 64)
    VB, VBSUM = B.aout(HD * TP * 16), B.aout(HD * 64)
    vc = C.dequant(accv, f32c(Wv.scale / s_o_hd), s_h_pad)
    B.expect(VB, vc, 16)
    B.expect(VBSUM, summaries(vc), 64)

    # RoPE: rows are (token, head) for q and (token) for k; cos / sin are per element, replicated per head
    cos_q = B.put_in(np.repeat(f32c(cos), H, 0), 32)
    sgn = np.concatenate([-sin[:, :HD // 2], sin[:, HD // 2:]], 1)         # the sign folded into sin
    sgn_q = B.put_in(np.repeat(f32c(sgn), H, 0), 32)
    cos_k, sgn_k = B.put_in(f32c(cos), 32), B.put_in(f32c(sgn), 32)
    QPA, QPASUM = B.aout(QR_ROWS * HD * 32), B.aout(QR_ROWS * 64)
    qpa, _ = R.rope_a(qc.reshape(QR_ROWS, HD), np.repeat(f32c(cos), H, 0))
    B.expect(QPA, qpa, 32)
    B.expect(QPASUM, [0] * QR_ROWS, 64)
    QR_, QRSUM = B.aout(QR_ROWS * HD * 16), B.aout(QR_ROWS * 64)
    qrow = qc.reshape(QR_ROWS, HD)
    part = np.concatenate([qrow[:, HD // 2:], qrow[:, :HD // 2]], 1)
    qr, _ = R.rope_b(part, np.repeat(f32c(sgn), H, 0), qpa)
    B.expect(QR_, qr, 16)
    B.expect(QRSUM, summaries(qr), 64)
    KPA, KPASUM = B.aout(T * HD * 32), B.aout(T * 64)
    kpa, _ = R.rope_a(kc, f32c(cos))
    B.expect(KPA, kpa, 32)
    B.expect(KPASUM, [0] * T, 64)
    KR_, KRSUM = B.aout(T * HD * 16), B.aout(T * 64)
    kpart = np.concatenate([kc[:, HD // 2:], kc[:, :HD // 2]], 1)
    kr, _ = R.rope_b(kpart, f32c(sgn), kpa)
    B.expect(KR_, kr, 16)
    B.expect(KRSUM, summaries(kr), 64)

    # key code region: prefix codes and scales are inputs (the prefix pass wrote them), the suffix rows are
    # quantised into the tail of the same region
    keys = np.zeros((NK, HD), np.int64)
    key_sums = np.zeros(NK, np.int64)
    if LMP:
        kpre_q = np.zeros((0, HD), np.int64)                  # no prefix inputs: every key is computed here
        kpre_s = []
    elif EXP_FULL:
        # the prefix pass's output, exactly as pi0_layer_lower.py computes it from the LM layer
        lm = LL.load_layer(a.layer, "paligemma_layer", a.alpha, "lm", gemms=("k", "v"))
        lm_in = z[f"lm.L{a.layer}.layer_in"].astype(np.float64)                   # (525, 2048)
        cos_p, sin_p = z["lm.rope_cos"].astype(np.float64), z["lm.rope_sin"].astype(np.float64)
        hlc = C.rms(R.bf16_codes(lm_in), f32c((1.0 + lm["g_in"]) / lm["s_qkv"]))
        hlq, s_hl = C.quant(hlc)
        kpc = C.dequant(C.gemm(hlq, lm["k"].codes), s_hl, f32c(lm["k"].scale))
        kpc = C.rope(kpc, f32c(cos_p), f32c(np.concatenate([-sin_p[:, :HD // 2], sin_p[:, HD // 2:]], 1)))
        kpre_q, kpre_s = C.quant(kpc)
        acc_vp = np.pad(C.gemm(hlq, lm["v"].codes).T, ((0, 0), (0, NPS - NPRE)))  # (256, NPS), column-major
        vpre_c = C.dequant(acc_vp, f32c(lm["v"].scale / s_o_hd), np.pad(s_hl, (0, NPS - NPRE)))
    else:
        kpre_c = R.bf16_codes(kpre_f)
        kpre_q, kpre_s = C.quant(kpre_c)
    keys[:NPRE] = kpre_q
    key_sums[:NPRE] = [int(v) for v in kpre_s]
    KEYS = B.put_in(keys & 0xFF, 8)                       # written whole, then the suffix part is overwritten
    KEYSUM = B.put_in(key_sums, 64)
    ksuf_q, ksuf_s = C.quant(kr)
    B.expect(KEYS + NPS * HD, ksuf_q & 0xFF, 8)
    B.expect(KEYSUM + NPS * 8, [int(v) for v in ksuf_s], 64)
    keys[NPS:NPS + T] = ksuf_q
    key_sums[NPS:NPS + T] = [int(v) for v in ksuf_s]
    QQ, QQSUM = B.aout(QR_ROWS * HD * 8), B.aout(QR_ROWS * 64)
    qq, s_q = C.quant(qr)
    B.expect(QQ, qq & 0xFF, 8)
    B.expect(QQSUM, [int(v) for v in s_q], 64)
    # V suffix channel rows (column-major): one QUANT row per channel
    VSQ, VSQSUM = B.aout(HD * TP * 8), B.aout(HD * 64)
    vsq, s_vs = C.quant(vc)
    B.expect(VSQ, vsq & 0xFF, 8)
    B.expect(VSQSUM, [int(v) for v in s_vs], 64)
    B.stage(VN,
            vu_record(OP_DEQUANT, T, QN, QB, QBSUM, desc(ACCQ, SH_E, F_INT32),
                      c=desc(HQSUM, SH_R, F_SUM, FIELD_RS0), d=desc(sq_base, SH_C, F_FP32))
            + vu_record(OP_DEQUANT, T, HD, KB, KBSUM, desc(ACCK, SH_E, F_INT32),
                        c=desc(HQSUM, SH_R, F_SUM, FIELD_RS0), d=desc(sk_base, SH_C, F_FP32))
            + vu_record(OP_DEQUANT, HD, TP, VB, VBSUM, desc(ACCV, SH_E, F_INT32),
                        c=desc(sv_base, SH_R, F_FP32), d=desc(shpad_base, SH_C, F_FP32))
            + vu_record(OP_ROPE_A, QR_ROWS, HD, QPA, QPASUM, desc(QB, SH_E, F_BF16), c=desc(cos_q, SH_E, F_FP32))
            + vu_record(OP_ROPE_B, QR_ROWS, HD, QR_, QRSUM, desc(QB, SH_E, F_BF16), c=desc(sgn_q, SH_E, F_FP32),
                        e=desc(QPA, SH_E, F_FP32), x_rot=1)
            + vu_record(OP_ROPE_A, T, HD, KPA, KPASUM, desc(KB, SH_E, F_BF16), c=desc(cos_k, SH_E, F_FP32))
            + vu_record(OP_ROPE_B, T, HD, KR_, KRSUM, desc(KB, SH_E, F_BF16), c=desc(sgn_k, SH_E, F_FP32),
                        e=desc(KPA, SH_E, F_FP32), x_rot=1)
            + vu_record(OP_QUANT, QR_ROWS, HD, QQ, QQSUM, desc(QR_, SH_E, F_BF16),
                        rs=desc(QRSUM, SH_R, F_SUM, FIELD_AMAX))
            + vu_record(OP_QUANT, T, HD, KEYS + NPS * HD, KEYSUM + NPS * 8, desc(KR_, SH_E, F_BF16),
                        rs=desc(KRSUM, SH_R, F_SUM, FIELD_AMAX))
            + vu_record(OP_QUANT, HD, TP, VSQ, VSQSUM, desc(VB, SH_E, F_BF16),
                        rs=desc(VBSUM, SH_R, F_SUM, FIELD_AMAX)) + [0])

    # ================= stage 3: QK^T =================
    LOGIT = B.aout(QR_ROWS * NK * 32)
    logit_acc = qq @ keys.T
    B.expect(LOGIT, logit_acc & 0xFFFFFFFF, 32)
    B.stage(CH8, cmds_loadx(KEYS, NK, W_HD, QQ, QR_ROWS, LOGIT) + [0])      # key code rows are the columns

    # ================= stage 4: logits -> uint8 P =================
    mask = np.zeros((QR_ROWS, NK), bool)
    if LMP:
        mchip = np.zeros((T, NK), bool)                       # the frame's prefix mask
        mchip[:, :T] = z[f"lm.L{a.layer}.mask"][:T, :T]
        mask[:] = np.repeat(mchip, H, 0)
    elif a.full:
        mchip = np.zeros((T, NK), bool)                       # the frame's own mask, on the slot layout
        mchip[:, :NPRE] = mask867[:, pre_idx]
        mchip[:, NPS:NPS + T] = mask867[:, 816:816 + T]
        mask[:] = np.repeat(mchip, H, 0)
    else:
        mask[:, :NPRE] = True
        for r in range(QR_ROWS):
            mask[r, NPRE:NPRE + (1 if r < H else T)] = True     # the state token sees only itself
    mask_base = B.put_in(mask.astype(np.int64), 16)
    LB, LSUM = B.aout(QR_ROWS * NK * 16), B.aout(QR_ROWS * 64)
    key_sum_codes = np.array([(v & 0xFFFFFFFF) for v in key_sums], np.int64)
    lc = C.dequant(logit_acc, s_q, key_sum_codes.astype(np.uint32))
    B.expect(LB, lc, 16)
    B.expect(LSUM, summaries(lc, mask), 64)
    PC, PSUM = B.aout(QR_ROWS * NK * 8), B.aout(QR_ROWS * 64)
    pq, s_p = C.smax_u8(lc, mask)
    B.expect(PC, pq & 0xFF, 8)
    B.expect(PSUM, [int(v) for v in s_p], 64)
    B.stage(VN,
            vu_record(OP_DEQUANT, QR_ROWS, NK, LB, LSUM, desc(LOGIT, SH_E, F_INT32),
                      c=desc(QQSUM, SH_R, F_SUM, FIELD_RS0), d=desc(KEYSUM, SH_C, F_SUM, FIELD_RS0),
                      mask=desc(mask_base, SH_E, F_BF16))
            + vu_record(OP_SMAX_Q8, QR_ROWS, NK, PC, PSUM, desc(LB, SH_E, F_BF16),
                        rs=desc(LSUM, SH_R, F_SUM, FIELD_MAX), mask=desc(mask_base, SH_E, F_BF16), b_fp32=1)
            + [0])

    # ================= stage 5: PV on the uint8 chain =================
    if not a.full and not LMP:
        vpre_f = LL.rms_fp(lm_in, g_in) @ Wv.w.T.astype(np.float64) / s_o_hd   # real prefix V (LM hidden slice)
        vpre_c = R.bf16_codes(vpre_f.T)                                        # column-major: one row per channel
    ctxs = pq[:, NPS:NPS + TP] @ vsq.T
    if NPS:
        vpre_q, s_vp = C.quant(vpre_c)
        VPQ = B.put_in(vpre_q & 0xFF, 8)
        VPSUM = B.put_in([int(v) for v in s_vp], 64)
        ctxp = pq[:, :NPS] @ vpre_q.T
        CTXP, CTXS = B.aout(QR_ROWS * HD * 32), B.aout(QR_ROWS * HD * 32)
        B.expect(CTXP, ctxp & 0xFFFFFFFF, 32)
        B.expect(CTXS, ctxs & 0xFFFFFFFF, 32)
        B.stage(CHU, cmds_loadx(VPQ, HD, W_PRE, PC, QR_ROWS, CTXP, stride=W_NK)      # V channel rows are the columns
                + cmds_loadx(VSQ, HD, W_SUF, PC + NPS, QR_ROWS, CTXS, stride=W_NK) + [0])
    else:
        CTXS = B.aout(QR_ROWS * HD * 32)
        B.expect(CTXS, ctxs & 0xFFFFFFFF, 32)
        B.stage(CHU, cmds_loadx(VSQ, HD, W_SUF, PC, QR_ROWS, CTXS, stride=W_NK) + [0])

    # ================= stage 6: context =================
    if NPS:
        svp_base = VPSUM
        CP, CPSUM = B.aout(QR_ROWS * HD * 16), B.aout(QR_ROWS * 64)
        cp = C.dequant(ctxp, s_p, np.asarray([int(v) & 0xFFFFFFFF for v in s_vp], np.uint32))
        B.expect(CP, cp, 16)
        B.expect(CPSUM, summaries(cp), 64)
    CS_, CSSUM = B.aout(QR_ROWS * HD * 16), B.aout(QR_ROWS * 64)
    cs_ = C.dequant(ctxs, s_p, np.asarray([int(v) & 0xFFFFFFFF for v in s_vs], np.uint32))
    B.expect(CS_, cs_, 16)
    B.expect(CSSUM, summaries(cs_), 64)
    # the ADD runs on token rows (T x QN, the same bytes as QR_ROWS x HD), so its summary amax is the one
    # the following QUANT needs: a token row's amax is not the amax of either head row
    CTX, CTXSUM = B.aout(QR_ROWS * HD * 16), B.aout(T * 64)
    ctx = C.add(cp, cs_) if NPS else C.add(cs_, np.zeros_like(cs_))     # no prefix part: ADD b = 0 for the amax
    ctx_tok = ctx.reshape(T, QN)
    B.expect(CTX, ctx_tok, 16)
    B.expect(CTXSUM, summaries(ctx_tok), 64)
    CTXQ, CTXQSUM = B.aout(T * QN * 8), B.aout(T * 64)
    ctxq, s_ctx = C.quant(ctx_tok)
    B.expect(CTXQ, ctxq & 0xFF, 8)
    B.expect(CTXQSUM, [int(v) for v in s_ctx], 64)
    B.stage(VN,
            (vu_record(OP_DEQUANT, QR_ROWS, HD, CP, CPSUM, desc(CTXP, SH_E, F_INT32),
                       c=desc(PSUM, SH_R, F_SUM, FIELD_RS0), d=desc(svp_base, SH_C, F_SUM, FIELD_RS0)) if NPS else [])
            + vu_record(OP_DEQUANT, QR_ROWS, HD, CS_, CSSUM, desc(CTXS, SH_E, F_INT32),
                        c=desc(PSUM, SH_R, F_SUM, FIELD_RS0), d=desc(VSQSUM, SH_C, F_SUM, FIELD_RS0))
            + (vu_record(OP_ADD, T, QN, CTX, CTXSUM, desc(CP, SH_E, F_BF16), b=desc(CS_, SH_E, F_BF16)) if NPS else
               vu_record(OP_ADD, T, QN, CTX, CTXSUM, desc(CS_, SH_E, F_BF16), b=K(0)))
            + vu_record(OP_QUANT, T, QN, CTXQ, CTXQSUM, desc(CTX, SH_E, F_BF16),
                        rs=desc(CTXSUM, SH_R, F_SUM, FIELD_AMAX)) + [0])

    # ================= stage 7: o GEMM =================
    acco = C.gemm(ctxq, Wo.codes)
    ACCO = B.aout(T * D * 32)
    B.expect(ACCO, acco & 0xFFFFFFFF, 32)
    B.stage(CH8, gemm_cmds(CTXQ, T, QN // 16, Wo.codes, ACCO, a.tiles) + [0])

    # ================= stage 8: o dequant + residual =================
    so_base = B.put_in(f32c(Wo.scale), 32)
    OB, OSUM = B.aout(T * D * 16), B.aout(T * 64)
    oc = C.dequant(acco, s_ctx, f32c(Wo.scale))
    B.expect(OB, oc, 16)
    B.expect(OSUM, summaries(oc), 64)
    YB, YSUM = B.aout(T * D * 16), B.aout(T * 64)
    yc = C.add(oc, xc)
    B.expect(YB, yc, 16)
    B.expect(YSUM, summaries(yc), 64)
    B.stage(VN,
            vu_record(OP_DEQUANT, T, D, OB, OSUM, desc(ACCO, SH_E, F_INT32),
                      c=desc(CTXQSUM, SH_R, F_SUM, FIELD_RS0), d=desc(so_base, SH_C, F_FP32))
            + vu_record(OP_ADD, T, D, YB, YSUM, desc(OB, SH_E, F_BF16), b=desc(x_base, SH_E, F_BF16)) + [0])

    # ================= stages 9-12: the MLP half (post-attention norm, gate/up, GeGLU, down, residual) =======
    if FF:
        W_FF = FF // 16
        gain2_base = B.put_in(f32c((1.0 + g_post) / s_gu), 32)
        y_amax = R.row_amax_code(yc)
        R2SUM = B.aout(T * 64)
        r2 = R.rms_stat(yc, y_amax, 1.0 / D)
        B.expect(R2SUM, [int(v) for v in r2], 64)
        H2, H2SUM = B.aout(T * D * 16), B.aout(T * 64)
        h2c, _ = R.rms_apply(yc, r2, f32c((1.0 + g_post) / s_gu))
        B.expect(H2, h2c, 16)
        B.expect(H2SUM, summaries(h2c), 64)
        H2Q, H2QSUM = B.aout(T * D * 8), B.aout(T * 64)
        h2q, s_h2 = C.quant(h2c)
        B.expect(H2Q, h2q & 0xFF, 8)
        B.expect(H2QSUM, [int(v) for v in s_h2], 64)
        B.stage(VN,
                vu_record(OP_RMS_STAT, T, D, 0, R2SUM, desc(YB, SH_E, F_BF16),
                          rs=desc(YSUM, SH_R, F_SUM, FIELD_AMAX), k=int(np.float32(1.0 / D).view(np.uint32)))
                + vu_record(OP_RMS_APPLY, T, D, H2, H2SUM, desc(YB, SH_E, F_BF16),
                            c=desc(R2SUM, SH_R, F_SUM, FIELD_RS0), d=desc(gain2_base, SH_C, F_FP32))
                + vu_record(OP_QUANT, T, D, H2Q, H2QSUM, desc(H2, SH_E, F_BF16),
                            rs=desc(H2SUM, SH_R, F_SUM, FIELD_AMAX)) + [0])

        # sub-blocks: a vector row holds at most 2^SLOT_BITS elements (one block when FF fits)
        FB = min(FF, 1 << a.slot_bits)
        assert FF % FB == 0
        NB, W_FB = FF // FB, FB // 16
        accg, accu = C.gemm(h2q, Wg.codes), C.gemm(h2q, Wu.codes)
        ACCG = [B.aout(T * FB * 32) for _ in range(NB)]
        ACCU = [B.aout(T * FB * 32) for _ in range(NB)]
        for j in range(NB):
            B.expect(ACCG[j], accg[:, j * FB:(j + 1) * FB] & 0xFFFFFFFF, 32)
        for j in range(NB):
            B.expect(ACCU[j], accu[:, j * FB:(j + 1) * FB] & 0xFFFFFFFF, 32)
        B.stage(CH8, gemm_cmds(H2Q, T, W_IN, Wg.codes, ACCG, a.tiles, NB)
                + gemm_cmds(H2Q, T, W_IN, Wu.codes, ACCU, a.tiles, NB) + [0])

        sg_base = B.put_in(f32c(Wg.scale), 32)
        su_base = B.put_in(f32c(Wu.scale.astype(np.float64) / s_d), 32)     # the down smoothing folded into up
        recs, MQ, MQSUM, mq, s_m = [], [], [], [], []
        for j in range(NB):
            cols = slice(j * FB, (j + 1) * FB)
            GB, GBSUM = B.aout(T * FB * 16), B.aout(T * 64)
            gc_ = C.dequant(accg[:, cols], s_h2, f32c(Wg.scale[cols]))
            B.expect(GB, gc_, 16)
            B.expect(GBSUM, summaries(gc_), 64)
            UB, UBSUM = B.aout(T * FB * 16), B.aout(T * 64)
            uc_ = C.dequant(accu[:, cols], s_h2, f32c(Wu.scale[cols].astype(np.float64) / s_d[cols]))
            B.expect(UB, uc_, 16)
            B.expect(UBSUM, summaries(uc_), 64)
            GG, GGSUM = B.aout(T * FB * 16), B.aout(T * 64)
            gg = C.geglu(gc_, uc_)
            B.expect(GG, gg, 16)
            B.expect(GGSUM, summaries(gg), 64)
            MQ.append(B.aout(T * FB * 8))
            MQSUM.append(B.aout(T * 64))
            mq_j, s_m_j = C.quant(gg)
            mq.append(mq_j)
            s_m.append(s_m_j)
            B.expect(MQ[j], mq_j & 0xFF, 8)
            B.expect(MQSUM[j], [int(v) for v in s_m_j], 64)
            recs += (vu_record(OP_DEQUANT, T, FB, GB, GBSUM, desc(ACCG[j], SH_E, F_INT32),
                               c=desc(H2QSUM, SH_R, F_SUM, FIELD_RS0), d=desc(sg_base + j * FB * 4, SH_C, F_FP32))
                     + vu_record(OP_DEQUANT, T, FB, UB, UBSUM, desc(ACCU[j], SH_E, F_INT32),
                                 c=desc(H2QSUM, SH_R, F_SUM, FIELD_RS0), d=desc(su_base + j * FB * 4, SH_C, F_FP32))
                     + vu_record(OP_GEGLU, T, FB, GG, GGSUM, desc(GB, SH_E, F_BF16), d=desc(UB, SH_E, F_BF16))
                     + vu_record(OP_QUANT, T, FB, MQ[j], MQSUM[j], desc(GG, SH_E, F_BF16),
                                 rs=desc(GGSUM, SH_R, F_SUM, FIELD_AMAX)))
        B.stage(VN, recs + [0])

        # down: one GEMM per sub-block of its input (a column of W_FB words fits a stage BRAM72K)
        accd = [C.gemm(mq[j], Wd.codes[:, j * FB:(j + 1) * FB]) for j in range(NB)]
        ACCD = [B.aout(T * D * 32) for _ in range(NB)]
        for j in range(NB):
            B.expect(ACCD[j], accd[j] & 0xFFFFFFFF, 32)
        B.stage(CH8, sum((gemm_cmds(MQ[j], T, W_FB, Wd.codes[:, j * FB:(j + 1) * FB], ACCD[j], a.tiles)
                          for j in range(NB)), []) + [0])

        # the partial sums combine inside DEQUANT: fp32 outputs chained through the bias, the last one bf16
        sd_base = B.put_in(f32c(Wd.scale), 32)
        recs, bias, bias_base = [], None, None
        for j in range(NB):
            last = j == NB - 1
            MB, MBSUM = B.aout(T * D * (16 if last else 32)), B.aout(T * 64)
            y, _ = R.op_dequant(accd[j], s_m[j], f32c(Wd.scale), bias,
                                "bf16" if last else "fp32")
            C._n("DEQUANT", accd[j].size)
            B.expect(MB, y, 16 if last else 32)
            B.expect(MBSUM, summaries(y) if last else [0] * T, 64)
            recs += vu_record(OP_DEQUANT, T, D, MB, MBSUM, desc(ACCD[j], SH_E, F_INT32),
                              c=desc(MQSUM[j], SH_R, F_SUM, FIELD_RS0), d=desc(sd_base, SH_C, F_FP32),
                              e=desc(bias_base, SH_E, F_FP32) if bias is not None else None,
                              bias_en=int(bias is not None), out_fp32=int(not last))
            bias, bias_base = y, MB
        mc_ = y
        ZB, ZSUM = B.aout(T * D * 16), B.aout(T * 64)
        zc = C.add(mc_, yc)
        B.expect(ZB, zc, 16)
        B.expect(ZSUM, summaries(zc), 64)
        B.stage(VN, recs + vu_record(OP_ADD, T, D, ZB, ZSUM, desc(MB, SH_E, F_BF16), b=desc(YB, SH_E, F_BF16)) + [0])

    # ================= fp32 reference of the same block =================
    h = LL.rms_fp(x, g_in)
    q = (h @ Wq.w.T.astype(np.float64)).reshape(T, H, HD)
    k = h @ Wk.w.T.astype(np.float64)
    v = h @ Wv.w.T.astype(np.float64)
    qr_f = LL.rope_fp(q, cos[:, None, :], sin[:, None, :]).reshape(QR_ROWS, HD)
    kr_f = LL.rope_fp(k, cos, sin)
    if LMP:
        kpre_f = vpre_f = np.zeros((0, HD))
    elif EXP_FULL:
        hl_f = LL.rms_fp(lm_in, lm["g_in"])
        kpre_f = LL.rope_fp(hl_f @ lm["k"].w.T.astype(np.float64), cos_p, sin_p)
        vpre_f = (hl_f @ lm["v"].w.T.astype(np.float64)) / s_o_hd
    Kall = np.zeros((NK, HD))
    Kall[:NPRE], Kall[NPS:NPS + T] = kpre_f, kr_f
    Vall = np.zeros((NK, HD))
    Vall[:NPRE], Vall[NPS:NPS + T] = vpre_f * s_o_hd, v
    logits = (qr_f @ Kall.T) * scaling
    p = LL.softmax_masked(logits, mask)
    ctx_f = (p @ Vall).reshape(T, QN)
    o_f = ctx_f @ Wo.w.T.astype(np.float64)
    y_f = x + o_f
    if FF:
        h2_f = LL.rms_fp(y_f, g_post)
        g_f, u_f = h2_f @ Wg.w.T.astype(np.float64), h2_f @ Wu.w.T.astype(np.float64)
        m_f = (LL.gelu_tanh(g_f) * u_f) @ Wd.w.T.astype(np.float64)
        z_f = y_f + m_f
    rel = LL.rel
    # every vector op must fit the node it is for: a row may hold 2^SLOT_BITS elements
    widest = 0
    for n, _, words in B.stages:
        if n != VN:
            continue
        for i in range(0, len(words) - 1, 6):
            widest = max(widest, (words[i] >> 60) & 0xFFFF)
    assert widest <= (1 << a.slot_bits), f"a vector op is {widest} wide; build the vector node with SLOT_BITS >= {widest.bit_length()}"
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if a.barrier:
        # one program per node instead of one per stage: each stage waits for the previous stage's flag in
        # GDDR6 and posts its own, so the host starts every node once and never steps in again
        FLAGS = B.ain(len(B.stages) * 256)
        wait_rec = lambda n, w: ([(0xB << 124) | (1 << 8), w, 0, 0, 0, 0] if n == VN      # noqa: E731
                                 else [(7 << 124) | (1 << 64) | w])
        post_rec = lambda n, w: ([(0xC << 124) | (1 << 8), w, 0, 0, 0, 0] if n == VN      # noqa: E731
                                 else [(8 << 124) | (1 << 64) | w])
        per_node: dict[int, list[int]] = {}
        for i, (n, _, words) in enumerate(B.stages):
            body = words[:-1] if words and words[-1] == 0 else words      # drop the stage's END
            prog = per_node.setdefault(n, [])
            if i > 0:
                prog += wait_rec(n, FLAGS + 32 * (i - 1))
            prog += body
            prog += post_rec(n, FLAGS + 32 * i)
            B.expect(FLAGS + 32 * i, [1], 256, name=f"flag{i}")
        lines = []
        base = PROG_BASE + B.PROG_AREA
        for n, prog in sorted(per_node.items()):
            prog.append(0)                                               # END
            B.mem.put(base, prog, 128)
            lines.append(f"{n} {base:x}\n")
            base += max(1, -(-len(prog) * 16 // 0x8000)) * 0x8000        # 0x8000 apart, or more for a long one
        (out / "barrier.txt").write_text("".join(lines))
    elif (out / "barrier.txt").exists():
        (out / "barrier.txt").unlink()          # a stale one would make the testbench run the barrier programs
    B.mem.write(out / "mem_in.hex")
    B.exp.write(out / "mem_exp.hex")
    (out / "stages.txt").write_text("".join(f"{n} {b:x}\n" for n, b, _ in B.stages))
    (out / "regions.txt").write_text("".join(f"{n} {b:x} {sz} {st}\n" for n, b, sz, st in B.regions))
    (out / "info.txt").write_text(
        f"part={a.part} T={T} D={D} H={H} HD={HD} NPRE={NPRE} NPS={NPS} TP={TP} NK={NK} FF={FF} stages={len(B.stages)} "
        f"widest vector row={widest} (SLOT_BITS {a.slot_bits})\n"
        f"image {len(B.mem.beats)} beats, expected {len(B.exp.beats)} beats\n"
        f"chip vs fp32: q_roped {rel(bf16v(qr), qr_f * scaling / LN2):.4f} "
        f"ctx {rel(bf16v(ctx).reshape(T, QN), ctx_f / np.tile(s_o_hd, H)):.4f} "
        f"o {rel(bf16v(oc), o_f):.4f} attn_block {rel(bf16v(yc) - x, y_f - x):.4f}"
        + (f" mlp {rel(bf16v(mc_), m_f):.4f} layer {rel(bf16v(zc) - x, z_f - x):.4f}\n" if FF else "\n")
        + f"max|acc| 2^{math.log2(max(C.max_acc, 1)):.1f}\n")
    print((out / "info.txt").read_text().strip())
    print(f"stages: " + ", ".join(f"{n}@{b:#x}" for n, b, _ in B.stages))


if __name__ == "__main__":
    main()
