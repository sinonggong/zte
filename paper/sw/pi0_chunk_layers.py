#!/usr/bin/env python3
"""D3, part 1: the layer builders of the whole chunk -- transcriptions of the verified per-layer goldens
(pi0_attn_golden.py --full / --part lm, pi0_siglip_golden.py --o-proj token, pi0_vision_ends_golden.py,
pi0_action_head_golden.py) with the chunk's data flow made explicit:

  * inputs are on-chip activations (`Act`: codes, region base, summary base) produced by the previous builder,
    not the captured frame, so the chip numerics (pi0_layer_lower.Chip / vector_unit_ref) propagate through the
    chunk exactly as the nodes will compute them;
  * every static tensor (weights, scales, gains, RoPE, tables, masks) is allocated once and shared by the layer
    instances that use it (the two cameras of a SigLIP layer, the 10 steps of an expert layer);
  * regions live in one pi0_chunk_layout.ChunkMem (D2) instead of a per-layer Build.

Stage lists, commands and records are those of the goldens, so every op runs exactly as it ran bit-exact in
the per-layer RTL gates.  Where the chunk differs from a per-layer test it is noted at the spot:
  - a Gemma layer's RMS_STAT takes its row amax from the previous ADD's summaries (F_SUM) instead of a bf16 amax
    input (the goldens' `ln_stage(..., xsum_is_input)` distinction);
  - the prefix layer writes a second V region scaled for the expert's o_proj smoothing (the expert layer's prefix
    V) and its key region is the head of the expert layer's key region (544 prefix + 64 suffix slots);
  - the expert step's suffix rows are assembled from the state token (host) and the action head's 50 tokens by an
    ADD b = 0 pass that also gives the 51 rows their summaries.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

import vector_unit_ref as R
import pi0_layer_lower as LL
import pi0_attn_golden as AG
from pi0_attn_golden import Tiler, vu_record, summaries, VN, CH8, CHU, LN2
from vu_node_golden import (desc, K, SH_E, SH_R, SH_C, F_BF16, F_FP32, F_INT32, F_SUM, FIELD_RS0, FIELD_MAX, FIELD_AMAX)
from pi0_chunk_layout import ChunkMem
from pi0_chunk_tiler import MixedTiler

OP_ADD, OP_RMS_STAT, OP_RMS_APPLY, OP_LN_STAT, OP_LN_APPLY = 0, 1, 2, 3, 4
OP_ROPE_A, OP_ROPE_B, OP_GELU, OP_GEGLU, OP_SILU = 5, 6, 7, 8, 9
OP_SMAX_Q8, OP_QUANT, OP_DEQUANT, OP_EULER = 12, 13, 14, 15
f32c, bf16v = LL.f32c, LL.bf16v


def slim(W: LL.QW) -> LL.QW:
    """keep only what the chip needs of a quantised GEMM weight: int8 codes and the fp32 scale (a prefix layer's
    int64 codes + fp32 weights are ~1.5 GB; 18 of them do not fit beside the rest of the chunk)"""
    W.codes = W.codes.astype(np.int16)          # int16, not int8: `codes & 0xFF` must stay legal under NumPy 2
    W.w = None
    return W


@dataclass
class Act:
    """an activation on chip: bf16 codes (rows x L) at `base`, one 64-bit summary per row at `sum_base` whose amax
    field is the row amax (what RMS_STAT / LN_STAT / QUANT read), or, when produced by the host, a bf16 amax row
    vector at `amax_base` (a bf16 R-shaped operand)"""
    codes: np.ndarray
    base: int
    sum_base: int = 0
    amax_base: int = 0

    def rs(self) -> int:
        return desc(self.sum_base, SH_R, F_SUM, FIELD_AMAX) if self.sum_base else desc(self.amax_base, SH_R, F_BF16)


@dataclass
class Stage:
    kind: int                   # VN / CH8 / CHU
    words: list
    name: str
    layer: str
    fuse: float = 1.0           # element passes left / before operator fusion (pi0_chunk_program.fuse_quant)


class ChunkBuild:
    """the goldens' Build interface on top of ChunkMem: put_in -> STATIC, aout -> SCRATCH, expect, stage"""

    def __init__(self, mem: ChunkMem):
        self.mem = mem
        self.stages: list[Stage] = []
        self.layer = ""
        self.regions: list[tuple] = []
        # chain depth classes per node kind, {CH8: [(32, n_deep), (16, n_shallow)], CHU: [(16, n_pv)]}; the
        # generator sets them from the array before any layer is built (pi0_chunk_tiler.MixedTiler)
        self.tiler_classes: dict = {CH8: [(AG.N_STAGE, 1)], CHU: [(AG.N_STAGE, 1)]}

    def put_in(self, values, width: int, name: str = "", area: str = "STATIC") -> int:
        vals = self._flat(values, width)
        r = self.mem.alloc(name or f"{self.layer}:in", len(vals) * width // 8, area)
        self.mem.put(r.base, vals, width)
        return r.base

    def alloc(self, nbits: int, name: str = "", area: str = "SCRATCH") -> int:
        return self.mem.alloc(name or f"{self.layer}:out", (nbits + 7) // 8, area).base

    def aout(self, nbits: int, name: str = "") -> int:
        return self.alloc(nbits, name, "SCRATCH")

    @staticmethod
    def _flat(values, width: int):
        if isinstance(values, np.ndarray) and values.dtype.kind in "iu" and width <= 64:
            return values.reshape(-1)
        return [int(v) for v in np.asarray(values, object).reshape(-1)]

    def expect(self, base: int, values, width: int, name: str = "") -> None:
        vals = self._flat(values, width)
        self.mem.expect(base, vals, width)
        self.regions.append((name or f"{self.layer}@{base:#x}", base, len(vals) * width // 8, len(self.stages)))

    def stage(self, node: int, words: list, name: str = "") -> None:
        self.stages.append(Stage(node, list(words), name, self.layer))


def add_pass(B: ChunkBuild, C: LL.Chip, x: Act, rows: int, L: int, name: str) -> Act:
    """ADD b = 0 over the rows: a copy with summaries (the amax the following norm / QUANT reads)"""
    B.layer = name
    Y, YSUM = B.aout(rows * L * 16, name), B.aout(rows * 64, name + ".sum")
    yc = C.add(x.codes, np.zeros_like(x.codes))
    B.expect(Y, yc, 16, name)
    B.expect(YSUM, summaries(yc), 64, name + ".sum")
    B.stage(VN, vu_record(OP_ADD, rows, L, Y, YSUM, desc(x.base, SH_E, F_BF16), b=K(0)) + [0], name)
    return Act(yc, Y, YSUM)


# ====================================================================== Gemma layers (expert and prefix)
class GemmaLayer:
    """one PaliGemma prefix layer (part 'lm': 525 tokens, hidden 2048, FF 16384) or one action-expert layer
    (part 'exp': 51 tokens, hidden 1024, FF 4096, 544 + 64 key slots); weights and scales allocated once."""

    def __init__(self, B: ChunkBuild, C: LL.Chip, L: int, part: str, alpha: float, slot_bits: int,
                 T: int, NPRE: int, ex_s_o_hd=None):
        self.B, self.C, self.L, self.part = B, C, L, part
        B.layer = f"{part}.L{L}"
        self.T = T
        self.D = 2048 if part == "lm" else 1024
        self.H, self.HD = LL.HEADS, LL.HD
        self.FF = 16384 if part == "lm" else 4096
        self.QN, self.QR_ROWS = self.H * self.HD, T * self.H
        if part == "lm":
            self.NPRE = self.NPS = 0
            self.TP = -(-T // 32) * 32
        else:
            self.NPRE = NPRE
            self.NPS = -(-NPRE // 32) * 32
            self.TP = -(-T // 32) * 32
        self.NK = self.NPS + self.TP
        D, HD, NPS, TP, NK = self.D, self.HD, self.NPS, self.TP, self.NK
        assert self.QN % 16 == 0 and HD % 16 == 0 and D % 16 == 0 and NPS % 16 == 0 and (NK // 16) % 2 == 0
        self.W_IN, self.W_HD, self.W_NK, self.W_PRE, self.W_SUF = D // 16, HD // 16, NK // 16, NPS // 16, TP // 16
        assert self.W_PRE % 2 == 0 and self.W_SUF % 2 == 0
        self.FB = min(self.FF, 1 << slot_bits)
        assert self.FF % self.FB == 0
        self.NB, self.W_FB = self.FF // self.FB, self.FB // 16
        gem = ("q", "k", "v", "o", "gate", "up", "down")
        ex = LL.load_layer(L, "paligemma_layer" if part == "lm" else "expert_layer", alpha, part, gemms=gem)
        self.g_in, self.s_qkv, self.s_o_hd = ex["g_in"], ex["s_qkv"], ex["s_o_hd"]
        self.g_post, self.s_gu, self.s_d = ex["g_post"], ex["s_gu"], ex["s_d"]
        # load_layer already quantised each GEMM with its smoothing factors (QW(w, s_in)); keep the int8 codes only
        self.Wq, self.Wk, self.Wv, self.Wo = (slim(ex[g]) for g in ("q", "k", "v", "o"))
        self.Wg, self.Wu, self.Wd = (slim(ex[g]) for g in ("gate", "up", "down"))
        del ex
        assert self.FF % AG.N_STAGE == 0
        TL = self.TL = MixedTiler(B, B.tiler_classes)
        # static: gains, scales, weight images (as the golden lays them out, once per layer)
        self.gain_base = B.put_in(f32c((1.0 + self.g_in) / self.s_qkv), 32, "gain_in")
        self.q_imgs = TL.put_images(self.Wq.codes, self.W_IN)
        self.k_imgs = TL.put_images(self.Wk.codes, self.W_IN)
        self.Wv_rows = B.put_in(np.concatenate([self.Wv.codes[c] for c in range(HD)]) & 0xFF, 8, "v_rows")
        self.sq_base = B.put_in(f32c(self.Wq.scale), 32, "sq")          # rewritten with the scaling in prepare()
        self.sk_base = B.put_in(f32c(self.Wk.scale), 32, "sk")
        self.sv_base = B.put_in(f32c(self.Wv.scale / self.s_o_hd), 32, "sv")
        # the expert's prefix V is the prefix layer's V scaled by the EXPERT's o_proj smoothing
        self.ex_s_o_hd = ex_s_o_hd
        if part == "lm" and ex_s_o_hd is not None:
            self.sv_ex_base = B.put_in(f32c(self.Wv.scale / ex_s_o_hd), 32, "sv_for_expert")
        self.so_base = B.put_in(f32c(self.Wo.scale), 32, "so")
        self.o_imgs = TL.put_images(self.Wo.codes, self.QN // 16, 1, 1)
        self.gain2_base = B.put_in(f32c((1.0 + self.g_post) / self.s_gu), 32, "gain_post")
        self.g_imgs = TL.put_images(self.Wg.codes, self.W_IN, 1, self.NB)
        self.u_imgs = TL.put_images(self.Wu.codes, self.W_IN, 1, self.NB)
        self.sg_base = B.put_in(f32c(self.Wg.scale), 32, "sg")
        self.su_base = B.put_in(f32c(self.Wu.scale.astype(np.float64) / self.s_d), 32, "su")
        self.d_imgs = [TL.put_images(self.Wd.codes[:, j * self.FB:(j + 1) * self.FB], self.W_FB) for j in range(self.NB)]
        self.sd_base = B.put_in(f32c(self.Wd.scale), 32, "sd")
        self.prepared = False

    def prepare(self, scaling: float, cos: np.ndarray, sin: np.ndarray, mask_chip: np.ndarray, shared: dict):
        """per-model constants: attention scaling (log2 domain folded into the q scale), RoPE tables and the key
        mask, shared between the layers of a part (allocated by the first layer that needs them)"""
        B, H, HD = self.B, self.H, self.HD
        B.layer = f"{self.part}.L{self.L}"
        self.scaling = scaling
        sqv = f32c(self.Wq.scale.astype(np.float64) * (scaling / LN2))
        self.B.mem.put(self.sq_base, sqv, 32)                            # overwrite the placeholder scale
        self.sq_vals = sqv
        key = (self.part, "rope")
        if key not in shared:
            sgn = np.concatenate([-sin[:, :HD // 2], sin[:, HD // 2:]], 1)
            shared[key] = dict(cos_q=B.put_in(np.repeat(f32c(cos), H, 0), 32, "rope_cos_q"),
                               sgn_q=B.put_in(np.repeat(f32c(sgn), H, 0), 32, "rope_sgn_q"),
                               cos_k=B.put_in(f32c(cos), 32, "rope_cos_k"), sgn_k=B.put_in(f32c(sgn), 32, "rope_sgn_k"),
                               cos=f32c(cos), sgn=f32c(sgn))
        self.rope = shared[key]
        key = (self.part, "mask")
        if key not in shared:
            mask = np.repeat(mask_chip, H, 0)                            # (QR_ROWS, NK)
            shared[key] = dict(base=B.put_in(mask.astype(np.int64), 16, "mask", area="IO"), mask=mask)
        self.mask, self.mask_base = shared[key]["mask"], shared[key]["base"]
        self.prepared = True

    # ------------------------------------------------------------------ one layer (one step for the expert)
    def run(self, x: Act, prefix=None, keys_region=None, name: str = "") -> dict:
        """x: the residual stream (T x D bf16 codes with summaries).  prefix (expert only): dict with the prefix
        keys / V of this layer as the prefix layer produced them (KEYS region base, KEYSUM, VPQ, VPSUM, and the
        arrays keys_q (NPRE x HD), key_sums, vpre_q (HD x NPS), s_vp).  For the prefix layer keys_region is None
        (it allocates the shared key region, NK_expert slots wide, and returns it in the result)."""
        assert self.prepared
        B, C, TL = self.B, self.C, self.TL
        B.layer = f"{self.part}.L{self.L}{name}"
        T, D, H, HD, QN, QR_ROWS = self.T, self.D, self.H, self.HD, self.QN, self.QR_ROWS
        NPRE, NPS, TP, NK, FF, FB, NB = self.NPRE, self.NPS, self.TP, self.NK, self.FF, self.FB, self.NB
        W_IN, W_HD, W_NK, W_PRE, W_SUF, W_FB = self.W_IN, self.W_HD, self.W_NK, self.W_PRE, self.W_SUF, self.W_FB
        LMP = self.part == "lm"
        xc = x.codes
        x_base = x.base
        put_images, cmds_load, cmds_loadx, gemm_cmds = TL.put_images, TL.cmds_load, TL.cmds_loadx, TL.gemm_cmds
        out = {}

        # ================= stage 0: RMS + QUANT =================
        amax_x = R.row_amax_code(xc)
        RSUM = B.aout(T * 64, "rms")
        r_rms = R.rms_stat(xc, amax_x, 1.0 / D)
        B.expect(RSUM, [int(v) for v in r_rms], 64)
        HB, HSUM = B.aout(T * D * 16, "h"), B.aout(T * 64, "h.sum")
        hc, _ = R.rms_apply(xc, r_rms, f32c((1.0 + self.g_in) / self.s_qkv))
        B.expect(HB, hc, 16)
        B.expect(HSUM, summaries(hc), 64)
        HQ, HQSUM = B.aout(TP * D * 8, "hq"), B.aout(TP * 64, "hq.sum")
        hq, s_h = C.quant(hc)
        B.expect(HQ, hq & 0xFF, 8)
        B.expect(HQSUM, [int(v) for v in s_h], 64)
        B.mem.put(HQSUM, [0] * TP, 64)                                   # the zero tail (rows T..TP-1)
        if TP > T:
            # the pad rows of the quantised input are part of the IMAGE: the V GEMM and QK^T read all TP slots, and
            # only a simulator's memory starts at zero (silicon 2026-09-18: stale GDDR6 there made accv / logit / lb
            # differ in the masked pad columns only; harmless downstream, but not bit-exact)
            assert (T * D) % 32 == 0
            B.mem.put(HQ + T * D, np.zeros((TP - T) * D, np.int64), 8)
        B.stage(VN, vu_record(OP_RMS_STAT, T, D, 0, RSUM, desc(x_base, SH_E, F_BF16),
                              rs=x.rs(), k=int(np.float32(1.0 / D).view(np.uint32)))
                + vu_record(OP_RMS_APPLY, T, D, HB, HSUM, desc(x_base, SH_E, F_BF16),
                            c=desc(RSUM, SH_R, F_SUM, FIELD_RS0), d=desc(self.gain_base, SH_C, F_FP32))
                + vu_record(OP_QUANT, T, D, HQ, HQSUM, desc(HB, SH_E, F_BF16),
                            rs=desc(HSUM, SH_R, F_SUM, FIELD_AMAX)) + [0], "rms_quant")

        # ================= stage 1: q, k and the transposed v GEMM =================
        accq, acck = C.gemm(hq, self.Wq.codes), C.gemm(hq, self.Wk.codes)
        hq_pad = np.zeros((TP, D), np.int64)
        hq_pad[:T] = hq
        accv = LL.imatmul(self.Wv.codes, hq_pad.T)                                            # (HD, TP), column-major V
        ACCQ, ACCK, ACCV = B.aout(T * QN * 32, "accq"), B.aout(T * HD * 32, "acck"), B.aout(HD * TP * 32, "accv")
        B.expect(ACCQ, accq & 0xFFFFFFFF, 32)
        B.expect(ACCK, acck & 0xFFFFFFFF, 32)
        B.expect(ACCV, accv & 0xFFFFFFFF, 32)
        B.stage(CH8, cmds_load(HQ, T, W_IN, self.q_imgs, ACCQ)
                + cmds_load(HQ, T, W_IN, self.k_imgs, ACCK)
                + cmds_loadx(HQ, TP, W_IN, self.Wv_rows, HD, ACCV) + [0], "qkv_gemm")

        # ================= stage 2: dequant, RoPE, quant =================
        # the golden fed the V DEQUANT the padded s_h as an fp32 input; on the chip s_h exists only in the QUANT's
        # summaries, so the DEQUANT reads HQSUM as a C-shaped summary operand over TP columns.  HQSUM is TP
        # entries long with a zero tail (host-written), which is the golden's zero padding bit for bit.
        s_h_pad = np.pad(np.asarray(s_h, np.uint32), (0, TP - T))
        QB, QBSUM = B.aout(T * QN * 16, "qb"), B.aout(T * 64, "qb.sum")
        qc = C.dequant(accq, s_h, self.sq_vals)
        B.expect(QB, qc, 16)
        B.expect(QBSUM, summaries(qc), 64)
        KB, KBSUM = B.aout(T * HD * 16, "kb"), B.aout(T * 64, "kb.sum")
        kc = C.dequant(acck, s_h, f32c(self.Wk.scale))
        B.expect(KB, kc, 16)
        B.expect(KBSUM, summaries(kc), 64)
        VB, VBSUM = B.aout(HD * TP * 16, "vb"), B.aout(HD * 64, "vb.sum")
        vc = C.dequant(accv, f32c(self.Wv.scale / self.s_o_hd), s_h_pad)
        B.expect(VB, vc, 16)
        B.expect(VBSUM, summaries(vc), 64)
        rope = self.rope
        QPA, QPASUM = B.aout(QR_ROWS * HD * 32, "qpa"), B.aout(QR_ROWS * 64, "qpa.sum")
        qpa, _ = R.rope_a(qc.reshape(QR_ROWS, HD), np.repeat(rope["cos"], H, 0))
        B.expect(QPA, qpa, 32)
        B.expect(QPASUM, [0] * QR_ROWS, 64)
        QR_, QRSUM = B.aout(QR_ROWS * HD * 16, "qr"), B.aout(QR_ROWS * 64, "qr.sum")
        qrow = qc.reshape(QR_ROWS, HD)
        part = np.concatenate([qrow[:, HD // 2:], qrow[:, :HD // 2]], 1)
        qr, _ = R.rope_b(part, np.repeat(rope["sgn"], H, 0), qpa)
        B.expect(QR_, qr, 16)
        B.expect(QRSUM, summaries(qr), 64)
        KPA, KPASUM = B.aout(T * HD * 32, "kpa"), B.aout(T * 64, "kpa.sum")
        kpa, _ = R.rope_a(kc, rope["cos"])
        B.expect(KPA, kpa, 32)
        B.expect(KPASUM, [0] * T, 64)
        KR_, KRSUM = B.aout(T * HD * 16, "kr"), B.aout(T * 64, "kr.sum")
        kpart = np.concatenate([kc[:, HD // 2:], kc[:, :HD // 2]], 1)
        kr, _ = R.rope_b(kpart, rope["sgn"], kpa)
        B.expect(KR_, kr, 16)
        B.expect(KRSUM, summaries(kr), 64)

        # key code region: NK slots; the prefix layer owns rows [0, T) of it (its keys are the expert's prefix
        # keys, same numerics), the expert writes its suffix rows [NPS, NPS + T) and reads the prefix rows
        keys = np.zeros((NK, HD), np.int64)
        key_sums = np.zeros(NK, np.int64)
        if LMP:
            KEYS = B.alloc(keys_region["slots"] * HD * 8, "keys", "SCRATCH")
            KEYSUM = B.alloc(keys_region["slots"] * 64, "keys.sum", "SCRATCH")
            out["KEYS"], out["KEYSUM"] = KEYS, KEYSUM
            # the pad key slots (prefix rows T..NPS-1, expert suffix rows past its T) are read by QK^T but written by
            # no node: zero them in the image (silicon 2026-09-18: stale GDDR6 there changed the masked pad columns of
            # logit / lb and every row's lb.sum amax; harmless after the mask, but not bit-exact)
            B.mem.put(KEYS, np.zeros(keys_region["slots"] * HD, np.int64), 8)
            B.mem.put(KEYSUM, np.zeros(keys_region["slots"], np.int64), 64)
        else:
            KEYS, KEYSUM = prefix["KEYS"], prefix["KEYSUM"]
            keys[:NPRE] = prefix["keys_q"]
            key_sums[:NPRE] = prefix["key_sums"]
        ksuf_q, ksuf_s = C.quant(kr)
        B.expect(KEYS + NPS * HD, ksuf_q & 0xFF, 8)
        B.expect(KEYSUM + NPS * 8, [int(v) for v in ksuf_s], 64)
        keys[NPS:NPS + T] = ksuf_q
        key_sums[NPS:NPS + T] = [int(v) for v in ksuf_s]
        if LMP:
            out["keys_q"], out["key_sums"] = ksuf_q.copy(), [int(v) for v in ksuf_s]
        QQ, QQSUM = B.aout(QR_ROWS * HD * 8, "qq"), B.aout(QR_ROWS * 64, "qq.sum")
        qq, s_q = C.quant(qr)
        B.expect(QQ, qq & 0xFF, 8)
        B.expect(QQSUM, [int(v) for v in s_q], 64)
        VSQ, VSQSUM = B.aout(HD * TP * 8, "vsq"), B.aout(HD * 64, "vsq.sum")
        vsq, s_vs = C.quant(vc)
        B.expect(VSQ, vsq & 0xFF, 8)
        B.expect(VSQSUM, [int(v) for v in s_vs], 64)
        recs = (vu_record(OP_DEQUANT, T, QN, QB, QBSUM, desc(ACCQ, SH_E, F_INT32),
                          c=desc(HQSUM, SH_R, F_SUM, FIELD_RS0), d=desc(self.sq_base, SH_C, F_FP32))
                + vu_record(OP_DEQUANT, T, HD, KB, KBSUM, desc(ACCK, SH_E, F_INT32),
                            c=desc(HQSUM, SH_R, F_SUM, FIELD_RS0), d=desc(self.sk_base, SH_C, F_FP32))
                # V (channel rows x TP token columns): the column scale is the token QUANT's s_h, padded
                + vu_record(OP_DEQUANT, HD, TP, VB, VBSUM, desc(ACCV, SH_E, F_INT32),
                            c=desc(self.sv_base, SH_R, F_FP32), d=desc(HQSUM, SH_C, F_SUM, FIELD_RS0))
                + vu_record(OP_ROPE_A, QR_ROWS, HD, QPA, QPASUM, desc(QB, SH_E, F_BF16), c=desc(rope["cos_q"], SH_E, F_FP32))
                + vu_record(OP_ROPE_B, QR_ROWS, HD, QR_, QRSUM, desc(QB, SH_E, F_BF16), c=desc(rope["sgn_q"], SH_E, F_FP32),
                            e=desc(QPA, SH_E, F_FP32), x_rot=1)
                + vu_record(OP_ROPE_A, T, HD, KPA, KPASUM, desc(KB, SH_E, F_BF16), c=desc(rope["cos_k"], SH_E, F_FP32))
                + vu_record(OP_ROPE_B, T, HD, KR_, KRSUM, desc(KB, SH_E, F_BF16), c=desc(rope["sgn_k"], SH_E, F_FP32),
                            e=desc(KPA, SH_E, F_FP32), x_rot=1)
                + vu_record(OP_QUANT, QR_ROWS, HD, QQ, QQSUM, desc(QR_, SH_E, F_BF16),
                            rs=desc(QRSUM, SH_R, F_SUM, FIELD_AMAX))
                + vu_record(OP_QUANT, T, HD, KEYS + NPS * HD, KEYSUM + NPS * 8, desc(KR_, SH_E, F_BF16),
                            rs=desc(KRSUM, SH_R, F_SUM, FIELD_AMAX))
                + vu_record(OP_QUANT, HD, TP, VSQ, VSQSUM, desc(VB, SH_E, F_BF16),
                            rs=desc(VBSUM, SH_R, F_SUM, FIELD_AMAX)))
        if LMP and self.ex_s_o_hd is not None:
            # the same V, scaled for the expert's o_proj smoothing, quantised per channel: the expert's prefix V
            VBX, VBXSUM = B.aout(HD * TP * 16, "vb_ex"), B.aout(HD * 64, "vb_ex.sum")
            vcx = C.dequant(accv, f32c(self.Wv.scale / self.ex_s_o_hd), s_h_pad)
            B.expect(VBX, vcx, 16)
            B.expect(VBXSUM, summaries(vcx), 64)
            VPQ, VPSUM = B.aout(HD * TP * 8, "vpq"), B.aout(HD * 64, "vpq.sum")
            vpq, s_vp = C.quant(vcx)
            B.expect(VPQ, vpq & 0xFF, 8)
            B.expect(VPSUM, [int(v) for v in s_vp], 64)
            recs += (vu_record(OP_DEQUANT, HD, TP, VBX, VBXSUM, desc(ACCV, SH_E, F_INT32),
                               c=desc(self.sv_ex_base, SH_R, F_FP32), d=desc(HQSUM, SH_C, F_SUM, FIELD_RS0))
                     + vu_record(OP_QUANT, HD, TP, VPQ, VPSUM, desc(VBX, SH_E, F_BF16),
                                 rs=desc(VBXSUM, SH_R, F_SUM, FIELD_AMAX)))
            out.update(VPQ=VPQ, VPSUM=VPSUM, vpre_q=vpq, s_vp=[int(v) for v in s_vp], TP=TP)
        B.stage(VN, recs + [0], "dequant_rope_quant")

        # ================= stage 3: QK^T =================
        LOGIT = B.aout(QR_ROWS * NK * 32, "logit")
        logit_acc = LL.imatmul(qq, keys.T)
        B.expect(LOGIT, logit_acc & 0xFFFFFFFF, 32)
        B.stage(CH8, cmds_loadx(KEYS, NK, W_HD, QQ, QR_ROWS, LOGIT) + [0], "qk_gemm")

        # ================= stage 4: logits -> uint8 P =================
        mask = self.mask
        LB, LSUM = B.aout(QR_ROWS * NK * 16, "lb"), B.aout(QR_ROWS * 64, "lb.sum")
        key_sum_codes = np.array([(v & 0xFFFFFFFF) for v in key_sums], np.int64)
        lc = C.dequant(logit_acc, s_q, key_sum_codes.astype(np.uint32))
        B.expect(LB, lc, 16)
        B.expect(LSUM, summaries(lc, mask), 64)
        PC, PSUM = B.aout(QR_ROWS * NK * 8, "pc"), B.aout(QR_ROWS * 64, "pc.sum")
        pq, s_p = C.smax_u8(lc, mask)
        B.expect(PC, pq & 0xFF, 8)
        B.expect(PSUM, [int(v) for v in s_p], 64)
        B.stage(VN,
                vu_record(OP_DEQUANT, QR_ROWS, NK, LB, LSUM, desc(LOGIT, SH_E, F_INT32),
                          c=desc(QQSUM, SH_R, F_SUM, FIELD_RS0), d=desc(KEYSUM, SH_C, F_SUM, FIELD_RS0),
                          mask=desc(self.mask_base, SH_E, F_BF16))
                + vu_record(OP_SMAX_Q8, QR_ROWS, NK, PC, PSUM, desc(LB, SH_E, F_BF16),
                            rs=desc(LSUM, SH_R, F_SUM, FIELD_MAX), mask=desc(self.mask_base, SH_E, F_BF16), b_fp32=1)
                + [0], "logits_softmax")

        # ================= stage 5: PV on the uint8 chain =================
        ctxs = LL.imatmul(pq[:, NPS:NPS + TP], vsq.T)
        if NPS:
            vpre_q, s_vp = prefix["vpre_q"], prefix["s_vp"]
            VPQ, VPSUM = prefix["VPQ"], prefix["VPSUM"]
            ctxp = LL.imatmul(pq[:, :NPS], vpre_q[:, :NPS].T)
            CTXP, CTXS = B.aout(QR_ROWS * HD * 32, "ctxp"), B.aout(QR_ROWS * HD * 32, "ctxs")
            B.expect(CTXP, ctxp & 0xFFFFFFFF, 32)
            B.expect(CTXS, ctxs & 0xFFFFFFFF, 32)
            B.stage(CHU, cmds_loadx(VPQ, HD, W_PRE, PC, QR_ROWS, CTXP, stride=W_NK, kind=CHU)
                    + cmds_loadx(VSQ, HD, W_SUF, PC + NPS, QR_ROWS, CTXS, stride=W_NK, kind=CHU) + [0], "pv_gemm")
        else:
            CTXS = B.aout(QR_ROWS * HD * 32, "ctxs")
            B.expect(CTXS, ctxs & 0xFFFFFFFF, 32)
            B.stage(CHU, cmds_loadx(VSQ, HD, W_SUF, PC, QR_ROWS, CTXS, stride=W_NK, kind=CHU) + [0], "pv_gemm")

        # ================= stage 6: context =================
        if NPS:
            CP, CPSUM = B.aout(QR_ROWS * HD * 16, "cp"), B.aout(QR_ROWS * 64, "cp.sum")
            cp = C.dequant(ctxp, s_p, np.asarray([int(v) & 0xFFFFFFFF for v in s_vp], np.uint32))
            B.expect(CP, cp, 16)
            B.expect(CPSUM, summaries(cp), 64)
        CS_, CSSUM = B.aout(QR_ROWS * HD * 16, "cs"), B.aout(QR_ROWS * 64, "cs.sum")
        cs_ = C.dequant(ctxs, s_p, np.asarray([int(v) & 0xFFFFFFFF for v in s_vs], np.uint32))
        B.expect(CS_, cs_, 16)
        B.expect(CSSUM, summaries(cs_), 64)
        CTX, CTXSUM = B.aout(QR_ROWS * HD * 16, "ctx"), B.aout(T * 64, "ctx.sum")
        ctx = C.add(cp, cs_) if NPS else C.add(cs_, np.zeros_like(cs_))
        ctx_tok = ctx.reshape(T, QN)
        B.expect(CTX, ctx_tok, 16)
        B.expect(CTXSUM, summaries(ctx_tok), 64)
        CTXQ, CTXQSUM = B.aout(T * QN * 8, "ctxq"), B.aout(T * 64, "ctxq.sum")
        ctxq, s_ctx = C.quant(ctx_tok)
        B.expect(CTXQ, ctxq & 0xFF, 8)
        B.expect(CTXQSUM, [int(v) for v in s_ctx], 64)
        B.stage(VN,
                (vu_record(OP_DEQUANT, QR_ROWS, HD, CP, CPSUM, desc(CTXP, SH_E, F_INT32),
                           c=desc(PSUM, SH_R, F_SUM, FIELD_RS0), d=desc(VPSUM, SH_C, F_SUM, FIELD_RS0)) if NPS else [])
                + vu_record(OP_DEQUANT, QR_ROWS, HD, CS_, CSSUM, desc(CTXS, SH_E, F_INT32),
                            c=desc(PSUM, SH_R, F_SUM, FIELD_RS0), d=desc(VSQSUM, SH_C, F_SUM, FIELD_RS0))
                + (vu_record(OP_ADD, T, QN, CTX, CTXSUM, desc(CP, SH_E, F_BF16), b=desc(CS_, SH_E, F_BF16)) if NPS else
                   vu_record(OP_ADD, T, QN, CTX, CTXSUM, desc(CS_, SH_E, F_BF16), b=K(0)))
                + vu_record(OP_QUANT, T, QN, CTXQ, CTXQSUM, desc(CTX, SH_E, F_BF16),
                            rs=desc(CTXSUM, SH_R, F_SUM, FIELD_AMAX)) + [0], "context")

        # ================= stage 7: o GEMM =================
        acco = C.gemm(ctxq, self.Wo.codes)
        ACCO = B.aout(T * D * 32, "acco")
        B.expect(ACCO, acco & 0xFFFFFFFF, 32)
        B.stage(CH8, cmds_load(CTXQ, T, QN // 16, self.o_imgs, ACCO) + [0], "o_gemm")

        # ================= stage 8: o dequant + residual =================
        OB, OSUM = B.aout(T * D * 16, "ob"), B.aout(T * 64, "ob.sum")
        oc = C.dequant(acco, s_ctx, f32c(self.Wo.scale))
        B.expect(OB, oc, 16)
        B.expect(OSUM, summaries(oc), 64)
        YB, YSUM = B.aout(T * D * 16, "yb"), B.aout(T * 64, "yb.sum")
        yc = C.add(oc, xc)
        B.expect(YB, yc, 16)
        B.expect(YSUM, summaries(yc), 64)
        B.stage(VN,
                vu_record(OP_DEQUANT, T, D, OB, OSUM, desc(ACCO, SH_E, F_INT32),
                          c=desc(CTXQSUM, SH_R, F_SUM, FIELD_RS0), d=desc(self.so_base, SH_C, F_FP32))
                + vu_record(OP_ADD, T, D, YB, YSUM, desc(OB, SH_E, F_BF16), b=desc(x_base, SH_E, F_BF16)) + [0],
                "o_dequant_residual")

        # ================= stages 9-12: MLP =================
        y_amax = R.row_amax_code(yc)
        R2SUM = B.aout(T * 64, "rms2")
        r2 = R.rms_stat(yc, y_amax, 1.0 / D)
        B.expect(R2SUM, [int(v) for v in r2], 64)
        H2, H2SUM = B.aout(T * D * 16, "h2"), B.aout(T * 64, "h2.sum")
        h2c, _ = R.rms_apply(yc, r2, f32c((1.0 + self.g_post) / self.s_gu))
        B.expect(H2, h2c, 16)
        B.expect(H2SUM, summaries(h2c), 64)
        H2Q, H2QSUM = B.aout(T * D * 8, "h2q"), B.aout(T * 64, "h2q.sum")
        h2q, s_h2 = C.quant(h2c)
        B.expect(H2Q, h2q & 0xFF, 8)
        B.expect(H2QSUM, [int(v) for v in s_h2], 64)
        B.stage(VN,
                vu_record(OP_RMS_STAT, T, D, 0, R2SUM, desc(YB, SH_E, F_BF16),
                          rs=desc(YSUM, SH_R, F_SUM, FIELD_AMAX), k=int(np.float32(1.0 / D).view(np.uint32)))
                + vu_record(OP_RMS_APPLY, T, D, H2, H2SUM, desc(YB, SH_E, F_BF16),
                            c=desc(R2SUM, SH_R, F_SUM, FIELD_RS0), d=desc(self.gain2_base, SH_C, F_FP32))
                + vu_record(OP_QUANT, T, D, H2Q, H2QSUM, desc(H2, SH_E, F_BF16),
                            rs=desc(H2SUM, SH_R, F_SUM, FIELD_AMAX)) + [0], "rms2_quant")

        accg, accu = C.gemm(h2q, self.Wg.codes), C.gemm(h2q, self.Wu.codes)
        ACCG = [B.aout(T * FB * 32, f"accg{j}") for j in range(NB)]
        ACCU = [B.aout(T * FB * 32, f"accu{j}") for j in range(NB)]
        for j in range(NB):
            B.expect(ACCG[j], accg[:, j * FB:(j + 1) * FB] & 0xFFFFFFFF, 32)
        for j in range(NB):
            B.expect(ACCU[j], accu[:, j * FB:(j + 1) * FB] & 0xFFFFFFFF, 32)
        B.stage(CH8, cmds_load(H2Q, T, W_IN, self.g_imgs, ACCG) + cmds_load(H2Q, T, W_IN, self.u_imgs, ACCU) + [0],
                "gate_up_gemm")

        recs, MQ, MQSUM, mq, s_m = [], [], [], [], []
        for j in range(NB):
            cols = slice(j * FB, (j + 1) * FB)
            GB, GBSUM = B.aout(T * FB * 16, f"gb{j}"), B.aout(T * 64, f"gb{j}.sum")
            gc_ = C.dequant(accg[:, cols], s_h2, f32c(self.Wg.scale[cols]))
            B.expect(GB, gc_, 16)
            B.expect(GBSUM, summaries(gc_), 64)
            UB, UBSUM = B.aout(T * FB * 16, f"ub{j}"), B.aout(T * 64, f"ub{j}.sum")
            uc_ = C.dequant(accu[:, cols], s_h2, f32c(self.Wu.scale[cols].astype(np.float64) / self.s_d[cols]))
            B.expect(UB, uc_, 16)
            B.expect(UBSUM, summaries(uc_), 64)
            GG, GGSUM = B.aout(T * FB * 16, f"gg{j}"), B.aout(T * 64, f"gg{j}.sum")
            gg = C.geglu(gc_, uc_)
            B.expect(GG, gg, 16)
            B.expect(GGSUM, summaries(gg), 64)
            MQ.append(B.aout(T * FB * 8, f"mq{j}"))
            MQSUM.append(B.aout(T * 64, f"mq{j}.sum"))
            mq_j, s_m_j = C.quant(gg)
            mq.append(mq_j)
            s_m.append(s_m_j)
            B.expect(MQ[j], mq_j & 0xFF, 8)
            B.expect(MQSUM[j], [int(v) for v in s_m_j], 64)
            recs += (vu_record(OP_DEQUANT, T, FB, GB, GBSUM, desc(ACCG[j], SH_E, F_INT32),
                               c=desc(H2QSUM, SH_R, F_SUM, FIELD_RS0), d=desc(self.sg_base + j * FB * 4, SH_C, F_FP32))
                     + vu_record(OP_DEQUANT, T, FB, UB, UBSUM, desc(ACCU[j], SH_E, F_INT32),
                                 c=desc(H2QSUM, SH_R, F_SUM, FIELD_RS0), d=desc(self.su_base + j * FB * 4, SH_C, F_FP32))
                     + vu_record(OP_GEGLU, T, FB, GG, GGSUM, desc(GB, SH_E, F_BF16), d=desc(UB, SH_E, F_BF16))
                     + vu_record(OP_QUANT, T, FB, MQ[j], MQSUM[j], desc(GG, SH_E, F_BF16),
                                 rs=desc(GGSUM, SH_R, F_SUM, FIELD_AMAX)))
        B.stage(VN, recs + [0], "geglu_quant")

        accd = [C.gemm(mq[j], self.Wd.codes[:, j * FB:(j + 1) * FB]) for j in range(NB)]
        ACCD = [B.aout(T * D * 32, f"accd{j}") for j in range(NB)]
        for j in range(NB):
            B.expect(ACCD[j], accd[j] & 0xFFFFFFFF, 32)
        B.stage(CH8, sum((cmds_load(MQ[j], T, W_FB, self.d_imgs[j], ACCD[j]) for j in range(NB)), []) + [0], "down_gemm")

        recs, bias, bias_base = [], None, None
        for j in range(NB):
            last = j == NB - 1
            MB, MBSUM = B.aout(T * D * (16 if last else 32), f"mb{j}"), B.aout(T * 64, f"mb{j}.sum")
            y, _ = R.op_dequant(accd[j], s_m[j], f32c(self.Wd.scale), bias, "bf16" if last else "fp32")
            C._n("DEQUANT", accd[j].size)
            B.expect(MB, y, 16 if last else 32)
            B.expect(MBSUM, summaries(y) if last else [0] * T, 64)
            recs += vu_record(OP_DEQUANT, T, D, MB, MBSUM, desc(ACCD[j], SH_E, F_INT32),
                              c=desc(MQSUM[j], SH_R, F_SUM, FIELD_RS0), d=desc(self.sd_base, SH_C, F_FP32),
                              e=desc(bias_base, SH_E, F_FP32) if bias is not None else None,
                              bias_en=int(bias is not None), out_fp32=int(not last))
            bias, bias_base = y, MB
        mc_ = y
        ZB, ZSUM = B.aout(T * D * 16, "zb"), B.aout(T * 64, "zb.sum")
        zc = C.add(mc_, yc)
        B.expect(ZB, zc, 16)
        B.expect(ZSUM, summaries(zc), 64)
        B.stage(VN, recs + vu_record(OP_ADD, T, D, ZB, ZSUM, desc(MB, SH_E, F_BF16), b=desc(YB, SH_E, F_BF16)) + [0],
                "down_dequant_residual")
        out["z"] = Act(zc, ZB, ZSUM)
        out["y"] = Act(yc, YB, YSUM)
        return out


# ====================================================================== SigLIP encoder layer (token-major o_proj)
class SiglipLayer:
    """one SigLIP So400m/14 encoder layer at real size (pi0_siglip_golden.py --o-proj token): weights shared by the
    two cameras; run(x) emits the 14 stages for one camera's 256 tokens."""
    HEADS, HD, HDP = 16, 72, 96
    D, FF, FFP, NB, FB = 1152, 4304, 4352, 2, 2176

    def __init__(self, B: ChunkBuild, C: LL.Chip, L: int, scaling: float):
        import pi0_siglip_golden as SG
        self.B, self.C, self.L = B, C, L
        B.layer = f"vis.L{L}"
        HEADS, HD, HDP, D, FF, FFP, NB, FB = self.HEADS, self.HD, self.HDP, self.D, self.FF, self.FFP, self.NB, self.FB
        t = SG.load_vis_layer(L)
        self.t = t
        self.T = T = 256
        self.W_IN, self.W_HP, self.W_T, self.W_FB = D // 16, HDP // 16, T // 16, FB // 16
        self.QN = HEADS * HDP

        def qpad_out(name):
            q = LL.QW(t[name + ".w"], None)
            return (SG.pad_heads_out(q.codes), SG.pad_heads_out(q.scale), SG.pad_heads_out(t[name + ".b"]), q)
        self.cq, self.sq, self.bq, self.Wq = qpad_out("q_proj")
        self.ck, self.sk, self.bk, self.Wk = qpad_out("k_proj")
        self.cv, self.sv, self.bv, self.Wv = qpad_out("v_proj")
        self.Wo = LL.QW(t["out_proj.w"], None)
        self.co = SG.pad_heads_in(self.Wo.codes).astype(np.int16)
        W1 = LL.QW(t["fc1.w"], None)
        self.c1 = np.zeros((FFP, D), np.int16)
        self.c1[:FF] = W1.codes
        self.s1 = np.zeros(FFP, np.float32)
        self.s1[:FF] = W1.scale
        self.b1 = np.zeros(FFP, np.float32)
        self.b1[:FF] = t["fc1.b"]
        self.W2 = LL.QW(t["fc2.w"], None)
        self.c2 = np.zeros((D, FFP), np.int16)
        self.c2[:, :FF] = self.W2.codes
        self.cq, self.ck, self.cv = (c.astype(np.int16) for c in (self.cq, self.ck, self.cv))
        for W in (self.Wq, self.Wk, self.Wv, self.Wo, W1, self.W2):
            slim(W)
        for k in ("q_proj.w", "k_proj.w", "v_proj.w", "out_proj.w", "fc1.w", "fc2.w"):
            t[k] = None                                              # the fp32 weights are not needed any more
        TL = self.TL = MixedTiler(B, B.tiler_classes)
        # static allocations
        self.ln1_g, self.ln1_b = B.put_in(f32c(t["layer_norm1.w"]), 32, "ln1_g"), B.put_in(f32c(t["layer_norm1.b"]), 32, "ln1_b")
        self.ln2_g, self.ln2_b = B.put_in(f32c(t["layer_norm2.w"]), 32, "ln2_g"), B.put_in(f32c(t["layer_norm2.b"]), 32, "ln2_b")
        self.q_imgs = TL.put_images(self.cq, self.W_IN, 1, HEADS)
        self.k_imgs = TL.put_images(self.ck, self.W_IN, 1, HEADS)
        self.Wv_rows = B.put_in(self.cv.reshape(-1) & 0xFF, 8, "v_rows")
        self.sq_f = f32c(self.sq.astype(np.float64) * (scaling / LN2))
        self.bq_f = f32c(self.bq.astype(np.float64) * (scaling / LN2))
        self.sq_base, self.bq_base = B.put_in(self.sq_f, 32, "sq"), B.put_in(self.bq_f, 32, "bq")
        self.sk_base, self.bk_base = B.put_in(f32c(self.sk), 32, "sk"), B.put_in(f32c(self.bk), 32, "bk")
        self.sv_base, self.bv_base = B.put_in(f32c(self.sv), 32, "sv"), B.put_in(f32c(self.bv), 32, "bv")
        self.o_imgs = TL.put_images(self.co, self.QN // 16)
        self.so_base, self.bo_base = B.put_in(f32c(self.Wo.scale), 32, "so"), B.put_in(f32c(t["out_proj.b"]), 32, "bo")
        self.f1_imgs = TL.put_images(self.c1, self.W_IN, 1, NB)
        self.s1_base, self.b1_base = B.put_in(f32c(self.s1), 32, "s1"), B.put_in(f32c(self.b1), 32, "b1")
        self.f2_imgs = [TL.put_images(self.c2[:, FB * j:FB * (j + 1)], self.W_FB) for j in range(NB)]
        self.s2_base, self.b2_base = B.put_in(f32c(self.W2.scale), 32, "s2"), B.put_in(f32c(t["fc2.b"]), 32, "b2")

    def run(self, x: Act, name: str = "") -> dict:
        B, C, TL = self.B, self.C, self.TL
        B.layer = f"vis.L{self.L}{name}"
        HEADS, HD, HDP, D, FF, FFP, NB, FB, T, QN = (self.HEADS, self.HD, self.HDP, self.D, self.FF, self.FFP, self.NB,
                                                     self.FB, self.T, self.QN)
        W_IN, W_HP, W_T, W_FB = self.W_IN, self.W_HP, self.W_T, self.W_FB
        t = self.t
        xc = x.codes

        def dequant(acc, s_row, s_col, bias=None, out="bf16"):
            y, _ = R.op_dequant(acc, s_row, s_col, bias, out)
            C._n("DEQUANT", acc.size)
            return y

        def ln(codes, amax, gain, beta):
            mu, r = R.ln_stat(codes, amax, 1.0 / D)
            y, _ = R.ln_apply(codes, mu, r, f32c(gain), f32c(beta))
            C._n("LN", codes.size * 2)
            return mu, r, y

        def ln_stage(xa: Act, codes, gain, beta, g_base, b_base, tag):
            LS, LSSUM = B.aout(T * 32, tag + ".ls"), B.aout(T * 64, tag + ".ls.sum")
            mu, r, hc = ln(codes, R.row_amax_code(codes), gain, beta)
            B.expect(LS, r, 32)
            B.expect(LSSUM, [int(v) & 0xFFFFFFFF for v in mu], 64)
            HB, HSUM = B.aout(T * D * 16, tag + ".h"), B.aout(T * 64, tag + ".h.sum")
            B.expect(HB, hc, 16)
            B.expect(HSUM, summaries(hc), 64)
            HQ, HQSUM = B.aout(T * D * 8, tag + ".hq"), B.aout(T * 64, tag + ".hq.sum")
            hq, s_h = C.quant(hc)
            B.expect(HQ, hq & 0xFF, 8)
            B.expect(HQSUM, [int(v) for v in s_h], 64)
            B.stage(VN, vu_record(OP_LN_STAT, T, D, LS, LSSUM, desc(xa.base, SH_E, F_BF16), rs=xa.rs(),
                                  k=int(np.float32(1.0 / D).view(np.uint32)))
                    + vu_record(OP_LN_APPLY, T, D, HB, HSUM, desc(xa.base, SH_E, F_BF16),
                                b=desc(LSSUM, SH_R, F_SUM, FIELD_RS0), c=desc(LS, SH_R, F_FP32),
                                d=desc(g_base, SH_C, F_FP32), e=desc(b_base, SH_C, F_FP32))
                    + vu_record(OP_QUANT, T, D, HQ, HQSUM, desc(HB, SH_E, F_BF16),
                                rs=desc(HSUM, SH_R, F_SUM, FIELD_AMAX)) + [0], tag)
            return hc, hq, s_h, HQ, HQSUM

        # stage 0: LayerNorm 1 + QUANT
        hc, hq, s_h, HQ, HQSUM = ln_stage(x, xc, t["layer_norm1.w"], t["layer_norm1.b"], self.ln1_g, self.ln1_b, "ln1")
        # stage 1: q, k per head; v role-swapped
        accq, acck = C.gemm(hq, self.cq), C.gemm(hq, self.ck)
        accv = LL.imatmul(self.cv, hq.T)
        ACCQ = [B.aout(T * HDP * 32, f"accq{h}") for h in range(HEADS)]
        ACCK = [B.aout(T * HDP * 32, f"acck{h}") for h in range(HEADS)]
        ACCV = B.aout(QN * T * 32, "accv")
        for h in range(HEADS):
            B.expect(ACCQ[h], accq[:, HDP * h:HDP * (h + 1)] & 0xFFFFFFFF, 32)
        for h in range(HEADS):
            B.expect(ACCK[h], acck[:, HDP * h:HDP * (h + 1)] & 0xFFFFFFFF, 32)
        B.expect(ACCV, accv & 0xFFFFFFFF, 32)
        cmds = TL.cmds_load(HQ, T, W_IN, self.q_imgs, ACCQ) + TL.cmds_load(HQ, T, W_IN, self.k_imgs, ACCK)
        cmds += TL.cmds_loadx(HQ, T, W_IN, self.Wv_rows, QN, ACCV)
        B.stage(CH8, cmds + [0], "qkv_gemm")
        # stage 2: DEQUANT q / k / v (+ bias), QUANT
        recs = []
        QB, QBSUM = B.aout(HEADS * T * HDP * 16, "qb"), B.aout(HEADS * T * 64, "qb.sum")
        KB, KBSUM = B.aout(HEADS * T * HDP * 16, "kb"), B.aout(HEADS * T * 64, "kb.sum")
        qb, kb = np.zeros((HEADS * T, HDP), np.uint16), np.zeros((HEADS * T, HDP), np.uint16)
        for nm, acc, s_c, b_c, sb, bb, OUT, OSUM, dst, ACC in (
                ("q", accq, self.sq_f, self.bq_f, self.sq_base, self.bq_base, QB, QBSUM, qb, ACCQ),
                ("k", acck, f32c(self.sk), f32c(self.bk), self.sk_base, self.bk_base, KB, KBSUM, kb, ACCK)):
            for h in range(HEADS):
                cols = slice(HDP * h, HDP * (h + 1))
                y = dequant(acc[:, cols], s_h, s_c[cols], np.repeat(b_c[None, cols], T, 0))
                dst[h * T:(h + 1) * T] = y
                B.expect(OUT + h * T * HDP * 2, y, 16)
                B.expect(OSUM + h * T * 8, summaries(y), 64)
                recs += vu_record(OP_DEQUANT, T, HDP, OUT + h * T * HDP * 2, OSUM + h * T * 8, desc(ACC[h], SH_E, F_INT32),
                                  c=desc(HQSUM, SH_R, F_SUM, FIELD_RS0), d=desc(sb + h * HDP * 4, SH_C, F_FP32),
                                  e=desc(bb + h * HDP * 4, SH_C, F_FP32), bias_en=1)
        VB, VBSUM = B.aout(QN * T * 16, "vb"), B.aout(QN * 64, "vb.sum")
        s_h_u32 = np.asarray(s_h, np.uint32)
        vb = dequant(accv, f32c(self.sv), s_h_u32, np.repeat(f32c(self.bv)[:, None], T, 1))
        B.expect(VB, vb, 16)
        B.expect(VBSUM, summaries(vb), 64)
        recs += vu_record(OP_DEQUANT, QN, T, VB, VBSUM, desc(ACCV, SH_E, F_INT32), c=desc(self.sv_base, SH_R, F_FP32),
                          d=desc(HQSUM, SH_C, F_SUM, FIELD_RS0), e=desc(self.bv_base, SH_R, F_FP32), bias_en=1)
        QQ, QQSUM = B.aout(HEADS * T * HDP * 8, "qq"), B.aout(HEADS * T * 64, "qq.sum")
        qq, s_q = C.quant(qb)
        B.expect(QQ, qq & 0xFF, 8)
        B.expect(QQSUM, [int(v) for v in s_q], 64)
        KQ, KQSUM = B.aout(HEADS * T * HDP * 8, "kq"), B.aout(HEADS * T * 64, "kq.sum")
        kq, s_k = C.quant(kb)
        B.expect(KQ, kq & 0xFF, 8)
        B.expect(KQSUM, [int(v) for v in s_k], 64)
        VQ, VQSUM = B.aout(QN * T * 8, "vq"), B.aout(QN * 64, "vq.sum")
        vq, s_v = C.quant(vb)
        B.expect(VQ, vq & 0xFF, 8)
        B.expect(VQSUM, [int(v) for v in s_v], 64)
        recs += (vu_record(OP_QUANT, HEADS * T, HDP, QQ, QQSUM, desc(QB, SH_E, F_BF16), rs=desc(QBSUM, SH_R, F_SUM, FIELD_AMAX))
                 + vu_record(OP_QUANT, HEADS * T, HDP, KQ, KQSUM, desc(KB, SH_E, F_BF16), rs=desc(KBSUM, SH_R, F_SUM, FIELD_AMAX))
                 + vu_record(OP_QUANT, QN, T, VQ, VQSUM, desc(VB, SH_E, F_BF16), rs=desc(VBSUM, SH_R, F_SUM, FIELD_AMAX)))
        B.stage(VN, recs + [0], "dequant_quant")
        # stage 3: QK^T per head
        LOGIT = B.aout(HEADS * T * T * 32, "logit")
        logit = np.zeros((HEADS * T, T), np.int64)
        cmds = []
        for h in range(HEADS):
            rows = slice(h * T, (h + 1) * T)
            logit[rows] = LL.imatmul(qq[rows], kq[rows].T)
            cmds += TL.cmds_loadx(KQ + h * T * HDP, T, W_HP, QQ + h * T * HDP, T, LOGIT + h * T * T * 4)
        B.expect(LOGIT, logit & 0xFFFFFFFF, 32)
        B.stage(CH8, cmds + [0], "qk_gemm")
        # stage 4: DEQUANT logits per head, SMAX_Q8
        LB, LSUM = B.aout(HEADS * T * T * 16, "lb"), B.aout(HEADS * T * 64, "lb.sum")
        lb = np.zeros((HEADS * T, T), np.uint16)
        recs = []
        for h in range(HEADS):
            rows = slice(h * T, (h + 1) * T)
            y = dequant(logit[rows], np.asarray(s_q[rows], np.uint32), np.asarray(s_k[rows], np.uint32))
            lb[rows] = y
            recs += vu_record(OP_DEQUANT, T, T, LB + h * T * T * 2, LSUM + h * T * 8, desc(LOGIT + h * T * T * 4, SH_E, F_INT32),
                              c=desc(QQSUM + h * T * 8, SH_R, F_SUM, FIELD_RS0), d=desc(KQSUM + h * T * 8, SH_C, F_SUM, FIELD_RS0))
        B.expect(LB, lb, 16)
        B.expect(LSUM, summaries(lb), 64)
        PC, PSUM = B.aout(HEADS * T * T * 8, "pc"), B.aout(HEADS * T * 64, "pc.sum")
        pq, s_p = C.smax_u8(lb, np.ones(lb.shape, bool))
        B.expect(PC, pq & 0xFF, 8)
        B.expect(PSUM, [int(v) for v in s_p], 64)
        recs += vu_record(OP_SMAX_Q8, HEADS * T, T, PC, PSUM, desc(LB, SH_E, F_BF16), rs=desc(LSUM, SH_R, F_SUM, FIELD_MAX),
                          b_fp32=1)
        B.stage(VN, recs + [0], "logits_softmax")
        # stage 5: PV per head
        CTX = B.aout(HEADS * T * HDP * 32, "ctx")
        ctx = np.zeros((HEADS * T, HDP), np.int64)
        cmds = []
        for h in range(HEADS):
            rows = slice(h * T, (h + 1) * T)
            ctx[rows] = LL.imatmul(pq[rows], vq[HDP * h:HDP * (h + 1)].T)
            cmds += TL.cmds_loadx(VQ + h * HDP * T, HDP, W_T, PC + h * T * T, T, CTX + h * T * HDP * 4, kind=CHU)
        B.expect(CTX, ctx & 0xFFFFFFFF, 32)
        B.stage(CHU, cmds + [0], "pv_gemm")
        # stage 6: DEQUANT context per head into token-major rows, ADD b = 0, QUANT per token
        CBT, CBSUM = B.aout(T * QN * 16, "cbt"), B.aout(HEADS * T * 64, "cbt.sum")
        cbt = np.zeros((T, QN), np.uint16)
        recs = []
        for h in range(HEADS):
            rows = slice(h * T, (h + 1) * T)
            y = dequant(ctx[rows], np.asarray(s_p[rows], np.uint32), np.asarray(s_v[HDP * h:HDP * (h + 1)], np.uint32))
            cbt[:, HDP * h:HDP * (h + 1)] = y
            B.expect(CBSUM + h * T * 8, summaries(y), 64)
            recs += vu_record(OP_DEQUANT, T, HDP, CBT + h * HDP * 2, CBSUM + h * T * 8, desc(CTX + h * T * HDP * 4, SH_E, F_INT32),
                              c=desc(PSUM + h * T * 8, SH_R, F_SUM, FIELD_RS0), d=desc(VQSUM + h * HDP * 8, SH_C, F_SUM, FIELD_RS0),
                              blk_beats=HDP * 16 // 256, gap_bytes=(QN - HDP) * 2)
        B.expect(CBT, cbt, 16)
        CTXB, CTXSUM = B.aout(T * QN * 16, "ctxb"), B.aout(T * 64, "ctxb.sum")
        ctxb = C.add(cbt, np.zeros_like(cbt))
        B.expect(CTXB, ctxb, 16)
        B.expect(CTXSUM, summaries(ctxb), 64)
        CQ, CQSUM = B.aout(T * QN * 8, "cq"), B.aout(T * 64, "cq.sum")
        cqd, s_c = C.quant(ctxb)
        B.expect(CQ, cqd & 0xFF, 8)
        B.expect(CQSUM, [int(v) for v in s_c], 64)
        recs += (vu_record(OP_ADD, T, QN, CTXB, CTXSUM, desc(CBT, SH_E, F_BF16), b=K(0))
                 + vu_record(OP_QUANT, T, QN, CQ, CQSUM, desc(CTXB, SH_E, F_BF16), rs=desc(CTXSUM, SH_R, F_SUM, FIELD_AMAX)))
        B.stage(VN, recs + [0], "context")
        # stage 7: out_proj
        acco1 = C.gemm(cqd, self.co)
        ACCO = B.aout(T * D * 32, "acco")
        B.expect(ACCO, acco1 & 0xFFFFFFFF, 32)
        B.stage(CH8, TL.cmds_load(CQ, T, QN // 16, self.o_imgs, ACCO) + [0], "o_gemm")
        # stage 8: DEQUANT (+ out_proj bias), residual
        OB, OBSUM = B.aout(T * D * 16, "ob"), B.aout(T * 64, "ob.sum")
        oc = dequant(acco1, np.asarray(s_c, np.uint32), f32c(self.Wo.scale), np.repeat(f32c(t["out_proj.b"])[None, :], T, 0))
        B.expect(OB, oc, 16)
        B.expect(OBSUM, summaries(oc), 64)
        YB, YSUM = B.aout(T * D * 16, "yb"), B.aout(T * 64, "yb.sum")
        yc = C.add(oc, xc)
        B.expect(YB, yc, 16)
        B.expect(YSUM, summaries(yc), 64)
        B.stage(VN, vu_record(OP_DEQUANT, T, D, OB, OBSUM, desc(ACCO, SH_E, F_INT32), c=desc(CQSUM, SH_R, F_SUM, FIELD_RS0),
                              d=desc(self.so_base, SH_C, F_FP32), e=desc(self.bo_base, SH_C, F_FP32), bias_en=1)
                + vu_record(OP_ADD, T, D, YB, YSUM, desc(OB, SH_E, F_BF16), b=desc(x.base, SH_E, F_BF16)) + [0],
                "o_dequant_residual")
        # stage 9: LayerNorm 2 + QUANT
        y_act = Act(yc, YB, YSUM)
        h2c, h2q, s_h2, H2Q, H2QSUM = ln_stage(y_act, yc, t["layer_norm2.w"], t["layer_norm2.b"], self.ln2_g, self.ln2_b, "ln2")
        # stage 10: fc1 into two sub-block regions
        acc1 = C.gemm(h2q, self.c1)
        ACC1 = [B.aout(T * FB * 32, f"acc1_{j}") for j in range(NB)]
        for j in range(NB):
            B.expect(ACC1[j], acc1[:, FB * j:FB * (j + 1)] & 0xFFFFFFFF, 32)
        B.stage(CH8, TL.cmds_load(H2Q, T, W_IN, self.f1_imgs, ACC1) + [0], "fc1_gemm")
        # stage 11: DEQUANT (+ bias), GELU, QUANT per sub-block
        recs, MQ, MQSUM, mq, s_m = [], [], [], [], []
        for j in range(NB):
            cols = slice(FB * j, FB * (j + 1))
            G1, G1SUM = B.aout(T * FB * 16, f"g1_{j}"), B.aout(T * 64, f"g1_{j}.sum")
            g1 = dequant(acc1[:, cols], s_h2, f32c(self.s1)[cols], np.repeat(f32c(self.b1)[None, cols], T, 0))
            B.expect(G1, g1, 16)
            B.expect(G1SUM, summaries(g1), 64)
            GE, GESUM = B.aout(T * FB * 16, f"ge_{j}"), B.aout(T * 64, f"ge_{j}.sum")
            ge, _ = R.op_gelu(g1)
            C._n("GELU", g1.size)
            B.expect(GE, ge, 16)
            B.expect(GESUM, summaries(ge), 64)
            MQ.append(B.aout(T * FB * 8, f"mq_{j}"))
            MQSUM.append(B.aout(T * 64, f"mq_{j}.sum"))
            mq_j, s_m_j = C.quant(ge)
            mq.append(mq_j)
            s_m.append(np.asarray(s_m_j, np.uint32))
            B.expect(MQ[j], mq_j & 0xFF, 8)
            B.expect(MQSUM[j], [int(v) for v in s_m_j], 64)
            recs += (vu_record(OP_DEQUANT, T, FB, G1, G1SUM, desc(ACC1[j], SH_E, F_INT32),
                               c=desc(H2QSUM, SH_R, F_SUM, FIELD_RS0), d=desc(self.s1_base + FB * j * 4, SH_C, F_FP32),
                               e=desc(self.b1_base + FB * j * 4, SH_C, F_FP32), bias_en=1)
                     + vu_record(OP_GELU, T, FB, GE, GESUM, desc(G1, SH_E, F_BF16))
                     + vu_record(OP_QUANT, T, FB, MQ[j], MQSUM[j], desc(GE, SH_E, F_BF16),
                                 rs=desc(GESUM, SH_R, F_SUM, FIELD_AMAX)))
        B.stage(VN, recs + [0], "gelu_quant")
        # stage 12: fc2, K split per sub-block
        acc2 = []
        ACC2 = [B.aout(T * D * 32, f"acc2_{j}") for j in range(NB)]
        cmds = []
        for j in range(NB):
            acc2.append(C.gemm(mq[j], self.c2[:, FB * j:FB * (j + 1)]))
            B.expect(ACC2[j], acc2[j] & 0xFFFFFFFF, 32)
            cmds += TL.cmds_load(MQ[j], T, W_FB, self.f2_imgs[j], ACC2[j])
        B.stage(CH8, cmds + [0], "fc2_gemm")
        # stage 13: DEQUANT chain (+ fc2 bias), residual
        recs, prev, prev_base = [], None, None
        for j in range(NB):
            last = j == NB - 1
            OUT, OSUM = B.aout(T * D * (16 if last else 32), f"m_{j}"), B.aout(T * 64, f"m_{j}.sum")
            bias = np.repeat(f32c(t["fc2.b"])[None, :], T, 0) if prev is None else prev
            y = dequant(acc2[j], s_m[j], f32c(self.W2.scale), bias, "bf16" if last else "fp32")
            B.expect(OUT, y, 16 if last else 32)
            B.expect(OSUM, summaries(y) if last else [0] * T, 64)
            e = desc(self.b2_base, SH_C, F_FP32) if prev is None else desc(prev_base, SH_E, F_FP32)
            recs += vu_record(OP_DEQUANT, T, D, OUT, OSUM, desc(ACC2[j], SH_E, F_INT32),
                              c=desc(MQSUM[j], SH_R, F_SUM, FIELD_RS0), d=desc(self.s2_base, SH_C, F_FP32),
                              e=e, bias_en=1, out_fp32=int(not last))
            prev, prev_base = y, OUT
        mc, MB = prev, prev_base
        ZB, ZSUM = B.aout(T * D * 16, "zb"), B.aout(T * 64, "zb.sum")
        zc = C.add(mc, yc)
        B.expect(ZB, zc, 16)
        B.expect(ZSUM, summaries(zc), 64)
        B.stage(VN, recs + vu_record(OP_ADD, T, D, ZB, ZSUM, desc(MB, SH_E, F_BF16), b=desc(YB, SH_E, F_BF16)) + [0],
                "fc2_dequant_residual")
        return dict(z=Act(zc, ZB, ZSUM))


# ====================================================================== the vision ends
class VisionEnds:
    """patch embedding (256 patches x 608 padded pixel codes -> SigLIP layer 0 input) and post-LayerNorm +
    projector (SigLIP output -> 256 prefix image tokens), pi0_vision_ends_golden.py; weights allocated once."""
    T, DV, DL, KP, KPP = 256, 1152, 2048, 588, 608

    def __init__(self, B: ChunkBuild, C: LL.Chip):
        import pi0_vision_ends_golden as VE
        self.B, self.C = B, C
        B.layer = "vision"
        t = self.t = VE.load()
        T, DV, DL, KP, KPP = self.T, self.DV, self.DL, self.KP, self.KPP
        TL = self.TL = MixedTiler(B, B.tiler_classes)
        self.Wp = LL.QW(t["patch.w"], None)
        self.wp_codes = np.zeros((DV, KPP), np.int64)
        self.wp_codes[:, :KP] = self.Wp.codes
        self.p_imgs = TL.put_images(self.wp_codes, KPP // 16)
        self.sp_base, self.bp_base = B.put_in(f32c(self.Wp.scale), 32, "patch_s"), B.put_in(f32c(t["patch.b"]), 32, "patch_b")
        self.POS = B.put_in(f32c(t["pos"]), 32, "pos_table")
        self.pln_g, self.pln_b = B.put_in(f32c(t["pln.w"]), 32, "pln_g"), B.put_in(f32c(t["pln.b"]), 32, "pln_b")
        self.Wj = LL.QW(t["proj.w"], None)
        self.j_imgs = TL.put_images(self.Wj.codes, DV // 16)
        self.sj_base, self.bj_base = B.put_in(f32c(self.Wj.scale), 32, "proj_s"), B.put_in(f32c(t["proj.b"]), 32, "proj_b")

    def patch_embed(self, patches: np.ndarray, cam: int) -> Act:
        """patches (256, 588) fp32 pixels -> the camera's SigLIP layer 0 input (codes + summaries)"""
        B, C, TL, T, DV, KP, KPP, t = self.B, self.C, self.TL, self.T, self.DV, self.KP, self.KPP, self.t
        B.layer = f"patch.c{cam}"
        pc = np.zeros((T, KPP), np.uint16)
        pc[:, :KP] = R.bf16_codes(patches)
        P_BASE = B.put_in(pc, 16, "patches", area="IO")
        P_AMAX = B.put_in(R.row_amax_code(pc), 16, "patches.amax", area="IO")
        PQ, PQSUM = B.aout(T * KPP * 8, "pq"), B.aout(T * 64, "pq.sum")
        pq, s_p = C.quant(pc)
        B.expect(PQ, pq & 0xFF, 8)
        B.expect(PQSUM, [int(v) for v in s_p], 64)
        B.stage(VN, vu_record(OP_QUANT, T, KPP, PQ, PQSUM, desc(P_BASE, SH_E, F_BF16), rs=desc(P_AMAX, SH_R, F_BF16)) + [0],
                "patch_quant")
        acc = C.gemm(pq, self.wp_codes)
        ACC = B.aout(T * DV * 32, "patch_acc")
        B.expect(ACC, acc & 0xFFFFFFFF, 32)
        B.stage(CH8, TL.cmds_load(PQ, T, KPP // 16, self.p_imgs, ACC) + [0], "patch_gemm")
        PE, PESUM = B.aout(T * DV * 16, "pe"), B.aout(T * 64, "pe.sum")
        pe, _ = R.op_dequant(acc, s_p, f32c(self.Wp.scale), np.repeat(f32c(t["patch.b"])[None, :], T, 0), "bf16")
        B.expect(PE, pe, 16)
        B.expect(PESUM, summaries(pe), 64)
        X0, X0SUM = B.aout(T * DV * 16, "x0"), B.aout(T * 64, "x0.sum")
        x0, _ = R.op_add(pe, f32c(t["pos"]), "fp32")
        B.expect(X0, x0, 16)
        B.expect(X0SUM, summaries(x0), 64)
        B.stage(VN, vu_record(OP_DEQUANT, T, DV, PE, PESUM, desc(ACC, SH_E, F_INT32), c=desc(PQSUM, SH_R, F_SUM, FIELD_RS0),
                              d=desc(self.sp_base, SH_C, F_FP32), e=desc(self.bp_base, SH_C, F_FP32), bias_en=1)
                + vu_record(OP_ADD, T, DV, X0, X0SUM, desc(PE, SH_E, F_BF16), b=desc(self.POS, SH_E, F_FP32), b_fp32=1) + [0],
                "patch_dequant_pos")
        return Act(x0, X0, X0SUM)

    def projector(self, h: Act, cam: int, IE: int, IESUM: int) -> np.ndarray:
        """post_layernorm + projector of one camera's SigLIP output into rows [256 cam, 256 cam + 256) of the prefix
        input (IE / IESUM: that camera's part of the prefix x region and its summaries); returns the token codes"""
        B, C, TL, T, DV, DL, t = self.B, self.C, self.TL, self.T, self.DV, self.DL, self.t
        B.layer = f"proj.c{cam}"
        hc = h.codes
        LS, LSSUM = B.aout(T * 32, "ls"), B.aout(T * 64, "ls.sum")
        mu, r = R.ln_stat(hc, R.row_amax_code(hc), 1.0 / DV)
        B.expect(LS, r, 32)
        B.expect(LSSUM, [int(v) & 0xFFFFFFFF for v in mu], 64)
        NB, NBSUM = B.aout(T * DV * 16, "nb"), B.aout(T * 64, "nb.sum")
        nb, _ = R.ln_apply(hc, mu, r, f32c(t["pln.w"]), f32c(t["pln.b"]))
        B.expect(NB, nb, 16)
        B.expect(NBSUM, summaries(nb), 64)
        NQ, NQSUM = B.aout(T * DV * 8, "nq"), B.aout(T * 64, "nq.sum")
        nq, s_n = C.quant(nb)
        B.expect(NQ, nq & 0xFF, 8)
        B.expect(NQSUM, [int(v) for v in s_n], 64)
        B.stage(VN, vu_record(OP_LN_STAT, T, DV, LS, LSSUM, desc(h.base, SH_E, F_BF16), rs=h.rs(),
                              k=int(np.float32(1.0 / DV).view(np.uint32)))
                + vu_record(OP_LN_APPLY, T, DV, NB, NBSUM, desc(h.base, SH_E, F_BF16), b=desc(LSSUM, SH_R, F_SUM, FIELD_RS0),
                            c=desc(LS, SH_R, F_FP32), d=desc(self.pln_g, SH_C, F_FP32), e=desc(self.pln_b, SH_C, F_FP32))
                + vu_record(OP_QUANT, T, DV, NQ, NQSUM, desc(NB, SH_E, F_BF16), rs=desc(NBSUM, SH_R, F_SUM, FIELD_AMAX)) + [0],
                "post_ln_quant")
        acc = C.gemm(nq, self.Wj.codes)
        ACC2 = B.aout(T * DL * 32, "proj_acc")
        B.expect(ACC2, acc & 0xFFFFFFFF, 32)
        B.stage(CH8, TL.cmds_load(NQ, T, DV // 16, self.j_imgs, ACC2) + [0], "proj_gemm")
        ie, _ = R.op_dequant(acc, np.asarray(s_n, np.uint32), f32c(self.Wj.scale), np.repeat(f32c(t["proj.b"])[None, :], T, 0), "bf16")
        B.expect(IE, ie, 16)
        B.expect(IESUM, summaries(ie), 64)
        B.stage(VN, vu_record(OP_DEQUANT, T, DL, IE, IESUM, desc(ACC2, SH_E, F_INT32), c=desc(NQSUM, SH_R, F_SUM, FIELD_RS0),
                              d=desc(self.sj_base, SH_C, F_FP32), e=desc(self.bj_base, SH_C, F_FP32), bias_en=1) + [0],
                "proj_dequant")
        return ie


# ====================================================================== the action head
class ActionHead:
    """pi0's action head around the expert (pi0_action_head_golden.py): input side x_t -> 50 action tokens,
    output side final norm -> action_out_proj -> Euler; weights and the 10 per-step time biases allocated once."""
    NUM_STEPS, CHUNK, AD, D = 10, 50, 32, 1024

    def __init__(self, B: ChunkBuild, C: LL.Chip):
        import pi0_action_head_golden as AH
        self.B, self.C, self.AH = B, C, AH
        B.layer = "head"
        t = self.t = AH.load_head()
        D = self.D
        TL = self.TL = MixedTiler(B, B.tiler_classes)
        self.W_in = LL.QW(t["action_in_proj.w"], None)
        self.W_mi = LL.QW(t["action_time_mlp_in.w"][:, :D], None)
        self.W_mo = LL.QW(t["action_time_mlp_out.w"], None)
        self.W_out = LL.QW(t["action_out_proj.w"], None)
        self.in_imgs = TL.put_images(self.W_in.codes, self.AD // 16)
        self.mi_imgs = TL.put_images(self.W_mi.codes, D // 16)
        self.mo_imgs = TL.put_images(self.W_mo.codes, D // 16)
        self.out_imgs = TL.put_images(self.W_out.codes, D // 16)
        self.s_in, self.b_in = B.put_in(f32c(self.W_in.scale), 32, "in_s"), B.put_in(f32c(t["action_in_proj.b"]), 32, "in_b")
        self.s_mi = B.put_in(f32c(self.W_mi.scale), 32, "mi_s")
        self.dt = -1.0 / self.NUM_STEPS
        self.b_mi = []
        for step in range(self.NUM_STEPS):
            time = 1.0 + step * self.dt
            bias_mi = (t["action_time_mlp_in.b"].astype(np.float64)
                       + t["action_time_mlp_in.w"][:, D:].astype(np.float64) @ AH.time_embedding(time).astype(np.float64))
            self.b_mi.append((bias_mi, B.put_in(f32c(bias_mi), 32, f"mi_b_step{step}")))
        self.s_mo, self.b_mo = B.put_in(f32c(self.W_mo.scale), 32, "mo_s"), B.put_in(f32c(t["action_time_mlp_out.b"]), 32, "mo_b")
        self.norm_gain = B.put_in(f32c(1.0 + t["norm.w"].astype(np.float64)), 32, "final_norm_gain")
        self.s_out, self.b_out = B.put_in(f32c(self.W_out.scale), 32, "out_s"), B.put_in(f32c(t["action_out_proj.b"]), 32, "out_b")

    def _quant_rec(self, src, src_sum, rows, L, codes, tag):
        B, C = self.B, self.C
        Q, QSUM = B.aout(rows * L * 8, tag), B.aout(rows * 64, tag + ".sum")
        q, s = C.quant(codes)
        B.expect(Q, q & 0xFF, 8)
        B.expect(QSUM, [int(v) for v in s], 64)
        return q, np.asarray(s, np.uint32), Q, QSUM, vu_record(OP_QUANT, rows, L, Q, QSUM, desc(src, SH_E, F_BF16),
                                                                 rs=desc(src_sum, SH_R, F_SUM, FIELD_AMAX))

    def _dq_rec(self, acc, ACC, s_row, QSUM, scale, s_base, b_base, bias_f, rows, L, tag, out="bf16", OUT=None, OSUM=None):
        B = self.B
        if OUT is None:
            OUT, OSUM = B.aout(rows * L * (32 if out == "fp32" else 16), tag), B.aout(rows * 64, tag + ".sum")
        y, _ = R.op_dequant(acc, s_row, scale, np.repeat(bias_f[None, :], rows, 0), out)
        B.expect(OUT, y, 32 if out == "fp32" else 16)
        B.expect(OSUM, [0] * rows if out == "fp32" else summaries(y), 64)
        return y, OUT, OSUM, vu_record(OP_DEQUANT, rows, L, OUT, OSUM, desc(ACC, SH_E, F_INT32),
                                       c=desc(QSUM, SH_R, F_SUM, FIELD_RS0), d=desc(s_base, SH_C, F_FP32),
                                       e=desc(b_base, SH_C, F_FP32), bias_en=1, out_fp32=int(out == "fp32"))

    def _gemm_stage(self, q, Q, W, imgs, rows, tag):
        B, C, TL = self.B, self.C, self.TL
        acc = C.gemm(q, W.codes)
        ACC = B.aout(rows * W.codes.shape[0] * 32, tag)
        B.expect(ACC, acc & 0xFFFFFFFF, 32)
        B.stage(CH8, TL.cmds_load(Q, rows, W.codes.shape[1] // 16, imgs, ACC) + [0], tag)
        return acc, ACC

    def input_side(self, x_t: np.ndarray, XT: int, step: int, TOK: int, TOKSUM: int) -> np.ndarray:
        """x_t (50, 32) fp32 (the region XT holds its bits) -> the 50 action tokens written at TOK (bf16 codes)
        with summaries at TOKSUM (rows 1..50 of the expert's suffix, so TOK = X_exp + 2048 bytes); returns the codes"""
        B, C, T, AD, D, t = self.B, self.C, self.CHUNK, self.AD, self.D, self.t
        B.layer = f"head_in.s{step}"
        XB, XBSUM = B.aout(T * AD * 16, "xb"), B.aout(T * 64, "xb.sum")
        xb, _ = R.op_add(np.zeros((T, AD), np.uint16), x_t.view(np.uint32), "fp32")
        B.expect(XB, xb, 16)
        B.expect(XBSUM, summaries(xb), 64)
        xq, s_x, XQ, XQSUM, rq = self._quant_rec(XB, XBSUM, T, AD, xb, "xq")
        B.stage(VN, vu_record(OP_ADD, T, AD, XB, XBSUM, K(0), b=desc(XT, SH_E, F_FP32), b_fp32=1) + rq + [0], "xt_quant")
        acc, ACC = self._gemm_stage(xq, XQ, self.W_in, self.in_imgs, T, "in_proj_gemm")
        ae, AE, AESUM, r1 = self._dq_rec(acc, ACC, s_x, XQSUM, f32c(self.W_in.scale), self.s_in, self.b_in,
                                         f32c(t["action_in_proj.b"]), T, D, "ae")
        aq, s_a, AQ, AQSUM, r2 = self._quant_rec(AE, AESUM, T, D, ae, "aq")
        B.stage(VN, r1 + r2 + [0], "in_proj_dequant_quant")
        acc, ACC = self._gemm_stage(aq, AQ, self.W_mi, self.mi_imgs, T, "mlp_in_gemm")
        bias_mi, b_mi_base = self.b_mi[step]
        mi, MI, MISUM, r1 = self._dq_rec(acc, ACC, s_a, AQSUM, f32c(self.W_mi.scale), self.s_mi, b_mi_base,
                                         f32c(bias_mi), T, D, "mi")
        SI, SISUM = B.aout(T * D * 16, "si"), B.aout(T * 64, "si.sum")
        si, _ = R.op_silu(mi)
        B.expect(SI, si, 16)
        B.expect(SISUM, summaries(si), 64)
        sq, s_s, SQ, SQSUM, r3 = self._quant_rec(SI, SISUM, T, D, si, "sq")
        B.stage(VN, r1 + vu_record(OP_SILU, T, D, SI, SISUM, desc(MI, SH_E, F_BF16)) + r3 + [0], "mlp_in_dequant_silu_quant")
        acc, ACC = self._gemm_stage(sq, SQ, self.W_mo, self.mo_imgs, T, "mlp_out_gemm")
        tok, _, _, r1 = self._dq_rec(acc, ACC, s_s, SQSUM, f32c(self.W_mo.scale), self.s_mo, self.b_mo,
                                     f32c(t["action_time_mlp_out.b"]), T, D, "tok", OUT=TOK, OSUM=TOKSUM)
        B.stage(VN, r1 + [0], "mlp_out_dequant")
        return tok

    def output_side(self, h: Act, x_t: np.ndarray, XT: int, step: int, XN: int) -> np.ndarray:
        """h: the expert's final residual (51 x 1024) -> RMS norm -> action_out_proj -> v_t (fp32) -> EULER on rows
        1..50 with x_t (at XT) into XN (50 x 32 fp32); returns x_{t+dt} as fp32"""
        B, C, T, AD, D, t = self.B, self.C, self.CHUNK, self.AD, self.D, self.t
        B.layer = f"head_out.s{step}"
        hc = h.codes
        RS = B.aout(51 * 64, "rs")
        r_rms = R.rms_stat(hc, R.row_amax_code(hc), 1.0 / D)
        B.expect(RS, [int(v) for v in r_rms], 64)
        NB, NBSUM = B.aout(51 * D * 16, "nb"), B.aout(51 * 64, "nb.sum")
        nb, _ = R.rms_apply(hc, r_rms, f32c(1.0 + t["norm.w"].astype(np.float64)))
        B.expect(NB, nb, 16)
        B.expect(NBSUM, summaries(nb), 64)
        nq, s_n, NQ, NQSUM, r4 = self._quant_rec(NB, NBSUM, 51, D, nb, "nq")
        B.stage(VN, vu_record(OP_RMS_STAT, 51, D, 0, RS, desc(h.base, SH_E, F_BF16), rs=h.rs(),
                              k=int(np.float32(1.0 / D).view(np.uint32)))
                + vu_record(OP_RMS_APPLY, 51, D, NB, NBSUM, desc(h.base, SH_E, F_BF16), c=desc(RS, SH_R, F_SUM, FIELD_RS0),
                            d=desc(self.norm_gain, SH_C, F_FP32)) + r4 + [0], "final_norm_quant")
        acc, ACC = self._gemm_stage(nq, NQ, self.W_out, self.out_imgs, 51, "out_proj_gemm")
        vt51, VT, VTSUM, r1 = self._dq_rec(acc, ACC, s_n, NQSUM, f32c(self.W_out.scale), self.s_out, self.b_out,
                                           f32c(t["action_out_proj.b"]), 51, AD, "vt", out="fp32")
        vt = vt51[1:]
        xn, _ = R.op_euler(x_t.view(np.uint32), vt, self.dt)
        B.expect(XN, xn, 32)
        XNSUM = B.aout(T * 64, "xn.sum")
        B.expect(XNSUM, [0] * T, 64)
        # Two stages, not one: the EULER reads v_t one row up (rows 1..50 of the 51-row DEQUANT), so when a stage is
        # shared out by 16-row slices over several vector nodes, slice k of the EULER needs row 16k+16 of the DEQUANT,
        # which another node writes.  Same-row dependencies inside a stage stay on one node; this offset one does not
        # (found by the smoke chunk simulation: actions row 31 raced the other node's DEQUANT).
        B.stage(VN, r1 + [0], "out_proj_dequant")
        B.stage(VN, vu_record(OP_EULER, T, AD, XN, XNSUM, desc(VT + AD * 4, SH_E, F_FP32), e=desc(XT, SH_E, F_FP32),
                              k=int(np.float32(self.dt).view(np.uint32))) + [0], "euler")
        return np.asarray(xn, np.uint32).view(np.float32)
