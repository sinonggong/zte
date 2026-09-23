#!/usr/bin/env python3
"""GDDR6 image, node programs and expected beats for tb_pi0_attn.sv: one SigLIP So400m/14 ENCODER LAYER of the
real checkpoint at its real size (256 patch tokens of one camera, hidden 1152, 16 heads x 72, MLP 4304), on the
same three nodes as the expert and prefix layers (4-lane vector node, int8 chain, uint8 PV chain).

Numerics: the prefix W8A8 chip recipe for SigLIP (paper/data/prefix_w8a8/SUMMARY.md): W8 per output channel
(MSE clip, no smoothing), A8 per token dynamic, INT8 QK^T (int8 q per row x int8 k per key), PV uint8 P per row x
int8 V per channel; biases in DEQUANT; LayerNorm by LN_STAT / LN_APPLY; GELU (tanh form) by its table.  The
activations are the captured layer input of the frame (build/paper_vector_unit/acts_*.npz, vis.L{0,13,26}).

What is different from a Gemma layer, and how it is lowered without new RTL:

  16 KV heads.  Every head has its own keys, values and scales, so attention runs HEAD-MAJOR: q and k write one
  int32 region per head (column regions of the projection), V is produced column-major by the role-swapped GEMM
  (its channel rows are head-major already), and QK^T, the logits DEQUANT, PV and the context DEQUANT run once
  per head, with row and column scales taken at that head's offset.
  Head dim 72 -> 96.  LOADX needs rows of an even number of words (72 bytes is 4.5), so q, k and v are padded
  with 24 zero output channels per head (codes, scales and biases 0); o_proj gets matching zero input columns.
  The padding adds nothing to any dot product.
  o_proj on a head-major context.  Its input would be the token-major concatenation of the 16 heads, which no
  op produces.  Instead o_proj is split by K into one GEMM per head (K = 96: the head's context codes, quantised
  per (token, head)), and the 16 partial sums are combined by DEQUANT itself: fp32 outputs chained through the
  fp32 bias, the first one carrying out_proj's bias, the last one bf16 (as the prefix layer's down K split).
  MLP 4304 -> 4352.  The rows fit a vector row (<= 4096) in two sub-blocks of 2176 = 136 words, so fc1 gets 48
  zero output rows; GELU and QUANT run per sub-block and fc2 is K-split into two GEMMs combined by DEQUANT.

Stages (one node program each; the testbench starts them in order):
  0 VN   LN_STAT, LN_APPLY, QUANT
  1 CH8  q, k (16 head regions each), v (role-swapped, column-major, 1536 channel rows in row slices)
  2 VN   DEQUANT q x16 (x scaling/ln2, + bias), k x16 (+ bias), v (+ bias per channel); QUANT q, k, v
  3 CH8  QK^T x16 (LOADX of the head's key rows)
  4 VN   DEQUANT logits x16, SMAX_Q8 (uint8 P)
  5 CHU  PV x16 (LOADX of the head's V channel rows)
  6 VN   DEQUANT context x16, QUANT (per token, head)
  7 CH8  o_proj x16 (K split per head)
  8 VN   DEQUANT chain x16 (+ out_proj bias), ADD residual
  9 VN   LN_STAT, LN_APPLY, QUANT
 10 CH8  fc1 (2 sub-block regions)
 11 VN   DEQUANT (+ bias), GELU, QUANT per sub-block
 12 CH8  fc2 x2 (K split per sub-block)
 13 VN   DEQUANT chain x2 (+ fc2 bias), ADD residual

Usage: pi0_siglip_golden.py --out DIR [--layer 0|13|26] [--image 0|1] [--slot-bits 12]
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import vector_unit_ref as R                                              # noqa: E402
import pi0_layer_lower as LL                                            # noqa: E402
import pi0_attn_golden as AG                                            # noqa: E402
from pi0_attn_golden import Build, Tiler, vu_record, summaries, VN, CH8, CHU  # noqa: E402
from vu_node_golden import (desc, K, SH_E, SH_R, SH_C, F_BF16, F_FP32, F_INT32, F_SUM,  # noqa: E402
                            FIELD_RS0, FIELD_MAX, FIELD_AMAX)

OP_ADD, OP_LN_STAT, OP_LN_APPLY, OP_GELU = 0, 3, 4, 7
OP_SMAX_Q8, OP_QUANT, OP_DEQUANT = 12, 13, 14
LN2 = math.log(2.0)
VIS = "model.paligemma_with_expert.paligemma.model.vision_tower.vision_model.encoder.layers.{L}."
HEADS, HD, HDP = 16, 72, 96                  # head dim, padded to 6 words
D, FF, FFP = 1152, 4304, 4352                # MLP padded to two 2176-wide sub-blocks
NB, FB = 2, 2176


def load_vis_layer(L: int) -> dict:
    from safetensors import safe_open
    t = {}
    with safe_open(LL.CKPT, "np") as f:
        p = VIS.format(L=L)
        for n in ("layer_norm1", "layer_norm2"):
            t[n + ".w"] = f.get_tensor(p + n + ".weight").astype(np.float32)
            t[n + ".b"] = f.get_tensor(p + n + ".bias").astype(np.float32)
        for n in ("q_proj", "k_proj", "v_proj", "out_proj"):
            t[n + ".w"] = f.get_tensor(p + "self_attn." + n + ".weight").astype(np.float32)
            t[n + ".b"] = f.get_tensor(p + "self_attn." + n + ".bias").astype(np.float32)
        for n in ("fc1", "fc2"):
            t[n + ".w"] = f.get_tensor(p + "mlp." + n + ".weight").astype(np.float32)
            t[n + ".b"] = f.get_tensor(p + "mlp." + n + ".bias").astype(np.float32)
    return t


def pad_heads_out(a: np.ndarray) -> np.ndarray:
    """(16 x 72, ...) per-output-channel array -> (16 x 96, ...) with 24 zero channels after each head"""
    out = np.zeros((HEADS * HDP,) + a.shape[1:], a.dtype)
    for h in range(HEADS):
        out[HDP * h:HDP * h + HD] = a[HD * h:HD * h + HD]
    return out


def pad_heads_in(codes: np.ndarray) -> np.ndarray:
    """(N, 16 x 72) weight codes -> (N, 16 x 96) with zero input columns after each head"""
    out = np.zeros((codes.shape[0], HEADS * HDP), codes.dtype)
    for h in range(HEADS):
        out[:, HDP * h:HDP * h + HD] = codes[:, HD * h:HD * h + HD]
    return out


def rows96() -> np.ndarray:
    """the rows vector_unit_capture.py keeps of fc1_out: torch.linspace(0, 255, 96).long()"""
    import torch
    return torch.linspace(0, 255, 96).long().numpy()


def ln_fp(x, g, b, eps=1e-6):
    mu = x.mean(-1, keepdims=True)
    var = ((x - mu) ** 2).mean(-1, keepdims=True)
    return (x - mu) / np.sqrt(var + eps) * g.astype(np.float64) + b.astype(np.float64)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--layer", type=int, default=0, help="a captured SigLIP layer: 0, 13 or 26")
    ap.add_argument("--image", type=int, default=0, help="camera 0 or 1")
    ap.add_argument("--slot-bits", type=int, default=12)
    ap.add_argument("--o-proj", choices=["head", "token"], default="head",
                    help="head: out_proj K-split per head over (token, head)-quantised context, recombined by a "
                         "16-DEQUANT chain; token: the per-head context DEQUANTs write token-major rows (vector "
                         "row blocks, w5), one per-token QUANT, one out_proj GEMM, one DEQUANT")
    a = ap.parse_args()

    z = np.load(LL.CAPTURE)
    x = z[f"vis.L{a.layer}.layer_in"][a.image].astype(np.float64)                 # (256, 1152)
    T = x.shape[0]
    assert x.shape == (256, D) and T % 16 == 0
    scaling = float(z[f"vis.L{a.layer}.scaling"])
    t = load_vis_layer(a.layer)
    W_IN, W_HP, W_T, W_FB = D // 16, HDP // 16, T // 16, FB // 16

    # weights: quantise the real matrices, then pad (zero codes, scales and biases in the padding)
    def qpad_out(name):
        q = LL.QW(t[name + ".w"], None)
        return (pad_heads_out(q.codes), pad_heads_out(q.scale), pad_heads_out(t[name + ".b"]), q)
    cq, sq, bq, Wq = qpad_out("q_proj")
    ck, sk, bk, Wk = qpad_out("k_proj")
    cv, sv, bv, Wv = qpad_out("v_proj")
    Wo = LL.QW(t["out_proj.w"], None)
    co = pad_heads_in(Wo.codes)
    W1 = LL.QW(t["fc1.w"], None)
    c1 = np.zeros((FFP, D), np.int64)
    c1[:FF] = W1.codes
    s1 = np.zeros(FFP, np.float32)
    s1[:FF] = W1.scale
    b1 = np.zeros(FFP, np.float32)
    b1[:FF] = t["fc1.b"]
    W2 = LL.QW(t["fc2.w"], None)
    c2 = np.zeros((D, FFP), np.int64)
    c2[:, :FF] = W2.codes

    B, C = Build(), LL.Chip()
    TL = Tiler(B)
    f32c, bf16v = LL.f32c, LL.bf16v
    QN = HEADS * HDP

    def dequant(acc, s_row, s_col, bias=None, out="bf16"):
        y, _ = R.op_dequant(acc, s_row, s_col, bias, out)
        C._n("DEQUANT", acc.size)
        return y

    def ln(codes, amax, gain, beta):
        mu, r = R.ln_stat(codes, amax, 1.0 / D)
        y, _ = R.ln_apply(codes, mu, r, f32c(gain), f32c(beta))
        C._n("LN", codes.size * 2)
        return mu, r, y

    def ln_stage(x_base, xsum_base, xsum_is_input, codes, amax, gain, beta):
        """LN_STAT + LN_APPLY + QUANT on T x D rows at x_base; the row amax comes from xsum_base (a bf16 amax
        input, or the summaries of the op that wrote the rows)"""
        g_base, b_base = B.put_in(f32c(gain), 32), B.put_in(f32c(beta), 32)
        LS, LSSUM = B.aout(T * 32), B.aout(T * 64)
        mu, r, hc = ln(codes, amax, gain, beta)
        B.expect(LS, r, 32)
        B.expect(LSSUM, [int(v) & 0xFFFFFFFF for v in mu], 64)
        HB, HSUM = B.aout(T * D * 16), B.aout(T * 64)
        B.expect(HB, hc, 16)
        B.expect(HSUM, summaries(hc), 64)
        HQ, HQSUM = B.aout(T * D * 8), B.aout(T * 64)
        hq, s_h = C.quant(hc)
        B.expect(HQ, hq & 0xFF, 8)
        B.expect(HQSUM, [int(v) for v in s_h], 64)
        rs = desc(xsum_base, SH_R, F_BF16) if xsum_is_input else desc(xsum_base, SH_R, F_SUM, FIELD_AMAX)
        B.stage(VN, vu_record(OP_LN_STAT, T, D, LS, LSSUM, desc(x_base, SH_E, F_BF16), rs=rs,
                              k=int(np.float32(1.0 / D).view(np.uint32)))
                + vu_record(OP_LN_APPLY, T, D, HB, HSUM, desc(x_base, SH_E, F_BF16),
                            b=desc(LSSUM, SH_R, F_SUM, FIELD_RS0), c=desc(LS, SH_R, F_FP32),
                            d=desc(g_base, SH_C, F_FP32), e=desc(b_base, SH_C, F_FP32))
                + vu_record(OP_QUANT, T, D, HQ, HQSUM, desc(HB, SH_E, F_BF16),
                            rs=desc(HSUM, SH_R, F_SUM, FIELD_AMAX)) + [0])
        return hc, hq, s_h, HQ, HQSUM

    # ================= stage 0: LayerNorm 1 + QUANT =================
    xc = R.bf16_codes(x)
    x_base = B.put_in(xc, 16)
    amax_x = R.row_amax_code(xc)
    amax_base = B.put_in(amax_x, 16)
    hc, hq, s_h, HQ, HQSUM = ln_stage(x_base, amax_base, True, xc, amax_x, t["layer_norm1.w"], t["layer_norm1.b"])

    # ================= stage 1: q, k per head; v role-swapped =================
    accq, acck = C.gemm(hq, cq), C.gemm(hq, ck)                           # (T, 1536)
    accv = cv @ hq.T                                                      # (1536, T): channel rows
    ACCQ = [B.aout(T * HDP * 32) for _ in range(HEADS)]
    ACCK = [B.aout(T * HDP * 32) for _ in range(HEADS)]
    ACCV = B.aout(QN * T * 32)
    for h in range(HEADS):
        B.expect(ACCQ[h], accq[:, HDP * h:HDP * (h + 1)] & 0xFFFFFFFF, 32)
    for h in range(HEADS):
        B.expect(ACCK[h], acck[:, HDP * h:HDP * (h + 1)] & 0xFFFFFFFF, 32)
    B.expect(ACCV, accv & 0xFFFFFFFF, 32)
    cmds = TL.gemm_cmds(HQ, T, W_IN, cq, ACCQ, blocks=HEADS) + TL.gemm_cmds(HQ, T, W_IN, ck, ACCK, blocks=HEADS)
    Wv_rows = B.put_in(cv.reshape(-1) & 0xFF, 8)
    cmds += TL.cmds_loadx(HQ, T, W_IN, Wv_rows, QN, ACCV)                 # token code rows are the columns
    B.stage(CH8, cmds + [0])

    # ================= stage 2: DEQUANT q / k / v (+ bias), QUANT =================
    sq_f = f32c(sq.astype(np.float64) * (scaling / LN2))                  # log2-domain logits folded into q
    bq_f = f32c(bq.astype(np.float64) * (scaling / LN2))
    sq_base, bq_base = B.put_in(sq_f, 32), B.put_in(bq_f, 32)
    sk_base, bk_base = B.put_in(f32c(sk), 32), B.put_in(f32c(bk), 32)
    sv_base, bv_base = B.put_in(f32c(sv), 32), B.put_in(f32c(bv), 32)
    recs = []
    QB, QBSUM = B.aout(HEADS * T * HDP * 16), B.aout(HEADS * T * 64)
    KB, KBSUM = B.aout(HEADS * T * HDP * 16), B.aout(HEADS * T * 64)
    qb, kb = np.zeros((HEADS * T, HDP), np.uint16), np.zeros((HEADS * T, HDP), np.uint16)
    for nm, acc, s_c, b_c, sb, bb, OUT, OSUM, dst, ACC in (
            ("q", accq, sq_f, bq_f, sq_base, bq_base, QB, QBSUM, qb, ACCQ),
            ("k", acck, f32c(sk), f32c(bk), sk_base, bk_base, KB, KBSUM, kb, ACCK)):
        for h in range(HEADS):
            cols = slice(HDP * h, HDP * (h + 1))
            y = dequant(acc[:, cols], s_h, s_c[cols], np.repeat(b_c[None, cols], T, 0))
            dst[h * T:(h + 1) * T] = y
            B.expect(OUT + h * T * HDP * 2, y, 16)
            B.expect(OSUM + h * T * 8, summaries(y), 64)
            recs += vu_record(OP_DEQUANT, T, HDP, OUT + h * T * HDP * 2, OSUM + h * T * 8, desc(ACC[h], SH_E, F_INT32),
                              c=desc(HQSUM, SH_R, F_SUM, FIELD_RS0), d=desc(sb + h * HDP * 4, SH_C, F_FP32),
                              e=desc(bb + h * HDP * 4, SH_C, F_FP32), bias_en=1)
    VB, VBSUM = B.aout(QN * T * 16), B.aout(QN * 64)
    s_h_u32 = np.asarray(s_h, np.uint32)
    vb = dequant(accv, f32c(sv), s_h_u32, np.repeat(f32c(bv)[:, None], T, 1))
    B.expect(VB, vb, 16)
    B.expect(VBSUM, summaries(vb), 64)
    recs += vu_record(OP_DEQUANT, QN, T, VB, VBSUM, desc(ACCV, SH_E, F_INT32), c=desc(sv_base, SH_R, F_FP32),
                      d=desc(HQSUM, SH_C, F_SUM, FIELD_RS0), e=desc(bv_base, SH_R, F_FP32), bias_en=1)
    QQ, QQSUM = B.aout(HEADS * T * HDP * 8), B.aout(HEADS * T * 64)
    qq, s_q = C.quant(qb)
    B.expect(QQ, qq & 0xFF, 8)
    B.expect(QQSUM, [int(v) for v in s_q], 64)
    KQ, KQSUM = B.aout(HEADS * T * HDP * 8), B.aout(HEADS * T * 64)
    kq, s_k = C.quant(kb)
    B.expect(KQ, kq & 0xFF, 8)
    B.expect(KQSUM, [int(v) for v in s_k], 64)
    VQ, VQSUM = B.aout(QN * T * 8), B.aout(QN * 64)
    vq, s_v = C.quant(vb)
    B.expect(VQ, vq & 0xFF, 8)
    B.expect(VQSUM, [int(v) for v in s_v], 64)
    recs += (vu_record(OP_QUANT, HEADS * T, HDP, QQ, QQSUM, desc(QB, SH_E, F_BF16), rs=desc(QBSUM, SH_R, F_SUM, FIELD_AMAX))
             + vu_record(OP_QUANT, HEADS * T, HDP, KQ, KQSUM, desc(KB, SH_E, F_BF16), rs=desc(KBSUM, SH_R, F_SUM, FIELD_AMAX))
             + vu_record(OP_QUANT, QN, T, VQ, VQSUM, desc(VB, SH_E, F_BF16), rs=desc(VBSUM, SH_R, F_SUM, FIELD_AMAX)))
    B.stage(VN, recs + [0])

    # ================= stage 3: QK^T per head =================
    LOGIT = B.aout(HEADS * T * T * 32)
    logit = np.zeros((HEADS * T, T), np.int64)
    cmds = []
    for h in range(HEADS):
        rows = slice(h * T, (h + 1) * T)
        logit[rows] = qq[rows] @ kq[rows].T
        cmds += TL.cmds_loadx(KQ + h * T * HDP, T, W_HP, QQ + h * T * HDP, T, LOGIT + h * T * T * 4)
    B.expect(LOGIT, logit & 0xFFFFFFFF, 32)
    B.stage(CH8, cmds + [0])

    # ================= stage 4: DEQUANT logits per head, SMAX_Q8 =================
    LB, LSUM = B.aout(HEADS * T * T * 16), B.aout(HEADS * T * 64)
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
    PC, PSUM = B.aout(HEADS * T * T * 8), B.aout(HEADS * T * 64)
    pq, s_p = C.smax_u8(lb, np.ones(lb.shape, bool))
    B.expect(PC, pq & 0xFF, 8)
    B.expect(PSUM, [int(v) for v in s_p], 64)
    recs += vu_record(OP_SMAX_Q8, HEADS * T, T, PC, PSUM, desc(LB, SH_E, F_BF16), rs=desc(LSUM, SH_R, F_SUM, FIELD_MAX),
                      b_fp32=1)
    B.stage(VN, recs + [0])

    # ================= stage 5: PV per head =================
    CTX = B.aout(HEADS * T * HDP * 32)
    ctx = np.zeros((HEADS * T, HDP), np.int64)
    cmds = []
    for h in range(HEADS):
        rows = slice(h * T, (h + 1) * T)
        ctx[rows] = pq[rows] @ vq[HDP * h:HDP * (h + 1)].T
        cmds += TL.cmds_loadx(VQ + h * HDP * T, HDP, W_T, PC + h * T * T, T, CTX + h * T * HDP * 4)
    B.expect(CTX, ctx & 0xFFFFFFFF, 32)
    B.stage(CHU, cmds + [0])

    # the K-split recombination: fp32 DEQUANTs chained through the fp32 bias, the last one bf16
    def dequant_chain(accs, sums_bases, s_rows, scale, scale_base, bias0, bias0_base):
        recs, prev, prev_base = [], None, None
        for j, acc in enumerate(accs):
            last = j == len(accs) - 1
            OUT, OSUM = B.aout(T * D * (16 if last else 32)), B.aout(T * 64)
            bias = np.repeat(bias0[None, :], T, 0) if prev is None else prev
            y = dequant(acc, s_rows[j], scale, bias, "bf16" if last else "fp32")
            B.expect(OUT, y, 16 if last else 32)
            B.expect(OSUM, summaries(y) if last else [0] * T, 64)
            e = desc(bias0_base, SH_C, F_FP32) if prev is None else desc(prev_base, SH_E, F_FP32)
            recs += vu_record(OP_DEQUANT, T, D, OUT, OSUM, desc(sums_bases[j][0], SH_E, F_INT32),
                              c=desc(sums_bases[j][1], SH_R, F_SUM, FIELD_RS0), d=desc(scale_base, SH_C, F_FP32),
                              e=e, bias_en=1, out_fp32=int(not last))
            prev, prev_base = y, OUT
        return recs, prev, prev_base

    if a.o_proj == "token":
        # ================= stage 6: DEQUANT context per head into token-major rows, ADD b = 0, QUANT per token ======
        CBT, CBSUM = B.aout(T * QN * 16), B.aout(HEADS * T * 64)
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
        CTXB, CTXSUM = B.aout(T * QN * 16), B.aout(T * 64)
        ctxb = C.add(cbt, np.zeros_like(cbt))
        B.expect(CTXB, ctxb, 16)
        B.expect(CTXSUM, summaries(ctxb), 64)
        CQ, CQSUM = B.aout(T * QN * 8), B.aout(T * 64)
        cqd, s_c = C.quant(ctxb)
        B.expect(CQ, cqd & 0xFF, 8)
        B.expect(CQSUM, [int(v) for v in s_c], 64)
        recs += (vu_record(OP_ADD, T, QN, CTXB, CTXSUM, desc(CBT, SH_E, F_BF16), b=K(0))
                 + vu_record(OP_QUANT, T, QN, CQ, CQSUM, desc(CTXB, SH_E, F_BF16), rs=desc(CTXSUM, SH_R, F_SUM, FIELD_AMAX)))
        B.stage(VN, recs + [0])
        cb = np.concatenate([cbt[:, HDP * h:HDP * (h + 1)] for h in range(HEADS)])  # head-major, for the report

        # ================= stage 7: out_proj, one GEMM =================
        acco1 = C.gemm(cqd, co)
        ACCO = B.aout(T * D * 32)
        B.expect(ACCO, acco1 & 0xFFFFFFFF, 32)
        B.stage(CH8, TL.gemm_cmds(CQ, T, QN // 16, co, ACCO) + [0])

        # ================= stage 8: DEQUANT (+ out_proj bias), residual =================
        so_base, bo_base = B.put_in(f32c(Wo.scale), 32), B.put_in(f32c(t["out_proj.b"]), 32)
        OB, OBSUM = B.aout(T * D * 16), B.aout(T * 64)
        oc = dequant(acco1, np.asarray(s_c, np.uint32), f32c(Wo.scale), np.repeat(f32c(t["out_proj.b"])[None, :], T, 0))
        B.expect(OB, oc, 16)
        B.expect(OBSUM, summaries(oc), 64)
        YB, YSUM = B.aout(T * D * 16), B.aout(T * 64)
        yc = C.add(oc, xc)
        B.expect(YB, yc, 16)
        B.expect(YSUM, summaries(yc), 64)
        B.stage(VN, vu_record(OP_DEQUANT, T, D, OB, OBSUM, desc(ACCO, SH_E, F_INT32), c=desc(CQSUM, SH_R, F_SUM, FIELD_RS0),
                              d=desc(so_base, SH_C, F_FP32), e=desc(bo_base, SH_C, F_FP32), bias_en=1)
                    + vu_record(OP_ADD, T, D, YB, YSUM, desc(OB, SH_E, F_BF16), b=desc(x_base, SH_E, F_BF16)) + [0])


    else:
        # ================= stage 6: DEQUANT context per head, QUANT per (token, head) =================
        CB, CBSUM = B.aout(HEADS * T * HDP * 16), B.aout(HEADS * T * 64)
        cb = np.zeros((HEADS * T, HDP), np.uint16)
        recs = []
        for h in range(HEADS):
            rows = slice(h * T, (h + 1) * T)
            y = dequant(ctx[rows], np.asarray(s_p[rows], np.uint32), np.asarray(s_v[HDP * h:HDP * (h + 1)], np.uint32))
            cb[rows] = y
            recs += vu_record(OP_DEQUANT, T, HDP, CB + h * T * HDP * 2, CBSUM + h * T * 8, desc(CTX + h * T * HDP * 4, SH_E, F_INT32),
                              c=desc(PSUM + h * T * 8, SH_R, F_SUM, FIELD_RS0), d=desc(VQSUM + h * HDP * 8, SH_C, F_SUM, FIELD_RS0))
        B.expect(CB, cb, 16)
        B.expect(CBSUM, summaries(cb), 64)
        CQ, CQSUM = B.aout(HEADS * T * HDP * 8), B.aout(HEADS * T * 64)
        cqd, s_c = C.quant(cb)
        B.expect(CQ, cqd & 0xFF, 8)
        B.expect(CQSUM, [int(v) for v in s_c], 64)
        recs += vu_record(OP_QUANT, HEADS * T, HDP, CQ, CQSUM, desc(CB, SH_E, F_BF16), rs=desc(CBSUM, SH_R, F_SUM, FIELD_AMAX))
        B.stage(VN, recs + [0])

        # ================= stage 7: out_proj, K split per head =================
        acco = []
        ACCO = [B.aout(T * D * 32) for _ in range(HEADS)]
        cmds = []
        for h in range(HEADS):
            rows = slice(h * T, (h + 1) * T)
            wcols = co[:, HDP * h:HDP * (h + 1)]
            acco.append(C.gemm(cqd[rows], wcols))
            B.expect(ACCO[h], acco[h] & 0xFFFFFFFF, 32)
            cmds += TL.gemm_cmds(CQ + h * T * HDP, T, W_HP, wcols, ACCO[h])
        B.stage(CH8, cmds + [0])

        # ================= stage 8: DEQUANT chain (+ out_proj bias), residual =================
        so_base, bo_base = B.put_in(f32c(Wo.scale), 32), B.put_in(f32c(t["out_proj.b"]), 32)


        recs, oc, OB = dequant_chain(acco, [(ACCO[h], CQSUM + h * T * 8) for h in range(HEADS)],
                                     [np.asarray(s_c[h * T:(h + 1) * T], np.uint32) for h in range(HEADS)],
                                     f32c(Wo.scale), so_base, f32c(t["out_proj.b"]), bo_base)
        YB, YSUM = B.aout(T * D * 16), B.aout(T * 64)
        yc = C.add(oc, xc)
        B.expect(YB, yc, 16)
        B.expect(YSUM, summaries(yc), 64)
        B.stage(VN, recs + vu_record(OP_ADD, T, D, YB, YSUM, desc(OB, SH_E, F_BF16), b=desc(x_base, SH_E, F_BF16)) + [0])

    # ================= stage 9: LayerNorm 2 + QUANT =================
    h2c, h2q, s_h2, H2Q, H2QSUM = ln_stage(YB, YSUM, False, yc, R.row_amax_code(yc),
                                           t["layer_norm2.w"], t["layer_norm2.b"])

    # ================= stage 10: fc1 into two sub-block regions =================
    acc1 = C.gemm(h2q, c1)
    ACC1 = [B.aout(T * FB * 32) for _ in range(NB)]
    for j in range(NB):
        B.expect(ACC1[j], acc1[:, FB * j:FB * (j + 1)] & 0xFFFFFFFF, 32)
    B.stage(CH8, TL.gemm_cmds(H2Q, T, W_IN, c1, ACC1, blocks=NB) + [0])

    # ================= stage 11: DEQUANT (+ bias), GELU, QUANT per sub-block =================
    s1_base, b1_base = B.put_in(f32c(s1), 32), B.put_in(f32c(b1), 32)
    recs, MQ, MQSUM, mq, s_m = [], [], [], [], []
    for j in range(NB):
        cols = slice(FB * j, FB * (j + 1))
        G1, G1SUM = B.aout(T * FB * 16), B.aout(T * 64)
        g1 = dequant(acc1[:, cols], s_h2, f32c(s1)[cols], np.repeat(f32c(b1)[None, cols], T, 0))
        B.expect(G1, g1, 16)
        B.expect(G1SUM, summaries(g1), 64)
        GE, GESUM = B.aout(T * FB * 16), B.aout(T * 64)
        ge, _ = R.op_gelu(g1)
        C._n("GELU", g1.size)
        B.expect(GE, ge, 16)
        B.expect(GESUM, summaries(ge), 64)
        MQ.append(B.aout(T * FB * 8))
        MQSUM.append(B.aout(T * 64))
        mq_j, s_m_j = C.quant(ge)
        mq.append(mq_j)
        s_m.append(np.asarray(s_m_j, np.uint32))
        B.expect(MQ[j], mq_j & 0xFF, 8)
        B.expect(MQSUM[j], [int(v) for v in s_m_j], 64)
        recs += (vu_record(OP_DEQUANT, T, FB, G1, G1SUM, desc(ACC1[j], SH_E, F_INT32),
                           c=desc(H2QSUM, SH_R, F_SUM, FIELD_RS0), d=desc(s1_base + FB * j * 4, SH_C, F_FP32),
                           e=desc(b1_base + FB * j * 4, SH_C, F_FP32), bias_en=1)
                 + vu_record(OP_GELU, T, FB, GE, GESUM, desc(G1, SH_E, F_BF16))
                 + vu_record(OP_QUANT, T, FB, MQ[j], MQSUM[j], desc(GE, SH_E, F_BF16),
                             rs=desc(GESUM, SH_R, F_SUM, FIELD_AMAX)))
    B.stage(VN, recs + [0])

    # ================= stage 12: fc2, K split per sub-block =================
    acc2 = []
    ACC2 = [B.aout(T * D * 32) for _ in range(NB)]
    cmds = []
    for j in range(NB):
        wcols = c2[:, FB * j:FB * (j + 1)]
        acc2.append(C.gemm(mq[j], wcols))
        B.expect(ACC2[j], acc2[j] & 0xFFFFFFFF, 32)
        cmds += TL.gemm_cmds(MQ[j], T, W_FB, wcols, ACC2[j])
    B.stage(CH8, cmds + [0])

    # ================= stage 13: DEQUANT chain (+ fc2 bias), residual =================
    s2_base, b2_base = B.put_in(f32c(W2.scale), 32), B.put_in(f32c(t["fc2.b"]), 32)
    recs, mc, MB = dequant_chain(acc2, [(ACC2[j], MQSUM[j]) for j in range(NB)], s_m,
                                 f32c(W2.scale), s2_base, f32c(t["fc2.b"]), b2_base)
    ZB, ZSUM = B.aout(T * D * 16), B.aout(T * 64)
    zc = C.add(mc, yc)
    B.expect(ZB, zc, 16)
    B.expect(ZSUM, summaries(zc), 64)
    B.stage(VN, recs + vu_record(OP_ADD, T, D, ZB, ZSUM, desc(MB, SH_E, F_BF16), b=desc(YB, SH_E, F_BF16)) + [0])

    # ================= fp32 reference (unpadded) and the capture =================
    f64 = lambda v: v.astype(np.float64)                                  # noqa: E731
    h_f = ln_fp(x, t["layer_norm1.w"], t["layer_norm1.b"])
    q_f = (h_f @ f64(t["q_proj.w"]).T + f64(t["q_proj.b"])).reshape(T, HEADS, HD).transpose(1, 0, 2)
    k_f = (h_f @ f64(t["k_proj.w"]).T + f64(t["k_proj.b"])).reshape(T, HEADS, HD).transpose(1, 0, 2)
    v_f = (h_f @ f64(t["v_proj.w"]).T + f64(t["v_proj.b"])).reshape(T, HEADS, HD).transpose(1, 0, 2)
    logits_f = np.einsum("htd,hsd->hts", q_f, k_f) * scaling
    p_f = np.exp(logits_f - logits_f.max(-1, keepdims=True))
    p_f /= p_f.sum(-1, keepdims=True)
    ctx_f = np.einsum("hts,hsd->htd", p_f, v_f).transpose(1, 0, 2).reshape(T, HEADS * HD)
    o_f = ctx_f @ f64(t["out_proj.w"]).T + f64(t["out_proj.b"])
    y_f = x + o_f
    h2_f = ln_fp(y_f, t["layer_norm2.w"], t["layer_norm2.b"])
    g_f = h2_f @ f64(t["fc1.w"]).T + f64(t["fc1.b"])
    m_f = LL.gelu_tanh(g_f) @ f64(t["fc2.w"]).T + f64(t["fc2.b"])
    z_f = y_f + m_f
    rel = LL.rel
    cap = {"attn_out": rel(o_f, z[f"vis.L{a.layer}.attn_out"][a.image].astype(np.float64)),
           "layer_norm2.in": rel(y_f, z[f"vis.L{a.layer}.layer_norm2.in"][a.image].astype(np.float64)),
           "fc1_out (96 rows)": rel(g_f[rows96()], z[f"vis.L{a.layer}.fc1_out"][a.image].astype(np.float64))}

    widest = 0
    for n, _, words in B.stages:
        if n == VN:
            for i in range(0, len(words) - 1, 6):
                widest = max(widest, (words[i] >> 60) & 0xFFFF)
    assert widest <= (1 << a.slot_bits), f"a vector op is {widest} wide; SLOT_BITS >= {widest.bit_length()}"
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "barrier.txt").exists():
        (out / "barrier.txt").unlink()
    B.mem.write(out / "mem_in.hex")
    B.exp.write(out / "mem_exp.hex")
    (out / "stages.txt").write_text("".join(f"{n} {b:x}\n" for n, b, _ in B.stages))
    (out / "regions.txt").write_text("".join(f"{n} {b:x} {sz} {st}\n" for n, b, sz, st in B.regions))
    ctx_chip = bf16v(cb).reshape(HEADS, T, HDP)[:, :, :HD].transpose(1, 0, 2).reshape(T, HEADS * HD)
    (out / "info.txt").write_text(
        f"part=siglip layer={a.layer} image={a.image} T={T} D={D} heads={HEADS}x{HD} (padded {HDP}) FF={FF} "
        f"(padded {FFP}, {NB} sub-blocks) stages={len(B.stages)} widest vector row={widest} (SLOT_BITS {a.slot_bits})\n"
        f"image {len(B.mem.beats)} beats, expected {len(B.exp.beats)} beats\n"
        f"fp32 reference vs capture: " + ", ".join(f"{k} {v:.1e}" for k, v in cap.items()) + "\n"
        f"chip vs fp32: ctx {rel(ctx_chip, ctx_f):.4f} o {rel(bf16v(oc), o_f):.4f} "
        f"attn_block {rel(bf16v(yc) - x, y_f - x):.4f} mlp {rel(bf16v(mc), m_f):.4f} "
        f"layer {rel(bf16v(zc) - x, z_f - x):.4f}\n"
        f"max|acc| 2^{math.log2(max(C.max_acc, 1)):.1f}\n")
    print((out / "info.txt").read_text().strip())
    print("stages: " + ", ".join(f"{n}@{b:#x}" for n, b, _ in B.stages))


if __name__ == "__main__":
    main()
