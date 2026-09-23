#!/usr/bin/env python3
"""GDDR6 image, node programs and expected beats for tb_pi0_attn.sv: the two ends of pi0's vision path at their real
size, the only computations of a chunk not covered by the layer and action-head tests.

  patch embedding  256 patches x 588 pixel values (3 x 14 x 14, padded to 608 = 38 words) -> QUANT per patch
                   -> the Conv2d as a GEMM (1152 x 608) -> DEQUANT + bias -> ADD the fp32 position table (b fp32)
                   = SigLIP layer 0's input.  The capture has no pixels, so the image is a synthetic smooth one in
                   SigLIP's [-1, 1] range and the check is the fp32 convolution of the same pixels.
  projector        SigLIP's last hidden state before post_layernorm (captured, one camera) -> LN_STAT, LN_APPLY
                   -> QUANT -> multi_modal_projector GEMM (2048 x 1152) -> DEQUANT + bias = that camera's 256 image
                   tokens of the prefix, checked against the captured prefix layer 0 input.

Numerics: W8 per output channel (MSE clip), A8 per token dynamic.
Stages: 0 VN QUANT | 1 CH8 patch GEMM | 2 VN DEQUANT, ADD pos | 3 VN LN_STAT, LN_APPLY, QUANT | 4 CH8 projector |
        5 VN DEQUANT

Usage: pi0_vision_ends_golden.py --out DIR [--image 0|1]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import vector_unit_ref as R                                              # noqa: E402
import pi0_layer_lower as LL                                            # noqa: E402
from pi0_attn_golden import Build, Tiler, vu_record, summaries, VN, CH8    # noqa: E402
from vu_node_golden import desc, SH_E, SH_R, SH_C, F_BF16, F_FP32, F_INT32, F_SUM, FIELD_RS0, FIELD_AMAX  # noqa: E402

OP_ADD, OP_LN_STAT, OP_LN_APPLY, OP_QUANT, OP_DEQUANT = 0, 3, 4, 13, 14
VT = "model.paligemma_with_expert.paligemma.model.vision_tower.vision_model."
T, DV, DL, KP, KPP = 256, 1152, 2048, 588, 608


def load() -> dict:
    from safetensors import safe_open
    t = {}
    with safe_open(LL.CKPT, "np") as f:
        t["patch.w"] = f.get_tensor(VT + "embeddings.patch_embedding.weight").astype(np.float32).reshape(DV, KP)
        t["patch.b"] = f.get_tensor(VT + "embeddings.patch_embedding.bias").astype(np.float32)
        t["pos"] = f.get_tensor(VT + "embeddings.position_embedding.weight").astype(np.float32)
        t["pln.w"] = f.get_tensor(VT + "post_layernorm.weight").astype(np.float32)
        t["pln.b"] = f.get_tensor(VT + "post_layernorm.bias").astype(np.float32)
        p = "model.paligemma_with_expert.paligemma.model.multi_modal_projector.linear."
        t["proj.w"] = f.get_tensor(p + "weight").astype(np.float32)
        t["proj.b"] = f.get_tensor(p + "bias").astype(np.float32)
    return t


def synthetic_patches() -> np.ndarray:
    """a smooth 224 x 224 RGB image in [-1, 1], cut into 16 x 16 patches of 14 x 14, flattened (C, kh, kw) as the
    Conv2d weight is"""
    yy, xx = np.meshgrid(np.arange(224), np.arange(224), indexing="ij")
    img = np.stack([np.sin(xx / 17.0 + c) * np.cos(yy / 23.0 - c) for c in range(3)]).astype(np.float32)   # (3, 224, 224)
    patches = img.reshape(3, 16, 14, 16, 14).transpose(1, 3, 0, 2, 4).reshape(T, KP)
    return patches


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--image", type=int, default=0)
    a = ap.parse_args()
    z = np.load(LL.CAPTURE)
    t = load()
    B, C = Build(), LL.Chip()
    TL = Tiler(B)
    f32c, bf16v = LL.f32c, LL.bf16v

    def quant(src, rs, rows, L, codes):
        """QUANT record; rs: the row-amax operand (a bf16 amax input, or the summaries of the op before)"""
        Q, QSUM = B.aout(rows * L * 8), B.aout(rows * 64)
        q, s = C.quant(codes)
        B.expect(Q, q & 0xFF, 8)
        B.expect(QSUM, [int(v) for v in s], 64)
        return q, np.asarray(s, np.uint32), Q, QSUM, vu_record(OP_QUANT, rows, L, Q, QSUM, desc(src, SH_E, F_BF16), rs=rs)

    # ================= patch embedding =================
    pat = synthetic_patches()
    pc = np.zeros((T, KPP), np.uint16)
    pc[:, :KP] = R.bf16_codes(pat)
    P_BASE = B.put_in(pc, 16)
    P_AMAX = B.put_in(R.row_amax_code(pc), 16)
    pq, s_p, PQ, PQSUM, rq = quant(P_BASE, desc(P_AMAX, SH_R, F_BF16), T, KPP, pc)
    B.stage(VN, rq + [0])
    Wp = LL.QW(t["patch.w"], None)
    wp_codes = np.zeros((DV, KPP), np.int64)
    wp_codes[:, :KP] = Wp.codes
    acc = C.gemm(pq, wp_codes)
    ACC = B.aout(T * DV * 32)
    B.expect(ACC, acc & 0xFFFFFFFF, 32)
    B.stage(CH8, TL.gemm_cmds(PQ, T, KPP // 16, wp_codes, ACC) + [0])
    s_base, b_base = B.put_in(f32c(Wp.scale), 32), B.put_in(f32c(t["patch.b"]), 32)
    PE, PESUM = B.aout(T * DV * 16), B.aout(T * 64)
    pe, _ = R.op_dequant(acc, s_p, f32c(Wp.scale), np.repeat(f32c(t["patch.b"])[None, :], T, 0), "bf16")
    B.expect(PE, pe, 16)
    B.expect(PESUM, summaries(pe), 64)
    POS = B.put_in(f32c(t["pos"]), 32)
    X0, X0SUM = B.aout(T * DV * 16), B.aout(T * 64)
    x0, _ = R.op_add(pe, f32c(t["pos"]), "fp32")
    B.expect(X0, x0, 16)
    B.expect(X0SUM, summaries(x0), 64)
    B.stage(VN, vu_record(OP_DEQUANT, T, DV, PE, PESUM, desc(ACC, SH_E, F_INT32), c=desc(PQSUM, SH_R, F_SUM, FIELD_RS0),
                          d=desc(s_base, SH_C, F_FP32), e=desc(b_base, SH_C, F_FP32), bias_en=1)
            + vu_record(OP_ADD, T, DV, X0, X0SUM, desc(PE, SH_E, F_BF16), b=desc(POS, SH_E, F_FP32), b_fp32=1) + [0])

    # ================= post_layernorm + projector =================
    h = z["vis.post_ln.in"][a.image].astype(np.float64)
    hc = R.bf16_codes(h)
    H_BASE, H_AMAX = B.put_in(hc, 16), B.put_in(R.row_amax_code(hc), 16)
    g_base, be_base = B.put_in(f32c(t["pln.w"]), 32), B.put_in(f32c(t["pln.b"]), 32)
    LS, LSSUM = B.aout(T * 32), B.aout(T * 64)
    mu, r = R.ln_stat(hc, R.row_amax_code(hc), 1.0 / DV)
    B.expect(LS, r, 32)
    B.expect(LSSUM, [int(v) & 0xFFFFFFFF for v in mu], 64)
    NB, NBSUM = B.aout(T * DV * 16), B.aout(T * 64)
    nb, _ = R.ln_apply(hc, mu, r, f32c(t["pln.w"]), f32c(t["pln.b"]))
    B.expect(NB, nb, 16)
    B.expect(NBSUM, summaries(nb), 64)
    nq, s_n, NQ, NQSUM, rq = quant(NB, desc(NBSUM, SH_R, F_SUM, FIELD_AMAX), T, DV, nb)
    B.stage(VN, vu_record(OP_LN_STAT, T, DV, LS, LSSUM, desc(H_BASE, SH_E, F_BF16), rs=desc(H_AMAX, SH_R, F_BF16),
                          k=int(np.float32(1.0 / DV).view(np.uint32)))
            + vu_record(OP_LN_APPLY, T, DV, NB, NBSUM, desc(H_BASE, SH_E, F_BF16), b=desc(LSSUM, SH_R, F_SUM, FIELD_RS0),
                        c=desc(LS, SH_R, F_FP32), d=desc(g_base, SH_C, F_FP32), e=desc(be_base, SH_C, F_FP32))
            + rq + [0])
    Wj = LL.QW(t["proj.w"], None)
    acc = C.gemm(nq, Wj.codes)
    ACC2 = B.aout(T * DL * 32)
    B.expect(ACC2, acc & 0xFFFFFFFF, 32)
    B.stage(CH8, TL.gemm_cmds(NQ, T, DV // 16, Wj.codes, ACC2) + [0])
    sj_base, bj_base = B.put_in(f32c(Wj.scale), 32), B.put_in(f32c(t["proj.b"]), 32)
    IE, IESUM = B.aout(T * DL * 16), B.aout(T * 64)
    ie, _ = R.op_dequant(acc, s_n, f32c(Wj.scale), np.repeat(f32c(t["proj.b"])[None, :], T, 0), "bf16")
    B.expect(IE, ie, 16)
    B.expect(IESUM, summaries(ie), 64)
    B.stage(VN, vu_record(OP_DEQUANT, T, DL, IE, IESUM, desc(ACC2, SH_E, F_INT32), c=desc(NQSUM, SH_R, F_SUM, FIELD_RS0),
                          d=desc(sj_base, SH_C, F_FP32), e=desc(bj_base, SH_C, F_FP32), bias_en=1) + [0])

    # ================= fp32 references =================
    f64 = lambda v: v.astype(np.float64)                                  # noqa: E731
    x0_f = f64(pat) @ f64(t["patch.w"]).T + f64(t["patch.b"]) + f64(t["pos"])
    mu_f = h.mean(-1, keepdims=True)
    n_f = (h - mu_f) / np.sqrt(((h - mu_f) ** 2).mean(-1, keepdims=True) + 1e-6) * f64(t["pln.w"]) + f64(t["pln.b"])
    ie_f = n_f @ f64(t["proj.w"]).T + f64(t["proj.b"])
    lm_in = z["lm.L0.layer_in"][a.image * T:(a.image + 1) * T].astype(np.float64)
    rel = LL.rel
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "barrier.txt").exists():
        (out / "barrier.txt").unlink()
    B.mem.write(out / "mem_in.hex")
    B.exp.write(out / "mem_exp.hex")
    (out / "stages.txt").write_text("".join(f"{n} {b:x}\n" for n, b, _ in B.stages))
    (out / "regions.txt").write_text("".join(f"{n} {b:x} {sz} {st}\n" for n, b, sz, st in B.regions))
    scale = float(np.linalg.norm(lm_in) / max(np.linalg.norm(ie_f), 1e-30))
    (out / "info.txt").write_text(
        f"part=vision_ends image={a.image} stages={len(B.stages)}\n"
        f"image {len(B.mem.beats)} beats, expected {len(B.exp.beats)} beats\n"
        f"fp32 projector output vs captured prefix layer 0 input: {rel(ie_f, lm_in):.1e} (norm ratio {scale:.4f})\n"
        f"chip vs fp32: patch embedding + pos {rel(bf16v(x0), x0_f):.4f}  image tokens {rel(bf16v(ie), ie_f):.4f}\n")
    print((out / "info.txt").read_text().strip())


if __name__ == "__main__":
    main()
