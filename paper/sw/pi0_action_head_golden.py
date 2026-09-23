#!/usr/bin/env python3
"""GDDR6 image, node programs and expected beats for tb_pi0_attn.sv: pi0's ACTION HEAD around the expert, at its real
size, for one Euler step of the captured frame -- the parts of the 10-step loop that are not transformer layers.

  input side   x_t (fp32, 50 x 32; step 0 = the captured noise)
               -> ADD b fp32 (a = 0): bf16 codes + row amax                                     (vector)
               -> QUANT per token -> action_in_proj GEMM (32 -> 1024)                          (vector, chain)
               -> DEQUANT + bias -> QUANT -> action_time_mlp_in on the action half (1024 -> 1024)
               -> DEQUANT + bias, where the bias also carries W_in[:, 1024:] . time_emb(t): the time embedding is
                  the same for all 50 tokens, so its half of the concatenated input is a per-step constant
               -> SILU -> QUANT -> action_time_mlp_out (1024 -> 1024) -> DEQUANT + bias = the 50 action tokens
  output side  expert final norm input (51 x 1024, captured) -> RMS_STAT, RMS_APPLY -> QUANT (all 51 rows: operand
               regions start on a beat, so they cannot start at row 1) -> action_out_proj GEMM (1024 -> 32)
               -> DEQUANT + bias to fp32 -> EULER on rows 1..50 (an fp32 row is 4 beats): x_{t+dt} = x_t + dt v_t

Numerics: W8 per output channel (MSE clip), A8 per token dynamic, as the layers.  (The recipe's grouped scales for
action_in_proj, G = 4 over 8 columns, are not realisable as GEMMs of whole 16-byte words; this uses G = 1.)
Checked against the capture: the SiLU input (ex.silu_in) and v_t (ex.v_t) of the step.

Stages: 0 VN ADD, QUANT | 1 CH8 action_in_proj | 2 VN DEQUANT, QUANT | 3 CH8 mlp_in | 4 VN DEQUANT, SILU, QUANT |
        5 CH8 mlp_out | 6 VN DEQUANT, RMS_STAT, RMS_APPLY, QUANT | 7 CH8 action_out_proj | 8 VN DEQUANT fp32, EULER

Usage: pi0_action_head_golden.py --out DIR [--step 0|9]
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
from pi0_attn_golden import Build, Tiler, vu_record, summaries, VN, CH8    # noqa: E402
from vu_node_golden import desc, K, SH_E, SH_R, SH_C, F_BF16, F_FP32, F_INT32, F_SUM, FIELD_RS0, FIELD_AMAX  # noqa: E402

OP_ADD, OP_RMS_STAT, OP_RMS_APPLY, OP_SILU = 0, 1, 2, 9
OP_QUANT, OP_DEQUANT, OP_EULER = 13, 14, 15
NUM_STEPS, CHUNK, AD, D = 10, 50, 32, 1024
MIN_PERIOD, MAX_PERIOD = 4e-3, 4.0


def load_head() -> dict:
    from safetensors import safe_open
    t = {}
    with safe_open(LL.CKPT, "np") as f:
        for n in ("action_in_proj", "action_out_proj", "action_time_mlp_in", "action_time_mlp_out"):
            t[n + ".w"] = f.get_tensor(f"model.{n}.weight").astype(np.float32)
            t[n + ".b"] = f.get_tensor(f"model.{n}.bias").astype(np.float32)
        t["norm.w"] = f.get_tensor("model.paligemma_with_expert.gemma_expert.model.norm.weight").astype(np.float32)
    return t


def time_embedding(time: float, dim: int = D) -> np.ndarray:
    """create_sinusoidal_pos_embedding of LeRobot's pi0 (float64, then float32 as the model casts it)"""
    fraction = np.linspace(0.0, 1.0, dim // 2, dtype=np.float64)
    period = MIN_PERIOD * (MAX_PERIOD / MIN_PERIOD) ** fraction
    x = (1.0 / period * 2 * math.pi) * time
    return np.concatenate([np.sin(x), np.cos(x)]).astype(np.float32)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--step", type=int, default=0, help="0 or 9: the steps whose final-norm input was captured")
    a = ap.parse_args()
    z = np.load(LL.CAPTURE)
    t = load_head()
    dt = -1.0 / NUM_STEPS
    time = 1.0 + a.step * dt
    cap_idx = {0: 0, 9: 1}[a.step]
    T = CHUNK
    f32c, bf16v = LL.f32c, LL.bf16v

    # x_t of this step: the captured noise advanced by the captured v_t (fp32, as the model does)
    x_t = z["ex.noise"].astype(np.float32)
    for s in range(a.step):
        x_t = (x_t + np.float32(dt) * z["ex.v_t"][s]).astype(np.float32)

    W_in = LL.QW(t["action_in_proj.w"], None)
    W_mi = LL.QW(t["action_time_mlp_in.w"][:, :D], None)          # the action half of the concatenated input
    W_mo = LL.QW(t["action_time_mlp_out.w"], None)
    W_out = LL.QW(t["action_out_proj.w"], None)
    bias_mi = (t["action_time_mlp_in.b"].astype(np.float64)
               + t["action_time_mlp_in.w"][:, D:].astype(np.float64) @ time_embedding(time).astype(np.float64))

    B, C = Build(), LL.Chip()
    TL = Tiler(B)

    def dequant(acc, s_row, s_col, bias=None, out="bf16"):
        y, _ = R.op_dequant(acc, s_row, s_col, bias, out)
        return y

    def quant_rec(src, src_sum, rows, L, codes):
        Q, QSUM = B.aout(rows * L * 8), B.aout(rows * 64)
        q, s = C.quant(codes)
        B.expect(Q, q & 0xFF, 8)
        B.expect(QSUM, [int(v) for v in s], 64)
        return q, np.asarray(s, np.uint32), Q, QSUM, vu_record(OP_QUANT, rows, L, Q, QSUM, desc(src, SH_E, F_BF16),
                                                                 rs=desc(src_sum, SH_R, F_SUM, FIELD_AMAX))

    def dq_rec(acc, ACC, s_row, QSUM, W, rows, L, bias_f, out="bf16"):
        s_base, b_base = B.put_in(f32c(W.scale), 32), B.put_in(f32c(bias_f), 32)
        OUT, OSUM = B.aout(rows * L * (32 if out == "fp32" else 16)), B.aout(rows * 64)
        y = dequant(acc, s_row, f32c(W.scale), np.repeat(f32c(bias_f)[None, :], rows, 0), out)
        B.expect(OUT, y, 32 if out == "fp32" else 16)
        B.expect(OSUM, [0] * rows if out == "fp32" else summaries(y), 64)
        return y, OUT, OSUM, vu_record(OP_DEQUANT, rows, L, OUT, OSUM, desc(ACC, SH_E, F_INT32),
                                       c=desc(QSUM, SH_R, F_SUM, FIELD_RS0), d=desc(s_base, SH_C, F_FP32),
                                       e=desc(b_base, SH_C, F_FP32), bias_en=1, out_fp32=int(out == "fp32"))

    def gemm_stage(q, Q, W, rows):
        acc = C.gemm(q, W.codes)
        ACC = B.aout(rows * W.codes.shape[0] * 32)
        B.expect(ACC, acc & 0xFFFFFFFF, 32)
        B.stage(CH8, TL.gemm_cmds(Q, rows, W.codes.shape[1] // 16, W.codes, ACC) + [0])
        return acc, ACC

    # ================= stage 0: x_t (fp32) -> bf16 codes, QUANT =================
    XT = B.put_in(x_t.view(np.uint32), 32)
    XB, XBSUM = B.aout(T * AD * 16), B.aout(T * 64)
    xb, _ = R.op_add(np.zeros((T, AD), np.uint16), x_t.view(np.uint32), "fp32")
    B.expect(XB, xb, 16)
    B.expect(XBSUM, summaries(xb), 64)
    xq, s_x, XQ, XQSUM, rq = quant_rec(XB, XBSUM, T, AD, xb)
    B.stage(VN, vu_record(OP_ADD, T, AD, XB, XBSUM, K(0), b=desc(XT, SH_E, F_FP32), b_fp32=1) + rq + [0])

    # ================= stage 1-2: action_in_proj =================
    acc, ACC = gemm_stage(xq, XQ, W_in, T)
    ae, AE, AESUM, r1 = dq_rec(acc, ACC, s_x, XQSUM, W_in, T, D, t["action_in_proj.b"])
    aq, s_a, AQ, AQSUM, r2 = quant_rec(AE, AESUM, T, D, ae)
    B.stage(VN, r1 + r2 + [0])

    # ================= stage 3-4: action_time_mlp_in (the time half folded into the bias), SiLU =================
    acc, ACC = gemm_stage(aq, AQ, W_mi, T)
    mi, MI, MISUM, r1 = dq_rec(acc, ACC, s_a, AQSUM, W_mi, T, D, bias_mi)
    SI, SISUM = B.aout(T * D * 16), B.aout(T * 64)
    si, _ = R.op_silu(mi)
    B.expect(SI, si, 16)
    B.expect(SISUM, summaries(si), 64)
    sq, s_s, SQ, SQSUM, r3 = quant_rec(SI, SISUM, T, D, si)
    B.stage(VN, r1 + vu_record(OP_SILU, T, D, SI, SISUM, desc(MI, SH_E, F_BF16)) + r3 + [0])

    # ================= stage 5-6: action_time_mlp_out; the output head's norm =================
    acc, ACC = gemm_stage(sq, SQ, W_mo, T)
    tok, TOK, TOKSUM, r1 = dq_rec(acc, ACC, s_s, SQSUM, W_mo, T, D, t["action_time_mlp_out.b"])
    h = z["ex.final_norm.in"][cap_idx].astype(np.float64)                     # (51, 1024), captured
    hc = R.bf16_codes(h)
    H_BASE = B.put_in(hc, 16)
    amax = R.row_amax_code(hc)
    AMAX = B.put_in(amax, 16)
    RS = B.aout(51 * 64)
    r_rms = R.rms_stat(hc, amax, 1.0 / D)
    B.expect(RS, [int(v) for v in r_rms], 64)
    gain = B.put_in(f32c(1.0 + t["norm.w"].astype(np.float64)), 32)
    NB, NBSUM = B.aout(51 * D * 16), B.aout(51 * 64)
    nb, _ = R.rms_apply(hc, r_rms, f32c(1.0 + t["norm.w"].astype(np.float64)))
    B.expect(NB, nb, 16)
    B.expect(NBSUM, summaries(nb), 64)
    # all 51 rows (operand bases are beat aligned, so a region cannot start at row 1); the state row rides along
    nq, s_n, NQ, NQSUM, r4 = quant_rec(NB, NBSUM, 51, D, nb)
    B.stage(VN, r1 + vu_record(OP_RMS_STAT, 51, D, 0, RS, desc(H_BASE, SH_E, F_BF16), rs=desc(AMAX, SH_R, F_BF16),
                               k=int(np.float32(1.0 / D).view(np.uint32)))
            + vu_record(OP_RMS_APPLY, 51, D, NB, NBSUM, desc(H_BASE, SH_E, F_BF16), c=desc(RS, SH_R, F_SUM, FIELD_RS0),
                        d=desc(gain, SH_C, F_FP32)) + r4 + [0])

    # ================= stage 7-8: action_out_proj -> v_t (fp32), EULER on the 50 action rows =================
    acc, ACC = gemm_stage(nq, NQ, W_out, 51)
    vt51, VT, VTSUM, r1 = dq_rec(acc, ACC, s_n, NQSUM, W_out, 51, AD, t["action_out_proj.b"], out="fp32")
    vt = vt51[1:]
    XN, XNSUM = B.aout(T * AD * 32), B.aout(T * 64)
    xn, _ = R.op_euler(x_t.view(np.uint32), vt, dt)
    B.expect(XN, xn, 32)
    B.expect(XNSUM, [0] * T, 64)
    B.stage(VN, r1 + vu_record(OP_EULER, T, AD, XN, XNSUM, desc(VT + AD * 4, SH_E, F_FP32), e=desc(XT, SH_E, F_FP32),
                               k=int(np.float32(dt).view(np.uint32))) + [0])

    # ================= fp32 reference and the capture =================
    f64 = lambda v: v.astype(np.float64)                                  # noqa: E731
    ae_f = f64(x_t) @ f64(t["action_in_proj.w"]).T + f64(t["action_in_proj.b"])
    at_f = np.concatenate([ae_f, np.repeat(f64(time_embedding(time))[None, :], T, 0)], 1)
    mi_f = at_f @ f64(t["action_time_mlp_in.w"]).T + f64(t["action_time_mlp_in.b"])
    silu_f = mi_f / (1.0 + np.exp(-mi_f))
    tok_f = silu_f @ f64(t["action_time_mlp_out.w"]).T + f64(t["action_time_mlp_out.b"])
    n_f = LL.rms_fp(h, t["norm.w"])
    vt_f = n_f[1:] @ f64(t["action_out_proj.w"]).T + f64(t["action_out_proj.b"])
    rel = LL.rel
    vt_chip = np.asarray(vt, np.uint32).view(np.float32).astype(np.float64)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "barrier.txt").exists():
        (out / "barrier.txt").unlink()
    B.mem.write(out / "mem_in.hex")
    B.exp.write(out / "mem_exp.hex")
    (out / "stages.txt").write_text("".join(f"{n} {b:x}\n" for n, b, _ in B.stages))
    (out / "regions.txt").write_text("".join(f"{n} {b:x} {sz} {st}\n" for n, b, sz, st in B.regions))
    (out / "info.txt").write_text(
        f"part=action_head step={a.step} time={time:.2f} T={T} stages={len(B.stages)}\n"
        f"image {len(B.mem.beats)} beats, expected {len(B.exp.beats)} beats\n"
        f"fp32 reference vs capture: silu_in {rel(mi_f, z['ex.silu_in'][a.step].astype(np.float64)):.1e}, "
        f"v_t {rel(vt_f, z['ex.v_t'][a.step].astype(np.float64)):.1e}\n"
        f"chip vs fp32: silu_in {rel(bf16v(mi), mi_f):.4f} tokens {rel(bf16v(tok), tok_f):.4f} "
        f"v_t {rel(vt_chip, vt_f):.4f}\n")
    print((out / "info.txt").read_text().strip())


if __name__ == "__main__":
    main()
