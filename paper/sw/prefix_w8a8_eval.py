#!/usr/bin/env python3
"""Phase A numerics: does the pi0 PREFIX survive W8A8 INT8?  Measured on real frames, not assumed.

Runs the real LeRobot prefix -- SigLIP So400m/14 on both cameras + PaliGemma-2B (Gemma) over the
compact prefix, op for op what PI0FpgaPolicy._run_prefix_compact does -- in fp32 on the CPU, and
fake-quantises every GEMM in place (docs/PI0_FULL_CHIP_ARCHITECTURE_20260910.md section 4.1):

  weights      INT8 symmetric per output channel, MSE-optimal clip (the grid of
               pi0_deploy_model.mse_clip_per_column: 21 fractions of amax in [0.5, 1])
  activations  INT8 symmetric per token, dynamic (row amax / 127)        variants: per-tensor static
  fp32         LayerNorm / RMSNorm, softmax, GELU / GeGLU, RoPE, residuals, biases, token lookup
  GEMMs        SigLIP patch embedding (a GEMM on 14x14x3 patches), q/k/v/out/fc1/fc2 x 27 layers,
               the multi-modal projector, Gemma q/k/v/o/gate/up/down x 18 layers
  attention    fp32 by default; variant 'attn*' puts QK^T and softmax(QK^T)V in INT8 per row

Gates (architecture doc section 4.2):
  G0  SigLIP tokens (projector output = LM input embeddings), rel RMS per image       <= 1 %
      (the SigLIP tower output after post-LayerNorm is reported too)
  G1  prefix KV cache, rel RMS per layer, K and V separately, vs the fp32 capture      <= 1 %
      (+ the final prefix hidden state vs this script's own fp32 run)
  G3  the stock fp32 torch expert on the INT8 prefix KV, same noise as the capture:
      50x7 chunk rel RMS <= 3 %, cosine >= 0.998, every frame <= 5 %

Stages (--stages, comma separated, run in order in one process; each writes <out>/<stage>.json):
  selftest      harness checks: compact prefix == PI0FpgaPolicy._run_prefix_compact, fp32 KV vs the
                capture, capture KV through the expert == actions_fp32, the patch-GEMM path == conv,
                torch MSE clip == pi0_deploy_model.mse_clip_per_column, attention patch hit counts
  fp32          the fp32 floor of every metric (cached references, one prefix + expert per frame)
  base          (a) W8 per-channel MSE + A8 per-token dynamic on SigLIP + Gemma
  base_lm       (a) Gemma only, fed the fp32 SigLIP embeddings (isolates the LM)
  base_vis      (a) SigLIP + projector only (its effect on the KV)
  w8 / a8       weights only / activations only
  calib         calibration statistics on --calib frames (per-channel amax, per-tensor MSE grid)
  smooth        (b) = base + SmoothQuant alpha on the Gemma linears (qkv/gateup -> RMSNorm gains,
                o -> v columns tied per head_dim, down -> up columns; calibrated on --calib)
  smooth_lm     (b) Gemma only
  attn          (c) = base + INT8 QK^T and softmax(QK^T)V (Q, K per row; P per row; V per key)
  attn_only     (c) INT8 attention alone, fp32 GEMMs
  attn_qk / attn_pv   base + only one of the two attention GEMMs in INT8
  attn_vcol     base + INT8 attention with V per channel instead of per key
  static_amax / static_mse   (d) per-tensor static activation scales (calibration amax / MSE clip)
  local         per-GEMM local error on fp32 inputs (W only, A only, both) + activation outlier
                statistics + local INT8-attention error, on --local-frames
  iso_lm_layer / iso_lm_proj    only one Gemma layer / one projection class quantised (LM-only KV)
  iso_vis_layer / iso_vis_proj  only one SigLIP layer / projection class quantised (G0)
  fallback      greedy: the worst Gemma layers (iso_lm_layer G3 ranking) kept fp32 on top of base / smooth
                until G3 max < --fallback-target (1 %)
  e2e_int8_expert  the W8A8 prefix KV (base, smooth_lm) into the deployment INT8 expert (pi0_deploy_model
                rebuilt from the session's numerics .npz), G3 + joint-space error vs actions_fp32
  e2e_dyn_expert   the action expert fake-quantised with the prefix recipe (W8 MSE per channel, A8 per token
                dynamic, no output requant; + SmoothQuant / INT8 attention variants) on prefix KV fp32 / base /
                smooth_lm (--dyn-combos), G3 + G4 vs actions_fp32
  iso_exp_proj / iso_exp_layer  one expert projection class / layer quantised alone (fp32 prefix KV), G3
  custom        --custom-name N --custom-fp REGEX [--custom-base base|smooth|attn]: base with the
                GEMMs matching REGEX kept in fp32
  runs          --runs-file F.json [--runs-out NAME] [--run-filter REGEX]: a set of end-to-end runs, each
                prefix variant x prefix INT8 attention x expert (dyn recipes or the deployed static INT8
                expert) x expert INT8 attention x hardware formats, G3 (mean / median / p95 / max) + G4 and the
                paired distance to a base run; files in paper/sw/prefix_w8a8_runs/ (e2e_46, attn_loc, fmt_sens)
  attn_local    local INT8-attention error per module (SigLIP / Gemma / expert), QK vs PV, dead softmax mass,
                --local-prefix smooth --cmp-cfg '{"v_gran": "fold"}'
  fmt_selftest  the hardware-format emulation (bf16, row-scale encodings, the vector unit's 2048-entry
                tables for GELU / SiLU gates, softmax exp, rsqrt) against its references + plumbing hit counts

INT8 attention granularity (ATTN_INT8 v_gran): 'row' (V per key, the original variant) is not one integer
accumulation in PV -- the key index is the summation index, so its scale cannot be factored out as
acc x s_row x s_col; 'fold' (key scale folded into P before P is quantised) and 'col' (V per channel) are.

Frames: --eval / --calib / --local-frames take episode:frames specs (e.g. demo1_ep20:2,8 or
demo1_ep20:all); episodes from ~/pi0_glue/episodes/<ep>.npz, the fp32 reference from
~/pi0_glue/captures/<ep>/frame_NN.npz (glue/level3/capture_pi0_prefix_kv.py).

Memory: the checkpoint is memory-mapped (copy-on-write) into a meta-initialised policy, so the
fp32 weights live in the page cache; only the INT8 codes (~2.4 GB per weight set) are anonymous.
Run beside an ACE build only inside a capped, niced scope, e.g.

  systemd-run --user --scope -q -p MemoryHigh=14G nice -n 10 \\
      ~/lerobot/.venv/bin/python paper/sw/prefix_w8a8_eval.py --threads 8 \\
      --eval demo1_ep20:2,8 demo1_ep40:2,8 recov_pi0_ep00:2,8 recov_pi0_ep10:2,8 \\
      --calib demo1_ep01:5 demo1_ep02:5 demo1_ep03:5 demo1_ep04:5 demo1_ep05:5 demo1_ep06:5 demo1_ep07:5 demo1_ep08:5 \\
      --stages selftest,fp32,base,base_lm,calib,smooth,attn,static_amax
"""
from __future__ import annotations

import argparse
import glob
import json
import mmap
import os
import re
import resource
import struct
import subprocess
import sys
import time
from pathlib import Path


def _early_threads() -> int:
    t = sys.argv[sys.argv.index("--threads") + 1] if "--threads" in sys.argv else "8"
    for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[k] = t
    return int(t)


THREADS = _early_threads()
os.environ.setdefault("PI0_ACTION_EXPERT", "torch")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from torch import nn  # noqa: E402

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / "glue" / "level3"))
import pi0_deploy_model as D  # noqa: E402

CKPT = os.path.expanduser("~/pi0_ckpt/ur7e-demo-2-pi0-fsdp/010000/pretrained_model")
EPISODES = Path(os.path.expanduser("~/pi0_glue/episodes"))
CAPTURES = Path(os.path.expanduser("~/pi0_glue/captures"))
GATE_G0, GATE_G1, GATE_G3, GATE_G3_FRAME, GATE_G3_COS = 0.01, 0.01, 0.03, 0.05, 0.998
W_GRID = np.linspace(0.5, 1.0, 21)      # pi0_deploy_model.mse_clip_per_column
A_GRID = np.linspace(0.2, 1.0, 33)      # pi0_deploy_model.mse_scale_per_tensor
LM_SLOTS = ("q", "k", "v", "o", "gate", "up", "down")
VIS_SLOTS = ("q", "k", "v", "out", "fc1", "fc2")

torch.set_grad_enabled(False)
torch.set_num_threads(THREADS)


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def rss_gb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6


# ======================================================================================
# model loading: meta init + copy-on-write mmap of the safetensors file
# ======================================================================================
_MMAPS = []


def mmap_safetensors(path: str) -> dict:
    with open(path, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        header = json.loads(fh.read(n))
        mm = mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_COPY)
    _MMAPS.append(mm)
    header.pop("__metadata__", None)
    base = 8 + n
    out = {}
    for key, v in header.items():
        if v["dtype"] != "F32":
            raise ValueError(f"{key}: dtype {v['dtype']}, expected F32")
        s, e = v["data_offsets"]
        out[key] = torch.frombuffer(mm, dtype=torch.float32, count=(e - s) // 4, offset=base + s).view(v["shape"])
    return out


def load_policy(ckpt: str):
    from accelerate import init_empty_weights
    from lerobot.configs.policies import PreTrainedConfig
    from pi0_fpga_policy import PI0FpgaPolicy

    cfg = PreTrainedConfig.from_pretrained(pretrained_name_or_path=ckpt)
    cfg.device = "meta"
    with init_empty_weights(include_buffers=False):
        policy = PI0FpgaPolicy(cfg)
    cfg.device = "cpu"
    sd = mmap_safetensors(os.path.join(ckpt, "model.safetensors"))
    missing, unexpected = policy.load_state_dict(sd, strict=False, assign=True)
    still_meta = [n for n, p in list(policy.named_parameters()) + list(policy.named_buffers()) if p.device.type == "meta"]
    if still_meta:
        raise RuntimeError(f"parameters left on meta: {still_meta[:5]} ({len(still_meta)})")
    policy.eval()
    return policy, list(missing), list(unexpected)


# ======================================================================================
# quantisation primitives
# ======================================================================================
# ---- hardware formats of the chip (stages runs / fmt_selftest); all off = the plain fake-quant ----
#   row_enc     encoding of every dynamic row scale (GEMM activations, attention Q/K/P/V):
#               fp (float) | m8 / m6 / m4 (power of two x n-bit mantissa, rounded up) | pow2 (rounded up)
#   out_bf16    every quantised GEMM result (dequant acc x s_row x s_col, QK^T, PV) rounded to bf16
#   iface_bf16  bf16 at every vector-op interface: GEMM in/out, norms, GELU/SiLU, attention q/k/v/P/out,
#               decoder-layer outputs (the residual stream); implies out_bf16
#   tables      GELU/SiLU gates, softmax exp and RMSNorm/LayerNorm rsqrt through the vector unit's
#               2048-entry linearly interpolated tables (paper/sw/vector_unit_ref.py TABLES)
#   acc         an AccRec: records the exact integer accumulation |acc| of every INT8 GEMM
FMT = dict(row_enc="fp", out_bf16=False, iface_bf16=False, tables=False, acc=None)
TBL_HITS: dict = {}


def bf16_t(x: torch.Tensor) -> torch.Tensor:
    """Round to bf16 (sign, 8-bit exponent, 7-bit mantissa, RNE) and back (== pi0_ae_model.to_bf16)."""
    return x.to(torch.bfloat16).to(x.dtype)


def enc_scale(s: torch.Tensor) -> torch.Tensor:
    mode = FMT["row_enc"]
    if mode == "fp":
        return s
    if mode == "pow2":
        return torch.pow(2.0, torch.ceil(torch.log2(s))).to(s.dtype)
    bits = {"m8": 8, "m6": 6, "m4": 4}[mode]
    m, e = torch.frexp(s)                                     # s = m 2^e, m in [0.5, 1)
    return torch.ldexp(torch.ceil(m * (1 << bits)) / (1 << bits), e)


def q_rows(x: torch.Tensor, levels: int = 127, dim: int = -1):
    """Symmetric INT8 codes and scale, one scale per slice along `dim` (per token for dim=-1)."""
    amax = x.abs().amax(dim=dim, keepdim=True)
    s = enc_scale(torch.where(amax > 0, amax, torch.ones_like(amax)) / levels)
    return torch.round(x / s).clamp_(-levels, levels), s


def fq_rows(x: torch.Tensor, levels: int = 127, dim: int = -1) -> torch.Tensor:
    """Symmetric fake-quant with one scale per slice along `dim` (per token for dim=-1)."""
    c, s = q_rows(x, levels, dim)
    return c.mul_(s)


def q_groups(x: torch.Tensor, levels: int, g: int):
    """Codes and scales with one scale per group of g entries along the last dim."""
    T = x.shape[-1]
    xg = F.pad(x, (0, (-T) % g)).reshape(*x.shape[:-1], -1, g)
    c, s = q_rows(xg, levels)
    return c.reshape(*x.shape[:-1], -1)[..., :T], s.expand_as(xg).reshape(*x.shape[:-1], -1)[..., :T]


def gemm_class(name: str) -> str:
    p = name.split(".")
    return f"{p[0]}.{p[2]}" if len(p) == 3 and p[1].startswith("L") else name


class AccRec:
    """max |acc| of the exact integer accumulation per GEMM class, against the analytic bound 127^2 K."""

    def __init__(self):
        self.d = {}

    def add(self, name: str, acc: torch.Tensor, K: int, levels_a: int = 127) -> None:
        e = self.d.setdefault(gemm_class(name), {"max_abs_acc": 0.0, "K_max": 0, "bound_127sq_K": 0, "calls": 0,
                                                 "worst_gemm": None, "mean_abs_acc_sum": 0.0})
        mx = float(acc.abs().max())
        if mx > e["max_abs_acc"]:
            e["max_abs_acc"], e["worst_gemm"] = mx, name
        e["K_max"] = max(e["K_max"], int(K))
        e["bound_127sq_K"] = max(e["bound_127sq_K"], 127 * levels_a * int(K))
        e["calls"] += 1
        e["mean_abs_acc_sum"] += float(acc.abs().mean())

    def summary(self) -> dict:
        out = {}
        for c, e in sorted(self.d.items()):
            out[c] = {**{k: v for k, v in e.items() if k != "mean_abs_acc_sum"},
                      "mean_abs_acc": e["mean_abs_acc_sum"] / max(e["calls"], 1),
                      "log2_max_abs_acc": float(np.log2(max(e["max_abs_acc"], 1.0))),
                      "headroom_bits_vs_2^47": float(47 - np.log2(max(e["max_abs_acc"], 1.0)))}
        return out


# ---- the vector unit's tables, emulated in float64 with its index / fraction / value quantisation ----
LN2 = float(np.log(2.0))
_TBL: dict = {}


def _tbl(name: str):
    if name not in _TBL:
        import vector_unit_ref as VU
        v, d = VU.TABLES[name]
        _TBL[name] = (torch.from_numpy(np.asarray(v, np.int64)), torch.from_numpy(np.asarray(d, np.int64)))
    return _TBL[name]


def tbl_lookup(name: str, u: torch.Tensor) -> torch.Tensor:
    """u (int64, < 2^23) = 11-bit index into the 2048 entries . 12-bit fraction -> Q1.32 value as float64."""
    v, d = _tbl(name)
    idx = u >> 12
    return (v[idx] * 4096 + d[idx] * (u & 0xFFF)).double() / 4294967296.0


def tbl_gate(x: torch.Tensor, name: str) -> torch.Tensor:
    """phi_gelu(x) on [-8, 8) (h = 2^-7) or sigmoid(x) on [-16, 16) (h = 2^-6); saturated outside (vector_unit_ref.tbl_gate)."""
    shift, lim = (19, 8.0) if name == "gelu" else (18, 16.0)
    xd = x.double()
    z = torch.floor(xd.abs() * float(1 << shift)).clamp_(max=float(1 << 23)).to(torch.int64)
    u = torch.where(xd < 0, (1 << 22) - z, (1 << 22) + z).clamp_(0, (1 << 23) - 1)
    y = torch.where(xd.abs() >= lim, (xd > 0).double(), tbl_lookup(name, u))
    TBL_HITS[name] = TBL_HITS.get(name, 0) + x.numel()
    return y


def tbl_rsqrt(v: torch.Tensor, tag: str = "rsqrt") -> torch.Tensor:
    """rsqrt of a positive value: 24-bit mantissa, exponent parity selects [1,2) h=2^-10 or [2,4) h=2^-9."""
    m, ex = torch.frexp(v.double().clamp_min(1e-300))
    e = ex.to(torch.int64) - 1
    M = torch.floor(m * float(1 << 24)).to(torch.int64)       # [2^23, 2^24)
    odd = torch.remainder(e, 2)
    u = (odd * 1024 + ((M - (1 << 23)) >> 13)) * 4096 + ((M >> 1) & 0xFFF)
    TBL_HITS[tag] = TBL_HITS.get(tag, 0) + v.numel()
    return tbl_lookup("rsqrt", u) * torch.pow(2.0, -((e - odd) // 2).double())


def tbl_softmax(w: torch.Tensor) -> torch.Tensor:
    """softmax along -1 as the vector unit computes it: logits in the log2 domain in Q.23 (the 1/ln2 folded
    into the QK scaling), e_i = 2^-k * T_exp(f) (2^-f table, h = 2^-11), e_i = 0 for k >= 64, S accumulated
    in Q.40 without the terms with k >= 41, 1/S = rsqrt(S)^2 from the rsqrt table."""
    a = torch.trunc(w.double() * (float(1 << 23) / LN2))
    t = (a.amax(-1, keepdim=True) - a).clamp_(max=float(1 << 40)).to(torch.int64)
    k = t >> 23
    y = tbl_lookup("exp", t & ((1 << 23) - 1))
    e = torch.where(k < 64, y * torch.pow(2.0, -k.clamp(max=63).double()), torch.zeros_like(y))
    S = torch.where(k < 41, torch.floor(e * float(1 << 40)), torch.zeros_like(e)).sum(-1, keepdim=True) / float(1 << 40)
    TBL_HITS["exp"] = TBL_HITS.get("exp", 0) + w.numel()
    return (e * tbl_rsqrt(S, "rsqrt_smax").pow(2)).to(w.dtype)


def mse_clip_rows(w: torch.Tensor) -> torch.Tensor:
    """(out, in) weight -> per output channel clip minimising the channel's INT8 MSE.
    Same grid and tie rule as pi0_deploy_model.mse_clip_per_column (which is (K, N) column-major)."""
    w = w.float()
    n_out, n_in = w.shape
    amax = w.abs().amax(1)
    best = torch.full((n_out,), float("inf"))
    best_c = amax.clone()
    step = max(1, (1 << 23) // n_in)
    for f in W_GRID:
        c = torch.where(amax > 0, amax * float(f), torch.ones_like(amax))
        err = torch.empty(n_out)
        for r0 in range(0, n_out, step):
            wr = w[r0:r0 + step]
            s = (c[r0:r0 + step] / 127.0)[:, None]
            err[r0:r0 + step] = (torch.round(wr / s).clamp_(-127, 127).mul_(s).sub_(wr)).pow_(2).mean(1)
        better = err < best
        best = torch.where(better, err, best)
        best_c = torch.where(better, c, best_c)
    return torch.where(best_c > 0, best_c, torch.ones_like(best_c))


def quant_codes(w: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    n_out, n_in = w.shape
    codes = torch.empty((n_out, n_in), dtype=torch.int8)
    step = max(1, (1 << 23) // n_in)
    for r0 in range(0, n_out, step):
        codes[r0:r0 + step] = torch.round(w[r0:r0 + step] / scale[r0:r0 + step, None]).clamp_(-127, 127).to(torch.int8)
    return codes


# ======================================================================================
# GEMM wrappers
# ======================================================================================
class QLinear(nn.Module):
    """nn.Linear replacement: fp32, or INT8 weights (per output channel) x INT8 activations
    (per token dynamic | per tensor static), optional SmoothQuant factor s (x/s, W*s)."""

    def __init__(self, name: str, weight, bias):
        super().__init__()
        self.name = name
        self.weight = weight
        self.bias = bias
        self.out_features, self.in_features = int(weight.shape[0]), int(weight.shape[1])
        self.codes = None
        self.wscale = None
        self.wq = False
        self.aq = "none"
        self.smooth = None
        self.static_scale = None
        self.rec = None
        self.nclip = 0
        self.nel = 0
        self.a_groups = None   # per-token activation scale split into this many contiguous column groups

    def w2d(self) -> torch.Tensor:
        return self.weight

    def deq(self) -> torch.Tensor:
        return self.codes.to(torch.float32).mul_(self.wscale[:, None])

    def core(self, x: torch.Tensor) -> torch.Tensor:
        if self.rec is not None:
            self.rec(self, x)
        iface = FMT["iface_bf16"]
        if not self.wq and self.aq == "none" and self.smooth is None:
            if iface:
                return bf16_t(F.linear(bf16_t(x), self.w2d(), self.bias))
            return F.linear(x, self.w2d(), self.bias)
        if iface:
            x = bf16_t(x)
        xs = x if self.smooth is None else x / self.smooth
        codes_x = None
        if self.aq == "token":
            if self.a_groups:           # G sub-row passes on the chip, one scale (and DEQUANT) per group
                cg, sg = q_groups(xs, 127, -(-xs.shape[-1] // int(self.a_groups)))
                xs = cg * sg
            elif FMT["acc"] is not None and self.wq:
                codes_x, sx = q_rows(xs)
                xs = codes_x * sx
            else:
                xs = fq_rows(xs)
        elif self.aq == "static":
            q = torch.round(xs / self.static_scale)
            self.nclip += int((q.abs() > 127).sum())
            self.nel += q.numel()
            xs = q.clamp_(-127, 127).mul_(self.static_scale)
        if self.wq:
            W = self.deq()
        else:
            W = self.w2d() if self.smooth is None else self.w2d() * self.smooth
        if codes_x is not None:
            k_in = int(self.codes.shape[1])                      # QPatch: 588 taps, not the conv's 3 channels
            FMT["acc"].add(self.name, codes_x.reshape(-1, k_in).double() @ self.codes.double().T, k_in)
        if FMT["out_bf16"] or iface:
            y = bf16_t(F.linear(xs, W))
            if self.bias is not None:
                y = y + self.bias
            return bf16_t(y) if iface else y
        return F.linear(xs, W, self.bias)

    def forward(self, x):
        return self.core(x)


class QPatch(QLinear):
    """SigLIP patch embedding (Conv2d k=14 s=14) as the GEMM it is on the accelerator:
    unfold to 588-tap patch rows, per-token INT8, per-output-channel INT8 weights."""

    def __init__(self, name: str, conv: nn.Conv2d):
        super().__init__(name, conv.weight, conv.bias)
        self.k = conv.kernel_size[0]
        self.stride = conv.stride[0]

    def w2d(self):
        return self.weight.reshape(self.weight.shape[0], -1)

    def forward(self, pix):
        if self.rec is None and not self.wq and self.aq == "none":
            return F.conv2d(pix, self.weight, self.bias, stride=self.stride)
        B, _, H, _ = pix.shape
        patches = F.unfold(pix, self.k, stride=self.stride).transpose(1, 2)
        y = self.core(patches)
        g = H // self.stride
        return y.transpose(1, 2).reshape(B, -1, g, g)


# ======================================================================================
# attention patch (prefix modules only; the expert always runs the stock function)
# ======================================================================================
from transformers.models.gemma import modeling_gemma as MG  # noqa: E402
from transformers.models.siglip import modeling_siglip as MS  # noqa: E402

_ORIG_GEMMA = MG.eager_attention_forward
_ORIG_SIGLIP = MS.eager_attention_forward
ATTN = {"mode": "fp", "cfg": None, "names": {}, "calls": {}, "acc": {}, "siglip_cfgs": {}, "siglip_impl": {},
        "exp_names": {}, "exp_mode": "fp", "exp_cfg": None,
        "sel": None,          # regex on the module name (vis.L3 / lm.L3 / exp.L3): INT8 only where it matches
        "layer_cfg": {},      # module name -> cfg override (None = fp attention in that module)
        "cmp_cfg": None}      # base cfg of the compare mode (default ATTN_INT8)
# v_gran: row  = V per key (and head).  NOT one integer accumulation: in PV the key index is the summation
#                index, so a per-key scale cannot be factored out as acc x s_row x s_col (kept for continuity
#                with the 8-frame results)
#         fold = V per key with the key scale folded into P before P is quantised: P'_qt = P_qt s_V(t),
#                acc = sum_t q(P')_qt q(V)_td, dequant = s_P'(q)   -> one accumulation per output
#         col  = V per channel (head_dim column over the keys), dequant = s_P(q) s_V(d)
# p_levels 127 (int8) or 255 (uint8, P >= 0); p_group g: one P scale per group of g keys (g accumulations
# per output); p_top_fp k: the k largest P of each row taken out of the INT8 GEMM and added in float
ATTN_INT8 = dict(qk=True, pv=True, v_gran="row", p_levels=127)


def pv_int8(p, v, cfg, name=None, rec=None):
    levels = cfg["p_levels"]
    extra = None
    if cfg.get("p_top_fp"):
        vals, idx = p.topk(int(cfg["p_top_fp"]), dim=-1)
        extra = torch.matmul(torch.zeros_like(p).scatter_(-1, idx, vals), v)
        p = p.scatter(-1, idx, 0.0)
    gran = cfg["v_gran"]
    if gran == "col":
        cv, sv = q_rows(v, dim=-2)
    else:
        cv, sv = q_rows(v)
        if gran == "fold":
            p = p * sv.transpose(2, 3)
    cp, sp = q_groups(p, levels, int(cfg["p_group"])) if cfg.get("p_group") else q_rows(p, levels)
    out = torch.matmul(cp * sp, cv if gran == "fold" else cv * sv)
    if rec is not None:
        rec.add(name + ".attn_pv", torch.matmul(cp.double(), cv.double()), p.shape[-1], levels)
    if FMT["out_bf16"] or FMT["iface_bf16"]:
        out = bf16_t(out)
    return out if extra is None else out + extra


def attn_math(q, k, v, mask, scaling, cfg, name=None):
    iface = FMT["iface_bf16"]
    rec = FMT["acc"] if (cfg is not None and name is not None) else None
    if iface:
        q, k, v = bf16_t(q), bf16_t(k), bf16_t(v)
    if cfg is not None and cfg["qk"]:
        cq, sq = q_rows(q)
        ck, sk = q_rows(k)
        w = torch.matmul(cq * sq, (ck * sk).transpose(2, 3))
        if rec is not None:
            rec.add(name + ".attn_qk", torch.matmul(cq.double(), ck.double().transpose(2, 3)), q.shape[-1])
        if FMT["out_bf16"] or iface:
            w = bf16_t(w)
        w = w * scaling
    else:
        w = torch.matmul(q, k.transpose(2, 3)) * scaling
    if mask is not None:
        w = w + mask
    p = tbl_softmax(w) if FMT["tables"] else torch.softmax(w, dim=-1, dtype=torch.float32).to(q.dtype)
    if iface:
        p = bf16_t(p)
    out = pv_int8(p, v, cfg, name, rec) if (cfg is not None and cfg["pv"]) else torch.matmul(p, v)
    out = out.transpose(1, 2).contiguous()
    return bf16_t(out) if iface else out


def p_dead_mass(p, v, cfg) -> tuple:
    """Share of the softmax mass whose INT8 P code is 0 (per row, summed) under cfg."""
    pp = p
    if cfg["v_gran"] == "fold":
        pp = p * q_rows(v)[1].transpose(2, 3)
    cp, _ = q_groups(pp, cfg["p_levels"], int(cfg["p_group"])) if cfg.get("p_group") else q_rows(pp, cfg["p_levels"])
    dead = (p * (cp == 0)).sum(-1)
    return float(dead.double().sum()), float(dead.double().max()), int(dead.numel())


def patched_attention(module, query, key, value, attention_mask, scaling=None, dropout=0.0, **kwargs):
    name, mode, cfg = ATTN["names"].get(id(module)), ATTN["mode"], ATTN["cfg"]
    if name is None:                       # the action expert has its own switch (e2e_dyn_expert)
        name, mode, cfg = ATTN["exp_names"].get(id(module)), ATTN["exp_mode"], ATTN["exp_cfg"]
    orig = _ORIG_SIGLIP if isinstance(module, MS.SiglipAttention) else _ORIG_GEMMA
    if name is not None and mode == "int8":
        if ATTN["sel"] is not None and not re.search(ATTN["sel"], name):
            mode, cfg = "fp", None
        elif name in ATTN["layer_cfg"]:
            cfg = ATTN["layer_cfg"][name]
            mode = "fp" if cfg is None else mode
    if name is None or (mode == "fp" and not (FMT["tables"] or FMT["iface_bf16"])):
        return orig(module, query, key, value, attention_mask, scaling=scaling, dropout=dropout, **kwargs)
    ATTN["calls"][name] = ATTN["calls"].get(name, 0) + 1
    groups = getattr(module, "num_key_value_groups", 1)
    if groups > 1:
        key, value = MG.repeat_kv(key, groups), MG.repeat_kv(value, groups)
    if mode in ("int8", "fp"):
        return attn_math(query, key, value, attention_mask, scaling, cfg if mode == "int8" else None, name), None
    # compare: return fp32, accumulate the local INT8 error of this module
    base = ATTN["cmp_cfg"] or ATTN_INT8
    ref = attn_math(query, key, value, attention_mask, scaling, None)
    acc = ATTN["acc"].setdefault(name, {"y2": 0.0, "both": 0.0, "qk": 0.0, "pv": 0.0, "dead_sum": 0.0, "dead_max": 0.0,
                                        "rows": 0})
    acc["y2"] += float(ref.double().pow(2).sum())
    for tag, cfg in (("both", base), ("qk", {**base, "pv": False}), ("pv", {**base, "qk": False})):
        acc[tag] += float((attn_math(query, key, value, attention_mask, scaling, cfg) - ref).double().pow(2).sum())
    w = torch.matmul(query, key.transpose(2, 3)) * scaling
    if attention_mask is not None:
        w = w + attention_mask
    ds, dm, n = p_dead_mass(torch.softmax(w, dim=-1, dtype=torch.float32), value, base)
    acc["dead_sum"] += ds
    acc["dead_max"] = max(acc["dead_max"], dm)
    acc["rows"] += n
    return ref, None


MG.eager_attention_forward = patched_attention
MS.eager_attention_forward = patched_attention


def refresh_siglip_impl() -> None:
    eager = ATTN["mode"] != "fp" or FMT["tables"] or FMT["iface_bf16"]
    for key, c in ATTN["siglip_cfgs"].items():
        c._attn_implementation = "eager" if eager else ATTN["siglip_impl"][key]


def set_attn(mode: str, cfg=None) -> None:
    ATTN["mode"], ATTN["cfg"] = mode, cfg
    refresh_siglip_impl()


# ---- norms, GELU / SiLU and the residual stream under the hardware formats ----------------------
from lerobot.policies import pi_gemma as PG  # noqa: E402
from transformers import activations as TA  # noqa: E402

_ORIG_RMS_FWD = PG.PiGemmaRMSNorm.forward
_ORIG_LN_FWD = nn.LayerNorm.forward
_ORIG_GELU_FWD = TA.GELUTanh.forward
_ORIG_SILU = F.silu


def _rms_forward(self, x, cond=None):
    if FMT["tables"] and (cond is None or self.dense is None):
        var = torch.mean(torch.square(x.float()), dim=-1, keepdim=True)
        out = ((x * tbl_rsqrt(var + self.eps, "rsqrt_rms").to(x.dtype)) * (1.0 + self.weight.float())).type_as(x)
        gate = None
    else:
        out, gate = _ORIG_RMS_FWD(self, x, cond)
    return (bf16_t(out) if FMT["iface_bf16"] else out), gate


def _ln_forward(self, x):
    if FMT["tables"]:
        mu = x.mean(-1, keepdim=True)
        xc = x - mu
        y = xc * tbl_rsqrt(xc.pow(2).mean(-1, keepdim=True) + self.eps, "rsqrt_ln").to(x.dtype)
        if self.weight is not None:
            y = y * self.weight
        if self.bias is not None:
            y = y + self.bias
    else:
        y = _ORIG_LN_FWD(self, x)
    return bf16_t(y) if FMT["iface_bf16"] else y


def _gelu_forward(self, x):
    y = x * tbl_gate(x, "gelu").to(x.dtype) if FMT["tables"] else _ORIG_GELU_FWD(self, x)
    return bf16_t(y) if FMT["iface_bf16"] else y


def _silu(input, inplace=False):  # noqa: A002
    y = input * tbl_gate(input, "sigm").to(input.dtype) if FMT["tables"] else _ORIG_SILU(input, inplace=inplace)
    return bf16_t(y) if FMT["iface_bf16"] else y


PG.PiGemmaRMSNorm.forward = _rms_forward
nn.LayerNorm.forward = _ln_forward
TA.GELUTanh.forward = _gelu_forward
F.silu = _silu


def _bf16_out_hook(module, inputs, output):
    if not FMT["iface_bf16"]:
        return None
    if isinstance(output, torch.Tensor):
        return bf16_t(output)
    if isinstance(output, tuple) and output and isinstance(output[0], torch.Tensor):
        return (bf16_t(output[0]),) + tuple(output[1:])
    return None


# ======================================================================================
# global state
# ======================================================================================
class G:
    policy = None
    pwe = None
    QL: dict = {}
    QX: dict = {}          # action expert GEMMs (exp.L{l}.{slot}, exp.{state,act_in,tm_in,tm_out,act_out})
    WSX: dict = {}         # expert weight sets
    SMOOTH_X: dict = {}    # expert SmoothQuant factors
    WS: dict = {}          # "base" | "smooth" -> {name: (codes, scale)}
    SMOOTH: dict = {}      # name -> s (input channels)
    STATIC: dict = {}      # "amax" | "mse" -> {name: scale}
    CALIB = None
    pre = None
    replays: dict = {}
    inputs: dict = {}
    args = None
    out_dir: Path = None
    cache_dir: Path = None
    tower: list = []
    hm = None              # the deployed static INT8 expert (numpy HostModel), loaded on first use
    SMOOTH_X_ALPHA: dict = {}   # alpha -> expert SmoothQuant factors at that alpha (per-layer overrides)
    WSX_ALPHA: dict = {}        # (alpha, name) -> (factor, INT8 codes, scale)


def install_wrappers() -> None:
    pwe = G.pwe
    vm = pwe.paligemma.model.vision_tower.vision_model
    QL = {}
    emb = vm.embeddings
    emb.patch_embedding = QL.setdefault("vis.patch", QPatch("vis.patch", emb.patch_embedding))
    for li, layer in enumerate(vm.encoder.layers):
        for slot, owner, attr in (("q", layer.self_attn, "q_proj"), ("k", layer.self_attn, "k_proj"),
                                  ("v", layer.self_attn, "v_proj"), ("out", layer.self_attn, "out_proj"),
                                  ("fc1", layer.mlp, "fc1"), ("fc2", layer.mlp, "fc2")):
            lin = getattr(owner, attr)
            name = f"vis.L{li}.{slot}"
            QL[name] = QLinear(name, lin.weight, lin.bias)
            setattr(owner, attr, QL[name])
        ATTN["names"][id(layer.self_attn)] = f"vis.L{li}"
        layer.register_forward_hook(_bf16_out_hook)
        c = layer.self_attn.config
        ATTN["siglip_cfgs"][id(c)] = c
        ATTN["siglip_impl"][id(c)] = c._attn_implementation
    mmp = pwe.paligemma.model.multi_modal_projector
    QL["vis.mmproj"] = QLinear("vis.mmproj", mmp.linear.weight, mmp.linear.bias)
    mmp.linear = QL["vis.mmproj"]
    for li, pair in enumerate(pwe.joint_layers):
        layer = pair.paligemma_layer
        for slot, owner, attr in (("q", layer.self_attn, "q_proj"), ("k", layer.self_attn, "k_proj"),
                                  ("v", layer.self_attn, "v_proj"), ("o", layer.self_attn, "o_proj"),
                                  ("gate", layer.mlp, "gate_proj"), ("up", layer.mlp, "up_proj"),
                                  ("down", layer.mlp, "down_proj")):
            lin = getattr(owner, attr)
            name = f"lm.L{li}.{slot}"
            QL[name] = QLinear(name, lin.weight, lin.bias)
            setattr(owner, attr, QL[name])
        ATTN["names"][id(layer.self_attn)] = f"lm.L{li}"
        layer.register_forward_hook(_bf16_out_hook)
    G.QL = QL
    pwe.paligemma.model.vision_tower.register_forward_hook(lambda m, i, o: G.tower.append(o.last_hidden_state))


# ======================================================================================
# frames, references
# ======================================================================================
def parse_frames(specs) -> list:
    out = []
    for spec in specs or []:
        ep, fr = spec.split(":")
        if fr == "all":
            idx = range(len(glob.glob(str(CAPTURES / ep / "frame_*.npz"))))
        else:
            idx = [int(x) for x in fr.split(",")]
        out += [(ep, int(i)) for i in idx]
    return out


def fid(fr) -> str:
    return f"{fr[0]}:{fr[1]}"


def get_inputs(fr) -> dict:
    if fr in G.inputs:
        return G.inputs[fr]
    import capture_pi0_prefix_kv as CAP
    from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS

    if G.pre is None:
        G.pre, _ = CAP.make_preprocessor(G.policy, CKPT)
    ep, f = fr
    if ep not in G.replays:
        G.replays[ep] = np.load(EPISODES / f"{ep}.npz", allow_pickle=False)
    batch = CAP.build_batch(G.policy, G.pre, G.replays[ep], f)
    images, img_masks = G.policy._preprocess_images(batch)
    G.inputs[fr] = dict(images=images, img_masks=img_masks, lang_tokens=batch[OBS_LANGUAGE_TOKENS],
                        lang_masks=batch[OBS_LANGUAGE_ATTENTION_MASK], state=G.policy.prepare_state(batch))
    return G.inputs[fr]


def get_capture(fr) -> dict:
    ep, f = fr
    z = np.load(CAPTURES / ep / f"frame_{f:02d}.npz")
    valid = z["prefix_valid"].astype(bool)
    idx = np.nonzero(valid)[0]
    kv = z["kv"][:, :, idx, 0, :]                      # [L, 2, n_valid, D]
    return dict(K=torch.from_numpy(np.ascontiguousarray(kv[:, 0])), V=torch.from_numpy(np.ascontiguousarray(kv[:, 1])),
                valid=torch.from_numpy(valid), idx=idx, state=z["state"], noise=z["noise"],
                actions=z["actions_fp32"], lang_tokens=z["lang_tokens"], raw_state=z["raw_state"])


def prefix_compact(fin: dict, image_embs=None) -> dict:
    """PI0FpgaPolicy._run_prefix_compact, returning the pieces the gates need."""
    from lerobot.policies.pi0.modeling_pi0 import make_att_2d_masks

    model, pwe = G.policy.model, G.pwe
    G.tower.clear()
    embs, valid_parts, img_embs = [], [], []
    for img, img_mask in zip(fin["images"], fin["img_masks"], strict=True):
        if bool(img_mask[0]):
            emb = pwe.embed_image(img) if image_embs is None else image_embs[len(img_embs)]
            img_embs.append(emb)
            embs.append(emb)
            valid_parts.append(torch.ones(emb.shape[1], dtype=torch.bool))
        else:
            valid_parts.append(torch.zeros(G.policy._image_tokens(), dtype=torch.bool))
    lang_emb = pwe.embed_language_tokens(fin["lang_tokens"])
    lang_valid = fin["lang_masks"][0].to(torch.bool)
    embs.append(lang_emb[:, lang_valid])
    valid_parts.append(lang_valid)
    prefix_valid = torch.cat(valid_parts)
    compact = torch.cat(embs, dim=1)
    n = compact.shape[1]
    att_2d = make_att_2d_masks(torch.ones(1, n, dtype=torch.bool), torch.zeros(1, n, dtype=torch.bool))
    pwe.paligemma.model.language_model.config._attn_implementation = "eager"
    outs, pkv = pwe.forward(attention_mask=model._prepare_attention_masks_4d(att_2d),
                            position_ids=torch.arange(n)[None], past_key_values=None,
                            inputs_embeds=[compact, None], use_cache=True)
    entries = list(pkv)
    K = torch.stack([e[0][0, 0] for e in entries])       # [L, n, D] (one KV head)
    V = torch.stack([e[1][0, 0] for e in entries])
    return dict(img_embs=img_embs, tower=list(G.tower), hidden=outs[0][0], K=K, V=V, valid=prefix_valid)


def siglip_only(fin: dict) -> dict:
    G.tower.clear()
    embs = [G.pwe.embed_image(img) for img, m in zip(fin["images"], fin["img_masks"]) if bool(m[0])]
    return dict(img_embs=embs, tower=list(G.tower))


def run_expert(K, V, valid, cap) -> np.ndarray:
    from transformers.cache_utils import DynamicCache

    T = valid.shape[0]
    idx = torch.nonzero(valid)[:, 0]
    data = []
    for li in range(K.shape[0]):
        kf = torch.zeros(1, 1, T, K.shape[-1])
        vf = torch.zeros(1, 1, T, V.shape[-1])
        kf[0, 0, idx] = K[li]
        vf[0, 0, idx] = V[li]
        data.append((kf, vf))
    x = G.policy._denoise_torch(torch.from_numpy(cap["state"])[None], valid[None], DynamicCache(data),
                                torch.from_numpy(cap["noise"])[None], G.policy.config.num_inference_steps)
    return x[0].float().numpy()


def ref_path(fr) -> Path:
    return G.cache_dir / "ref" / f"{fr[0]}_f{fr[1]:02d}.pt"


def get_ref(fr) -> dict:
    p = ref_path(fr)
    if p.exists():
        return torch.load(p, weights_only=False)
    assert not (FMT["tables"] or FMT["iface_bf16"] or FMT["out_bf16"] or FMT["row_enc"] != "fp"), \
        "fp32 references must be computed with the hardware formats off"
    configure(dict(sel="none"))
    fin, cap = get_inputs(fr), get_capture(fr)
    t0 = time.time()
    out = prefix_compact(fin)
    t_prefix = time.time() - t0
    t0 = time.time()
    actions = run_expert(out["K"], out["V"], out["valid"], cap)
    ref = dict(img_embs=out["img_embs"], tower=out["tower"], hidden=out["hidden"], K=out["K"], V=out["V"],
               valid=out["valid"], actions=actions, t_prefix_s=t_prefix, t_expert_s=time.time() - t0)
    p.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ref, p)
    log(f"ref {fid(fr)}: prefix {t_prefix:.1f} s, expert {ref['t_expert_s']:.1f} s")
    return ref


# ======================================================================================
# configuration of a run
# ======================================================================================
def select(sel) -> set:
    names = set(G.QL)
    if sel == "all":
        return names
    if sel == "none":
        return set()
    if sel in ("lm", "vis"):
        return {n for n in names if n.startswith(sel + ".")}
    if isinstance(sel, str):
        rx = re.compile(sel)
        return {n for n in names if rx.search(n)}
    return set(sel)


def build_ws(kind: str) -> dict:
    path = G.cache_dir / f"ws_{kind}.pt"
    if path.exists():
        log(f"loading weight set {kind} from {path}")
        return torch.load(path, weights_only=False)
    t0 = time.time()
    ws = {}
    for name, ql in G.QL.items():
        if kind == "smooth" and name not in G.SMOOTH:
            base = get_ws("base")
            ws[name] = base[name]
            continue
        w = ql.w2d()
        if kind == "smooth":
            w = w * G.SMOOTH[name][None, :]
        clip = mse_clip_rows(w)
        scale = (clip / 127.0).float()
        ws[name] = (quant_codes(w.float(), scale), scale)
    log(f"weight set {kind}: {len(ws)} GEMMs quantised in {time.time() - t0:.0f} s, rss {rss_gb():.1f} GB")
    if G.args.save_ws:
        torch.save(ws, path)
    return ws


def get_ws(kind: str) -> dict:
    if kind not in G.WS:
        G.WS[kind] = build_ws(kind)
    return G.WS[kind]


def configure(spec: dict) -> None:
    names = select(spec.get("sel", "all"))
    smooth = bool(spec.get("smooth"))
    if smooth and not G.SMOOTH:
        load_calib()
    if spec.get("a") == "static" and not G.STATIC:
        load_calib()
    for name, ql in G.QL.items():
        on = name in names
        use_s = on and smooth and name in G.SMOOTH
        ql.wq = on and spec.get("w", True)
        ql.aq = spec.get("a", "token") if on else "none"
        ql.smooth = G.SMOOTH[name] if use_s else None
        if ql.wq:
            ql.codes, ql.wscale = get_ws("smooth" if use_s else "base")[name]
        else:
            ql.codes = ql.wscale = None
        ql.static_scale = G.STATIC[spec["static"]][name] if (on and ql.aq == "static") else None
        ql.nclip = ql.nel = 0
        ql.rec = None
    set_attn(spec.get("attn", "fp"), spec.get("attn_cfg"))


# ======================================================================================
# metrics
# ======================================================================================
def rel(a, b) -> float:
    a, b = a.double(), b.double()
    return float((a - b).norm() / b.norm())


def tok_rel_max(a, b) -> float:
    a, b = a.double().reshape(-1, a.shape[-1]), b.double().reshape(-1, b.shape[-1])
    return float(((a - b).norm(dim=1) / b.norm(dim=1).clamp_min(1e-30)).max())


def g0_metrics(out, ref) -> dict:
    return {"proj": [rel(a, b) for a, b in zip(out["img_embs"], ref["img_embs"])],
            "proj_tokmax": [tok_rel_max(a, b) for a, b in zip(out["img_embs"], ref["img_embs"])],
            "tower": [rel(a, b) for a, b in zip(out["tower"], ref["tower"])]}


def g1_metrics(out, ref, cap) -> dict:
    n = out["K"].shape[1]
    idx = cap["idx"]
    is_img = torch.from_numpy(idx < 768)
    assert int(out["valid"].sum()) == n == len(idx), "valid token count differs from the capture"
    m = {k: [] for k in ("K", "V", "K_img", "V_img", "K_lang", "V_lang", "K_tokmax", "V_tokmax", "K_own", "V_own")}
    for li in range(out["K"].shape[0]):
        for t in ("K", "V"):
            a, c = out[t][li], cap[t][li]
            m[t].append(rel(a, c))
            m[t + "_img"].append(rel(a[is_img], c[is_img]))
            m[t + "_lang"].append(rel(a[~is_img], c[~is_img]))
            m[t + "_tokmax"].append(tok_rel_max(a, c))
            m[t + "_own"].append(rel(a, ref[t][li]))
    m["hidden"] = rel(out["hidden"], ref["hidden"])
    m["hidden_lang"] = rel(out["hidden"][~is_img], ref["hidden"][~is_img])
    return m


def g3_metrics(actions, ref, cap) -> dict:
    e = D.chunk_error(actions, cap["actions"])
    e["rel_rms_7_vs_harness_fp32"] = float(np.linalg.norm(actions[:, :7] - ref["actions"][:, :7]) /
                                           np.linalg.norm(ref["actions"][:, :7]))
    return e


def eval_frames(spec: dict, frames: list, g3: bool = True, vis_only: bool = False, lm_only: bool = False,
                progress=None) -> dict:
    refs_needed = [fr for fr in frames if not ref_path(fr).exists()]
    for fr in refs_needed:
        get_ref(fr)
    configure(spec)
    recs = []
    for fr in frames:
        fin, ref = get_inputs(fr), get_ref(fr)
        cap = get_capture(fr)
        rec = {"frame": fid(fr)}
        t0 = time.time()
        if vis_only:
            out = siglip_only(fin)
        else:
            out = prefix_compact(fin, image_embs=ref["img_embs"] if lm_only else None)
        rec["t_prefix_s"] = time.time() - t0
        if not lm_only:
            rec["g0"] = g0_metrics(out, ref)
        if not vis_only:
            rec["g1"] = g1_metrics(out, ref, cap)
            if g3:
                t0 = time.time()
                rec["g3"] = g3_metrics(run_expert(out["K"], out["V"], out["valid"], cap), ref, cap)
                rec["t_expert_s"] = time.time() - t0
        if spec.get("a") == "static":
            rec["static_clip_frac"] = {n: q.nclip / q.nel for n, q in G.QL.items() if q.nel}
            for q in G.QL.values():
                q.nclip = q.nel = 0
        recs.append(rec)
        if progress:
            progress(recs)
    return {"records": recs, "summary": summarize(recs)}


def summarize(recs: list) -> dict:
    s = {"frames": len(recs)}
    if "g0" in recs[0]:
        proj = np.array([r["g0"]["proj"] for r in recs])
        tower = np.array([r["g0"]["tower"] for r in recs])
        s["g0"] = {"proj_mean": float(proj.mean()), "proj_max": float(proj.max()),
                   "tower_mean": float(tower.mean()), "tower_max": float(tower.max()),
                   "proj_tokmax_max": float(np.max([r["g0"]["proj_tokmax"] for r in recs])),
                   "images": int(proj.size), "pass": bool(proj.max() <= GATE_G0)}
    if "g1" in recs[0]:
        g1 = {}
        worst = 0.0
        for t in ("K", "V", "K_img", "V_img", "K_lang", "V_lang", "K_own", "V_own", "K_tokmax", "V_tokmax"):
            a = np.array([r["g1"][t] for r in recs])            # [frames, L]
            g1[t + "_layer_mean"] = a.mean(0).tolist()
            g1[t + "_layer_max"] = a.max(0).tolist()
            if t in ("K", "V"):
                worst = max(worst, float(a.max()))
                g1[t + "_worst_layer"] = int(a.max(0).argmax())
                g1[t + "_max"] = float(a.max())
                g1[t + "_layers_over_gate"] = int((a.max(0) > GATE_G1).sum())
        hid = np.array([r["g1"]["hidden"] for r in recs])
        g1["hidden_mean"], g1["hidden_max"] = float(hid.mean()), float(hid.max())
        g1["kv_max"] = worst
        g1["kv_last_layer_max"] = float(max(np.max([r["g1"]["K"][-1] for r in recs]), np.max([r["g1"]["V"][-1] for r in recs])))
        g1["pass"] = bool(worst <= GATE_G1)
        s["g1"] = g1
    if "g3" in recs[0]:
        e = np.array([r["g3"]["rel_rms_7"] for r in recs])
        c = np.array([r["g3"]["cos_7"] for r in recs])
        h = np.array([r["g3"]["rel_rms_7_vs_harness_fp32"] for r in recs])
        mx = np.array([r["g3"]["max_abs_7"] for r in recs])
        s["g3"] = {"rel_rms_7_mean": float(e.mean()), "rel_rms_7_max": float(e.max()), "cos_7_min": float(c.min()),
                   "max_abs_7_max": float(mx.max()), "vs_harness_mean": float(h.mean()), "vs_harness_max": float(h.max()),
                   "pass": bool(e.mean() <= GATE_G3 and e.max() <= GATE_G3_FRAME and c.min() >= GATE_G3_COS)}
    s["t_prefix_s_mean"] = float(np.mean([r["t_prefix_s"] for r in recs]))
    if "t_expert_s" in recs[0]:
        s["t_expert_s_mean"] = float(np.mean([r["t_expert_s"] for r in recs]))
    return s


# ======================================================================================
# output
# ======================================================================================
def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, (np.floating, np.integer, np.bool_)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, torch.Tensor):
        return o.tolist()
    return o


def write_json(stage: str, obj: dict) -> None:
    obj = {"stage": stage, "meta": meta(), **obj}
    p = G.out_dir / f"{stage}.json"
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(_jsonable(obj), indent=1))
    tmp.replace(p)


_META = {}


def meta() -> dict:
    if not _META:
        try:
            head = subprocess.run(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
        except Exception:  # noqa: BLE001
            head = "?"
        _META.update({"checkpoint": CKPT, "git_head": head, "torch": torch.__version__, "threads": THREADS,
                      "eval_frames": [fid(f) for f in G.args.eval_frames], "calib_frames": [fid(f) for f in G.args.calib_frames],
                      "argv": sys.argv, "weights": "INT8 sym per output channel, MSE clip grid 0.5..1.0 x21",
                      "activations": "INT8 sym per token dynamic unless the stage says static"})
    _META["written"] = time.strftime("%Y-%m-%d %H:%M:%S")
    return dict(_META)


# ======================================================================================
# stages
# ======================================================================================
VARIANTS = {
    "base": (dict(sel="all", w=True, a="token"), {}),
    "base_lm": (dict(sel="lm", w=True, a="token"), {"lm_only": True}),
    "base_vis": (dict(sel="vis", w=True, a="token"), {}),
    "w8": (dict(sel="all", w=True, a="none"), {}),
    "a8": (dict(sel="all", w=False, a="token"), {}),
    "smooth": (dict(sel="all", w=True, a="token", smooth=True), {}),
    "smooth_lm": (dict(sel="lm", w=True, a="token", smooth=True), {"lm_only": True}),
    "attn": (dict(sel="all", w=True, a="token", attn="int8", attn_cfg=ATTN_INT8), {}),
    "attn_only": (dict(sel="none", attn="int8", attn_cfg=ATTN_INT8), {}),
    "attn_qk": (dict(sel="all", w=True, a="token", attn="int8", attn_cfg={**ATTN_INT8, "pv": False}), {}),
    "attn_pv": (dict(sel="all", w=True, a="token", attn="int8", attn_cfg={**ATTN_INT8, "qk": False}), {}),
    "attn_vcol": (dict(sel="all", w=True, a="token", attn="int8", attn_cfg={**ATTN_INT8, "v_gran": "col"}), {}),
    "attn_p255": (dict(sel="all", w=True, a="token", attn="int8", attn_cfg={**ATTN_INT8, "p_levels": 255}), {}),
    "static_amax": (dict(sel="all", w=True, a="static", static="amax"), {}),
    "static_mse": (dict(sel="all", w=True, a="static", static="mse"), {}),
    "smooth_lm_attn": (dict(sel="lm", w=True, a="token", smooth=True, attn="int8", attn_cfg=ATTN_INT8), {"lm_only": True}),
    "smooth_attn": (dict(sel="all", w=True, a="token", smooth=True, attn="int8", attn_cfg=ATTN_INT8), {}),
}


def stage_variant(stage: str) -> None:
    spec, kw = VARIANTS[stage]
    frames = G.args.eval_frames
    t0 = time.time()
    log(f"stage {stage}: {spec} on {len(frames)} frames")

    def progress(recs):
        write_json(stage, {"spec": spec, "partial": True, "records": recs})
        r = recs[-1]
        bits = [f"{r['frame']}"]
        if "g0" in r:
            bits.append(f"G0 {max(r['g0']['proj']) * 100:.2f}%")
        if "g1" in r:
            bits.append(f"G1 K {max(r['g1']['K']) * 100:.2f}% V {max(r['g1']['V']) * 100:.2f}%")
        if "g3" in r:
            bits.append(f"G3 {r['g3']['rel_rms_7'] * 100:.2f}%")
        log("  " + " | ".join(bits) + f" | {r['t_prefix_s']:.1f}+{r.get('t_expert_s', 0):.1f} s")

    res = eval_frames(spec, frames, g3=not G.args.no_g3, progress=progress, **kw)
    extra = {}
    if spec.get("smooth"):
        extra["smooth_factor_range"] = {n: [float(s.min()), float(s.max())] for n, s in G.SMOOTH.items()}
        extra["alpha"] = G.args.alpha
    write_json(stage, {"spec": spec, "wall_s": time.time() - t0, **extra, **res})
    log(f"stage {stage} done in {time.time() - t0:.0f} s: {json.dumps(_jsonable(res['summary']))[:600]}")


def stage_fp32() -> None:
    t0 = time.time()
    recs = []
    for fr in G.args.eval_frames:
        ref, cap = get_ref(fr), get_capture(fr)
        rec = {"frame": fid(fr), "t_prefix_s": ref["t_prefix_s"], "t_expert_s": ref["t_expert_s"]}
        rec["g1"] = g1_metrics(dict(K=ref["K"], V=ref["V"], hidden=ref["hidden"], valid=ref["valid"]), ref, cap)
        rec["g3"] = g3_metrics(ref["actions"], ref, cap)
        recs.append(rec)
    write_json("fp32", {"note": "this script's own fp32 compact prefix vs the capture (the numerical floor)",
                        "wall_s": time.time() - t0, "records": recs, "summary": summarize(recs)})
    log(f"fp32 floor: {json.dumps(summarize(recs))[:500]}")


def stage_selftest() -> None:
    fr = G.args.eval_frames[0]
    fin, cap = get_inputs(fr), get_capture(fr)
    res = {"frame": fid(fr)}
    res["lang_tokens_match_capture"] = bool((fin["lang_tokens"][0].numpy() == cap["lang_tokens"]).all())
    res["state_max_abs_diff_vs_capture"] = float(np.abs(fin["state"][0].numpy() - cap["state"]).max())
    res["raw_state_match"] = bool(np.allclose(G.replays[fr[0]]["state"][fr[1]], cap["raw_state"]))
    configure(dict(sel="none"))
    t0 = time.time()
    kv_pol, valid_pol = G.policy._run_prefix_compact(fin["images"], fin["img_masks"], fin["lang_tokens"], fin["lang_masks"])
    res["t_policy_compact_s"] = time.time() - t0
    t0 = time.time()
    out = prefix_compact(fin)
    res["t_mine_compact_s"] = time.time() - t0
    idx = torch.nonzero(valid_pol[0])[:, 0]
    res["mine_vs_policy_compact_max_abs"] = max(float((kv_pol[:, 0, idx, 0] - out["K"]).abs().max()),
                                                float((kv_pol[:, 1, idx, 0] - out["V"]).abs().max()))
    res["valid_match"] = bool((valid_pol[0] == out["valid"]).all()) and bool((valid_pol[0].numpy() == cap["valid"].numpy()).all())
    res["fp32_vs_capture_K_rel_per_layer"] = [rel(out["K"][li], cap["K"][li]) for li in range(out["K"].shape[0])]
    res["fp32_vs_capture_V_rel_per_layer"] = [rel(out["V"][li], cap["V"][li]) for li in range(out["V"].shape[0])]
    t0 = time.time()
    act_cap = run_expert(cap["K"], cap["V"], cap["valid"], cap)
    res["t_expert_s"] = time.time() - t0
    res["expert_harness_on_capture_kv"] = D.chunk_error(act_cap, cap["actions"])
    # the patch-GEMM path (unfold + linear) against the stock conv
    q = G.QL["vis.patch"]
    img = fin["images"][0]
    ref_conv = F.conv2d(img, q.weight, q.bias, stride=q.stride)
    q.rec = lambda *_: None
    res["patch_gemm_vs_conv_rel"] = rel(q(img), ref_conv)
    q.rec = None
    # SigLIP eager attention (used by the INT8 attention variants) against the stock implementation
    res["siglip_attn_impl_stock"] = sorted({str(v) for v in ATTN["siglip_impl"].values()})
    set_attn("int8", {**ATTN_INT8, "qk": False, "pv": False})
    ATTN["calls"].clear()
    eager = siglip_only(fin)
    res["siglip_eager_vs_stock_rel"] = [rel(a, b) for a, b in zip(eager["img_embs"], out["img_embs"])]
    res["attn_patch_calls_siglip_only"] = len(ATTN["calls"])
    set_attn("fp")
    # torch MSE clip against the numpy deploy-model helper
    w = G.QL["vis.L0.q"].w2d()
    c_t = mse_clip_rows(w).numpy()
    c_n = D.mse_clip_per_column(w.T.numpy().astype(np.float32))
    res["mse_clip_torch_vs_numpy_max_rel"] = float(np.max(np.abs(c_t - c_n) / c_n))
    res["mse_clip_choice_differs"] = int((np.abs(c_t - c_n) / c_n > 1e-5).sum())
    write_json("selftest", res)
    log("selftest: " + json.dumps(_jsonable({k: v for k, v in res.items() if not k.startswith("fp32_vs")}))[:1500])
    log(f"selftest: fp32 vs capture K max {max(res['fp32_vs_capture_K_rel_per_layer']):.2e} "
        f"V max {max(res['fp32_vs_capture_V_rel_per_layer']):.2e}")


# ---- calibration ---------------------------------------------------------------------
class CalibAmax:
    def __init__(self):
        self.ch = {}

    def __call__(self, ql, x):
        ch = x.reshape(-1, x.shape[-1]).abs().amax(0)
        prev = self.ch.get(ql.name)
        self.ch[ql.name] = ch if prev is None else torch.maximum(prev, ch)


class CalibGrid:
    def __init__(self, amax: dict):
        self.amax = amax
        self.err = {}
        self.n = {}

    def __call__(self, ql, x):
        a = self.amax[ql.name]
        xf = x.reshape(-1).float()
        e = self.err.setdefault(ql.name, np.zeros(len(A_GRID)))
        for i, f in enumerate(A_GRID):
            s = a * float(f) / 127.0
            e[i] += float(torch.round(xf / s).clamp_(-127, 127).mul_(s).sub_(xf).pow_(2).sum())
        self.n[ql.name] = self.n.get(ql.name, 0) + xf.numel()


def run_fp_pass(frames, rec) -> None:
    configure(dict(sel="none"))
    for ql in G.QL.values():
        ql.rec = rec
    for fr in frames:
        t0 = time.time()
        prefix_compact(get_inputs(fr))
        log(f"  calib pass {type(rec).__name__} {fid(fr)} {time.time() - t0:.1f} s")
    for ql in G.QL.values():
        ql.rec = None


def stage_calib() -> None:
    frames = G.args.calib_frames
    overlap = {fid(f) for f in frames} & {fid(f) for f in G.args.eval_frames}
    assert not overlap, f"calibration and evaluation frames overlap: {overlap}"
    t0 = time.time()
    amax = CalibAmax()
    run_fp_pass(frames, amax)
    tensor_amax = {n: float(c.max()) for n, c in amax.ch.items()}
    grid = CalibGrid(tensor_amax)
    run_fp_pass(frames, grid)
    mse = {n: tensor_amax[n] * float(A_GRID[int(np.argmin(grid.err[n]))]) for n in grid.err}
    torch.save({"ch_amax": amax.ch, "tensor_amax": tensor_amax, "mse_clip": mse, "grid_err": grid.err, "grid_n": grid.n,
                "frames": [fid(f) for f in frames]}, G.cache_dir / "calib.pt")
    write_json("calib", {"frames": [fid(f) for f in frames], "wall_s": time.time() - t0,
                         "tensor_amax": tensor_amax, "mse_clip": mse,
                         "mse_clip_fraction_of_amax": {n: mse[n] / tensor_amax[n] for n in mse},
                         "channel_amax_ratio_max_over_median": {n: float(c.max() / c.median().clamp_min(1e-12)) for n, c in amax.ch.items()}})
    log(f"calib done in {time.time() - t0:.0f} s")
    G.CALIB = None
    G.SMOOTH, G.STATIC = {}, {}
    load_calib()


def load_calib() -> None:
    p = G.cache_dir / "calib.pt"
    if not p.exists():
        raise RuntimeError("run the calib stage first")
    G.CALIB = torch.load(p, weights_only=False)
    G.STATIC = {"amax": {n: a / 127.0 for n, a in G.CALIB["tensor_amax"].items()},
                "mse": {n: c / 127.0 for n, c in G.CALIB["mse_clip"].items()}}
    ch, alpha = G.CALIB["ch_amax"], G.args.alpha
    S = {}
    W = lambda n: G.QL[n].w2d()  # noqa: E731
    for li in range(len(G.pwe.joint_layers)):
        p_ = f"lm.L{li}."
        wr = torch.maximum(torch.maximum(W(p_ + "q").abs().amax(0), W(p_ + "k").abs().amax(0)), W(p_ + "v").abs().amax(0))
        s = torch.from_numpy(D.smoothing_factors(ch[p_ + "q"].numpy(), wr.numpy(), alpha))
        for slot in ("q", "k", "v"):
            S[p_ + slot] = s
        wr = torch.maximum(W(p_ + "gate").abs().amax(0), W(p_ + "up").abs().amax(0))
        s = torch.from_numpy(D.smoothing_factors(ch[p_ + "gate"].numpy(), wr.numpy(), alpha))
        S[p_ + "gate"] = S[p_ + "up"] = s
        heads = G.pwe.joint_layers[li].paligemma_layer.self_attn.config.num_attention_heads
        xa = ch[p_ + "o"].reshape(heads, -1).amax(0).numpy()
        wr = W(p_ + "o").abs().amax(0).reshape(heads, -1).amax(0).numpy()
        S[p_ + "o"] = torch.from_numpy(np.tile(D.smoothing_factors(xa, wr, alpha), heads))
        S[p_ + "down"] = torch.from_numpy(D.smoothing_factors(ch[p_ + "down"].numpy(), W(p_ + "down").abs().amax(0).numpy(), alpha))
    G.SMOOTH = S


# ---- localisation ----------------------------------------------------------------------
class LocalStats:
    def __init__(self):
        self.d = {}

    def __call__(self, ql, x):
        x2 = x.reshape(-1, x.shape[-1]).float()
        e = self.d.setdefault(ql.name, dict(y2=0.0, w=0.0, a=0.0, both=0.0, ratios=[], ch=None, amax=0.0, sumsq=0.0,
                                             n=0, peak_tok=[]))
        W, Wq = ql.w2d(), get_ws("base")[ql.name]
        Wq = Wq[0].to(torch.float32).mul_(Wq[1][:, None])
        xq = fq_rows(x2)
        y = x2 @ W.T
        e["y2"] += float(y.double().pow(2).sum())
        e["w"] += float((x2 @ Wq.T - y).double().pow(2).sum())
        e["a"] += float((xq @ W.T - y).double().pow(2).sum())
        e["both"] += float((xq @ Wq.T - y).double().pow(2).sum())
        tok_amax = x2.abs().amax(1)
        tok_rms = x2.pow(2).mean(1).sqrt().clamp_min(1e-12)
        e["ratios"].append((tok_amax / tok_rms).numpy())
        e["peak_tok"].append(int(tok_amax.argmax()))
        ch = x2.abs().amax(0)
        e["ch"] = ch if e["ch"] is None else torch.maximum(e["ch"], ch)
        e["amax"] = max(e["amax"], float(tok_amax.max()))
        e["sumsq"] += float(x2.double().pow(2).sum())
        e["n"] += x2.numel()


def stage_local() -> None:
    frames = G.args.local_frames or G.args.eval_frames[:2]
    t0 = time.time()
    get_ws("base")
    st = LocalStats()
    configure(dict(sel="none"))
    for ql in G.QL.values():
        ql.rec = st
    set_attn("compare")
    ATTN["acc"].clear()
    for fr in frames:
        t1 = time.time()
        prefix_compact(get_inputs(fr))
        log(f"  local {fid(fr)} {time.time() - t1:.1f} s")
    for ql in G.QL.values():
        ql.rec = None
    set_attn("fp")
    table = {}
    for n, e in st.d.items():
        r = np.concatenate(e["ratios"])
        ch = e["ch"]
        top = torch.topk(ch, min(5, ch.numel()))
        rms = (e["sumsq"] / e["n"]) ** 0.5
        table[n] = {"local_rel_both": (e["both"] / e["y2"]) ** 0.5, "local_rel_w": (e["w"] / e["y2"]) ** 0.5,
                    "local_rel_a": (e["a"] / e["y2"]) ** 0.5, "amax": e["amax"], "rms": rms,
                    "amax_over_rms": e["amax"] / rms, "tok_peak_ratio_p50": float(np.percentile(r, 50)),
                    "tok_peak_ratio_p99": float(np.percentile(r, 99)), "tok_peak_ratio_max": float(r.max()),
                    "ch_amax_max_over_median": float(ch.max() / ch.median().clamp_min(1e-12)),
                    "top_channels": top.indices.tolist(), "top_channel_amax": top.values.tolist(),
                    "peak_token_index_per_call": e["peak_tok"]}
    attn = {n: {"local_rel_both": (a["both"] / a["y2"]) ** 0.5, "local_rel_qk": (a["qk"] / a["y2"]) ** 0.5,
                "local_rel_pv": (a["pv"] / a["y2"]) ** 0.5} for n, a in ATTN["acc"].items()}
    write_json("local", {"frames": [fid(f) for f in frames], "wall_s": time.time() - t0, "gemm": table, "attention": attn,
                         "note": "local errors: GEMM output (no bias) with INT8 W / A / both on the fp32 input of the "
                                 "fp32 model, rel = sqrt(sum err^2 / sum y^2) over all calls; tok_peak_ratio = token "
                                 "amax / token rms (per-token INT8 noise ~ ratio / 440); peak_token_index = argmax "
                                 "row in each call (LM: compact position, 512.. = language tokens)"})
    log(f"local done in {time.time() - t0:.0f} s")


def stage_iso(stage: str) -> None:
    frames = G.args.iso_frames or G.args.eval_frames[:2]
    t0 = time.time()
    nL = len(G.pwe.joint_layers)
    nV = len(G.pwe.paligemma.model.vision_tower.vision_model.encoder.layers)
    if stage == "iso_lm_layer":
        runs = {f"lm.L{li}": f"^lm\\.L{li}\\." for li in range(nL)}
    elif stage == "iso_lm_proj":
        runs = {f"lm.*.{s}": f"^lm\\.L\\d+\\.{s}$" for s in LM_SLOTS}
    elif stage == "iso_vis_layer":
        runs = {f"vis.L{li}": f"^vis\\.L{li}\\." for li in range(nV)}
    else:
        runs = {f"vis.*.{s}": f"^vis\\.L\\d+\\.{s}$" for s in VIS_SLOTS}
        runs.update({"vis.patch": "^vis\\.patch$", "vis.mmproj": "^vis\\.mmproj$"})
    vis = stage.startswith("iso_vis")
    out = {}
    for run, rx in runs.items():
        t1 = time.time()
        res = eval_frames(dict(sel=rx, w=True, a="token"), frames, g3=not vis and not G.args.no_g3, vis_only=vis, lm_only=not vis)
        out[run] = {"regex": rx, "gemms": len(select(rx)), **res}
        s = res["summary"]
        msg = f"G0 max {s['g0']['proj_max'] * 100:.3f}%" if vis else \
            f"KV max {s['g1']['kv_max'] * 100:.3f}% last {s['g1']['kv_last_layer_max'] * 100:.3f}% hidden {s['g1']['hidden_max'] * 100:.3f}%" + \
            (f" G3 max {s['g3']['rel_rms_7_max'] * 100:.3f}%" if "g3" in s else "")
        log(f"  {stage} {run}: {msg} ({time.time() - t1:.0f} s)")
        write_json(stage, {"frames": [fid(f) for f in frames], "partial": True, "runs": out})
    write_json(stage, {"frames": [fid(f) for f in frames], "wall_s": time.time() - t0, "runs": out})
    log(f"{stage} done in {time.time() - t0:.0f} s")


def stage_fallback() -> None:
    """Greedy fp32 fallback toward G3 max < --fallback-target.  Gemma layers are taken in the order of
    their damage when quantised alone (iso_lm_layer: G3 max on the iso frames, then KV max); for each
    --fallback-bases variant the top k = --fallback-k layers are kept fp32, stopping at the first k that
    meets the target on the evaluation frames (k = 0 is the variant's own JSON)."""
    iso = json.loads((G.out_dir / "iso_lm_layer.json").read_text())["runs"]

    def damage(r):
        s = iso[r]["summary"]
        return (s.get("g3", {}).get("rel_rms_7_max", 0.0), s["g1"]["kv_max"])

    rank = sorted(iso, key=damage, reverse=True)
    target = G.args.fallback_target
    out = {"target_g3_rel_rms_7_max": target,
           "ranking": [{"layer": r, "iso_g3_max": damage(r)[0], "iso_kv_max": damage(r)[1]} for r in rank],
           "runs": {}, "minimal": {}}
    t0 = time.time()
    for base in G.args.fallback_bases:
        bj = G.out_dir / f"{base}.json"
        if bj.exists():
            s = json.loads(bj.read_text())["summary"]
            out["runs"][f"{base}_top0"] = {"base": base, "fp32_layers": [], "fp32_gemms": 0, "summary": s}
            if s.get("g3", {}).get("rel_rms_7_max", 1.0) < target:
                out["minimal"][base] = {"k": 0, "fp32_layers": [], "g3_max": s["g3"]["rel_rms_7_max"], "g3_mean": s["g3"]["rel_rms_7_mean"]}
                continue
        for k in G.args.fallback_k:
            fp = rank[:k]
            keep = select("^(" + "|".join(re.escape(r) + "\\." for r in fp) + ")")
            spec = {**VARIANTS[base][0], "sel": set(G.QL) - keep}
            log(f"fallback {base} top{k}: fp32 {fp}")
            res = eval_frames(spec, G.args.eval_frames, g3=True)
            out["runs"][f"{base}_top{k}"] = {"base": base, "fp32_layers": fp, "fp32_gemms": len(keep), **res}
            g3 = res["summary"]["g3"]
            log(f"  {base} top{k}: G3 mean {g3['rel_rms_7_mean'] * 100:.3f}% max {g3['rel_rms_7_max'] * 100:.3f}% "
                f"KV max {res['summary']['g1']['kv_max'] * 100:.2f}%")
            write_json("fallback", {**out, "partial": True})
            if g3["rel_rms_7_max"] < target:
                out["minimal"][base] = {"k": k, "fp32_layers": fp, "g3_max": g3["rel_rms_7_max"], "g3_mean": g3["rel_rms_7_mean"]}
                break
    out["wall_s"] = time.time() - t0
    write_json("fallback", out)


def stage_custom() -> None:
    a = G.args
    base = {"base": VARIANTS["base"][0], "smooth": VARIANTS["smooth"][0], "attn": VARIANTS["attn"][0]}[a.custom_base]
    keep = select(a.custom_fp)
    spec = {**base, "sel": set(G.QL) - keep}
    log(f"custom {a.custom_name}: {len(keep)} GEMMs fp32 ({a.custom_fp}) on top of {a.custom_base}")
    t0 = time.time()
    res = eval_frames(spec, a.eval_frames, g3=not a.no_g3)
    write_json(f"custom_{a.custom_name}", {"base": a.custom_base, "fp32_regex": a.custom_fp, "fp32_gemms": sorted(keep),
                                           "wall_s": time.time() - t0, **res})
    log(f"custom {a.custom_name}: {json.dumps(_jsonable(res['summary']))[:500]}")


# ---- end to end: the W8A8 prefix into the deployment INT8 expert ------------------------
def load_int8_expert(numerics_path: str):
    """The deployed INT8 expert as pi0_deploy_model.HostModel, rebuilt from the numerics .npz the session
    was exported from (receipt.json) -- the table build_deploy_model returned, without its 3 GB pickle."""
    import pi0_ae_model as M

    w = M.load_weights_lerobot(CKPT)
    z = np.load(os.path.expanduser(numerics_path))
    meta_n = json.loads(str(z["meta"]))
    cfg = meta_n["config"]
    lin = {}
    for L in range(M.DEPTH):
        for slot in D.LAYER_SLOTS:
            p = f"layers/{L}/{slot}/"
            wq = D.layer_matrix(w, L, slot) * z[p + "row_scale"][:, None] * z[p + "col_scale"][None, :]
            dl = D.DeployLinear(f"L{L}.{slot}", wq, z[p + "w_clip"], float(z[p + "in_scale"]), z[p + "out_scale"],
                                encode=cfg["encode"], out_bf16=cfg["host_bf16"])
            dl.bias48 = z[p + "bias48"].astype(np.float64)
            lin[(L, slot)] = dl
    for slot in D.PROJ_SLOTS:
        mat, _ = D.proj_matrix(w, slot)
        p = f"proj/{slot}/"
        lin[slot] = D.DeployLinear(slot, mat, z[p + "w_clip"], float(z[p + "in_scale"]), z[p + "out_scale"],
                                   host_bias=z[p + "host_bias"], bias_in_hw=False, encode=cfg["encode"], out_bf16=False)
    bf = D.bf16 if cfg["host_bf16"] else (lambda x: x)
    attn = np.stack([z[f"layers/{L}/attn_norm"] for L in range(M.DEPTH)])
    ffw = np.stack([z[f"layers/{L}/ffw_norm"] for L in range(M.DEPTH)])
    hm = D.HostModel(lin=lin, attn_gain=bf(attn), ffw_gain=bf(ffw), final_gain=bf(z["final_norm"]),
                     prefix_v_scale=z["prefix_v_scale"], host_bf16=cfg["host_bf16"])
    return hm, meta_n


def int8_expert_actions(hm, K, V, valid, cap) -> np.ndarray:
    """Valid-token KV [L, n, D] -> the host's bf16 816-token cache -> the 10-step INT8 expert chunk."""
    import pi0_ae_model as M

    valid = np.asarray(valid).astype(bool)
    idx = np.nonzero(valid)[0]
    kv = []
    for L in range(K.shape[0]):
        kf = np.zeros((valid.shape[0], K.shape[-1]), np.float32)
        vf = np.zeros_like(kf)
        kf[idx], vf[idx] = K[L], V[L]
        kv.append((M.to_bf16(kf), M.to_bf16(vf)))
    ep = M.Episode(state=cap["state"].astype(np.float32), noise=cap["noise"].astype(np.float32), prefix_kv=kv,
                   prefix_valid=valid, actions_ref=cap["actions"])
    return D.denoise_host(hm, ep)


def joint_metrics(a, ref, std) -> dict:
    d = (np.asarray(a, np.float64)[:, :7] - np.asarray(ref, np.float64)[:, :7]) * std[None, :7]
    j = d[:, :6]
    return {"joint_max_abs_rad": float(np.abs(j).max()), "joint_rms_rad": float(np.sqrt((j ** 2).mean())),
            "gripper_max_abs": float(np.abs(d[:, 6]).max())}


def stage_e2e_int8_expert() -> None:
    frames, variants = G.args.eval_frames, list(G.args.e2e_variants)
    kvdir = G.cache_dir / "e2e_kv"
    kvdir.mkdir(parents=True, exist_ok=True)
    kvp = lambda v, fr: kvdir / f"{v}_{fr[0]}_f{fr[1]:02d}.pt"  # noqa: E731
    t0 = time.time()
    for fr in frames:
        get_ref(fr)
    for v in variants:
        spec, kw = VARIANTS[v]
        todo = [fr for fr in frames if not kvp(v, fr).exists()]
        if not todo:
            continue
        configure(spec)
        for fr in todo:
            ref = get_ref(fr)
            t1 = time.time()
            out = prefix_compact(get_inputs(fr), image_embs=ref["img_embs"] if kw.get("lm_only") else None)
            torch.save({"K": out["K"], "V": out["V"], "valid": out["valid"]}, kvp(v, fr))
            log(f"  e2e KV {v} {fid(fr)} {time.time() - t1:.1f} s")
    configure(dict(sel="none"))
    G.WS.clear()
    import gc
    gc.collect()
    t1 = time.time()
    hm, meta_n = load_int8_expert(G.args.numerics)
    t_build = time.time() - t1
    log(f"INT8 expert rebuilt from {G.args.numerics} ({meta_n.get('recipe')}) in {t_build:.0f} s, rss {rss_gb():.1f} GB")
    from safetensors import safe_open
    with safe_open(os.path.join(CKPT, "policy_postprocessor_step_0_unnormalizer_processor.safetensors"), "np") as f:
        std = f.get_tensor([k for k in f.keys() if k.endswith("std")][0]).astype(np.float64)
    held = {}
    hp = HERE / "results" / "deploy_calib_heldout.json"
    if hp.exists():
        hj = json.loads(hp.read_text())
        rec_m = [r for r in hj["results"] if r["name"] == meta_n.get("recipe")]
        if rec_m and len(hj["validation_frames"]) == len(rec_m[0]["frames"]):
            for path, fr_rec in zip(hj["validation_frames"], rec_m[0]["frames"]):
                held["/".join(Path(path).parts[-2:])] = fr_rec["rel_rms_7"]
    torch_ref = {}
    for v in variants:
        p = G.out_dir / f"{v}.json"
        if p.exists():
            torch_ref[v] = {r["frame"]: r["g3"]["rel_rms_7"] for r in json.loads(p.read_text())["records"] if "g3" in r}
    recs = []
    for fr in frames:
        cap = get_capture(fr)
        rec = {"frame": fid(fr)}
        sources = [("fp32_prefix", cap["K"].numpy(), cap["V"].numpy(), cap["valid"].numpy())]
        for v in variants:
            z = torch.load(kvp(v, fr), weights_only=False)
            sources.append((v, z["K"].numpy(), z["V"].numpy(), z["valid"].numpy()))
        for name, K, V, valid in sources:
            t1 = time.time()
            a = int8_expert_actions(hm, K, V, valid, cap)
            e = D.chunk_error(a, cap["actions"])
            e.update(joint_metrics(a, cap["actions"], std))
            e["t_s"] = time.time() - t1
            if name == "fp32_prefix":
                e["heldout_json_rel_rms_7"] = held.get(f"{fr[0]}/frame_{fr[1]:02d}.npz")
            else:
                tr = torch_ref.get(name, {}).get(fid(fr))
                e["torch_fp32_expert_rel_rms_7"] = tr
                if tr is not None:
                    e["rss_of_stages_rel_rms_7"] = float(np.hypot(tr, rec["fp32_prefix"]["rel_rms_7"]))
            rec[name] = e
        recs.append(rec)
        log(f"  e2e {fid(fr)}: " + " | ".join(f"{n} {rec[n]['rel_rms_7'] * 100:.2f}% {rec[n]['joint_max_abs_rad']:.4f} rad"
                                              for n, *_ in sources) + f" ({rec['fp32_prefix']['t_s']:.1f} s/chunk)")
        write_json("e2e_int8_expert", {"partial": True, "records": recs})
    summ = {}
    for name in ["fp32_prefix"] + variants:
        e = np.array([r[name]["rel_rms_7"] for r in recs])
        c = np.array([r[name]["cos_7"] for r in recs])
        jm = np.array([r[name]["joint_max_abs_rad"] for r in recs])
        jr = np.array([r[name]["joint_rms_rad"] for r in recs])
        s = {"rel_rms_7_mean": float(e.mean()), "rel_rms_7_max": float(e.max()), "cos_7_min": float(c.min()),
             "joint_max_abs_rad_mean": float(jm.mean()), "joint_max_abs_rad_max": float(jm.max()),
             "joint_rms_rad_mean": float(jr.mean()), "joint_rms_rad_max": float(jr.max()),
             "pass_g3": bool(e.mean() <= GATE_G3 and e.max() <= GATE_G3_FRAME and c.min() >= GATE_G3_COS),
             "pass_g4_rms": bool(jr.max() <= 0.005), "frames_joint_max_over_0p02_rad": int((jm > 0.02).sum())}
        if name != "fp32_prefix":
            tr = [r[name]["torch_fp32_expert_rel_rms_7"] for r in recs if r[name].get("torch_fp32_expert_rel_rms_7") is not None]
            rs = [r[name]["rss_of_stages_rel_rms_7"] for r in recs if r[name].get("rss_of_stages_rel_rms_7") is not None]
            if tr:
                s.update({"torch_fp32_expert_rel_rms_7_mean": float(np.mean(tr)), "torch_fp32_expert_rel_rms_7_max": float(np.max(tr)),
                          "rss_of_stages_mean": float(np.mean(rs)), "rss_of_stages_max": float(np.max(rs))})
        summ[name] = s
    val = [abs(r["fp32_prefix"]["rel_rms_7"] - r["fp32_prefix"]["heldout_json_rel_rms_7"]) for r in recs
           if r["fp32_prefix"].get("heldout_json_rel_rms_7") is not None]
    write_json("e2e_int8_expert", {"numerics": G.args.numerics, "recipe": meta_n.get("recipe"), "variants": variants,
                                   "rebuild_check_max_abs_diff_vs_heldout_json": max(val) if val else None,
                                   "rebuild_check_frames": len(val), "build_s": t_build, "wall_s": time.time() - t0,
                                   "note": "G3 of the deployment INT8 expert (pi0_deploy_model, bf16 host ops, bf16 KV) vs the "
                                           "torch fp32 capture chunk; fp32_prefix = capture KV (expert alone); joint errors "
                                           "in rad through the policy unnormaliser std", "records": recs, "summary": summ})
    log(f"e2e_int8_expert done in {time.time() - t0:.0f} s: {json.dumps(_jsonable(summ))[:900]}")


# ---- the action expert with the prefix recipe: W8 per channel MSE, A8 per token dynamic ----
EXP_PROJ = (("state", "state_proj"), ("act_in", "action_in_proj"), ("tm_in", "action_time_mlp_in"),
            ("tm_out", "action_time_mlp_out"), ("act_out", "action_out_proj"))
EXPERT_SPECS = {
    "fp32": dict(sel="none"),
    "dyn": dict(sel="all", w=True, a="token"),
    "dyn_smooth": dict(sel="all", w=True, a="token", smooth=True),
    "dyn_smooth_attn": dict(sel="all", w=True, a="token", smooth=True, attn="int8"),
    "dyn_attn": dict(sel="all", w=True, a="token", attn="int8"),
    # float fallbacks for the expert's most sensitive GEMMs (iso_exp_layer: L0; iso_exp_proj: act_in)
    "dyn_smooth_L0fp": dict(sel=r"^exp\.(?!L0\.)", w=True, a="token", smooth=True),
    "dyn_smooth_L0fp_actin_fp": dict(sel=r"^exp\.(?!L0\.|act_in$)", w=True, a="token", smooth=True),
    "dyn_smooth_L0L17fp": dict(sel=r"^exp\.(?!L0\.|L17\.)", w=True, a="token", smooth=True),
}
EXPERT_LABEL = {
    "dyn": "W8 per-channel MSE + A8 per-token dynamic on all 131 expert GEMMs (126 layer + 5 projections), attention fp32",
    "dyn_smooth": "dyn + SmoothQuant a=0.5 on the 126 expert layer GEMMs (calibrated on --calib frames)",
    "dyn_smooth_attn": "dyn_smooth + INT8 QK^T and PV in the expert (Q,K,P per row; V per key; prefix keys included)",
    "dyn_attn": "dyn + INT8 QK^T and PV in the expert",
}


def install_expert_wrappers() -> None:
    QX = {}
    for li, pair in enumerate(G.pwe.joint_layers):
        layer = pair.expert_layer
        for slot, owner, attr in (("q", layer.self_attn, "q_proj"), ("k", layer.self_attn, "k_proj"),
                                  ("v", layer.self_attn, "v_proj"), ("o", layer.self_attn, "o_proj"),
                                  ("gate", layer.mlp, "gate_proj"), ("up", layer.mlp, "up_proj"),
                                  ("down", layer.mlp, "down_proj")):
            lin = getattr(owner, attr)
            name = f"exp.L{li}.{slot}"
            QX[name] = QLinear(name, lin.weight, lin.bias)
            setattr(owner, attr, QX[name])
        ATTN["exp_names"][id(layer.self_attn)] = f"exp.L{li}"
        layer.register_forward_hook(_bf16_out_hook)
    for slot, attr in EXP_PROJ:
        lin = getattr(G.policy.model, attr)
        name = f"exp.{slot}"
        QX[name] = QLinear(name, lin.weight, lin.bias)
        setattr(G.policy.model, attr, QX[name])
    G.QX = QX


def select_x(sel) -> set:
    names = set(G.QX)
    if sel == "all":
        return names
    if sel == "none":
        return set()
    if isinstance(sel, str):
        rx = re.compile(sel)
        return {n for n in names if rx.search(n)}
    return set(sel)


def layer_smoothing(prefix: str, layers, QD: dict, ch: dict, alpha: float) -> dict:
    """SmoothQuant factors of pi0_deploy_model.build_deploy_model for a stack of Gemma layers:
    qkv and gate/up share their RMSNorm input, o is tied per head_dim (one KV head), down alone."""
    S = {}
    W = lambda n: QD[n].w2d()  # noqa: E731
    for li, layer in enumerate(layers):
        p_ = f"{prefix}.L{li}."
        wr = torch.maximum(torch.maximum(W(p_ + "q").abs().amax(0), W(p_ + "k").abs().amax(0)), W(p_ + "v").abs().amax(0))
        s = torch.from_numpy(D.smoothing_factors(ch[p_ + "q"].numpy(), wr.numpy(), alpha))
        for slot in ("q", "k", "v"):
            S[p_ + slot] = s
        wr = torch.maximum(W(p_ + "gate").abs().amax(0), W(p_ + "up").abs().amax(0))
        s = torch.from_numpy(D.smoothing_factors(ch[p_ + "gate"].numpy(), wr.numpy(), alpha))
        S[p_ + "gate"] = S[p_ + "up"] = s
        heads = layer.self_attn.config.num_attention_heads
        xa = ch[p_ + "o"].reshape(heads, -1).amax(0).numpy()
        wr = W(p_ + "o").abs().amax(0).reshape(heads, -1).amax(0).numpy()
        S[p_ + "o"] = torch.from_numpy(np.tile(D.smoothing_factors(xa, wr, alpha), heads))
        S[p_ + "down"] = torch.from_numpy(D.smoothing_factors(ch[p_ + "down"].numpy(), W(p_ + "down").abs().amax(0).numpy(), alpha))
    return S


def load_calib_exp() -> None:
    p = G.cache_dir / "calib_exp.pt"
    if not p.exists():
        frames = G.args.calib_frames
        overlap = {fid(f) for f in frames} & {fid(f) for f in G.args.eval_frames}
        assert not overlap, f"calibration and evaluation frames overlap: {overlap}"
        configure_expert(dict(sel="none"))
        rec = CalibAmax()
        for ql in G.QX.values():
            ql.rec = rec
        t0 = time.time()
        for fr in frames:
            cap = get_capture(fr)
            run_expert(cap["K"], cap["V"], cap["valid"], cap)
        for ql in G.QX.values():
            ql.rec = None
        torch.save({"ch_amax": rec.ch, "frames": [fid(f) for f in frames]}, p)
        log(f"expert calibration on {len(frames)} frames (fp32 capture KV) in {time.time() - t0:.0f} s")
    ch = torch.load(p, weights_only=False)["ch_amax"]
    G.SMOOTH_X = layer_smoothing("exp", [pair.expert_layer for pair in G.pwe.joint_layers], G.QX, ch, G.args.alpha)


def get_ws_x(kind: str) -> dict:
    if kind in G.WSX:
        return G.WSX[kind]
    path = G.cache_dir / f"ws_exp_{kind}.pt"
    if path.exists():
        G.WSX[kind] = torch.load(path, weights_only=False)
        return G.WSX[kind]
    t0 = time.time()
    sm = G.SMOOTH_X if kind == "smooth" else {}
    ws = {}
    for name, ql in G.QX.items():
        w = ql.w2d()
        if name in sm:
            w = w * sm[name][None, :]
        scale = (mse_clip_rows(w) / 127.0).float()
        ws[name] = (quant_codes(w.float(), scale), scale)
    log(f"expert weight set {kind}: {len(ws)} GEMMs in {time.time() - t0:.0f} s")
    if G.args.save_ws:
        torch.save(ws, path)
    G.WSX[kind] = ws
    return ws


def expert_alpha_override(name: str, alpha: float) -> tuple:
    """SmoothQuant factor, INT8 codes and scale of one expert layer GEMM at a different alpha (cached)."""
    key = (round(float(alpha), 4), name)
    if key not in G.WSX_ALPHA:
        if key[0] not in G.SMOOTH_X_ALPHA:
            ch = torch.load(G.cache_dir / "calib_exp.pt", weights_only=False)["ch_amax"]
            G.SMOOTH_X_ALPHA[key[0]] = layer_smoothing("exp", [pair.expert_layer for pair in G.pwe.joint_layers], G.QX, ch, key[0])
        s = G.SMOOTH_X_ALPHA[key[0]][name]
        w = G.QX[name].w2d() * s[None, :]
        scale = (mse_clip_rows(w) / 127.0).float()
        G.WSX_ALPHA[key] = (s, quant_codes(w.float(), scale), scale)
    return G.WSX_ALPHA[key]


def configure_expert(spec: dict) -> None:
    """spec keys beyond sel/w/a/smooth/attn: alpha_layers {"0": 0.3} = SmoothQuant alpha override per expert layer;
    a_groups {regex: G} = per-token activation scales split into G contiguous column groups for matching GEMMs."""
    names = select_x(spec.get("sel", "none"))
    smooth = bool(spec.get("smooth"))
    alpha_layers = {str(k): float(v) for k, v in (spec.get("alpha_layers") or {}).items()}
    a_groups = spec.get("a_groups") or {}
    if smooth and not G.SMOOTH_X:
        load_calib_exp()
    for name, ql in G.QX.items():
        on = name in names
        use_s = on and smooth and name in G.SMOOTH_X
        ql.wq = on and spec.get("w", True)
        ql.aq = spec.get("a", "token") if on else "none"
        m = re.match(r"exp\.L(\d+)\.", name)
        a_ovr = alpha_layers.get(m.group(1)) if (use_s and m is not None) else None
        if a_ovr is not None:
            s_fac, codes, wscale = expert_alpha_override(name, a_ovr)
            ql.smooth = s_fac
            ql.codes, ql.wscale = (codes, wscale) if ql.wq else (None, None)
        else:
            ql.smooth = G.SMOOTH_X[name] if use_s else None
            ql.codes, ql.wscale = get_ws_x("smooth" if use_s else "base")[name] if ql.wq else (None, None)
        ql.a_groups = next((int(g) for rx, g in a_groups.items() if on and re.search(rx, name)), None)
        ql.static_scale, ql.rec = None, None
    ATTN["exp_mode"] = spec.get("attn", "fp")
    ATTN["exp_cfg"] = spec.get("attn_cfg", ATTN_INT8)


_STD = {}


def unnorm_std() -> np.ndarray:
    if "std" not in _STD:
        from safetensors import safe_open
        with safe_open(os.path.join(CKPT, "policy_postprocessor_step_0_unnormalizer_processor.safetensors"), "np") as f:
            _STD["std"] = f.get_tensor([k for k in f.keys() if k.endswith("std")][0]).astype(np.float64)
    return _STD["std"]


def e2e_kv(v: str, fr) -> tuple:
    """Prefix KV of variant v for frame fr: the capture for fp32, else the e2e cache (computed if missing)."""
    if v == "fp32":
        cap = get_capture(fr)
        return cap["K"], cap["V"], cap["valid"]
    p = G.cache_dir / "e2e_kv" / f"{v}_{fr[0]}_f{fr[1]:02d}.pt"
    if not p.exists():
        spec, kw = VARIANTS[v]
        ref = get_ref(fr)
        configure(spec)
        out = prefix_compact(get_inputs(fr), image_embs=ref["img_embs"] if kw.get("lm_only") else None)
        configure(dict(sel="none"))
        p.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"K": out["K"], "V": out["V"], "valid": out["valid"]}, p)
    z = torch.load(p, weights_only=False)
    return z["K"], z["V"], z["valid"]


def expert_eval(frames, prefix_v: str, espec: dict) -> list:
    configure_expert(espec)
    recs = []
    for fr in frames:
        cap = get_capture(fr)
        K, V, valid = e2e_kv(prefix_v, fr)
        t1 = time.time()
        a = run_expert(K, V, valid, cap)
        e = D.chunk_error(a, cap["actions"])
        e.update(joint_metrics(a, cap["actions"], unnorm_std()))
        e["t_s"] = time.time() - t1
        recs.append({"frame": fid(fr), **e})
    configure_expert(dict(sel="none"))
    return recs


def summarize_chunks(recs: list) -> dict:
    e = np.array([r["rel_rms_7"] for r in recs])
    c = np.array([r["cos_7"] for r in recs])
    jm = np.array([r["joint_max_abs_rad"] for r in recs])
    jr = np.array([r["joint_rms_rad"] for r in recs])
    return {"frames": len(recs), "rel_rms_7_mean": float(e.mean()), "rel_rms_7_max": float(e.max()), "cos_7_min": float(c.min()),
            "joint_max_abs_rad_mean": float(jm.mean()), "joint_max_abs_rad_max": float(jm.max()),
            "joint_rms_rad_mean": float(jr.mean()), "joint_rms_rad_max": float(jr.max()),
            "pass_g3": bool(e.mean() <= GATE_G3 and e.max() <= GATE_G3_FRAME and c.min() >= GATE_G3_COS),
            "pass_g3_target_1pct": bool(e.max() <= 0.01),
            "pass_g4_rms": bool(jr.max() <= 0.005), "pass_g4_max": bool(jm.max() <= 0.02),
            "frames_joint_max_over_0p02_rad": int((jm > 0.02).sum()), "t_s_mean": float(np.mean([r["t_s"] for r in recs]))}


def stage_e2e_dyn_expert() -> None:
    frames = G.args.eval_frames
    combos = [tuple(c.split(":")) for c in G.args.dyn_combos]
    t0 = time.time()
    out = {"note": "action expert fake-quantised in torch with the prefix recipe (W8 per output channel MSE clip, A8 per token "
                   "dynamic, dequant = acc x s_row x s_col, no INT8 output requant, biases fp32); attention, norms, SiLU/GELU, "
                   "RoPE, residuals and the Euler update fp32; prefix KV from the capture (fp32) or the e2e cache; G3 vs "
                   "actions_fp32 of the capture, joint errors in rad through the unnormaliser std",
           "expert_specs": {k: {"spec": EXPERT_SPECS[k], "label": EXPERT_LABEL.get(k, "")} for k in {e for _, e in combos}},
           "combos": {}}
    for prefix_v, expert_v in combos:
        name = f"{prefix_v}+{expert_v}"
        t1 = time.time()
        recs = expert_eval(frames, prefix_v, EXPERT_SPECS[expert_v])
        s = summarize_chunks(recs)
        out["combos"][name] = {"prefix": prefix_v, "expert": expert_v, "wall_s": time.time() - t1, "records": recs, "summary": s}
        log(f"  e2e_dyn {name}: G3 mean {s['rel_rms_7_mean'] * 100:.3f}% max {s['rel_rms_7_max'] * 100:.3f}% | joint max "
            f"{s['joint_max_abs_rad_max']:.4f} rms {s['joint_rms_rad_max']:.4f} rad | {s['t_s_mean']:.1f} s/chunk")
        write_json("e2e_dyn_expert", {**out, "partial": True})
    sp = G.out_dir / "e2e_int8_expert.json"
    if sp.exists():
        out["static_int8_expert_reference"] = json.loads(sp.read_text())["summary"]
    out["wall_s"] = time.time() - t0
    write_json("e2e_dyn_expert", out)


def stage_iso_exp(stage: str) -> None:
    nL = len(G.pwe.joint_layers)
    if stage == "iso_exp_proj":
        frames = G.args.eval_frames
        runs = {f"exp.*.{s}": f"^exp\\.L\\d+\\.{s}$" for s in LM_SLOTS}
        runs.update({f"exp.{s}": f"^exp\\.{s}$" for s, _ in EXP_PROJ})
    else:
        frames = G.args.iso_exp_frames or G.args.eval_frames
        runs = {f"exp.L{li}": f"^exp\\.L{li}\\." for li in range(nL)}
    t0 = time.time()
    out = {}
    for run, rx in runs.items():
        recs = expert_eval(frames, "fp32", dict(sel=rx, w=True, a="token"))
        s = summarize_chunks(recs)
        out[run] = {"regex": rx, "gemms": len(select_x(rx)), "records": recs, "summary": s}
        log(f"  {stage} {run}: G3 mean {s['rel_rms_7_mean'] * 100:.3f}% max {s['rel_rms_7_max'] * 100:.3f}%")
        write_json(stage, {"frames": [fid(f) for f in frames], "prefix_kv": "fp32 capture", "partial": True, "runs": out})
    write_json(stage, {"frames": [fid(f) for f in frames], "prefix_kv": "fp32 capture", "wall_s": time.time() - t0, "runs": out})


# ---- generic runs: prefix variant x prefix attention x expert x expert attention x hardware formats ----
# A runs file is JSON {"note": str, "runs": {name: spec}}, spec keys:
#   prefix   fp32 (capture KV unless pattn/fmt are set) | fp32h (this harness's fp32 prefix KV) | a VARIANTS name
#   pattn    null | {"cfg": {ATTN_INT8 overrides}, "sel": regex on vis.L/lm.L names, "layer_cfg": {name: cfg|null}}
#   expert   an EXPERT_SPECS name | int8_static (the deployed numpy INT8 expert, --numerics)
#   eattn    like pattn, for the expert (exp.L names)
#   fmt      null | {"row_enc": .., "out_bf16": bool, "iface_bf16": bool, "tables": bool, "acc": bool}
#   base     name of another run in the same file: adds the paired chunk distance to that run's actions
class fmt_ctx:
    def __init__(self, fmt: dict | None, rec: AccRec | None = None):
        self.fmt, self.rec = dict(fmt or {}), rec

    def __enter__(self):
        FMT.update(row_enc=self.fmt.get("row_enc", "fp"), out_bf16=bool(self.fmt.get("out_bf16")),
                   iface_bf16=bool(self.fmt.get("iface_bf16")), tables=bool(self.fmt.get("tables")),
                   acc=self.rec if self.fmt.get("acc") else None)
        refresh_siglip_impl()
        return self

    def __exit__(self, *exc):
        FMT.update(row_enc="fp", out_bf16=False, iface_bf16=False, tables=False, acc=None)
        refresh_siglip_impl()
        return False


def _attn_spec(attn: dict | None) -> tuple:
    if attn is None:
        return None, None, {}
    return {**ATTN_INT8, **attn.get("cfg", {})}, attn.get("sel"), attn.get("layer_cfg", {})


def prefix_kv_run(prefix: str, pattn, fr, fmt, rec=None) -> tuple:
    fmt = {k: v for k, v in (fmt or {}).items() if v not in (None, False, "fp")}
    if prefix == "fp32" and pattn is None and not fmt:
        cap = get_capture(fr)
        return cap["K"], cap["V"], cap["valid"]
    if prefix == "fp32h" and pattn is None and not fmt:
        ref = get_ref(fr)
        return ref["K"], ref["V"], ref["valid"]
    kvdir = G.cache_dir / "e2e_kv"
    if pattn is None and not fmt:
        p = kvdir / f"{prefix}_{fr[0]}_f{fr[1]:02d}.pt"                   # the e2e_int8_expert cache
    else:
        import hashlib
        key = hashlib.sha1(json.dumps([prefix, pattn, fmt], sort_keys=True).encode()).hexdigest()[:12]
        p = kvdir / f"{prefix}~{key}_{fr[0]}_f{fr[1]:02d}.pt"
    if p.exists() and not fmt.get("acc"):
        z = torch.load(p, weights_only=False)
        return z["K"], z["V"], z["valid"]
    ref = get_ref(fr)
    spec, kw = (dict(sel="none"), {}) if prefix in ("fp32", "fp32h") else (dict(VARIANTS[prefix][0]), VARIANTS[prefix][1])
    cfg, sel, layer_cfg = _attn_spec(pattn)
    if cfg is not None:
        spec.update(attn="int8", attn_cfg=cfg)
    with fmt_ctx(fmt, rec):
        configure(spec)
        ATTN["sel"], ATTN["layer_cfg"] = sel, layer_cfg
        out = prefix_compact(get_inputs(fr), image_embs=ref["img_embs"] if kw.get("lm_only") else None)
        ATTN["sel"], ATTN["layer_cfg"] = None, {}
        configure(dict(sel="none"))
    if not fmt.get("acc"):
        p.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"K": out["K"], "V": out["V"], "valid": out["valid"]}, p)
    return out["K"], out["V"], out["valid"]


def expert_actions_run(expert: str, eattn, fr, K, V, valid, fmt, rec=None, opts=None) -> np.ndarray:
    cap = get_capture(fr)
    if expert == "int8_static":
        if G.hm is None:
            G.hm, _ = load_int8_expert(G.args.numerics)
        tn = lambda t: t.numpy() if isinstance(t, torch.Tensor) else np.asarray(t)  # noqa: E731
        return int8_expert_actions(G.hm, tn(K), tn(V), tn(valid), cap)
    espec = dict(EXPERT_SPECS[expert])
    espec.update(opts or {})
    cfg, sel, layer_cfg = _attn_spec(eattn)
    if cfg is not None:
        espec.update(attn="int8", attn_cfg=cfg)
    with fmt_ctx(fmt, rec):
        configure_expert(espec)
        ATTN["sel"], ATTN["layer_cfg"] = sel, layer_cfg
        a = run_expert(K, V, valid, cap)
        ATTN["sel"], ATTN["layer_cfg"] = None, {}
        configure_expert(dict(sel="none"))
    return a


def summarize_run(recs: list) -> dict:
    s = summarize_chunks(recs)
    e = np.array([r["rel_rms_7"] for r in recs])
    jm = np.array([r["joint_max_abs_rad"] for r in recs])
    jr = np.array([r["joint_rms_rad"] for r in recs])
    s.update(rel_rms_7_p95=float(np.percentile(e, 95)), rel_rms_7_median=float(np.median(e)),
             joint_max_abs_rad_p95=float(np.percentile(jm, 95)), joint_rms_rad_p95=float(np.percentile(jr, 95)),
             worst_frame=recs[int(e.argmax())]["frame"])
    if "vs_base_rel_rms_7" in recs[0]:
        d = np.array([r["vs_base_rel_rms_7"] for r in recs])
        dg = np.array([r["delta_g3_vs_base"] for r in recs])
        s.update(vs_base_rel_rms_7_mean=float(d.mean()), vs_base_rel_rms_7_max=float(d.max()),
                 delta_g3_mean=float(dg.mean()), delta_g3_max=float(dg.max()), delta_g3_min=float(dg.min()))
    return s


def _acts_path(out_name: str, run: str) -> Path:
    return G.cache_dir / "actions" / out_name / f"{run}.npz"


def stage_runs() -> None:
    a = G.args
    spec_all = json.loads(Path(a.runs_file).read_text())
    runs = spec_all["runs"]
    out_name = a.runs_out or Path(a.runs_file).stem
    frames = a.eval_frames
    fids = [fid(f) for f in frames]
    names = [n for n in runs if (not a.run_filter or re.search(a.run_filter, n))]
    p = G.out_dir / f"{out_name}.json"
    out = {"runs_file": str(a.runs_file), "note": spec_all.get("note", ""), "frames": fids, "runs": {}}
    if p.exists():
        old = json.loads(p.read_text())
        if old.get("frames") == fids:
            out["runs"] = {n: v for n, v in old.get("runs", {}).items() if n in runs and v.get("spec") == runs[n]
                           and _acts_path(out_name, n).exists()}
    t0 = time.time()

    def do(name: str):
        if name in out["runs"]:
            return
        r = runs[name]
        base = r.get("base")
        base_acts = None
        if base:
            do(base)
            base_acts = dict(np.load(_acts_path(out_name, base)))
        rec = AccRec() if (r.get("fmt") or {}).get("acc") else None
        TBL_HITS.clear()
        ATTN["calls"].clear()
        t1 = time.time()
        recs, acts = [], {}
        for fr in frames:
            cap = get_capture(fr)
            t2 = time.time()
            K, V, valid = prefix_kv_run(r["prefix"], r.get("pattn"), fr, r.get("fmt"), rec)
            t_prefix = time.time() - t2
            t2 = time.time()
            act = expert_actions_run(r["expert"], r.get("eattn"), fr, K, V, valid, r.get("fmt"), rec, r.get("expert_opts"))
            e = D.chunk_error(act, cap["actions"])
            e.update(joint_metrics(act, cap["actions"], unnorm_std()))
            e.update(t_s=time.time() - t2, t_prefix_s=t_prefix)
            key = f"{fr[0]}_f{fr[1]:02d}"
            if base_acts is not None:
                b = base_acts[key]
                e["vs_base_rel_rms_7"] = float(np.linalg.norm(act[:, :7] - b[:, :7]) / np.linalg.norm(cap["actions"][:, :7]))
                e["delta_g3_vs_base"] = e["rel_rms_7"] - float(np.linalg.norm(b[:, :7] - cap["actions"][:, :7]) /
                                                                np.linalg.norm(cap["actions"][:, :7]))
            acts[key] = act
            recs.append({"frame": fid(fr), **e})
        s = summarize_run(recs)
        _acts_path(out_name, name).parent.mkdir(parents=True, exist_ok=True)
        np.savez(_acts_path(out_name, name), **acts)
        out["runs"][name] = {"spec": r, "wall_s": time.time() - t1, "records": recs, "summary": s,
                             "attn_calls": len(ATTN["calls"]), "table_hits": dict(TBL_HITS)}
        if rec is not None:
            out["runs"][name]["acc"] = rec.summary()
        log(f"  run {name}: G3 mean {s['rel_rms_7_mean'] * 100:.3f}% p95 {s['rel_rms_7_p95'] * 100:.3f}% max "
            f"{s['rel_rms_7_max'] * 100:.3f}% | joint max {s['joint_max_abs_rad_max']:.4f} rms {s['joint_rms_rad_max']:.4f} rad"
            + (f" | vs base {s['vs_base_rel_rms_7_mean'] * 100:.3f}% dG3 {s['delta_g3_mean'] * 100:+.3f}%" if base else "")
            + f" | {time.time() - t1:.0f} s")
        write_json(out_name, {**out, "partial": True})

    for n in names:
        do(n)
    out["wall_s"] = time.time() - t0
    write_json(out_name, out)


def stage_attn_local() -> None:
    """Local INT8-attention error per module (SigLIP, Gemma, expert) on the W8A8 pipeline: prefix --local-prefix
    with every attention in compare mode (fp32 result propagated), then expert dyn_smooth in compare mode on that
    KV.  cfg = ATTN_INT8 with --cmp-cfg overrides; rel = sqrt(sum err^2 / sum y^2) over calls; dead = softmax
    mass per row whose P code is 0."""
    a = G.args
    frames = a.local_frames or a.eval_frames
    base = {**ATTN_INT8, **json.loads(a.cmp_cfg)}
    ATTN["cmp_cfg"] = base
    ATTN["acc"].clear()
    t0 = time.time()
    spec, kw = dict(VARIANTS[a.local_prefix][0]), VARIANTS[a.local_prefix][1]
    spec["attn"] = "compare"
    for fr in frames:
        ref, cap = get_ref(fr), get_capture(fr)
        configure(spec)
        out = prefix_compact(get_inputs(fr), image_embs=ref["img_embs"] if kw.get("lm_only") else None)
        configure(dict(sel="none"))
        configure_expert({**EXPERT_SPECS["dyn_smooth"], "attn": "compare"})
        run_expert(out["K"], out["V"], out["valid"], cap)
        configure_expert(dict(sel="none"))
        log(f"  attn_local {fid(fr)} {time.time() - t0:.0f} s")
    ATTN["cmp_cfg"] = None
    res = {n: {"local_rel_both": (v["both"] / v["y2"]) ** 0.5, "local_rel_qk": (v["qk"] / v["y2"]) ** 0.5,
               "local_rel_pv": (v["pv"] / v["y2"]) ** 0.5, "p_dead_mass_mean": v["dead_sum"] / max(v["rows"], 1),
               "p_dead_mass_max": v["dead_max"]} for n, v in ATTN["acc"].items()}
    name = a.runs_out or "attn_local"
    write_json(name, {"frames": [fid(f) for f in frames], "prefix": a.local_prefix, "expert": "dyn_smooth", "cfg": base,
                      "wall_s": time.time() - t0, "modules": res})
    log(f"{name} done in {time.time() - t0:.0f} s")


def stage_fmt_selftest() -> None:
    """The hardware-format emulation against its references: bf16_t == pi0_ae_model.to_bf16, the table gates and
    rsqrt against the bit-level numpy vector unit (paper/sw/vector_unit_ref.py), tbl_softmax against torch,
    row-scale encodings bounds, and FMT off == the plain fake-quant."""
    import pi0_ae_model as M
    import vector_unit_ref as VU

    res = {}
    g = torch.Generator().manual_seed(0)
    x = torch.randn(200000, generator=g) * torch.exp(torch.randn(200000, generator=g) * 3)
    res["bf16_t_vs_to_bf16_max_abs_diff"] = float(np.abs(bf16_t(x).numpy() - M.to_bf16(x.numpy())).max())
    xs = (torch.randn(100000, generator=g) * 4).float()
    xb = torch.from_numpy(M.to_bf16(xs.numpy()))
    codes = (xb.numpy().astype(np.float32).view(np.uint32) >> 16).astype(np.uint16)    # the same rounded bf16 value
    for nm, op, gate in (("gelu", VU.op_gelu, "gelu"), ("silu", VU.op_silu, "sigm")):
        _, y = op(codes)
        mine = (xb.double() * tbl_gate(xb, gate)).numpy()
        ref_f = xb.double().numpy() * (VU._phi_gelu(xb.double().numpy()) if gate == "gelu" else VU._sigmoid(xb.double().numpy()))
        res[f"{nm}_tbl_vs_vector_unit_max_abs"] = float(np.abs(mine - y.value()).max())
        res[f"{nm}_tbl_vs_exact_max_abs"] = float(np.abs(mine - ref_f).max())
        res[f"{nm}_vector_unit_vs_exact_max_abs"] = float(np.abs(y.value() - ref_f).max())
    v = torch.exp(torch.randn(100000, generator=g, dtype=torch.float64) * 5).float()
    vx = VU.unpack_fp32(v.numpy().astype(np.float32).view(np.uint32))
    mine = tbl_rsqrt(v).numpy()
    res["rsqrt_tbl_vs_vector_unit_max_rel"] = float(np.abs(mine / VU.rsqrt(vx).value() - 1).max())
    res["rsqrt_tbl_vs_exact_max_rel"] = float(np.abs(mine * np.sqrt(v.double().numpy()) - 1).max())
    w = torch.randn(4, 8, 51, 867, generator=g) * 6
    w[..., 700:] = torch.finfo(torch.float32).min
    ps, pt = tbl_softmax(w), torch.softmax(w, -1)
    res["softmax_tbl_vs_torch_max_abs"] = float((ps - pt).abs().max())
    res["softmax_tbl_vs_torch_max_rel_where_p_gt_1e-6"] = float(((ps - pt).abs() / pt)[pt > 1e-6].max())
    res["softmax_tbl_row_sum_max_dev"] = float((ps.double().sum(-1) - 1).abs().max())
    s = torch.exp(torch.randn(10000, generator=g) * 4)
    for mode, lim in (("m8", 1 + 1 / 128), ("m6", 1 + 1 / 32), ("m4", 1 + 1 / 8), ("pow2", 2.0)):
        FMT["row_enc"] = mode
        r = (enc_scale(s) / s).double()
        res[f"row_enc_{mode}_ratio_min_max"] = [float(r.min()), float(r.max())]
        res[f"row_enc_{mode}_ok"] = bool(r.min() >= 1 - 1e-6 and r.max() <= lim * (1 + 1e-6))
    FMT["row_enc"] = "fp"
    # plumbing on one frame: every table and bf16 site is hit, and FMT off is the plain fake-quant
    fr = G.args.eval_frames[0]
    cap = get_capture(fr)
    K, V, valid = prefix_kv_run("smooth", None, fr, None)
    a0 = expert_actions_run("dyn_smooth", None, fr, K, V, valid, None)
    out_old = json.loads((HERE.parent / "data" / "prefix_w8a8" / "e2e_dyn_expert.json").read_text())
    rec_old = {r["frame"]: r["rel_rms_7"] for r in out_old["combos"].get("smooth_lm+dyn_smooth", {}).get("records", [])}
    K1, V1, valid1 = prefix_kv_run("smooth_lm", None, fr, None)
    a1 = expert_actions_run("dyn_smooth", None, fr, K1, V1, valid1, None)
    res["fmt_off_smooth_lm_dyn_smooth_rel_rms_7"] = D.chunk_error(a1, cap["actions"])["rel_rms_7"]
    res["same_in_e2e_dyn_expert_json"] = rec_old.get(fid(fr))
    TBL_HITS.clear()
    ATTN["calls"].clear()
    full = {"row_enc": "m8", "iface_bf16": True, "tables": True, "acc": True}
    rec = AccRec()
    K2, V2, valid2 = prefix_kv_run("smooth", {"cfg": {"v_gran": "fold"}}, fr, full, rec)
    res["table_hits_prefix"] = dict(TBL_HITS)
    res["attn_calls_prefix"] = len(ATTN["calls"])
    TBL_HITS.clear()
    ATTN["calls"].clear()
    a2 = expert_actions_run("dyn_smooth", {"cfg": {"v_gran": "fold"}}, fr, K2, V2, valid2, full, rec)
    res["table_hits_expert"] = dict(TBL_HITS)
    res["attn_calls_expert"] = len(ATTN["calls"])
    res["acc"] = rec.summary()
    res["smooth_dyn_smooth_rel_rms_7"] = D.chunk_error(a0, cap["actions"])["rel_rms_7"]
    res["smooth_fold_all_formats_rel_rms_7"] = D.chunk_error(a2, cap["actions"])["rel_rms_7"]
    res["frame"] = fid(fr)
    write_json("fmt_selftest", res)
    log("fmt_selftest: " + json.dumps(_jsonable({k: v for k, v in res.items() if k != "acc"}))[:3000])


# ======================================================================================
def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--eval", nargs="+", default=["demo1_ep20:2,8", "demo1_ep40:2,8", "recov_pi0_ep00:2,8", "recov_pi0_ep10:2,8"])
    p.add_argument("--calib", nargs="+", default=[f"demo1_ep0{i}:5" for i in range(1, 9)])
    p.add_argument("--local-frames", nargs="+", default=None)
    p.add_argument("--iso-frames", nargs="+", default=None)
    p.add_argument("--stages", default="selftest,fp32,base")
    p.add_argument("--alpha", type=float, default=0.5)
    p.add_argument("--fallback-k", type=int, nargs="+", default=[1, 2, 4, 8])
    p.add_argument("--fallback-bases", nargs="+", default=["base", "smooth"])
    p.add_argument("--fallback-target", type=float, default=0.01, help="G3 rel RMS 7 max the fallback must reach")
    p.add_argument("--e2e-variants", nargs="+", default=["base", "smooth_lm"])
    p.add_argument("--dyn-combos", nargs="+", default=["fp32:dyn", "base:dyn", "smooth_lm:dyn", "fp32:dyn_smooth",
                                                         "smooth_lm:dyn_smooth", "smooth_lm:dyn_smooth_attn"],
                   help="prefix_variant:expert_variant pairs for e2e_dyn_expert")
    p.add_argument("--iso-exp-frames", nargs="+", default=None)
    p.add_argument("--numerics", default=os.path.expanduser("~/pi0_glue/numerics/ur7e_m2mse.npz"),
                   help="expert numerics the deployed session was exported from (see its receipt.json)")
    p.add_argument("--custom-name", default="custom")
    p.add_argument("--custom-fp", default="^$")
    p.add_argument("--custom-base", default="base", choices=("base", "smooth", "attn"))
    p.add_argument("--runs-file", default=None, help="stage runs: JSON {note, runs: {name: spec}} (see stage_runs)")
    p.add_argument("--runs-out", default=None, help="stage runs / attn_local: output JSON name (default: runs file stem)")
    p.add_argument("--run-filter", default=None, help="stage runs: regex on run names (bases are pulled in)")
    p.add_argument("--local-prefix", default="smooth", help="stage attn_local: prefix variant")
    p.add_argument("--cmp-cfg", default="{}", help="stage attn_local: JSON overrides of ATTN_INT8 for the compare mode")
    p.add_argument("--no-g3", action="store_true")
    p.add_argument("--save-ws", action="store_true", help="cache the INT8 weight sets under --cache (2.4 GB each)")
    p.add_argument("--out", default=str(REPO / "paper" / "data" / "prefix_w8a8"))
    p.add_argument("--cache", default=os.environ.get("PREFIX_W8A8_CACHE", "/tmp/prefix_w8a8_cache"))
    a = p.parse_args()
    a.eval_frames = parse_frames(a.eval)
    a.calib_frames = parse_frames(a.calib)
    a.local_frames = parse_frames(a.local_frames) if a.local_frames else None
    a.iso_frames = parse_frames(a.iso_frames) if a.iso_frames else None
    a.iso_exp_frames = parse_frames(a.iso_exp_frames) if a.iso_exp_frames else None
    G.args = a
    G.out_dir = Path(a.out)
    G.out_dir.mkdir(parents=True, exist_ok=True)
    G.cache_dir = Path(a.cache)
    G.cache_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    G.policy, missing, unexpected = load_policy(CKPT)
    G.pwe = G.policy.model.paligemma_with_expert
    install_wrappers()
    install_expert_wrappers()
    log(f"policy mmap-loaded in {time.time() - t0:.0f} s (missing {len(missing)}, unexpected {len(unexpected)}), "
        f"{len(G.QL)} GEMM wrappers, threads {torch.get_num_threads()}, rss {rss_gb():.1f} GB")
    for stage in a.stages.split(","):
        stage = stage.strip()
        ts = time.time()
        if stage == "selftest":
            stage_selftest()
        elif stage == "fp32":
            stage_fp32()
        elif stage == "calib":
            stage_calib()
        elif stage in VARIANTS:
            stage_variant(stage)
        elif stage == "local":
            stage_local()
        elif stage in ("iso_exp_proj", "iso_exp_layer"):   # before the generic iso_ branch (it used to swallow these)
            stage_iso_exp(stage)
        elif stage.startswith("iso_"):
            stage_iso(stage)
        elif stage == "fallback":
            stage_fallback()
        elif stage == "custom":
            stage_custom()
        elif stage == "e2e_int8_expert":
            stage_e2e_int8_expert()
        elif stage == "e2e_dyn_expert":
            stage_e2e_dyn_expert()
        elif stage in ("iso_exp_proj", "iso_exp_layer"):
            stage_iso_exp(stage)
        elif stage == "runs":
            stage_runs()
        elif stage == "attn_local":
            stage_attn_local()
        elif stage == "fmt_selftest":
            stage_fmt_selftest()
        else:
            raise ValueError(f"unknown stage {stage}")
        log(f"=== {stage} finished in {time.time() - ts:.0f} s, peak rss {rss_gb():.1f} GB")
    log("ALL_DONE")


if __name__ == "__main__":
    main()
