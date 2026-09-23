# pi0 full-model W8A8 numerics (workstream A, 2026-09-15)

Harness: `paper/sw/prefix_w8a8_eval.py` (torch fake-quant of the real LeRobot pi0; stages `runs`, `attn_local`,
`fmt_selftest` added for this section). Run files: `paper/sw/prefix_w8a8_runs/`. [M] = measured, [I] = inferred.

Scheme under test: weights INT8 per output channel (MSE clip), activations INT8 per token dynamic, dequant
acc x s_row x s_col, SmoothQuant alpha = 0.5 on the Gemma LM (and, `dyn_smooth`, on the expert).

Frames:
- 46 held-out frames = the deployed model's validation set (`paper/sw/results/deploy_calib_heldout.json`):
  demo1_ep20, demo1_ep40, recov_pi0_ep00, recov_pi0_ep10, all frames.
- 8-frame subset: frames 2 and 8 of the same episodes.
- SmoothQuant calibration: demo1_ep01..08 frame 5, disjoint from both (asserted).

G3 = 50x7 action chunk rel RMS vs `actions_fp32` of the capture. G4 = joint-space error in rad through the
unnormaliser std. "Chunk distance" = rel RMS of (chunk - base chunk) / |fp32 chunk|. It is NOT a cost measure:
dynamic per-token rounding is discontinuous, so any change that flips rounding decisions re-draws the quantisation
noise of every downstream GEMM and moves the chunk by ~0.7-0.9 % on its own. Measured: the vector unit's tables move
the fp32 chunk by 1.7e-6 but the W8A8 chunk by 0.83 %, at ΔG3 -0.006 %. The frame-averaged ΔG3 against a base run is
the cost; chunk distance is reported beside it as the noise level.

## 1. The whole pipeline on 46 frames [M]

| pipeline | G3 mean / p95 / max % | G4 joint max-abs max, rad | G4 joint RMS max, rad |
|---|---|---|---|
| fp32 (harness vs capture) | 0.000 / 0.000 / 0.000 | 0.0000 | 0.0000 |
| smooth_lm prefix + dyn expert | 1.080 / 1.822 / 2.988 | 0.0154 | 0.0041 |
| smooth_lm + dyn_smooth | 0.792 / 1.261 / 1.899 | 0.0130 | 0.0031 |
| smooth_lm + dyn_smooth + INT8 attention V per key, expert only | 1.177 / 1.867 / 4.146 | 0.0168 | 0.0044 |
| smooth_lm + dyn_smooth + INT8 attention V per key, prefix + expert | 1.202 / 2.022 / 2.703 | 0.0141 | 0.0045 |
| smooth_lm + dyn_smooth + INT8 attention uint8 P / V per channel, expert only | 0.941 / 1.453 / 2.725 | 0.0156 | 0.0037 |
| smooth_lm + dyn_smooth + INT8 attention uint8 P / V per channel, prefix + expert | 0.930 / 1.469 / 2.092 | 0.0150 | 0.0041 |
| smooth (SigLIP W8A8 too) + dyn_smooth | 0.830 / 1.339 / 1.625 | 0.0129 | 0.0031 |
| **chip candidate: smooth + dyn_smooth + INT8 attention uint8 P / V per channel everywhere** | **0.938 / 1.507 / 1.686** | **0.0132** | **0.0034** |
| deployed static INT8 expert, fp32 prefix | 1.859 / 3.406 / 3.841 | 0.0265 | 0.0062 |
| deployed static INT8 expert, smooth_lm prefix | 1.866 / 2.894 / 3.948 | 0.0281 | 0.0057 |

- The all-W8A8 chip candidate halves the error of today's deployed static INT8 expert: mean G3 0.94 vs 1.86 %,
  worst frame 1.69 vs 3.84 %, joint max 0.013 vs 0.027 rad. It passes G3 on every frame, and no dynamic-recipe
  frame reaches 0.02 rad (inferred from the row maxima).
- The static expert limits its own pipeline: 1.86 % on the fp32 prefix and 1.87 % on the W8A8 prefix.
- SigLIP in W8A8 is not visible at the chunk: 0.79 → 0.83 % mean, max 1.90 → 1.63 %.
- INT8 attention in the realisable form (uint8 P, V per channel) costs +0.11 % mean and does not raise the worst
  frame. The old per-key V form costs about +0.4 % and is not one accumulation.
- 8 vs 46 frames: the 8-frame subset contains the hard frames; the 46-frame means are lower
  (smooth_lm + dyn_smooth 0.97 → 0.79 %).

## 2. Where INT8 attention costs [M]

- Realisability [I]: in PV = softmax(QK^T)·V the key index is the summation index, so V quantised per key
  (the earlier `attn*` variants) is not one integer accumulation. Two forms are: V per channel (column scale
  s_V(d)) and V per key folded into P before P is quantised. QK^T with Q and K per row is realisable.
- The cost is PV, not QK^T:
  - QK^T alone: ΔG3 −0.07 % (expert), +0.01 % (Gemma), +0.07 % (SigLIP), all at the perturbation floor.
  - Local error of QK^T ≤ 0.7 % in every module. Local PV error: SigLIP 3.6 % mean (max 12.8 %, L0),
    Gemma 8.7 % (max 25 %, L4/L5), expert 3.9 % (max 7.4 %, L1).
- Mechanism: with int8 P per row, a peaked softmax row sets the scale and the long tail rounds to code 0.
  The zero-code softmax mass is 3.0 % (SigLIP), 10.4 % (Gemma), 7.5 % (expert) on average, up to 74 % in a row.
- By part (PV, V per channel, ΔG3 / chunk distance): expert +0.32 / 1.33 %, Gemma +0.11 / 0.92 %,
  SigLIP (folded) +0.11 / 0.92 %. The expert carries most of it.
- By layer: one expert layer's attention INT8 alone is at the floor for every layer except L17 PV
  (+0.11 % / 0.91 %). The expert cost is spread over layers and compounds, so a worst-layer bf16 fallback
  does not remove it.
- Fixes (expert attention only, 8 frames, ΔG3 / chunk distance):
  - P as uint8 (255 levels): folded V +0.08 / 1.07 %, V per channel +0.10 / 0.99 %.
  - One P scale per 16 / 64 keys: −0.05 / 0.82 %, −0.04 / 0.80 %. Free, but one accumulation per group [I].
  - Top-4 P per row outside the INT8 GEMM: −0.03 / 0.82 %.
  - PV in float: −0.07 / 0.77 %.
- Choice: uint8 P per row x int8 V per channel. It is one accumulation per output, and ACX_MLP72 multmode
  5'h13 (A unsigned x B signed 8x8, UG086 Table 123) matches; ACE acceptance of that value is still open.

### 2.1 Expert GEMM localisation (per-token dynamic W8A8, fp32 prefix KV, 8 frames) [M]

The earlier `iso_exp_*.json` quantised nothing, because a dispatch bug sent those stages into the prefix iso code.
These are the re-run results.

One projection class quantised alone (G3 mean / max %):

| class | G3 mean / max % |
|---|---:|
| up | 0.585 / 1.026 |
| act_in | 0.443 / 0.733 |
| down | 0.398 / 0.646 |
| act_out | 0.389 / 0.698 |
| state | 0.328 / 0.499 |
| gate | 0.223 / 0.405 |
| v | 0.163 / 0.317 |
| tm_out | 0.157 / 0.238 |
| o | 0.148 / 0.246 |
| tm_in | 0.126 / 0.194 |
| q | 0.105 / 0.251 |
| k | 0.065 / 0.132 |

One layer quantised alone:
- **L0:** 0.751 / 1.126 %, the outlier.
- **L17:** 0.302 / 0.625 %.
- **L5:** 0.165 %; **L16:** 0.109 %.
- **All others:** 0.038–0.097 %.

Reading:
- The expert's error concentrates in its first layer, which sees the raw suffix embedding, and to a lesser degree its last layer.
- Among projections it concentrates in `up`/`down` and the action/state input and output projections [I].

Float fallback on the chip candidate (`exp_fallback`, 8 frames; float GEMMs are not realisable on the INT8 chain, so this is an upper bound):
- The chip itself gives G3 1.050 / 1.540 / 1.548 %.
- Expert L0 in float: 1.021 / 1.724 %, ΔG3 −0.028 %.
- Expert L0 + L17 in float: 1.030 / 1.761 %, ΔG3 −0.020 %.
- Expert L0 + `action_in_proj` in float: 0.926 / 1.664 %, ΔG3 −0.123 %, joint max 0.0087 rad.
- Reading: layer 0 costs 0.75 % when quantised alone, but in the full pipeline restoring it gains at most ~0.03 %.
  The per-token rounding noise of the other ~120 GEMMs dominates [I]. `action_in_proj` is worth ~0.1 %.

### 2.2 Realisable fixes on the chip candidate (`exp_l0fix`, 8 frames) [M]

Float GEMMs cannot run on the INT8 chain, so the fixes below are the realisable ones: a different SmoothQuant alpha
(folded into gains and weights, free) and per-token activation scales split into G contiguous column groups
(G sub-row passes + one DEQUANT per group + ADD).  Base `chip.m8` = the chip candidate with 2^e x m8 row scales:
G3 0.971 / 1.482 / 1.634 %, joint max 0.0087 rad.

| fix | G3 mean / p95 / max % | joint max rad | ΔG3 |
|---|---|---:|---:|
| **groups G = 16 on `action_in_proj` + `state_proj` inputs** | **0.929 / 1.405 / 1.509** | 0.0087 | **−0.042 %** |
| **groups G = 4 on the same two inputs** | **0.943 / 1.399 / 1.538** | 0.0119 | **−0.028 %** |
| float expert L0 (upper bound, NOT realisable) | 0.943 / 1.450 / 1.608 | 0.0101 | −0.028 % |
| L0 groups G = 16 + io groups G = 4 | 0.947 / 1.440 / 1.641 | 0.0129 | −0.023 % |
| SmoothQuant alpha 0.65 on expert L0 | 0.967 / 1.367 / 1.492 | 0.0100 | −0.004 % |
| L0 groups G = 64 / 16 / 4 | 0.978 / 0.980 / 1.005 mean | 0.0109–0.0122 | +0.007 / +0.009 / +0.034 % |
| SmoothQuant alpha 0.3 / 0.8 on expert L0 | 1.019 / 1.023 mean | 0.0146 / 0.0090 | +0.048 / +0.052 % |

- **The fix that works is on the input projections, not on layer 0.** Grouped scales on the 32-column `action_in_proj`
  and `state_proj` inputs recover the whole float-L0 gain at G = 4 and beat it at G = 16, with the best worst frame.
  Cost: 4 (or 16) sub-row passes on two small projections, not in the 1024-wide layer stack.
- **No fix for expert layer 0 is needed or works.** Every alpha and group size lands between −0.004 % and +0.052 %,
  while float L0 itself buys only 0.028 %; `chip.m8` is already within that of the bound. Adding L0 groups on top of
  the io groups makes it worse (−0.023 % vs −0.042 %).
- **alpha 0.65 on L0** is free on the chip and gives the best single worst frame (1.492 %) with an unchanged mean; optional.
- Why [I]: L0 is the worst layer in isolation, but once all ~130 GEMMs are quantised the summed rounding noise
  dominates, and a better scale inside one layer cannot remove noise made elsewhere. The action/state projections are
  different: they sit at the input of every denoising step, so their error enters all 10 steps.

## 3. Hardware-format sensitivity [M, 8 frames]

### 3.1 48-bit accumulation [M]

The run `A.acc` recorded the exact integer accumulation of every INT8 GEMM:
- 8 frames, SigLIP + Gemma + expert.
- Recording changes nothing: the chunks match `A` bit for bit.

| GEMM class group | measured max \|acc\| (log2) | worst class | analytic bound 127²·K (log2) |
|---|---:|---|---:|
| SigLIP (patch, q/k/v/out, fc1/fc2, mmproj) | 21.8 | vis.patch (K = 588: 40 % of its bound) | 23.2–26.0 |
| Gemma (q/k/v/o, gate/up/down) | 20.5 | lm.L0.o | 25.0–28.0 (down, K = 16384) |
| expert (layers + projections) | 20.6 | exp.L17.o | 19.0–26.0 |
| attention PV, uint8 P × int8 V per channel (`C.acc`) | 22.2 | lm.L1 PV (K = 525 keys: 28 % of its bound); expert PV 22.0 (K = 867), SigLIP PV 21.5 (K = 256) | 23.0–24.7 (255·127·T_k) |
| attention QK^T, int8 × int8 (`C.acc`) | 18.6 | vis.L0 QK (K = head_dim 72); Gemma / expert ≤ 18.4 (K = 256) | 20.1–22.0 |

- The largest accumulation over every GEMM and attention class is 2^22.2, measured in `C.acc` on 8 frames.
  That leaves at least 24.8 bits of headroom to 2^47.
- The analytic worst case over all GEMMs is 2^28 (Gemma down, K = 16384); over attention it is 2^24.7
  (expert PV with 867 keys). The headroom is therefore at least 19 bits in any case.
- A 48-bit accumulator cannot overflow; 32 bits would already hold the worst case [I].

### 3.2 Number formats [M, 8 frames]

**Criteria**
- A format is free if the frame-averaged ΔG3 ≤ 0.1 %; the worst frame is also checked.
- Chunk distance is not a cost measure. Any change that flips rounding decisions re-draws the per-token INT8
  rounding noise and moves the chunk by ~0.7–0.9 %. Measured: the tables move the fp32 chunk by 1.7e-6 but the
  W8A8 chunk by 0.83 %, with ΔG3 −0.006 %.

**Bases**
- A = smooth prefix (SigLIP W8A8 + Gemma W8A8 SmoothQuant) + dyn_smooth expert, attention in float.
  G3 0.927 / 1.527 / 1.625 % (mean / p95 / max), joint max 0.0101 rad.
- C = A + INT8 QK^T and PV (uint8 P per row, V per channel) in SigLIP, Gemma and the expert (the chip candidate).

C itself: G3 1.050 / 1.540 / 1.548 %, joint max 0.0101 rad, ΔG3 +0.122 % vs A.
With int8 P instead (B): 1.354 / 2.063 / 2.076 %, ΔG3 +0.427 %.

| format | fp32 model, G3 | A: ΔG3 mean | A: G3 max % | A: joint max rad | C: ΔG3 mean | C: G3 max % | C: joint max rad | free? |
|---|---|---:|---:|---:|---:|---:|---:|---|
| 2048-entry tables (GELU/SiLU gate, softmax exp, rsqrt) | 1.7e-6 | −0.006 % | 1.424 | 0.0100 | −0.006 % | 1.567 | 0.0115 | yes |
| bf16 at every vector-op interface (incl. residual stream) | 0.27 % mean / 0.41 % max | +0.049 % | 1.719 | 0.0100 | +0.024 % | 1.565 | 0.0095 | yes |
| GEMM (and QK/PV) result rounded to bf16 | — | +0.043 % | 1.703 | 0.0111 | +0.043 % | 1.833 | 0.0102 | yes |
| row scale 2^e × 8-bit mantissa (rounded up) | — | +0.024 % | 1.585 | 0.0106 | −0.079 % | 1.634 | 0.0087 | yes |
| row scale 2^e × 6-bit mantissa | — | −0.003 % | 1.456 | 0.0101 | — | — | — | yes |
| row scale 2^e × 4-bit mantissa | — | +0.018 % | 1.711 | **0.0173** | −0.041 % | 1.652 | **0.0134** | mean yes, worst frame no |
| row scale 2^e only | — | **+0.750 %** | **4.193** | **0.0207** | **+0.673 %** | **3.863** | 0.0150 | **no** |
| all at once: m8 + bf16 interfaces + tables | — | +0.053 % | 1.714 | 0.0094 | −0.033 % | 1.718 | 0.0107 | yes |

Findings:
- Everything tested is free except the pure power-of-two row scale.
- 2^e alone wastes up to one bit of INT8 range [I], and costs +0.75 % G3 with a joint max of 0.021 rad.
- A 6–8 bit mantissa removes that cost. 4 bits is free on the mean, but the worst-frame joint error rises
  from 0.010 to 0.017 rad.

- With INT8 attention (C) the picture is the same. The complete hardware format set costs nothing on top of the
  chip candidate: C.all_m8 gives 1.017 / 1.597 / 1.718 %, joint max 0.0107 rad.
- With uint8 P, INT8 attention itself costs about 0.1 % mean G3: +0.12 % on 8 frames, +0.11 % on 46.
  int8 P costs +0.43 %.

0.978 / 1.571 / 1.718 % G3 on 46 frames, joint max 0.0150 rad, joint RMS ≤ 0.0034 rad, 0 frames > 0.02 rad (ΔG3 +0.039 % vs the same pipeline without formats)

## 4. Recipe for the chip

**Recipe for the chip.** All three models run W8A8, SigLIP included. Measured on 46 held-out frames unless marked.

**Weights and activations**
- Weights: INT8 symmetric per output channel, MSE-optimal clip.
- Activations: INT8 symmetric per token, dynamic. The row scale is amax/127, stored as 2^e × an 8-bit mantissa,
  rounded up.
  - 6 bits is also free; 4 bits is borderline on the worst frame [8 frames].
  - A pure power-of-two scale costs +0.7 % G3 and is ruled out [8 frames].
- SmoothQuant α = 0.5 is folded into Gemma and the action expert (calibration: 8 frames of other episodes).
- No static activation scales and no INT8 output requant anywhere.
- One exception to the per-token scale: the expert's `action_in_proj` and `state_proj` inputs (32 columns) take one
  scale per group of 8 columns (G = 4), which is 4 sub-row passes on two small projections. It buys −0.028 % G3 and a
  better worst frame, as much as expert layer 0 in float, which the chain cannot do. G = 16 buys −0.042 % for 16 passes.
- No layer needs a float or bf16 fallback: in the full pipeline even the worst layer (expert L0) is worth ≤ 0.03 %.

**Attention on the chains**
- QK^T: int8 × int8 with per-row scales.
- PV: uint8 P per row × int8 V per channel, one accumulation per output.
  - ACX_MLP72 multmode 5'h13 (A unsigned × B signed); ACE acceptance is still to be confirmed.
  - The remaining ~0.1 % G3 of attention comes from the softmax tail rounding to zero. One P scale per 16–64 keys
    removes it [8 frames], at the price of extra accumulations.
  - int8 P costs 4× more, and V per key is not realisable.

**Arithmetic and interfaces**
- Accumulator: 48 bits. The measured max |acc| is 2^22.2, the analytic worst case 2^28, so there is no overflow risk.
- GEMM output: bf16(acc × s_row × s_col).
- bf16 at every vector-op interface, including the residual stream; fp32 inside the vector unit.
- GELU/SiLU gates, softmax exp (log2 domain) and rsqrt: 2048-entry linearly interpolated BRAM tables.

**Accuracy**
- Chip candidate: G3 0.938 % mean / 1.507 % p95 / 1.686 % max, joint max-abs 0.0132 rad, joint RMS ≤ 0.0034 rad,
  0 of 46 frames above 0.02 rad.
- With every hardware format applied: 0.978 / 1.571 / 1.718 % G3 on 46 frames, joint max 0.0150 rad, joint RMS ≤ 0.0034 rad, 0 frames > 0.02 rad (ΔG3 +0.039 % vs the same pipeline without formats); on 8 frames 1.017 / 1.597 / 1.718 %, joint max 0.0107 rad.
- This is half the error of today's deployed static INT8 expert on an fp32 prefix: 1.859 / 3.406 / 3.841 %,
  joint max 0.027 rad.

**Open**
- ACE acceptance of multmode 5'h13.
- The G1 per-layer KV gate (1 %) is not met by any per-token scheme (K/V errors of 8–15 %); it should stay a
  diagnostic, with G3/G4 the acceptance gates.
- The held-out set is 46 frames from 4 episodes of one task, with a single noise draw per frame.


---

# pi0 prefix W8A8: numerics measured before it is built (phase A, 2026-09-15)

Question (docs/PI0_FULL_CHIP_ARCHITECTURE_20260910.md §4): can the pi0 prefix run in W8A8 INT8 on the FPGA? The prefix is SigLIP So400m on two cameras plus PaliGemma-2B Gemma over the 525-token compact prefix.

Scheme: weights INT8 per output channel with MSE clip. Activations INT8 per token, dynamic. Norms, softmax, GELU, RoPE, residuals and biases stay in fp32.

Harness: `paper/sw/prefix_w8a8_eval.py` fake-quantises the real LeRobot torch prefix in place, 290 GEMMs.

Harness checks (`selftest.json`):
- My compact prefix is bit-identical to `PI0FpgaPolicy._run_prefix_compact`.
- fp32 KV vs the capture: ≤ 1.4e-6 relative.
- Capture KV through the torch expert reproduces `actions_fp32` to 6e-8.
- The rebuilt deployment INT8 expert reproduces `paper/sw/results/deploy_calib_heldout.json` exactly on all 8 frames.

## Findings

1. **G0 narrowly misses: 1.21 % max vs 1 %.**
   - The multi-modal projector alone accounts for 0.86 %.
   - Keeping only the projector in fp32 (`custom_down_mmproj_fp`) brings G0 to 0.87 %.
   - Weights alone give 0.31 %, so the loss is activations.
2. **G1 fails by 10–30× in every INT8 variant, in all 18 layers.**
   - `base`: K 14.7 %, V 27.0 % (layer 1).
   - Weights alone (`w8`): 4.6 %. Activations alone (`a8`): 26 %.
   - SmoothQuant α = 0.5 on Gemma: 14.9 %; LM-only 12.3 %.
   - Per-token INT8 activations cannot hold the KV cache within 1 %. The cause is the Gemma RMSNorm outputs, which carry fixed outlier channels (738, 604, 50) in every token with token amax/rms ≈ 30–43. The KV-driving projections are `v` (13.5 % KV when quantised alone), `k` and `q`.
3. **G3 with the fp32 expert passes for every per-token variant, but G3 max < 1 % needs help.**
   - `base`: 0.79 % mean / 1.82 % max.
   - `smooth`: 0.48 % / 1.03 %.
   - `smooth_lm` (fp32 SigLIP): 0.28 % / 0.57 %.
   - G1 is therefore far stricter than what the action chunk needs.
   - Per-tensor static activation scales are unusable: amax 14.6 / 42.9 %, MSE 6.5 / 25.4 %.
4. **Localisation.**
   - The dominant G3 cost is Gemma `down_proj`: quantised alone, G3 max 2.09 % and KV 18.2 %.
     - Its inputs reach amax 1,198 on the first language token (compact position 512).
     - Token amax/rms reaches 128, and channel amax max/median is ≈ 1,900.
   - Next are `v`, 0.51 %, and `up`, 0.44 %.
   - The worst single layers by G3 are L14, L12 and L11. Layer 0 has the largest KV damage (20 %) but small G3 damage.
   - SigLIP: projector 0.86 %, fc2 0.62 %, worst layer L26 at 0.49 % (G0 when quantised alone).
5. **INT8 attention**, both QKᵀ and softmax(QKᵀ)·V per row, costs little extra on G3: `attn` 0.86 / 1.79 % vs `base` 0.79 / 1.82 %. It does worsen G1 (V 31.7 %) and G0 (1.34 %).
   - Locally, almost all of the loss is in softmax·V: 8.1 % Gemma, 5.2 % SigLIP. QKᵀ alone is 0.7 %.
   - Worst layers: lm.L1 21.7 %, lm.L5 17.9 %, vis.L19 10.1 %.
6. **Minimal fallback for G3 max < 1 % (fp32 expert):**
   - Found: SmoothQuant α = 0.5 on Gemma **plus Gemma layer 14 in fp32**. G3 0.41 % mean / 0.76 % max.
   - On `base`, whole-layer fallback needs more than 8 layers: top-8 gives 1.010 % max.
   - Keeping all 18 `down_proj` in fp32 is not enough: 0.47 / 1.28 %. With the projector also fp32: 0.45 / 1.11 %.
7. **End to end with the deployment INT8 expert** (`ur7e_m2mse` numerics, bf16 host and KV), compounding measured:

   | prefix feeding the INT8 expert | G3 mean / max | verdict | joint max-abs max |
   |---|---|---|---|
   | fp32 prefix (expert alone) | 2.06 % / 3.84 % | | 0.0265 rad |
   | `base` W8A8 prefix | 2.51 % / 5.00 % | fails the every-frame ≤ 5 % bound by 0.001 % | 0.0258 rad |
   | `smooth_lm` | 2.13 % / 3.95 % | passes G3 | 0.0216 rad |

   - Errors add roughly in quadrature. For `base`, the root-sum-square of the two stages measured alone is 2.22 / 4.25 %; the measured result is 2.51 / 5.00 %.
   - With SmoothQuant the INT8 expert still dominates the full-model error.

**Recommendation**
- Gemma: W8A8 with SmoothQuant α = 0.5, plus a small bf16 fallback (layer 14, or the `down_proj`s).
- SigLIP: W8A8, with the multi-modal projector in bf16 to meet G0.
- Drop G1 as a hard per-layer 1 % gate for per-token INT8. Judge the prefix by G3 through the INT8 expert instead.

**Caveats**
- 8 evaluation frames only. The worst frames are demo1_ep20:8 and recov_pi0_ep00:2.
- The fallback ranking and localisation use 2 frames.

## Frames

- Evaluation (held-out episodes): demo1_ep20, demo1_ep40, recov_pi0_ep00, recov_pi0_ep10, frames 2 and 8.
- SmoothQuant / static calibration: demo1_ep01..08, frame 5. Disjoint from evaluation; the script asserts it.
- Local stats and SigLIP isolation: demo1_ep20:2, recov_pi0_ep10:8.
- Gemma isolation (layers and projections, which rank the fallback): demo1_ep20:2, recov_pi0_ep00:2.

References:
- fp32: `~/pi0_glue/captures/<ep>/frame_NN.npz` (LeRobot torch fp32 prefix KV and `actions_fp32`, same noise).
- Checkpoint: `~/pi0_ckpt/ur7e-demo-2-pi0-fsdp/010000`.

## Reproduce

```
S="systemd-run --user --scope -q -p MemoryHigh=16G nice -n 10 ~/lerobot/.venv/bin/python paper/sw/prefix_w8a8_eval.py --threads 8 --cache <cache> --save-ws"
$S --stages selftest,fp32,base,base_lm,base_vis,w8,a8,calib,smooth,smooth_lm,attn,attn_only,attn_qk,attn_pv,static_amax,static_mse
$S --local-frames demo1_ep20:2 recov_pi0_ep10:8 --iso-frames demo1_ep20:2 recov_pi0_ep10:8 --stages local,iso_vis_layer,iso_vis_proj
$S --iso-frames demo1_ep20:2 recov_pi0_ep00:2 --stages iso_lm_layer,iso_lm_proj,fallback
$S --stages e2e_int8_expert
$S --stages custom --custom-name down_fp --custom-fp '^lm\.L[0-9]+\.down$'
$S --stages custom --custom-name down_mmproj_fp --custom-fp '^(lm\.L[0-9]+\.down|vis\.mmproj)$'
~/lerobot/.venv/bin/python paper/sw/prefix_w8a8_summary.py --notes <this notes file>
```

Runtime on fics beside an ACE build, 8 threads, nice 10:
- About 11 s per quantised prefix and 2.2 s for the fp32 expert.
- About 8–13 s per INT8-expert chunk.
- 104 s to build a weight set.
- Peak RSS 16.9 GB, of which about 10 GB is page cache.
- Total about 1 h.

### Variant comparison (all gates)

| variant | frames | G0 proj max % | G0 tower max % | G1 K max % (layer) | G1 V max % (layer) | layers > 1 % (K/V) | hidden max % | G3 rel RMS 7 mean / max % | cos min | G3 pass |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `fp32` fp32 floor (this harness vs the capture) | 8 | — | — | 0.00 (L17) | 0.00 (L16) | 0/0 | 0.00 | 0.00 / 0.00 | 1.00000 | yes |
| `base` (a) W8 per-channel MSE + A8 per-token dynamic, SigLIP + Gemma | 8 | 1.21 | 3.27 | 14.73 (L17) | 27.01 (L1) | 18/18 | 16.05 | 0.79 / 1.82 | 0.99985 | yes |
| `base_vis` (a) SigLIP + projector only | 8 | 1.21 | 3.27 | 8.42 (L17) | 10.76 (L17) | 18/18 | 9.07 | 0.30 / 0.68 | 0.99998 | yes |
| `base_lm` (a) Gemma only (fp32 SigLIP embeddings) | 8 | — | — | 12.35 (L17) | 25.75 (L1) | 18/18 | 14.15 | 0.73 / 2.03 | 0.99981 | yes |
| `w8` W8 only (A fp32) | 8 | 0.31 | 1.35 | 3.42 (L17) | 4.57 (L16) | 17/18 | 4.87 | 0.15 / 0.25 | 1.00000 | yes |
| `a8` A8 per-token only (W fp32) | 8 | 1.17 | 2.68 | 15.14 (L17) | 26.36 (L1) | 18/18 | 16.13 | 1.00 / 2.21 | 0.99977 | yes |
| `smooth` (b) base + SmoothQuant a=0.5 on Gemma | 8 | 1.21 | 3.27 | 11.29 (L17) | 14.88 (L17) | 18/18 | 12.52 | 0.48 / 1.03 | 0.99995 | yes |
| `smooth_lm` (b) Gemma only + SmoothQuant | 8 | — | — | 7.62 (L17) | 12.28 (L16) | 17/18 | 8.95 | 0.28 / 0.57 | 0.99998 | yes |
| `attn` (c) base + INT8 QK^T and PV (Q,K,P per row; V per key) | 8 | 1.34 | 4.33 | 17.88 (L17) | 31.73 (L1) | 18/18 | 19.59 | 0.86 / 1.79 | 0.99986 | yes |
| `attn_only` (c) INT8 attention only, fp32 GEMMs | 8 | 0.79 | 3.40 | 10.65 (L17) | 14.02 (L17) | 18/18 | 11.61 | 0.52 / 1.07 | 0.99994 | yes |
| `attn_qk` (c) base + INT8 QK^T only | 8 | 1.22 | 3.14 | 15.48 (L17) | 27.11 (L1) | 18/18 | 16.35 | 0.80 / 1.73 | 0.99987 | yes |
| `attn_pv` (c) base + INT8 PV only | 8 | 1.35 | 4.32 | 17.92 (L17) | 31.74 (L1) | 18/18 | 19.71 | 0.80 / 1.81 | 0.99986 | yes |
| `static_amax` (d) W8 + A8 per-tensor static, calibration amax | 8 | 2.61 | 7.94 | 75.78 (L17) | 86.79 (L15) | 18/18 | 87.66 | 14.62 / 42.92 | 0.90997 | no |
| `static_mse` (d) W8 + A8 per-tensor static, calibration MSE clip | 8 | 3.19 | 10.92 | 61.20 (L17) | 75.84 (L17) | 18/18 | 73.57 | 6.50 / 25.39 | 0.96722 | no |

### G0 per image, variant `base`

| frame | front: proj % | front: tower % | wrist: proj % | wrist: tower % | worst token % |
|---|---:|---:|---:|---:|---:|
| demo1_ep20:2 | 1.08 | 2.74 | 1.18 | 3.03 | 3.94 |
| demo1_ep20:8 | 1.02 | 2.39 | 1.12 | 2.88 | 2.34 |
| demo1_ep40:2 | 1.08 | 2.64 | 1.12 | 2.84 | 4.11 |
| demo1_ep40:8 | 1.04 | 2.51 | 1.13 | 3.01 | 3.25 |
| recov_pi0_ep00:2 | 1.13 | 2.79 | 1.15 | 2.93 | 3.16 |
| recov_pi0_ep00:8 | 1.12 | 2.90 | 1.10 | 2.96 | 3.37 |
| recov_pi0_ep10:2 | 1.20 | 3.27 | 1.21 | 3.12 | 5.62 |
| recov_pi0_ep10:8 | 1.11 | 3.00 | 1.12 | 2.83 | 3.76 |

### G1 per layer (max over frames), K / V

| layer | `fp32` K / V max % | `base` K / V max % | `base_lm` K / V max % | `smooth` K / V max % | `attn` K / V max % | `static_amax` K / V max % |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.00 / 0.00 | 3.18 / 7.23 | 1.48 / 2.88 | 2.91 / 6.81 | 3.47 / 7.69 | 6.63 / 14.17 |
| 1 | 0.00 / 0.00 | 8.10 / 27.01 | 7.73 / 25.75 | 3.10 / 11.02 | 9.58 / 31.73 | 13.10 / 41.51 |
| 2 | 0.00 / 0.00 | 8.33 / 18.13 | 7.78 / 16.44 | 3.99 / 9.64 | 9.61 / 21.20 | 15.51 / 34.52 |
| 3 | 0.00 / 0.00 | 6.94 / 20.46 | 6.42 / 19.09 | 3.61 / 9.54 | 7.89 / 22.21 | 14.01 / 41.64 |
| 4 | 0.00 / 0.00 | 6.46 / 15.23 | 5.70 / 13.80 | 3.73 / 8.35 | 7.58 / 17.03 | 22.19 / 51.59 |
| 5 | 0.00 / 0.00 | 6.49 / 12.64 | 5.48 / 11.15 | 4.06 / 7.80 | 7.63 / 14.77 | 22.02 / 41.92 |
| 6 | 0.00 / 0.00 | 6.43 / 11.76 | 5.48 / 10.20 | 4.31 / 7.85 | 7.81 / 13.30 | 26.34 / 41.80 |
| 7 | 0.00 / 0.00 | 7.20 / 11.11 | 5.89 / 9.51 | 4.99 / 7.72 | 8.71 / 13.06 | 32.82 / 45.53 |
| 8 | 0.00 / 0.00 | 7.13 / 10.78 | 5.89 / 9.15 | 5.07 / 8.07 | 8.56 / 12.48 | 39.55 / 61.54 |
| 9 | 0.00 / 0.00 | 8.38 / 14.73 | 7.60 / 13.47 | 5.03 / 9.37 | 9.54 / 16.40 | 38.99 / 67.83 |
| 10 | 0.00 / 0.00 | 8.38 / 13.72 | 7.23 / 11.98 | 5.70 / 9.41 | 10.14 / 15.84 | 46.50 / 68.88 |
| 11 | 0.00 / 0.00 | 7.96 / 15.60 | 7.24 / 13.84 | 4.93 / 9.84 | 9.32 / 17.80 | 42.31 / 71.96 |
| 12 | 0.00 / 0.00 | 8.32 / 14.20 | 7.09 / 12.59 | 6.24 / 10.35 | 10.31 / 16.23 | 52.24 / 73.78 |
| 13 | 0.00 / 0.00 | 8.49 / 12.85 | 7.25 / 10.97 | 6.60 / 9.94 | 10.20 / 15.47 | 61.63 / 80.14 |
| 14 | 0.00 / 0.00 | 8.86 / 13.61 | 7.03 / 12.14 | 7.12 / 10.64 | 10.77 / 17.54 | 61.86 / 79.62 |
| 15 | 0.00 / 0.00 | 10.70 / 17.60 | 8.77 / 16.41 | 8.62 / 12.75 | 13.06 / 21.96 | 67.84 / 86.79 |
| 16 | 0.00 / 0.00 | 13.58 / 20.27 | 11.10 / 18.63 | 10.62 / 14.41 | 16.41 / 24.69 | 74.91 / 86.65 |
| 17 | 0.00 / 0.00 | 14.73 / 19.88 | 12.35 / 17.10 | 11.29 / 14.88 | 17.88 / 24.11 | 75.78 / 82.05 |

### G1 `base`: image vs language tokens, worst token

| layer | K img mean % | K lang mean % | V img mean % | V lang mean % | K worst token % | V worst token % |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 3.04 | 2.42 | 7.88 | 2.34 | 22.29 | 47.10 |
| 1 | 7.93 | 4.48 | 26.20 | 9.64 | 19.95 | 173.75 |
| 2 | 8.19 | 5.59 | 17.56 | 7.77 | 21.94 | 106.44 |
| 3 | 6.89 | 5.36 | 19.89 | 10.64 | 22.33 | 64.52 |
| 4 | 6.42 | 5.56 | 14.49 | 8.68 | 22.82 | 52.40 |
| 5 | 6.37 | 5.24 | 12.05 | 9.69 | 21.52 | 49.74 |
| 6 | 6.31 | 5.46 | 11.43 | 12.06 | 32.19 | 61.55 |
| 7 | 7.06 | 5.91 | 10.67 | 10.28 | 35.08 | 57.07 |
| 8 | 7.00 | 5.79 | 10.24 | 10.95 | 30.51 | 58.10 |
| 9 | 8.30 | 7.28 | 14.12 | 12.12 | 26.86 | 47.27 |
| 10 | 8.10 | 7.26 | 12.85 | 11.21 | 30.31 | 47.27 |
| 11 | 7.54 | 7.25 | 14.50 | 11.70 | 25.72 | 52.62 |
| 12 | 7.88 | 7.33 | 13.42 | 14.23 | 30.48 | 46.75 |
| 13 | 7.74 | 7.24 | 11.63 | 15.32 | 47.71 | 100.75 |
| 14 | 7.56 | 6.65 | 12.55 | 11.42 | 61.55 | 111.54 |
| 15 | 9.27 | 10.40 | 16.50 | 18.72 | 62.03 | 111.24 |
| 16 | 11.99 | 11.88 | 18.68 | 23.73 | 74.89 | 109.37 |
| 17 | 12.63 | 13.07 | 17.60 | 19.10 | 74.75 | 115.81 |

### G3 per frame

| frame | `fp32` rel RMS 7 % / max-abs | `base` rel RMS 7 % / max-abs | `smooth` rel RMS 7 % / max-abs | `attn` rel RMS 7 % / max-abs | `static_amax` rel RMS 7 % / max-abs |
|---|---:|---:|---:|---:|---:|
| demo1_ep20:2 | 0.00 / 0.0000 | 0.35 / 0.0085 | 0.20 / 0.0055 | 0.48 / 0.0147 | 6.03 / 0.1252 |
| demo1_ep20:8 | 0.00 / 0.0000 | 1.47 / 0.0468 | 0.73 / 0.0180 | 1.00 / 0.0214 | 18.61 / 0.4031 |
| demo1_ep40:2 | 0.00 / 0.0000 | 0.33 / 0.0077 | 0.23 / 0.0042 | 0.58 / 0.0099 | 4.21 / 0.0820 |
| demo1_ep40:8 | 0.00 / 0.0000 | 1.06 / 0.0178 | 0.75 / 0.0152 | 0.86 / 0.0186 | 42.92 / 0.8672 |
| recov_pi0_ep00:2 | 0.00 / 0.0000 | 1.82 / 0.0388 | 1.03 / 0.0181 | 1.79 / 0.0368 | 27.57 / 0.5000 |
| recov_pi0_ep00:8 | 0.00 / 0.0000 | 0.65 / 0.0575 | 0.48 / 0.0413 | 1.10 / 0.0931 | 10.20 / 0.4441 |
| recov_pi0_ep10:2 | 0.00 / 0.0000 | 0.44 / 0.0168 | 0.26 / 0.0072 | 0.70 / 0.0253 | 4.80 / 0.2223 |
| recov_pi0_ep10:8 | 0.00 / 0.0000 | 0.23 / 0.0075 | 0.11 / 0.0046 | 0.37 / 0.0090 | 2.61 / 0.0819 |

### Localisation: one Gemma layer quantised alone (LM-only, fp32 SigLIP embeddings)

| quantised alone | GEMMs | KV max over layers % | KV last layer % | hidden max % | G3 mean / max % |
|---|---:|---:|---:|---:|---:|
| lm.L14 | 7 | 9.554 | 6.675 | 6.568 | 0.379 / 0.577 |
| lm.L12 | 7 | 7.221 | 4.607 | 3.819 | 0.275 / 0.494 |
| lm.L11 | 7 | 9.622 | 5.262 | 4.858 | 0.239 / 0.384 |
| lm.L7 | 7 | 5.818 | 5.247 | 4.818 | 0.204 / 0.351 |
| lm.L15 | 7 | 6.091 | 5.008 | 4.466 | 0.175 / 0.305 |
| lm.L0 | 7 | 20.071 | 7.999 | 6.221 | 0.228 / 0.295 |
| lm.L13 | 7 | 5.375 | 4.462 | 3.999 | 0.182 / 0.278 |
| lm.L8 | 7 | 5.349 | 2.171 | 1.824 | 0.143 / 0.244 |
| lm.L6 | 7 | 7.032 | 2.238 | 1.947 | 0.146 / 0.236 |
| lm.L16 | 7 | 5.999 | 5.544 | 5.484 | 0.137 / 0.230 |
| lm.L3 | 7 | 13.134 | 1.693 | 1.429 | 0.102 / 0.176 |
| lm.L10 | 7 | 8.123 | 2.923 | 2.654 | 0.114 / 0.175 |
| lm.L2 | 7 | 8.991 | 2.105 | 1.751 | 0.099 / 0.163 |
| lm.L5 | 7 | 6.040 | 1.849 | 1.526 | 0.093 / 0.162 |
| lm.L9 | 7 | 10.217 | 2.382 | 2.056 | 0.082 / 0.122 |
| lm.L4 | 7 | 8.677 | 1.926 | 1.611 | 0.075 / 0.113 |
| lm.L1 | 7 | 11.869 | 1.872 | 1.347 | 0.054 / 0.077 |
| lm.L17 | 7 | 6.814 | 6.814 | 4.413 | 0.047 / 0.065 |

### Localisation: one Gemma projection class quantised alone (all 18 layers)

| quantised alone | GEMMs | KV max over layers % | KV last layer % | hidden max % | G3 mean / max % |
|---|---:|---:|---:|---:|---:|
| lm.*.down | 18 | 18.169 | 11.522 | 10.532 | 1.215 / 2.086 |
| lm.*.v | 18 | 13.474 | 10.004 | 7.110 | 0.309 / 0.507 |
| lm.*.up | 18 | 5.821 | 5.239 | 4.782 | 0.257 / 0.443 |
| lm.*.k | 18 | 6.332 | 4.878 | 2.448 | 0.161 / 0.272 |
| lm.*.o | 18 | 2.108 | 1.769 | 1.522 | 0.134 / 0.246 |
| lm.*.gate | 18 | 4.952 | 4.612 | 4.226 | 0.145 / 0.219 |
| lm.*.q | 18 | 3.159 | 2.874 | 2.668 | 0.106 / 0.186 |

### Localisation: one SigLIP layer quantised alone (top 10)

| quantised alone | GEMMs | G0 proj max % | G0 tower max % |
|---|---:|---:|---:|
| vis.L26 | 6 | 0.485 | 1.163 |
| vis.L1 | 6 | 0.241 | 0.925 |
| vis.L24 | 6 | 0.203 | 1.037 |
| vis.L0 | 6 | 0.198 | 0.775 |
| vis.L25 | 6 | 0.162 | 0.799 |
| vis.L2 | 6 | 0.159 | 0.597 |
| vis.L3 | 6 | 0.154 | 0.596 |
| vis.L23 | 6 | 0.145 | 0.732 |
| vis.L7 | 6 | 0.131 | 0.509 |
| vis.L8 | 6 | 0.118 | 0.470 |

### Localisation: one SigLIP projection class quantised alone

| quantised alone | GEMMs | G0 proj max % | G0 tower max % |
|---|---:|---:|---:|
| vis.mmproj | 1 | 0.863 | 0.000 |
| vis.*.fc2 | 27 | 0.623 | 2.423 |
| vis.*.v | 27 | 0.262 | 1.158 |
| vis.*.fc1 | 27 | 0.237 | 0.950 |
| vis.*.k | 27 | 0.190 | 0.742 |
| vis.*.out | 27 | 0.136 | 0.583 |
| vis.*.q | 27 | 0.118 | 0.450 |
| vis.patch | 1 | 0.094 | 0.341 |

### Local GEMM error by class (fp32 inputs) and local attention error

| class | layers | local rel W8A8 % mean / max | W only % mean | A only % mean | token amax/rms max |
|---|---:|---:|---:|---:|---:|
| lm.down | 18 | 5.17 / 8.18 | 1.33 | 4.85 | 128.0 |
| lm.gate | 18 | 2.28 / 3.26 | 0.50 | 2.22 | 43.1 |
| lm.k | 18 | 3.93 / 6.27 | 0.70 | 3.86 | 42.7 |
| lm.o | 18 | 1.21 / 1.94 | 0.66 | 1.01 | 23.4 |
| lm.q | 18 | 4.32 / 6.69 | 0.68 | 4.26 | 42.7 |
| lm.up | 18 | 4.80 / 15.06 | 0.92 | 4.71 | 43.1 |
| lm.v | 18 | 7.42 / 13.49 | 1.18 | 7.30 | 42.7 |
| vis.fc1 | 27 | 1.26 / 2.32 | 0.50 | 1.15 | 33.4 |
| vis.fc2 | 27 | 2.58 / 5.41 | 0.83 | 2.39 | 65.6 |
| vis.k | 27 | 2.23 / 4.20 | 0.73 | 2.10 | 33.7 |
| vis.mmproj | 1 | 0.86 / 0.86 | 0.15 | 0.85 | 28.6 |
| vis.out | 27 | 1.06 / 1.65 | 0.52 | 0.92 | 20.3 |
| vis.patch | 1 | 0.34 / 0.34 | 0.29 | 0.18 | 7.5 |
| vis.q | 27 | 2.29 / 3.62 | 0.75 | 2.16 | 33.7 |
| vis.v | 27 | 2.78 / 3.91 | 0.85 | 2.65 | 33.7 |

| attention (local, INT8 vs fp32 on the same q/k/v) | both % mean / max | QK only % mean | PV only % mean |
|---|---:|---:|---:|
| vis (27 layers) | 5.19 / 10.09 | 0.63 | 5.15 |
| lm (18 layers) | 8.11 / 21.68 | 0.70 | 8.08 |

worst attention layers (local both %): lm.L1 21.68, lm.L5 17.88, lm.L2 15.57, lm.L4 12.23, lm.L10 11.07, vis.L19 10.09

### Worst Gemma GEMMs, local error + activation outliers

| GEMM | local rel W8A8 % | W only % | A only % | token amax/rms p50 / p99 / max | channel amax max/median | top channels | global amax |
|---|---:|---:|---:|---|---:|---|---:|
| lm.L0.up | 15.06 | 1.69 | 14.96 | 36.2 / 37.8 / 38.2 | 66.7 | [738, 604, 50] | 58.7 |
| lm.L3.v | 13.49 | 1.76 | 13.38 | 31.2 / 33.2 / 34.1 | 31.2 | [738, 604, 674] | 22.6 |
| lm.L2.up | 13.08 | 1.63 | 12.97 | 32.1 / 33.4 / 36.4 | 28.6 | [738, 604, 674] | 28.9 |
| lm.L1.v | 12.20 | 1.76 | 12.08 | 29.9 / 33.3 / 35.0 | 26.9 | [50, 738, 604] | 15.3 |
| lm.L9.v | 10.10 | 1.37 | 10.02 | 32.1 / 35.0 / 40.5 | 18.5 | [738, 50, 674] | 24.4 |
| lm.L2.v | 9.46 | 1.29 | 9.37 | 29.6 / 31.7 / 32.1 | 19.8 | [738, 604, 1967] | 15.5 |
| lm.L11.v | 9.45 | 1.18 | 9.37 | 33.6 / 37.5 / 38.3 | 12.6 | [738, 50, 674] | 25.4 |
| lm.L4.v | 8.74 | 1.28 | 8.64 | 25.8 / 31.5 / 35.5 | 35.6 | [738, 604, 50] | 27.7 |
| lm.L7.down | 8.18 | 6.41 | 5.00 | 31.3 / 62.0 / 128.0 | 1877.5 | [12669, 12622, 13674] | 1197.5 |
| lm.L10.v | 7.90 | 1.10 | 7.82 | 30.8 / 34.7 / 42.2 | 17.8 | [738, 50, 674] | 30.3 |
| lm.L0.down | 7.81 | 0.75 | 7.71 | 95.9 / 100.1 / 118.8 | 742.0 | [9760, 15024, 4993] | 52.8 |
| lm.L14.down | 7.69 | 2.96 | 7.27 | 79.9 / 121.1 / 121.9 | 222.0 | [5248, 4402, 9172] | 717.2 |

### Worst SigLIP GEMMs, local error + activation outliers

| GEMM | local rel W8A8 % | W only % | A only % | token amax/rms p50 / p99 / max | channel amax max/median | top channels | global amax |
|---|---:|---:|---:|---|---:|---|---:|
| vis.L17.fc2 | 5.41 | 1.50 | 5.18 | 39.5 / 56.6 / 64.0 | 434.5 | [438, 3902, 639] | 63.3 |
| vis.L15.fc2 | 4.40 | 0.92 | 4.31 | 22.8 / 45.1 / 59.9 | 29.8 | [3236, 1786, 204] | 23.5 |
| vis.L26.k | 4.20 | 0.87 | 4.11 | 17.1 / 21.5 / 30.6 | 38.9 | [1076, 192, 725] | 35.5 |
| vis.L16.fc2 | 4.14 | 0.92 | 4.06 | 27.2 / 48.9 / 61.9 | 114.1 | [625, 709, 3353] | 30.6 |
| vis.L3.v | 3.91 | 1.13 | 3.75 | 13.8 / 22.8 / 33.1 | 17.6 | [1076, 889, 626] | 18.1 |
| vis.L18.fc2 | 3.69 | 1.03 | 3.54 | 33.8 / 53.2 / 64.0 | 196.0 | [1334, 1033, 939] | 14.3 |
| vis.L13.fc2 | 3.64 | 0.80 | 3.54 | 19.3 / 34.2 / 38.9 | 8.8 | [3113, 2956, 726] | 7.4 |
| vis.L2.v | 3.64 | 1.23 | 3.42 | 13.5 / 21.5 / 33.0 | 17.2 | [1076, 889, 738] | 13.8 |
| vis.L26.q | 3.62 | 0.74 | 3.54 | 17.1 / 21.5 / 30.6 | 38.9 | [1076, 192, 725] | 35.5 |
| vis.L25.q | 3.36 | 0.90 | 3.23 | 13.1 / 20.8 / 30.1 | 24.2 | [1076, 192, 725] | 35.5 |
| vis.L14.fc2 | 3.34 | 0.82 | 3.24 | 18.2 / 40.0 / 52.4 | 7.1 | [3379, 212, 1914] | 8.2 |
| vis.L7.v | 3.34 | 0.84 | 3.23 | 14.0 / 23.8 / 33.6 | 19.6 | [1076, 961, 586] | 30.8 |

### End to end: W8A8 prefix into the deployment INT8 expert (compounding)

Expert = deployment INT8 numerics `/home/sngong/pi0_glue/numerics/ur7e_m2mse.npz` (M out per-column margin 2.0 + in mse + W mse) in `pi0_deploy_model` (bf16 host ops, bf16 KV); rebuild check vs `paper/sw/results/deploy_calib_heldout.json` on the fp32 capture KV: max |Δ rel RMS| = 0.0 over 8 frames.

| prefix KV → INT8 expert | G3 rel RMS 7 mean / max % | cos min | G3 pass | joint max-abs rad mean / max | joint RMS rad mean / max | same KV → fp32 expert, mean / max % | RSS of the two stages alone, mean / max % |
|---|---:|---:|---|---:|---:|---:|---:|
| fp32 prefix (capture KV) = INT8 expert alone | 2.06 / 3.84 | 0.99930 | yes | 0.0176 / 0.0265 | 0.0038 / 0.0051 | — | — |
| W8A8 prefix `base` | 2.51 / 5.00 | 0.99875 | no | 0.0179 / 0.0258 | 0.0040 / 0.0054 | 0.79 / 1.82 | 2.22 / 4.25 |
| W8A8 Gemma + SmoothQuant `smooth_lm` (fp32 SigLIP) | 2.13 / 3.95 | 0.99926 | yes | 0.0170 / 0.0216 | 0.0039 / 0.0043 | 0.28 / 0.57 | 2.09 / 3.88 |

| frame | `fp32_prefix` rel RMS 7 % / joint max-abs rad | `base` rel RMS 7 % / joint max-abs rad | `smooth_lm` rel RMS 7 % / joint max-abs rad |
|---|---:|---:|---:|
| demo1_ep20:2 | 1.36 / 0.0160 | 1.40 / 0.0151 | 1.46 / 0.0145 |
| demo1_ep20:8 | 3.51 / 0.0166 | 5.00 / 0.0205 | 3.85 / 0.0178 |
| demo1_ep40:2 | 1.59 / 0.0138 | 2.09 / 0.0258 | 1.77 / 0.0168 |
| demo1_ep40:8 | 2.59 / 0.0265 | 2.69 / 0.0192 | 2.31 / 0.0159 |
| recov_pi0_ep00:2 | 3.84 / 0.0145 | 4.87 / 0.0161 | 3.95 / 0.0129 |
| recov_pi0_ep00:8 | 1.21 / 0.0149 | 1.59 / 0.0164 | 1.20 / 0.0216 |
| recov_pi0_ep10:2 | 1.22 / 0.0205 | 1.29 / 0.0163 | 1.18 / 0.0200 |
| recov_pi0_ep10:8 | 1.19 / 0.0178 | 1.16 / 0.0141 | 1.35 / 0.0162 |

### End to end: the expert with the prefix recipe (per-token dynamic W8A8)

action expert fake-quantised in torch with the prefix recipe (W8 per output channel MSE clip, A8 per token dynamic, dequant = acc x s_row x s_col, no INT8 output requant, biases fp32); attention, norms, SiLU/GELU, RoPE, residuals and the Euler update fp32; prefix KV from the capture (fp32) or the e2e cache; G3 vs actions_fp32 of the capture, joint errors in rad through the unnormaliser std.

- expert `dyn`: W8 per-channel MSE + A8 per-token dynamic on all 131 expert GEMMs (126 layer + 5 projections), attention fp32
- expert `dyn_smooth`: dyn + SmoothQuant a=0.5 on the 126 expert layer GEMMs (calibrated on --calib frames)
- expert `dyn_smooth_attn`: dyn_smooth + INT8 QK^T and PV in the expert (Q,K,P per row; V per key; prefix keys included)

| prefix KV | expert | G3 mean / max % | cos min | G3 pass (≤3 % mean, ≤5 % frame) | G3 max ≤ 1 % | joint max-abs rad mean / max | joint RMS rad mean / max | G4 RMS ≤ 0.005 | G4 max ≤ 0.02 |
|---|---|---:|---:|---|---|---:|---:|---|---|
| `fp32` | `dyn` | 1.06 / 1.68 | 0.99986 | yes | no | 0.0096 / 0.0136 | 0.0021 / 0.0027 | yes | yes |
| `base` | `dyn` | 1.36 / 2.82 | 0.99962 | yes | no | 0.0107 / 0.0131 | 0.0023 / 0.0028 | yes | yes |
| `smooth_lm` | `dyn` | 1.12 / 1.83 | 0.99984 | yes | no | 0.0086 / 0.0104 | 0.0020 / 0.0028 | yes | yes |
| `fp32` | `dyn_smooth` | 0.81 / 1.14 | 0.99994 | yes | no | 0.0076 / 0.0123 | 0.0017 / 0.0032 | yes | yes |
| `base` | `dyn_smooth` | 1.14 / 2.28 | 0.99976 | yes | no | 0.0072 / 0.0093 | 0.0019 / 0.0026 | yes | yes |
| `smooth_lm` | `dyn_smooth` | 0.97 / 1.90 | 0.99983 | yes | no | 0.0081 / 0.0130 | 0.0018 / 0.0031 | yes | yes |
| `fp32` | `dyn_smooth_attn` | 1.28 / 2.21 | 0.99976 | yes | no | 0.0091 / 0.0123 | 0.0021 / 0.0029 | yes | yes |
| `smooth_lm` | `dyn_smooth_attn` | 1.31 / 2.02 | 0.99980 | yes | no | 0.0093 / 0.0128 | 0.0021 / 0.0027 | yes | yes |
| `fp32` | deployed static INT8 (`ur7e_m2mse`) | 2.06 / 3.84 | 0.99930 | yes | no | 0.0176 / 0.0265 | 0.0038 / 0.0051 | no | no |
| `base` | deployed static INT8 (`ur7e_m2mse`) | 2.51 / 5.00 | 0.99875 | no | no | 0.0179 / 0.0258 | 0.0040 / 0.0054 | no | no |
| `smooth_lm` | deployed static INT8 (`ur7e_m2mse`) | 2.13 / 3.95 | 0.99926 | yes | no | 0.0170 / 0.0216 | 0.0039 / 0.0043 | yes | no |

| frame | `fp32+dyn` rel RMS 7 % / joint max rad | `base+dyn` rel RMS 7 % / joint max rad | `smooth_lm+dyn` rel RMS 7 % / joint max rad | `fp32+dyn_smooth` rel RMS 7 % / joint max rad | `base+dyn_smooth` rel RMS 7 % / joint max rad | `smooth_lm+dyn_smooth` rel RMS 7 % / joint max rad | `fp32+dyn_smooth_attn` rel RMS 7 % / joint max rad | `smooth_lm+dyn_smooth_attn` rel RMS 7 % / joint max rad |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| demo1_ep20:2 | 0.83 / 0.0101 | 0.90 / 0.0098 | 0.84 / 0.0074 | 0.69 / 0.0060 | 0.70 / 0.0055 | 0.76 / 0.0063 | 0.71 / 0.0067 | 0.71 / 0.0079 |
| demo1_ep20:8 | 1.46 / 0.0124 | 1.79 / 0.0085 | 1.83 / 0.0092 | 1.14 / 0.0069 | 1.50 / 0.0087 | 1.90 / 0.0098 | 1.66 / 0.0110 | 1.68 / 0.0112 |
| demo1_ep40:2 | 1.00 / 0.0070 | 1.13 / 0.0131 | 0.98 / 0.0067 | 0.67 / 0.0063 | 0.72 / 0.0056 | 0.60 / 0.0060 | 1.02 / 0.0092 | 1.09 / 0.0079 |
| demo1_ep40:8 | 1.24 / 0.0062 | 1.43 / 0.0112 | 1.28 / 0.0097 | 0.86 / 0.0048 | 1.22 / 0.0089 | 1.09 / 0.0077 | 2.21 / 0.0091 | 2.02 / 0.0089 |
| recov_pi0_ep00:2 | 1.68 / 0.0083 | 2.82 / 0.0111 | 1.80 / 0.0104 | 1.09 / 0.0068 | 2.28 / 0.0055 | 1.27 / 0.0059 | 1.52 / 0.0077 | 1.60 / 0.0069 |
| recov_pi0_ep00:8 | 0.96 / 0.0090 | 1.14 / 0.0116 | 0.98 / 0.0098 | 0.75 / 0.0123 | 0.98 / 0.0087 | 0.84 / 0.0130 | 1.56 / 0.0123 | 1.72 / 0.0128 |
| recov_pi0_ep10:2 | 0.69 / 0.0136 | 0.84 / 0.0116 | 0.63 / 0.0087 | 0.81 / 0.0104 | 1.07 / 0.0093 | 0.82 / 0.0086 | 1.00 / 0.0081 | 0.97 / 0.0082 |
| recov_pi0_ep10:8 | 0.64 / 0.0098 | 0.79 / 0.0087 | 0.61 / 0.0070 | 0.49 / 0.0073 | 0.69 / 0.0051 | 0.49 / 0.0079 | 0.56 / 0.0090 | 0.70 / 0.0102 |

### Expert localisation: one projection class quantised alone (per-token dynamic W8A8)

Frames: demo1_ep20:2, demo1_ep20:8, demo1_ep40:2, demo1_ep40:8, recov_pi0_ep00:2, recov_pi0_ep00:8, recov_pi0_ep10:2, recov_pi0_ep10:8; prefix KV = fp32 capture; everything else in the expert fp32.

| quantised alone | GEMMs | G3 mean / max % | joint max-abs rad max | joint RMS rad max |
|---|---:|---:|---:|---:|
| exp.*.up | 18 | 0.585 / 1.026 | 0.0078 | 0.0016 |
| exp.act_in | 1 | 0.443 / 0.733 | 0.0038 | 0.0011 |
| exp.act_out | 1 | 0.389 / 0.698 | 0.0034 | 0.0008 |
| exp.*.down | 18 | 0.398 / 0.646 | 0.0043 | 0.0011 |
| exp.state | 1 | 0.328 / 0.499 | 0.0041 | 0.0019 |
| exp.*.gate | 18 | 0.223 / 0.405 | 0.0023 | 0.0005 |
| exp.*.v | 18 | 0.163 / 0.317 | 0.0021 | 0.0007 |
| exp.*.q | 18 | 0.105 / 0.251 | 0.0013 | 0.0003 |
| exp.*.o | 18 | 0.148 / 0.246 | 0.0016 | 0.0004 |
| exp.tm_out | 1 | 0.157 / 0.238 | 0.0022 | 0.0004 |
| exp.tm_in | 1 | 0.126 / 0.194 | 0.0021 | 0.0003 |
| exp.*.k | 18 | 0.065 / 0.132 | 0.0006 | 0.0001 |

### Expert localisation: one layer quantised alone (per-token dynamic W8A8)

Frames: demo1_ep20:2, demo1_ep20:8, demo1_ep40:2, demo1_ep40:8, recov_pi0_ep00:2, recov_pi0_ep00:8, recov_pi0_ep10:2, recov_pi0_ep10:8; prefix KV = fp32 capture; everything else in the expert fp32.

| quantised alone | GEMMs | G3 mean / max % | joint max-abs rad max | joint RMS rad max |
|---|---:|---:|---:|---:|
| exp.L0 | 7 | 0.751 / 1.126 | 0.0075 | 0.0019 |
| exp.L17 | 7 | 0.302 / 0.625 | 0.0025 | 0.0006 |
| exp.L5 | 7 | 0.165 / 0.267 | 0.0017 | 0.0003 |
| exp.L16 | 7 | 0.109 / 0.264 | 0.0009 | 0.0002 |
| exp.L9 | 7 | 0.087 / 0.189 | 0.0014 | 0.0003 |
| exp.L6 | 7 | 0.097 / 0.164 | 0.0010 | 0.0002 |
| exp.L11 | 7 | 0.063 / 0.159 | 0.0008 | 0.0002 |
| exp.L15 | 7 | 0.072 / 0.155 | 0.0007 | 0.0001 |
| exp.L2 | 7 | 0.093 / 0.130 | 0.0011 | 0.0003 |
| exp.L10 | 7 | 0.066 / 0.129 | 0.0009 | 0.0001 |
| exp.L7 | 7 | 0.072 / 0.114 | 0.0010 | 0.0002 |
| exp.L3 | 7 | 0.071 / 0.114 | 0.0009 | 0.0002 |
| exp.L12 | 7 | 0.045 / 0.095 | 0.0005 | 0.0001 |
| exp.L8 | 7 | 0.052 / 0.087 | 0.0007 | 0.0001 |
| exp.L14 | 7 | 0.046 / 0.081 | 0.0007 | 0.0001 |
| exp.L4 | 7 | 0.047 / 0.072 | 0.0006 | 0.0001 |
| exp.L13 | 7 | 0.038 / 0.071 | 0.0006 | 0.0001 |
| exp.L1 | 7 | 0.040 / 0.044 | 0.0008 | 0.0002 |

### Fallback: minimal fp32 Gemma-layer set for G3 max < 1 %

Target: G3 rel RMS 7 max < 1.00 % on the evaluation frames. Layer order = damage when quantised alone (iso_lm_layer G3 max, then KV max): lm.L14 (0.58 %), lm.L12 (0.49 %), lm.L11 (0.38 %), lm.L7 (0.35 %), lm.L15 (0.31 %), lm.L0 (0.30 %), lm.L13 (0.28 %), lm.L8 (0.24 %)

| base | Gemma layers kept fp32 | GEMMs fp32 | G0 max % | G1 KV max % | layers > 1 % (K/V) | G3 mean / max % | cos min |
|---|---|---:|---:|---:|---:|---:|---:|
| `base` | none | 0 | 1.21 | 27.01 | 18/18 | 0.79 / 1.82 | 0.99985 |
| `base` | lm.L14 | 7 | 1.21 | 27.01 | 18/18 | 0.69 / 1.44 | 0.99990 |
| `base` | lm.L14, lm.L12 | 14 | 1.21 | 27.01 | 18/18 | 0.57 / 1.18 | 0.99993 |
| `base` | lm.L14, lm.L12, lm.L11, lm.L7 | 28 | 1.21 | 27.01 | 18/18 | 0.60 / 1.35 | 0.99991 |
| `base` | lm.L14, lm.L12, lm.L11, lm.L7, lm.L15, lm.L0, lm.L13, lm.L8 | 56 | 1.21 | 17.34 | 18/18 | 0.44 / 1.01 | 0.99995 |
| `smooth` | none | 0 | 1.21 | 14.88 | 18/18 | 0.48 / 1.03 | 0.99995 |
| `smooth` | lm.L14 | 7 | 1.21 | 14.51 | 18/18 | 0.41 / 0.76 | 0.99997 |

- minimal fp32 set on `smooth`: lm.L14 (k = 1), G3 mean 0.41 % / max 0.76 %
- `base`: target not reached with the tried k

### Custom fallbacks

| custom run | base | fp32 GEMMs | G0 max % | G1 KV max % | layers > 1 % (K/V) | G3 rel RMS 7 mean / max % |
|---|---|---:|---:|---:|---:|---:|
| `custom_down_fp` `^lm\.L[0-9]+\.down$` | base | 18 | 1.21 | 17.76 | 18/18 | 0.47 / 1.28 |
| `custom_down_mmproj_fp` `^(lm\.L[0-9]+\.down|vis\.mmproj)$` | base | 19 | 0.87 | 16.89 | 18/18 | 0.45 / 1.11 |
| `custom_g0_mmproj_fp` `^vis\.mmproj$` | base | 1 | 0.87 | 26.44 | 18/18 | 0.85 / 2.08 |

### 46 held-out frames: the whole pipeline (e2e_46)

Frames: 46. 46 held-out frames (demo1_ep20, demo1_ep40, recov_pi0_ep00, recov_pi0_ep10, all frames; the deployed model's validation set in paper/sw/results/deploy_calib_heldout.json). SmoothQuant calibration: demo1_ep01..08 frame 5 (inside the deployed model's calibration set, disjoint from these). G3 = 50x7 chunk rel RMS vs actions_fp32 of the capture, G4 = joint-space error in rad through the unnormaliser std. attn_key = INT8 QK^T + PV with V per key (the 8-frame definition, not one accumulation); _exp = expert attention only; attn_col_p255 = INT8 QK^T (Q, K per row) + PV with P as uint8 per row and V per channel (one accumulation, ACX_MLP72 multmode A unsigned x B signed); smooth+dyn_smooth+attn_col_p255 = the all-W8A8 chip candidate (SigLIP, Gemma with SmoothQuant, expert with SmoothQuant, attention INT8 everywhere)

| run | G3 mean / median / p95 / max % | cos min | G3 pass | joint max-abs rad mean / p95 / max | joint RMS rad mean / max | frames > 0.02 rad |
|---|---:|---:|---|---:|---:|---:|
| `fp32` | 0.00 / 0.00 / 0.00 / 0.00 | 1.00000 | yes | 0.0000 / 0.0000 / 0.0000 | 0.0000 / 0.0000 | 0 |
| `smooth_lm+dyn` | 1.08 / 0.98 / 1.82 / 2.99 | 0.99959 | yes | 0.0099 / 0.0148 / 0.0154 | 0.0024 / 0.0041 | 0 |
| `smooth_lm+dyn_smooth` | 0.79 / 0.72 / 1.26 / 1.90 | 0.99983 | yes | 0.0073 / 0.0101 / 0.0130 | 0.0017 / 0.0031 | 0 |
| `smooth_lm+dyn_smooth+attn_key_exp` | 1.18 / 1.07 / 1.87 / 4.15 | 0.99920 | yes | 0.0099 / 0.0151 / 0.0168 | 0.0024 / 0.0044 | 0 |
| `smooth_lm+dyn_smooth+attn_key` | 1.20 / 1.08 / 2.02 / 2.70 | 0.99965 | yes | 0.0095 / 0.0137 / 0.0141 | 0.0024 / 0.0045 | 0 |
| `smooth+dyn_smooth` | 0.83 / 0.80 / 1.34 / 1.63 | 0.99987 | yes | 0.0076 / 0.0117 / 0.0129 | 0.0017 / 0.0031 | 0 |
| `fp32+int8_static` | 1.86 / 1.58 / 3.41 / 3.84 | 0.99930 | yes | 0.0182 / 0.0239 / 0.0265 | 0.0041 / 0.0062 | 15 |
| `smooth_lm+int8_static` | 1.87 / 1.65 / 2.89 / 3.95 | 0.99926 | yes | 0.0172 / 0.0258 / 0.0281 | 0.0040 / 0.0057 | 10 |
| `smooth_lm+dyn_smooth+attn_col_p255_exp` | 0.94 / 0.83 / 1.45 / 2.72 | 0.99968 | yes | 0.0085 / 0.0126 / 0.0156 | 0.0020 / 0.0037 | 0 |
| `smooth_lm+dyn_smooth+attn_col_p255` | 0.93 / 0.87 / 1.47 / 2.09 | 0.99980 | yes | 0.0079 / 0.0111 / 0.0150 | 0.0019 / 0.0041 | 0 |
| `smooth+dyn_smooth+attn_col_p255` | 0.94 / 0.90 / 1.51 / 1.69 | 0.99986 | yes | 0.0080 / 0.0113 / 0.0132 | 0.0019 / 0.0034 | 0 |

### INT8 attention localisation: part x op x V granularity (attn_loc)

Frames: 8. INT8-attention localisation on the smooth_lm prefix + dyn_smooth expert (SigLIP rows on the smooth prefix): which op (QK^T / PV), which part (SigLIP / Gemma / expert), which expert layers; V granularity key = per key (not one accumulation), fold = key scale folded into P, col = per channel

| run | G3 mean / median / p95 / max % | cos min | G3 pass | joint max-abs rad mean / p95 / max | joint RMS rad mean / max | frames > 0.02 rad | ΔG3 vs base mean / max % | chunk distance to base mean / max % |
|---|---:|---:|---|---:|---:|---:|---:|---:|
| `base` | 0.97 / 0.83 / 1.68 / 1.90 | 0.99983 | yes | 0.0081 / 0.0119 / 0.0130 | 0.0018 / 0.0031 | 0 | — | — |
| `exp.both.key` | 1.31 / 1.35 / 1.91 / 2.02 | 0.99980 | yes | 0.0093 / 0.0122 / 0.0128 | 0.0021 / 0.0027 | 0 | +0.341 / +0.925 | 1.322 / 2.199 |
| `exp.pv.key` | 1.35 / 1.29 / 2.32 / 2.64 | 0.99966 | yes | 0.0084 / 0.0131 / 0.0136 | 0.0021 / 0.0033 | 0 | +0.376 / +1.545 | 1.407 / 2.360 |
| `exp.both.fold` | 1.43 / 1.34 / 2.42 / 2.68 | 0.99965 | yes | 0.0094 / 0.0132 / 0.0136 | 0.0022 / 0.0033 | 0 | +0.460 / +1.592 | 1.498 / 2.712 |
| `exp.pv.fold` | 1.40 / 1.38 / 2.19 / 2.39 | 0.99972 | yes | 0.0086 / 0.0123 / 0.0124 | 0.0022 / 0.0033 | 0 | +0.428 / +1.294 | 1.458 / 2.778 |
| `exp.both.col` | 1.29 / 1.29 / 1.95 / 2.09 | 0.99978 | yes | 0.0084 / 0.0106 / 0.0110 | 0.0020 / 0.0026 | 0 | +0.322 / +0.996 | 1.334 / 2.215 |
| `exp.pv.col` | 1.29 / 1.28 / 1.99 / 2.15 | 0.99977 | yes | 0.0084 / 0.0118 / 0.0120 | 0.0020 / 0.0026 | 0 | +0.323 / +1.064 | 1.328 / 2.238 |
| `exp.qk` | 0.90 / 0.78 / 1.45 / 1.60 | 0.99987 | yes | 0.0066 / 0.0099 / 0.0106 | 0.0017 / 0.0029 | 0 | -0.068 / +0.035 | 0.772 / 1.248 |
| `lm.pv.key` | 1.11 / 0.90 / 1.89 / 2.10 | 0.99979 | yes | 0.0081 / 0.0128 / 0.0138 | 0.0020 / 0.0032 | 0 | +0.141 / +0.300 | 0.940 / 1.436 |
| `lm.pv.fold` | 1.13 / 0.93 / 1.84 / 2.01 | 0.99981 | yes | 0.0081 / 0.0117 / 0.0127 | 0.0020 / 0.0030 | 0 | +0.163 / +0.435 | 0.982 / 1.476 |
| `lm.pv.col` | 1.08 / 0.92 / 1.83 / 2.05 | 0.99980 | yes | 0.0073 / 0.0116 / 0.0131 | 0.0019 / 0.0034 | 0 | +0.112 / +0.196 | 0.915 / 1.411 |
| `lm.qk` | 0.98 / 0.80 / 1.56 / 1.56 | 0.99988 | yes | 0.0071 / 0.0098 / 0.0106 | 0.0018 / 0.0026 | 0 | +0.007 / +0.290 | 0.818 / 1.461 |
| `lm.both.fold` | 1.14 / 0.95 / 2.00 / 2.34 | 0.99974 | yes | 0.0082 / 0.0120 / 0.0122 | 0.0019 / 0.0028 | 0 | +0.167 / +0.441 | 0.978 / 1.457 |
| `lm+exp.both.fold` | 1.46 / 1.23 / 2.75 / 3.15 | 0.99954 | yes | 0.0083 / 0.0130 / 0.0133 | 0.0022 / 0.0039 | 0 | +0.490 / +2.060 | 1.567 / 2.806 |
| `lm+exp.both.key` | 1.38 / 1.28 / 2.25 / 2.45 | 0.99972 | yes | 0.0085 / 0.0108 / 0.0110 | 0.0021 / 0.0029 | 0 | +0.406 / +1.355 | 1.421 / 2.215 |
| `base_vis` | 0.93 / 0.84 / 1.53 / 1.63 | 0.99987 | yes | 0.0071 / 0.0097 / 0.0101 | 0.0017 / 0.0025 | 0 | — | — |
| `vis.qk` | 1.00 / 0.90 / 1.59 / 1.63 | 0.99987 | yes | 0.0074 / 0.0117 / 0.0127 | 0.0018 / 0.0028 | 0 | +0.072 / +0.288 | 0.887 / 1.665 |
| `vis.pv.fold` | 1.03 / 0.85 / 1.82 / 1.90 | 0.99982 | yes | 0.0079 / 0.0106 / 0.0113 | 0.0019 / 0.0023 | 0 | +0.106 / +0.315 | 0.917 / 1.707 |
| `vis.pv.key` | 1.10 / 0.91 / 2.13 / 2.31 | 0.99975 | yes | 0.0080 / 0.0137 / 0.0153 | 0.0019 / 0.0032 | 0 | +0.177 / +0.689 | 0.966 / 1.928 |

### INT8 attention localisation: one expert layer INT8 alone, by ΔG3 vs the base (attn_loc)

Frames: 8. INT8-attention localisation on the smooth_lm prefix + dyn_smooth expert (SigLIP rows on the smooth prefix): which op (QK^T / PV), which part (SigLIP / Gemma / expert), which expert layers; V granularity key = per key (not one accumulation), fold = key scale folded into P, col = per channel

| run | G3 mean / median / p95 / max % | cos min | G3 pass | joint max-abs rad mean / p95 / max | joint RMS rad mean / max | frames > 0.02 rad | ΔG3 vs base mean / max % | chunk distance to base mean / max % |
|---|---:|---:|---|---:|---:|---:|---:|---:|
| `iso.exp.L17.pv.fold` | 1.08 / 0.89 / 1.73 / 1.80 | 0.99984 | yes | 0.0088 / 0.0118 / 0.0122 | 0.0019 / 0.0028 | 0 | +0.107 / +0.524 | 0.906 / 1.413 |
| `iso.exp.L16.pv.fold` | 0.98 / 0.87 / 1.52 / 1.61 | 0.99988 | yes | 0.0078 / 0.0108 / 0.0121 | 0.0018 / 0.0032 | 0 | +0.011 / +0.263 | 0.798 / 1.412 |
| `iso.exp.L11.pv.fold` | 0.97 / 0.84 / 1.62 / 1.79 | 0.99985 | yes | 0.0079 / 0.0115 / 0.0121 | 0.0018 / 0.0029 | 0 | -0.001 / +0.032 | 0.742 / 1.200 |
| `iso.exp.L17.qk` | 0.96 / 0.83 / 1.69 / 1.93 | 0.99982 | yes | 0.0078 / 0.0108 / 0.0112 | 0.0018 / 0.0031 | 0 | -0.008 / +0.027 | 0.704 / 1.133 |
| `iso.exp.L3.qk` | 0.96 / 0.80 / 1.72 / 1.92 | 0.99982 | yes | 0.0073 / 0.0105 / 0.0110 | 0.0017 / 0.0026 | 0 | -0.010 / +0.083 | 0.722 / 1.225 |
| `iso.exp.L5.pv.fold` | 0.96 / 0.81 / 1.57 / 1.74 | 0.99985 | yes | 0.0074 / 0.0095 / 0.0096 | 0.0017 / 0.0025 | 0 | -0.012 / +0.079 | 0.771 / 1.210 |
| `iso.exp.L2.qk` | 0.96 / 0.82 / 1.63 / 1.83 | 0.99984 | yes | 0.0072 / 0.0108 / 0.0113 | 0.0017 / 0.0028 | 0 | -0.014 / +0.014 | 0.739 / 1.218 |
| `iso.exp.L14.pv.fold` | 0.95 / 0.89 / 1.51 / 1.64 | 0.99987 | yes | 0.0076 / 0.0113 / 0.0126 | 0.0018 / 0.0030 | 0 | -0.016 / +0.066 | 0.753 / 1.214 |
| `iso.exp.L13.qk` | 0.95 / 0.80 / 1.61 / 1.78 | 0.99985 | yes | 0.0071 / 0.0106 / 0.0113 | 0.0018 / 0.0030 | 0 | -0.017 / +0.036 | 0.730 / 1.198 |
| `iso.exp.L9.qk` | 0.95 / 0.81 / 1.58 / 1.75 | 0.99985 | yes | 0.0077 / 0.0104 / 0.0114 | 0.0018 / 0.0030 | 0 | -0.017 / +0.032 | 0.717 / 1.205 |
| `iso.exp.L12.pv.fold` | 0.95 / 0.82 / 1.59 / 1.73 | 0.99986 | yes | 0.0076 / 0.0101 / 0.0109 | 0.0018 / 0.0030 | 0 | -0.017 / +0.070 | 0.737 / 1.215 |
| `iso.exp.L10.qk` | 0.95 / 0.83 / 1.58 / 1.72 | 0.99986 | yes | 0.0075 / 0.0114 / 0.0126 | 0.0018 / 0.0030 | 0 | -0.019 / +0.037 | 0.727 / 1.188 |
| `iso.exp.L12.qk` | 0.95 / 0.80 / 1.60 / 1.77 | 0.99985 | yes | 0.0079 / 0.0104 / 0.0106 | 0.0018 / 0.0029 | 0 | -0.020 / +0.032 | 0.727 / 1.177 |
| `iso.exp.L8.pv.fold` | 0.95 / 0.82 / 1.59 / 1.75 | 0.99985 | yes | 0.0070 / 0.0103 / 0.0114 | 0.0018 / 0.0030 | 0 | -0.020 / +0.033 | 0.716 / 1.179 |
| `iso.exp.L7.qk` | 0.95 / 0.82 / 1.55 / 1.68 | 0.99986 | yes | 0.0071 / 0.0106 / 0.0120 | 0.0018 / 0.0030 | 0 | -0.021 / +0.058 | 0.731 / 1.216 |
| `iso.exp.L15.qk` | 0.95 / 0.79 / 1.59 / 1.76 | 0.99985 | yes | 0.0079 / 0.0111 / 0.0124 | 0.0018 / 0.0030 | 0 | -0.021 / +0.023 | 0.737 / 1.154 |
| `iso.exp.L16.qk` | 0.95 / 0.81 / 1.61 / 1.80 | 0.99985 | yes | 0.0075 / 0.0113 / 0.0120 | 0.0018 / 0.0031 | 0 | -0.022 / +0.029 | 0.723 / 1.151 |
| `iso.exp.L5.qk` | 0.95 / 0.81 / 1.57 / 1.71 | 0.99986 | yes | 0.0078 / 0.0118 / 0.0139 | 0.0018 / 0.0031 | 0 | -0.025 / +0.044 | 0.731 / 1.241 |
| `iso.exp.L13.pv.fold` | 0.94 / 0.83 / 1.56 / 1.71 | 0.99986 | yes | 0.0078 / 0.0121 / 0.0138 | 0.0017 / 0.0029 | 0 | -0.026 / +0.039 | 0.729 / 1.228 |
| `iso.exp.L4.qk` | 0.94 / 0.78 / 1.67 / 1.89 | 0.99983 | yes | 0.0074 / 0.0114 / 0.0121 | 0.0018 / 0.0032 | 0 | -0.026 / +0.018 | 0.719 / 1.142 |
| `iso.exp.L15.pv.fold` | 0.94 / 0.81 / 1.56 / 1.71 | 0.99986 | yes | 0.0074 / 0.0113 / 0.0127 | 0.0018 / 0.0030 | 0 | -0.027 / +0.026 | 0.739 / 1.161 |
| `iso.exp.L14.qk` | 0.94 / 0.82 / 1.60 / 1.74 | 0.99985 | yes | 0.0071 / 0.0102 / 0.0115 | 0.0018 / 0.0031 | 0 | -0.029 / +0.082 | 0.722 / 1.123 |
| `iso.exp.L11.qk` | 0.94 / 0.81 / 1.59 / 1.73 | 0.99985 | yes | 0.0082 / 0.0118 / 0.0131 | 0.0018 / 0.0031 | 0 | -0.029 / +0.053 | 0.721 / 1.267 |
| `iso.exp.L6.qk` | 0.94 / 0.78 / 1.63 / 1.82 | 0.99984 | yes | 0.0070 / 0.0105 / 0.0115 | 0.0017 / 0.0030 | 0 | -0.029 / +0.017 | 0.720 / 1.169 |
| `iso.exp.L10.pv.fold` | 0.94 / 0.77 / 1.59 / 1.73 | 0.99986 | yes | 0.0075 / 0.0104 / 0.0114 | 0.0018 / 0.0030 | 0 | -0.029 / +0.039 | 0.744 / 1.209 |
| `iso.exp.L9.pv.fold` | 0.94 / 0.81 / 1.45 / 1.57 | 0.99988 | yes | 0.0081 / 0.0113 / 0.0122 | 0.0018 / 0.0031 | 0 | -0.031 / +0.111 | 0.777 / 1.215 |
| `iso.exp.L3.pv.fold` | 0.94 / 0.80 / 1.52 / 1.69 | 0.99986 | yes | 0.0076 / 0.0108 / 0.0119 | 0.0018 / 0.0027 | 0 | -0.034 / +0.040 | 0.748 / 1.267 |
| `iso.exp.L1.pv.fold` | 0.93 / 0.80 / 1.52 / 1.66 | 0.99987 | yes | 0.0072 / 0.0103 / 0.0114 | 0.0018 / 0.0029 | 0 | -0.037 / +0.043 | 0.746 / 1.253 |
| `iso.exp.L8.qk` | 0.93 / 0.82 / 1.52 / 1.64 | 0.99987 | yes | 0.0076 / 0.0104 / 0.0117 | 0.0018 / 0.0030 | 0 | -0.045 / +0.011 | 0.731 / 1.200 |
| `iso.exp.L1.qk` | 0.92 / 0.82 / 1.49 / 1.61 | 0.99987 | yes | 0.0070 / 0.0099 / 0.0106 | 0.0017 / 0.0028 | 0 | -0.045 / +0.028 | 0.729 / 1.157 |
| `iso.exp.L6.pv.fold` | 0.92 / 0.80 / 1.53 / 1.69 | 0.99986 | yes | 0.0076 / 0.0116 / 0.0127 | 0.0017 / 0.0029 | 0 | -0.045 / +0.028 | 0.739 / 1.154 |
| `iso.exp.L2.pv.fold` | 0.92 / 0.82 / 1.48 / 1.58 | 0.99988 | yes | 0.0076 / 0.0104 / 0.0105 | 0.0018 / 0.0029 | 0 | -0.046 / +0.031 | 0.765 / 1.228 |
| `iso.exp.L7.pv.fold` | 0.92 / 0.81 / 1.51 / 1.67 | 0.99987 | yes | 0.0075 / 0.0108 / 0.0110 | 0.0017 / 0.0029 | 0 | -0.049 / +0.032 | 0.750 / 1.203 |
| `iso.exp.L0.pv.fold` | 0.92 / 0.77 / 1.44 / 1.52 | 0.99989 | yes | 0.0070 / 0.0116 / 0.0133 | 0.0018 / 0.0032 | 0 | -0.049 / +0.224 | 0.848 / 1.412 |
| `iso.exp.L4.pv.fold` | 0.92 / 0.81 / 1.43 / 1.48 | 0.99990 | yes | 0.0070 / 0.0092 / 0.0095 | 0.0017 / 0.0024 | 0 | -0.054 / +0.067 | 0.767 / 1.275 |
| `iso.exp.L0.qk` | 0.90 / 0.79 / 1.39 / 1.46 | 0.99990 | yes | 0.0069 / 0.0104 / 0.0117 | 0.0017 / 0.0028 | 0 | -0.072 / +0.024 | 0.770 / 1.286 |

### INT8 attention: local error per module (attn_local)

Frames: demo1_ep20:2, demo1_ep20:8, demo1_ep40:2, demo1_ep40:8, recov_pi0_ep00:2, recov_pi0_ep00:8, recov_pi0_ep10:2, recov_pi0_ep10:8; prefix `smooth`, expert `dyn_smooth`; cfg {'qk': True, 'pv': True, 'v_gran': 'fold', 'p_levels': 127}. Local rel error of the attention output with INT8 on the same q/k/v; dead mass = softmax mass per row whose P code is 0.

| part | layers | both % mean / max | QK only % mean / max | PV only % mean / max | dead P mass mean / max % |
|---|---:|---:|---:|---:|---:|
| vis | 27 | 3.69 / 12.66 | 0.63 / 0.95 | 3.62 / 12.76 | 3.02 / 50.33 |
| lm | 18 | 8.77 / 25.04 | 0.71 / 0.88 | 8.72 / 25.03 | 10.37 / 68.31 |
| exp | 18 | 3.96 / 7.40 | 0.36 / 0.56 | 3.94 / 7.42 | 7.45 / 74.06 |

| module | both % | QK % | PV % | dead P mass mean / max % |
|---|---:|---:|---:|---:|
| lm.L4 | 25.04 | 0.85 | 24.98 | 28.23 / 56.83 |
| lm.L5 | 25.02 | 0.66 | 25.03 | 17.85 / 46.63 |
| lm.L3 | 13.82 | 0.65 | 13.78 | 20.26 / 50.29 |
| lm.L9 | 13.19 | 0.85 | 13.18 | 22.94 / 55.28 |
| vis.L0 | 12.66 | 0.95 | 12.76 | 6.41 / 50.33 |
| vis.L22 | 10.08 | 0.54 | 10.08 | 6.90 / 37.99 |
| lm.L2 | 9.48 | 0.65 | 9.44 | 9.43 / 34.84 |
| lm.L11 | 9.16 | 0.85 | 9.09 | 9.10 / 29.24 |
| lm.L1 | 8.79 | 0.68 | 8.80 | 6.75 / 36.91 |
| lm.L10 | 8.59 | 0.81 | 8.57 | 14.07 / 43.11 |
| exp.L1 | 7.40 | 0.56 | 7.42 | 6.21 / 60.07 |
| lm.L0 | 7.17 | 0.27 | 7.17 | 10.09 / 35.16 |

### INT8 attention: fixes (attn_fix)

Frames: 8. INT8-attention fixes, expert attention only (smooth_lm prefix + dyn_smooth expert): p255 = P as uint8; gN = one P scale per group of N keys (N accumulations per output); topK = the K largest P of each row outside the INT8 GEMM (float); col = V per channel; qk_only = PV in float

| run | G3 mean / median / p95 / max % | cos min | G3 pass | joint max-abs rad mean / p95 / max | joint RMS rad mean / max | frames > 0.02 rad | ΔG3 vs base mean / max % | chunk distance to base mean / max % |
|---|---:|---:|---|---:|---:|---:|---:|---:|
| `base` | 0.97 / 0.83 / 1.68 / 1.90 | 0.99983 | yes | 0.0081 / 0.0119 / 0.0130 | 0.0018 / 0.0031 | 0 | — | — |
| `exp.fold` | 1.43 / 1.34 / 2.42 / 2.68 | 0.99965 | yes | 0.0094 / 0.0132 / 0.0136 | 0.0022 / 0.0033 | 0 | +0.460 / +1.592 | 1.498 / 2.712 |
| `exp.fold.p255` | 1.05 / 1.08 / 1.49 / 1.56 | 0.99988 | yes | 0.0076 / 0.0102 / 0.0111 | 0.0019 / 0.0028 | 0 | +0.077 / +0.468 | 1.070 / 1.959 |
| `exp.fold.g64` | 0.93 / 0.82 / 1.50 / 1.62 | 0.99987 | yes | 0.0072 / 0.0109 / 0.0126 | 0.0018 / 0.0031 | 0 | -0.036 / +0.084 | 0.796 / 1.240 |
| `exp.fold.g16` | 0.92 / 0.83 / 1.47 / 1.57 | 0.99988 | yes | 0.0072 / 0.0105 / 0.0112 | 0.0018 / 0.0030 | 0 | -0.048 / +0.045 | 0.816 / 1.284 |
| `exp.fold.top1` | 0.99 / 0.94 / 1.53 / 1.64 | 0.99987 | yes | 0.0068 / 0.0096 / 0.0097 | 0.0018 / 0.0025 | 0 | +0.024 / +0.549 | 0.994 / 1.789 |
| `exp.fold.top4` | 0.94 / 0.86 / 1.40 / 1.49 | 0.99989 | yes | 0.0076 / 0.0107 / 0.0116 | 0.0018 / 0.0029 | 0 | -0.028 / +0.124 | 0.819 / 1.354 |
| `exp.col` | 1.29 / 1.29 / 1.95 / 2.09 | 0.99978 | yes | 0.0084 / 0.0106 / 0.0110 | 0.0020 / 0.0026 | 0 | +0.322 / +0.996 | 1.334 / 2.215 |
| `exp.col.p255` | 1.07 / 1.11 / 1.45 / 1.46 | 0.99989 | yes | 0.0080 / 0.0114 / 0.0126 | 0.0019 / 0.0032 | 0 | +0.097 / +0.537 | 0.986 / 1.593 |
| `exp.fold.qk_only` | 0.90 / 0.78 / 1.45 / 1.60 | 0.99987 | yes | 0.0066 / 0.0099 / 0.0106 | 0.0017 / 0.0029 | 0 | -0.068 / +0.035 | 0.772 / 1.248 |

### Hardware-format sensitivity (fmt_sens)

Frames: 8. Hardware-format sensitivity. A = smooth prefix (SigLIP W8A8 + Gemma W8A8 SmoothQuant) + dyn_smooth expert, attention fp; B = A + INT8 QK^T and PV (V per channel, P int8 per row); C = the chip candidate: A + INT8 QK^T and PV with P uint8 per row and V per channel in SigLIP, Gemma and the expert. Formats: out_bf16 = quantised GEMM results rounded to bf16; iface_bf16 = bf16 at every vector-op interface incl. the residual stream; row_* = dynamic row scales as 2^e x n-bit mantissa (m8/m6/m4) or 2^e (pow2), rounded up; tables = GELU/SiLU gates, softmax exp, RMSNorm/LayerNorm rsqrt through the vector unit's 2048-entry interpolated tables; acc = record max |acc| (no numeric change). fp32.* = the format alone on the fp32 model. ΔG3 = paired G3 change vs the base run; chunk distance = rel RMS of (chunk - base chunk) / |fp32 chunk|

| run | G3 mean / median / p95 / max % | cos min | G3 pass | joint max-abs rad mean / p95 / max | joint RMS rad mean / max | frames > 0.02 rad | ΔG3 vs base mean / max % | chunk distance to base mean / max % |
|---|---:|---:|---|---:|---:|---:|---:|---:|
| `fp32` | 0.00 / 0.00 / 0.00 / 0.00 | 1.00000 | yes | 0.0000 / 0.0000 / 0.0000 | 0.0000 / 0.0000 | 0 | — | — |
| `fp32.tables` | 0.00 / 0.00 / 0.00 / 0.00 | 1.00000 | yes | 0.0000 / 0.0000 / 0.0000 | 0.0000 / 0.0000 | 0 | +0.000 / +0.000 | 0.000 / 0.000 |
| `fp32.iface_bf16` | 0.27 / 0.25 / 0.38 / 0.41 | 0.99999 | yes | 0.0019 / 0.0035 / 0.0040 | 0.0005 / 0.0011 | 0 | +0.269 / +0.407 | 0.269 / 0.407 |
| `A` | 0.93 / 0.84 / 1.53 / 1.63 | 0.99987 | yes | 0.0071 / 0.0097 / 0.0101 | 0.0017 / 0.0025 | 0 | — | — |
| `A.acc` | 0.93 / 0.84 / 1.53 / 1.63 | 0.99987 | yes | 0.0071 / 0.0097 / 0.0101 | 0.0017 / 0.0025 | 0 | -0.000 / +0.000 | 0.000 / 0.000 |
| `A.out_bf16` | 0.97 / 0.87 / 1.58 / 1.70 | 0.99987 | yes | 0.0078 / 0.0108 / 0.0111 | 0.0019 / 0.0033 | 0 | +0.043 / +0.154 | 0.932 / 1.835 |
| `A.iface_bf16` | 0.98 / 0.82 / 1.57 / 1.72 | 0.99985 | yes | 0.0074 / 0.0097 / 0.0100 | 0.0018 / 0.0024 | 0 | +0.049 / +0.281 | 0.987 / 1.629 |
| `A.row_m8` | 0.95 / 0.81 / 1.49 / 1.58 | 0.99988 | yes | 0.0080 / 0.0103 / 0.0106 | 0.0018 / 0.0022 | 0 | +0.024 / +0.242 | 0.907 / 1.492 |
| `A.row_m6` | 0.92 / 0.93 / 1.33 / 1.46 | 0.99990 | yes | 0.0079 / 0.0100 / 0.0101 | 0.0018 / 0.0029 | 0 | -0.003 / +0.308 | 1.085 / 1.665 |
| `A.row_m4` | 0.95 / 0.86 / 1.53 / 1.71 | 0.99985 | yes | 0.0096 / 0.0151 / 0.0173 | 0.0020 / 0.0038 | 0 | +0.018 / +0.114 | 1.183 / 1.988 |
| `A.row_pow2` | 1.68 / 1.48 / 3.41 / 4.19 | 0.99913 | yes | 0.0118 / 0.0193 / 0.0207 | 0.0029 / 0.0054 | 1 | +0.750 / +2.567 | 1.694 / 3.551 |
| `A.tables` | 0.92 / 0.79 / 1.39 / 1.42 | 0.99990 | yes | 0.0076 / 0.0097 / 0.0100 | 0.0018 / 0.0023 | 0 | -0.006 / +0.154 | 0.830 / 1.545 |
| `A.all_m8` | 0.98 / 0.75 / 1.68 / 1.71 | 0.99986 | yes | 0.0078 / 0.0090 / 0.0094 | 0.0017 / 0.0023 | 0 | +0.053 / +0.280 | 1.005 / 1.496 |
| `B` | 1.35 / 1.33 / 2.06 / 2.08 | 0.99979 | yes | 0.0085 / 0.0124 / 0.0134 | 0.0022 / 0.0031 | 0 | +0.427 / +1.088 | 1.368 / 2.096 |
| `C` | 1.05 / 0.96 / 1.54 / 1.55 | 0.99988 | yes | 0.0080 / 0.0100 / 0.0101 | 0.0019 / 0.0026 | 0 | +0.122 / +0.538 | 1.073 / 1.877 |
| `C.acc` | 1.05 / 0.96 / 1.54 / 1.55 | 0.99988 | yes | 0.0080 / 0.0100 / 0.0101 | 0.0019 / 0.0026 | 0 | +0.000 / +0.000 | 0.000 / 0.000 |
| `C.out_bf16` | 1.09 / 1.02 / 1.70 / 1.83 | 0.99985 | yes | 0.0073 / 0.0100 / 0.0102 | 0.0018 / 0.0023 | 0 | +0.043 / +0.285 | 0.883 / 1.716 |
| `C.iface_bf16` | 1.07 / 1.11 / 1.51 / 1.57 | 0.99989 | yes | 0.0075 / 0.0094 / 0.0095 | 0.0019 / 0.0028 | 0 | +0.024 / +0.421 | 0.982 / 1.726 |
| `C.row_m8` | 0.97 / 0.92 / 1.48 / 1.63 | 0.99987 | yes | 0.0068 / 0.0086 / 0.0087 | 0.0017 / 0.0023 | 0 | -0.079 / +0.086 | 0.979 / 1.833 |
| `C.row_m4` | 1.01 / 0.93 / 1.53 / 1.65 | 0.99987 | yes | 0.0084 / 0.0123 / 0.0134 | 0.0020 / 0.0033 | 0 | -0.041 / +0.144 | 1.203 / 2.265 |
| `C.row_pow2` | 1.72 / 1.24 / 3.48 / 3.86 | 0.99928 | yes | 0.0111 / 0.0148 / 0.0150 | 0.0029 / 0.0045 | 0 | +0.673 / +2.337 | 1.641 / 2.926 |
| `C.tables` | 1.04 / 1.07 / 1.49 / 1.57 | 0.99988 | yes | 0.0077 / 0.0107 / 0.0115 | 0.0018 / 0.0028 | 0 | -0.006 / +0.108 | 0.845 / 1.535 |
| `C.all_m8` | 1.02 / 0.92 / 1.60 / 1.72 | 0.99986 | yes | 0.0075 / 0.0105 / 0.0107 | 0.0018 / 0.0025 | 0 | -0.033 / +0.170 | 1.022 / 1.885 |

### Hardware-format sensitivity on 46 frames (fmt_sens_46)

Frames: 46. 46 held-out frames: the chip candidate (smooth prefix + dyn_smooth expert, INT8 QK^T + PV with uint8 P per row and V per channel everywhere) without and with all hardware formats (row scales 2^e x 8-bit mantissa, bf16 at every vector-op interface, 2048-entry interpolated tables)

| run | G3 mean / median / p95 / max % | cos min | G3 pass | joint max-abs rad mean / p95 / max | joint RMS rad mean / max | frames > 0.02 rad | ΔG3 vs base mean / max % | chunk distance to base mean / max % |
|---|---:|---:|---|---:|---:|---:|---:|---:|
| `chip` | 0.94 / 0.90 / 1.51 / 1.69 | 0.99986 | yes | 0.0080 / 0.0113 / 0.0132 | 0.0019 / 0.0034 | 0 | — | — |
| `chip.all_m8` | 0.98 / 0.87 / 1.57 / 1.72 | 0.99986 | yes | 0.0085 / 0.0119 / 0.0150 | 0.0021 / 0.0034 | 0 | +0.039 / +0.791 | 0.905 / 1.885 |

### 48-bit accumulation: max |acc| per GEMM class (fmt_sens)

Run `A.acc`; |acc| = exact integer sum of int8 x int8 products of one GEMM output element, over 8 frames; bound = 127 x levels x K (K = inputs per output).

| GEMM class | K max | calls | max abs acc | log2 max | bound 127²K | log2 bound | headroom to 2^47, bits (measured / worst case) | worst GEMM |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| exp.act_in | 32 | 80 | 104639 | 16.7 | 516128 | 19.0 | 30.3 / 28.0 | exp.act_in |
| exp.act_out | 1024 | 80 | 115941 | 16.8 | 16516096 | 24.0 | 30.2 / 23.0 | exp.act_out |
| exp.down | 4096 | 1440 | 1173785 | 20.2 | 66064384 | 26.0 | 26.8 / 21.0 | exp.L17.down |
| exp.gate | 1024 | 1440 | 420913 | 18.7 | 16516096 | 24.0 | 28.3 / 23.0 | exp.L0.gate |
| exp.k | 1024 | 1440 | 249443 | 17.9 | 16516096 | 24.0 | 29.1 / 23.0 | exp.L15.k |
| exp.o | 2048 | 1440 | 1557979 | 20.6 | 33032192 | 25.0 | 26.4 / 22.0 | exp.L17.o |
| exp.q | 1024 | 1440 | 435853 | 18.7 | 16516096 | 24.0 | 28.3 / 23.0 | exp.L0.q |
| exp.state | 32 | 80 | 48533 | 15.6 | 516128 | 19.0 | 31.4 / 28.0 | exp.state |
| exp.tm_in | 2048 | 80 | 1130959 | 20.1 | 33032192 | 25.0 | 26.9 / 22.0 | exp.tm_in |
| exp.tm_out | 1024 | 80 | 1198878 | 20.2 | 16516096 | 24.0 | 26.8 / 23.0 | exp.tm_out |
| exp.up | 1024 | 1440 | 411641 | 18.7 | 16516096 | 24.0 | 28.3 / 23.0 | exp.L15.up |
| exp.v | 1024 | 1440 | 404658 | 18.6 | 16516096 | 24.0 | 28.4 / 23.0 | exp.L3.v |
| lm.down | 16384 | 144 | 596731 | 19.2 | 264257536 | 28.0 | 27.8 / 19.0 | lm.L1.down |
| lm.gate | 2048 | 144 | 491727 | 18.9 | 33032192 | 25.0 | 28.1 / 22.0 | lm.L16.gate |
| lm.k | 2048 | 144 | 581341 | 19.1 | 33032192 | 25.0 | 27.9 / 22.0 | lm.L16.k |
| lm.o | 2048 | 144 | 1451735 | 20.5 | 33032192 | 25.0 | 26.5 / 22.0 | lm.L0.o |
| lm.q | 2048 | 144 | 794443 | 19.6 | 33032192 | 25.0 | 27.4 / 22.0 | lm.L0.q |
| lm.up | 2048 | 144 | 750034 | 19.5 | 33032192 | 25.0 | 27.5 / 22.0 | lm.L16.up |
| lm.v | 2048 | 144 | 387178 | 18.6 | 33032192 | 25.0 | 28.4 / 22.0 | lm.L17.v |
| vis.fc1 | 1152 | 432 | 1227333 | 20.2 | 18580608 | 24.1 | 26.8 / 22.9 | vis.L18.fc1 |
| vis.fc2 | 4304 | 432 | 3101539 | 21.6 | 69419216 | 26.0 | 25.4 / 21.0 | vis.L2.fc2 |
| vis.k | 1152 | 432 | 292855 | 18.2 | 18580608 | 24.1 | 28.8 / 22.9 | vis.L0.k |
| vis.mmproj | 1152 | 16 | 242138 | 17.9 | 18580608 | 24.1 | 29.1 / 22.9 | vis.mmproj |
| vis.out | 1152 | 432 | 694862 | 19.4 | 18580608 | 24.1 | 27.6 / 22.9 | vis.L19.out |
| vis.patch | 588 | 16 | 3746132 | 21.8 | 9483852 | 23.2 | 25.2 / 23.8 | vis.patch |
| vis.q | 1152 | 432 | 320011 | 18.3 | 18580608 | 24.1 | 28.7 / 22.9 | vis.L13.q |
| vis.v | 1152 | 432 | 242353 | 17.9 | 18580608 | 24.1 | 29.1 / 22.9 | vis.L13.v |

### Expert float fallback for the most sensitive GEMMs on the chip candidate (exp_fallback)

Frames: 8. Float fallback for the expert's most sensitive GEMMs on the chip candidate (smooth prefix + dyn_smooth expert, INT8 QK^T + PV with uint8 P / V per channel everywhere): expert layer 0 (7 GEMMs), layer 0 + action_in_proj, and layers 0 + 17 (14 GEMMs) kept in float (stands in for bf16; bf16 on every interface of the whole model costs +0.02-0.05 % G3)

| run | G3 mean / median / p95 / max % | cos min | G3 pass | joint max-abs rad mean / p95 / max | joint RMS rad mean / max | frames > 0.02 rad | ΔG3 vs base mean / max % | chunk distance to base mean / max % |
|---|---:|---:|---|---:|---:|---:|---:|---:|
| `chip` | 1.05 / 0.96 / 1.54 / 1.55 | 0.99988 | yes | 0.0080 / 0.0100 / 0.0101 | 0.0019 / 0.0026 | 0 | — | — |
| `chip.expL0_fp` | 1.02 / 0.92 / 1.62 / 1.72 | 0.99985 | yes | 0.0075 / 0.0114 / 0.0127 | 0.0018 / 0.0028 | 0 | -0.028 / +0.198 | 0.751 / 1.196 |
| `chip.expL0_actin_fp` | 0.93 / 0.85 / 1.52 / 1.66 | 0.99986 | yes | 0.0059 / 0.0081 / 0.0087 | 0.0015 / 0.0024 | 0 | -0.123 / +0.138 | 0.652 / 1.052 |
| `chip.expL0L17_fp` | 1.03 / 0.92 / 1.64 / 1.76 | 0.99985 | yes | 0.0069 / 0.0093 / 0.0099 | 0.0018 / 0.0026 | 0 | -0.020 / +0.235 | 0.770 / 1.301 |

### Realisable fixes for expert layer 0: SmoothQuant alpha, grouped activation scales (exp_l0fix)

Frames: 8. Realisable fixes for the expert's layer 0 on the chip candidate (smooth prefix + dyn_smooth expert, INT8 QK^T + PV with uint8 P / V per channel everywhere), 8 frames. chip.m8 = chip with every dynamic row scale as 2^e x 8-bit mantissa (the chip's format) and is the base of every fix. Upper bound: expert L0 in float (not realisable on the INT8 chain). alphaA = SmoothQuant alpha A for expert L0 only (the rest 0.5); gG = the per-token activation scale of the L0 GEMM inputs split into G contiguous column groups, each its own 2^e x m8 scale (G sub-row passes + one DEQUANT per group + ADD on the chip); io_gG = the same for action_in_proj / state_proj inputs (32 columns)

| run | G3 mean / median / p95 / max % | cos min | G3 pass | joint max-abs rad mean / p95 / max | joint RMS rad mean / max | frames > 0.02 rad | ΔG3 vs base mean / max % | chunk distance to base mean / max % |
|---|---:|---:|---|---:|---:|---:|---:|---:|
| `chip` | 1.05 / 0.96 / 1.54 / 1.55 | 0.99988 | yes | 0.0080 / 0.0100 / 0.0101 | 0.0019 / 0.0026 | 0 | — | — |
| `chip.expL0_fp` | 1.02 / 0.92 / 1.62 / 1.72 | 0.99985 | yes | 0.0075 / 0.0114 / 0.0127 | 0.0018 / 0.0028 | 0 | -0.028 / +0.198 | 0.751 / 1.196 |
| `chip.m8` | 0.97 / 0.92 / 1.48 / 1.63 | 0.99987 | yes | 0.0068 / 0.0086 / 0.0087 | 0.0017 / 0.0023 | 0 | -0.079 / +0.086 | 0.979 / 1.833 |
| `chip.m8.expL0_fp` | 0.94 / 0.88 / 1.45 / 1.61 | 0.99987 | yes | 0.0061 / 0.0092 / 0.0101 | 0.0016 / 0.0025 | 0 | -0.028 / +0.052 | 0.793 / 1.342 |
| `chip.m8.L0_alpha0.3` | 1.02 / 0.94 / 1.49 / 1.65 | 0.99987 | yes | 0.0083 / 0.0125 / 0.0146 | 0.0019 / 0.0032 | 0 | +0.048 / +0.223 | 0.910 / 1.389 |
| `chip.m8.L0_alpha0.65` | 0.97 / 0.94 / 1.37 / 1.49 | 0.99989 | yes | 0.0066 / 0.0094 / 0.0100 | 0.0017 / 0.0023 | 0 | -0.004 / +0.136 | 0.828 / 1.338 |
| `chip.m8.L0_alpha0.8` | 1.02 / 0.94 / 1.56 / 1.72 | 0.99986 | yes | 0.0072 / 0.0087 / 0.0090 | 0.0018 / 0.0022 | 0 | +0.052 / +0.113 | 0.848 / 1.326 |
| `chip.m8.L0_g4` | 1.01 / 0.95 / 1.52 / 1.69 | 0.99986 | yes | 0.0071 / 0.0094 / 0.0100 | 0.0018 / 0.0026 | 0 | +0.034 / +0.079 | 0.824 / 1.341 |
| `chip.m8.L0_g16` | 0.98 / 0.91 / 1.54 / 1.75 | 0.99985 | yes | 0.0077 / 0.0115 / 0.0122 | 0.0018 / 0.0029 | 0 | +0.009 / +0.121 | 0.805 / 1.302 |
| `chip.m8.L0_g64` | 0.98 / 0.88 / 1.55 / 1.76 | 0.99985 | yes | 0.0073 / 0.0102 / 0.0109 | 0.0018 / 0.0027 | 0 | +0.007 / +0.124 | 0.822 / 1.302 |
| `chip.m8.io_g4` | 0.94 / 0.90 / 1.40 / 1.54 | 0.99989 | yes | 0.0072 / 0.0106 / 0.0119 | 0.0017 / 0.0021 | 0 | -0.028 / +0.042 | 0.742 / 1.161 |
| `chip.m8.io_g16` | 0.93 / 0.89 / 1.40 / 1.51 | 0.99990 | yes | 0.0067 / 0.0083 / 0.0087 | 0.0017 / 0.0021 | 0 | -0.042 / +0.104 | 0.812 / 1.192 |
| `chip.m8.L0_g16.io_g4` | 0.95 / 0.94 / 1.44 / 1.64 | 0.99987 | yes | 0.0072 / 0.0109 / 0.0129 | 0.0018 / 0.0031 | 0 | -0.023 / +0.132 | 0.769 / 1.175 |

