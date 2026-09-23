# pi0 vector-unit lane: op table

Workstream C of `docs/PI0_FULL_MODEL_ON_CHIP_DESIGN_20260916.md`.

- **Bit-exact spec:** `paper/sw/vector_unit_ref.py`.
- **RTL:** `paper/rtl/vector_unit/vu_lane.sv`, built from the units `vu_add.sv`, `vu_mul.sv`, `vu_tbl.sv`, `vu_round.sv`, `vu_qtab.sv` and `vu_pkg.sv`.
- **Verification:** `paper/rtl/run_vu_lane_sim.sh`.

Numbers tagged **[M]** are measured by a tool (simulation, synthesis, numpy on real activations). **[D]** marks a design decision, and **[P]** marks a count derived from the model shapes.

## 1. Formats and units

Stream I/O uses these formats:

- bf16 codes (the top half of an IEEE fp32) for activations.
- fp32 codes for per-column constants, row scalars and the Euler state.
- int8 codes out of QUANT and SMAX_Q8.
- int48 GEMM sums into DEQUANT.

Inside the lane, every value is an `xf`: sign, 12-bit signed exponent and a 24-bit mantissa (fp32 precision). Every unit truncates to 24 bits; only the stream output rounds, round-to-nearest-even. The function tables are BRAM72K ROMs of 2048 × 36-bit words {V Q1.20, D = V[i+1]−V[i]}, linearly interpolated with a 12-bit fraction. They cover the GELU gate φ (domain [−8, 8)), the sigmoid ([−16, 16)), 2^−f ([0, 1)) and rsqrt (two binades). Workstream A's harness emulates these layouts, so they are fixed.

| unit | latency (cycles) | function |
|---|---:|---|
| U | 3 | unpack bf16/fp32, or int (≤ 54 bit) → xf (n64: abs, leading zeros, normalise) |
| ADD | 6 | compare, swap + clamp, align (3 guard bits), add/sub, leading zeros, normalise |
| PREP | 5 | table address / fraction / exponent offset for gate, exp or rsqrt |
| TBL + POST | 6 + 1 | BRAM read, select, slope × fraction, interpolate, leading zeros, normalise; gate saturation |
| MUL | 4 | 24×24 product (synthesis infers an MLP72), normalise |
| ROUND | 3 | xf → bf16 / fp32 / int8 (RNE, saturating) |

Every token takes the same path:

`S0 → U → ADD1 ∥ (PREP → TBL → POST) → MUL1 → MUL2 → ADD2 → ROUND → FIFO`

A unit an op does not use gets an identity operand: `add(x, 0)` and `mul(x, 1.0)` are exact in the ref. The stream code leaves the pipeline 32 cycles after S0 [D]. Flow control is by credits into a 64-deep output FIFO.

## 2. Ops

The table lists each op's composition, in lane stage order. U, ADD1 and TBL act on the element x; MUL1 is P × Q, MUL2 multiplies by Q2, and ADD2 adds E2. b, c, d and e are the lane's per-element operand ports (fp32 unless marked bf16). Row scalars are handed over as fp32 codes, constant along the row. The last column is the per-row control overhead in cycles, at no input gaps and no back-pressure [M]. It comes from `build/paper_vu_sim/run_tp.log`: 1 row versus 8 rows of 256 elements, all bit-exact.

| op | composition | output beats | row reduction | overhead / row [M] |
|---|---|---|---|---:|
| ADD | a (bf16) + b (bf16, or fp32 position table) | bf16 + summary{amax, max} | no | 1 |
| RMS_STAT | acc2 = Σ m8²≪(2(e−E)+24); microcode s0: n64(acc2, Q.38, 2^2E) × K(=1/N) + ε → T; s1: rsqrt(T) → r | summary{rs0 = r} | yes, 2 steps | 71 |
| RMS_APPLY | a × r (c) × (1+w) (d) | bf16 + summary | no | 1 |
| LN_STAT | acc1 = Σ ±m8≪((e−E)+25), acc2 as above; s0: n64(acc1) × K → μ (rs0); s1: μ × μ; s2: n64(acc2) × K − μ² → var; s3: var + ε (ADD1); s4: rsqrt(var+ε, or ε if ≤ 0) → r | kind 2 {r}, summary{rs0 = μ} | yes, 5 steps | 169 |
| LN_APPLY | (a − μ (b)) × r (c) × γ (d) + β (e) | bf16 + summary | no | 1 |
| ROPE_A | x × cos (c) | fp32 (pass-A word) | no | 1 |
| ROPE_B | partner × ±sin (c) + pass-A word (e) | bf16 + summary | no | 1 |
| GELU | a × φ_gelu(a) (table; \|a\| ≥ 8 saturates) | bf16 + summary | no | 1 |
| GEGLU | g × φ_gelu(g) × up (d, bf16) | bf16 + summary | no | 1 |
| SILU | a × σ(a) (table; \|a\| ≥ 16 saturates) | bf16 + summary | no | 1 |
| SMAX_SUM | e_i = 2^−(fix(max)−fix(a)) (exp table, Q.23 log2 domain; 0 if masked or k ≥ 64); S = Σ y≪(8−k); s0: n64(S, Q.40) → rsqrt → r × r = 1/S | summary{rs0 = 1/S} | yes, 1 step | 50 |
| SMAX_OUT | e_i × 1/S (c) | bf16 + summary | no | 1 |
| SMAX_Q8 | e_i × 127 → int8 (one pass); S as SMAX_SUM; s0: 1/S × fp32(1/127) → s_row | int8 + summary{rs0 = s_row} | yes, 1 step | 50 |
| QUANT | a × R, R = 127/amax from `vu_tbl_quant.mem` (256 entries by amax mantissa and exponent parity, equal to the ref's rsqrt(amax)²·127 bit for bit) → int8 | int8 + summary{rs0 = s_row = amax·fp32(1/127)} | no (uses the row amax) | 1 |
| DEQUANT | n64(int48 sum) × s_row (c) × s_col (d) [+ bias (e)] | bf16 or fp32 + summary | no | 1 |
| EULER | v × dt (K) + x_t (e) | fp32 | no | 1 |

- **Row reductions** are RMS_STAT, LN_STAT, SMAX_SUM and SMAX_Q8. Their accumulators are fixed point and relative to the row statistic. They finish with microcode tokens through the same datapath, and the next row is accepted after the summary beat. Each microcode step costs ~32 cycles.
- **All other ops stream rows back to back.** The only gap is one idle cycle, which gives the summary beat its FIFO slot.

### 2.1 Row-statistic handover

- **RMS_STAT, LN_STAT and QUANT** need the amax code of their input row on `i_rs`. RMS_STAT and LN_STAT use only its exponent E, as the fixed point of the accumulators; QUANT uses the whole code.
- **SMAX_\*** need the signed maximum code over the valid keys.
- **Who supplies it:** every bf16-output op emits `{amax, max over mask}` of its output row in its summary beat. The sequencer keeps that pair with the row buffer and presents it to the next op. The softmax logits come from the QKᵀ DEQUANT, whose summary max already honours the key mask.
- **Rows that no lane op produced** need their amax from somewhere else. The only such case is the SigLIP patch rows from the host; the host sends their amax, or an `ADD b = 0` pass produces it.

### 2.2 Row-scalar handover

- RMS_STAT r → RMS_APPLY c
- LN_STAT μ, r → LN_APPLY b, c
- SMAX_SUM 1/S → SMAX_OUT c
- QUANT s_row → DEQUANT c of the GEMM that consumed the codes
- SMAX_Q8 s_row → the PV dequant

## 3. What the three models need

### 3.1 Model op to lane ops

- **SigLIP So400m/14** (27 layers, 2 images × 256 tokens)
  - Patch rows (588 taps): QUANT → chain → DEQUANT (+bias) → ADD (fp32 position table).
  - Each layer runs: LN_STAT, LN_APPLY → QUANT (q/k/v input, shared) → chain ×3 → DEQUANT ×3 (+bias).
  - Attention: QUANT of q, k, v per head row (72) → chain QKᵀ → DEQUANT logits → SMAX_Q8 → chain PV → DEQUANT.
  - The logit column scale folds in 1/√72 and 1/ln 2, because the softmax table works on log2-domain logits.
  - Rest of the layer: QUANT → chain out_proj → DEQUANT (+bias) → ADD (residual) → LN_STAT, LN_APPLY → QUANT → chain fc1 → DEQUANT (+bias) → GELU → QUANT → chain fc2 → DEQUANT (+bias) → ADD.
  - After the last layer: LN_STAT, LN_APPLY (post_layernorm) → QUANT → projector chain → DEQUANT (+bias). The 1/√2048 image-feature scale is folded into the projector.
- **PaliGemma prefix** (Gemma 2B, 18 layers, 525 tokens)
  - RMS_STAT, RMS_APPLY with d = fp32(1+w). SmoothQuant α = 0.5 is folded into w and the GEMM weights by the exporter [D].
  - QUANT → chain q/k/v → DEQUANT → ROPE_A, ROPE_B on q (8 heads) and k.
  - Attention: QUANT per head/token → chain QKᵀ → DEQUANT (column scale per key, incl. 1/(√256·ln 2)) → SMAX_Q8 → chain PV → DEQUANT.
  - Then: QUANT → chain o → DEQUANT → ADD → RMS_STAT, RMS_APPLY → QUANT → chain gate, up → DEQUANT ×2 → GEGLU → QUANT → chain down → DEQUANT → ADD.
  - K (after RoPE, bf16) and V (bf16) go to the KV cache, together with their INT8 per-token copies from QUANT.
- **Action expert** (Gemma 300M, 18 layers, 51 tokens, 867 keys, 10 Euler steps)
  - Embedding: QUANT (state / action rows) → chain → DEQUANT (+bias); QUANT (action ⧺ time) → chain → DEQUANT (+bias) → SILU → QUANT → chain → DEQUANT (+bias).
  - Layers: as the prefix layers, with 867 keys and the prefix padding masked through `i_mask`.
  - Final RMS pair → QUANT → chain action_out_proj → DEQUANT (+bias, fp32 out) → EULER.

Not lane ops (offline constants or folds): token embedding lookup and its √D normaliser, the sinusoidal time embedding (one row per step), concatenations, padding, and slicing the 7 action dimensions.

Attention alternative: with bf16 softmax probabilities, SMAX_Q8 becomes SMAX_SUM + SMAX_OUT and the attention QUANT disappears (second visit total in §3.2).

**Open numerics interface.** "V per key" INT8 in PV (the variant measured by workstream A) does not factor as `acc × s_row × s_col`: the per-key V scale sits inside the sum over keys. PV through this DEQUANT needs V scaled per column (head-dim channel): `s_row = s_P`, `s_col = s_V[j]`. That is the `attn_vcol` variant of `paper/sw/prefix_w8a8_eval.py`, which has to be measured, or V has to be rescaled before the chain. QKᵀ with K per key factors fine (s_col = s_K[key]).

### 3.2 Elements per 50-action chunk [P]

Source: `vector_unit_ref.inventory()`, written to `paper/data/vector_unit/inventory.json`. Design v2 attention is INT8 QKᵀ/PV with a one-pass SMAX_Q8. Element visits are elements × passes; the RMS and LN pairs are two passes over the same elements.

| lane op | SigLIP | LM prefix | expert (10 steps) | total |
|---|---:|---:|---:|---:|
| ADD | 32,440,320 | 38,707,200 | 18,800,640 | 89,948,160 |
| RMS_STAT | 0 | 38,707,200 | 19,312,640 | 58,019,840 |
| RMS_APPLY | 0 | 38,707,200 | 19,312,640 | 58,019,840 |
| LN_STAT | 32,440,320 | 0 | 0 | 32,440,320 |
| LN_APPLY | 32,440,320 | 0 | 0 | 32,440,320 |
| ROPE_A | 0 | 21,772,800 | 21,150,720 | 42,923,520 |
| ROPE_B | 0 | 21,772,800 | 21,150,720 | 42,923,520 |
| GELU | 59,498,496 | 0 | 0 | 59,498,496 |
| GEGLU | 0 | 154,828,800 | 37,601,280 | 192,430,080 |
| SILU | 0 | 0 | 512,000 | 512,000 |
| SMAX_Q8 | 56,623,104 | 39,690,000 | 63,672,480 | 159,985,584 |
| QUANT | 155,940,864 | 237,081,600 | 100,767,680 | 493,790,144 |
| EULER | 0 | 0 | 16,000 | 16,000 |
| **vector unit** | 369,383,424 | 591,267,600 | 302,296,800 | **1,262,947,824** |
| DEQUANT (GEMM results, fabric side of the chains) | 213,311,488 | 431,600,400 | 201,539,360 | 846,451,248 |
| **all** | | | | **2,109,399,072** |

- **Row widths:**
  - QUANT: 32 to 16,384 (the SigLIP head rows are 72).
  - SMAX_Q8: 256 / 525 / 867.
  - RMS: 1024 / 2048; LN: 1152.
  - ROPE: 256.
  - GELU: 4304; GEGLU: 4096 / 16,384.
  - EULER: 32.
- **bf16 attention:** 1,327,464,864 vector-unit visits (SMAX_SUM + SMAX_OUT, no attention QUANT).

### 3.3 Latency

- **Pipeline:** every op has the same pipeline latency, 32 cycles from S0 to the stream code. The FIFO write and head register add 2 more.
- **First row:** in simulation, a 256-element row finishes 292 cycles after its descriptor is loaded (256 + 36) [M].
- **Following rows:** the control overhead is the last column of §2 [M].
- **Steady state:** a lane processes one element per clock.

With the chunk's row mix, control overhead adds these fractions to the element count [M]:

| op class | overhead |
|---|---:|
| SMAX_Q8 | 11.6 % |
| LN pair | 7.4 % |
| RMS pair | 2.3 % |
| QUANT, GEGLU, ADD, GELU, ROPE, DEQUANT | ≤ 0.4 % |

Throughput projection: `paper/data/vector_unit/projection.json`.

## 4. Lane interface (`vu_lane.sv`)

| group | signals |
|---|---|
| descriptor (load while `o_idle`) | `i_op[3:0]`, `i_b_fp32`, `i_bias_en`, `i_out_fp32`, `i_k[31:0]` (fp32: 1/N or dt) |
| input stream | `i_valid`/`o_ready`, `i_last`, `i_mask`, `i_x[47:0]` (bf16 \| fp32 \| int48), `i_rs[15:0]`, `i_b/c/d/e[31:0]` |
| output stream | `o_valid`/`i_ready`, `o_kind[1:0]` (0 element, 1 summary, 2 LN r), `o_data[63:0]` (element code; summary {amax, max, rs0}) |

## 5. Limits of the arithmetic (by construction, matched in the ref)

- bf16/fp32 codes with exponent 255 are treated as finite numbers (no inf/NaN). Subnormal inputs read as zero. Outputs saturate to the largest finite value.
- Row statistics must be consistent with the row: amax ≥ every |element|, and the SMAX max taken over valid keys. The RTL and the ref clamp identically when they are not.
- A softmax row with no valid key is undefined; the sequencer must not emit one.
- The internal exponent is 12 bits. Its only wrap is in the discarded rsqrt of zero, which is masked for QUANT and never reached by the other ops.
