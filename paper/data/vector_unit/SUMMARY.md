# Vector unit (workstream C): summary

Written by the main session from workstream C's final report (2026-09-16); the subagent could not
write this file itself. Tags: **[M]** measured, **[P]** projected.

## Arithmetic and accuracy

Reference: `paper/sw/vector_unit_ref.py` (`--tables`, `--inventory`, `--measure`). Row scalars are
handed between passes as fp32 codes, so the RTL mirrors the reference exactly. The table layouts
are unchanged (2048 entries, 12-bit fraction, Q1.20), and WS-A's `fmt_selftest` still applies.

Per-op error against the float torch op, same bf16 inputs, frame demo1_ep20:2 [M]
(`errors.json`). Element errors are taken over |y| ≥ 1e-3 of the row max:

| op | element rel err max / RMS | abs err / row max | output row rel RMS (bf16 floor) |
|---|---|---|---|
| LayerNorm | 5.7e-4 / 8.1e-6 | 1.4e-6 | 2.89e-3 (2.89e-3) |
| RMSNorm | 1.2e-6 / 4.1e-7 | 1.1e-6 | 2.96e-3 (2.96e-3) |
| RoPE | 3.4e-5 / 4.2e-7 | 2.4e-7 | 2.55e-3 (2.55e-3) |
| softmax (256 / 525 / 867 keys, masked) | 2.8e-6 / 8.4e-7 | 1.9e-6 | 3.52e-3 (3.52e-3) |
| GELU | 2.2e-3 / 9.4e-5 | 4.3e-6 | 2.20e-3 (2.20e-3) |
| GeGLU | 1.2e-3 / 1.2e-5 | 3.2e-6 | 2.60e-3 (2.60e-3) |
| SiLU | 3.4e-4 / 4.4e-6 | 1.4e-6 | 1.79e-3 (1.79e-3) |
| residual / position add | 1.2e-7 | 1.2e-7 | codes identical to ideal bf16 |
| DEQUANT (real INT8 sums) | 3.0e-5 with bias, 2.3e-7 without | 2.3e-7 | at the floor |
| DEQUANT, fp32 out | 3.0e-7 | – | – |
| Euler, 10 chained steps | – | 8.9e-7 | – |

- The gate ops' 1e-3 element errors sit in the gate tails, where outputs are near 1e-3 of the row max.
- QUANT: 1.6e-4 of 13.8 M codes differ from torch by one step; s_row is within 1 fp32 ULP.
- INT8 softmax codes match 0.99969 of the time.

Chunk level, 8 held-out frames, full SigLIP → prefix → 10-step expert [M] (`e2e_chunk.json`).
GEMMs stay fp32 and every vector-unit op is patched:

| comparison | chunk rel RMS mean / max |
|---|---|
| bf16 interface vs fp32 | 0.20 % / 0.44 % |
| lane arithmetic vs fp32 | 0.19 % / 0.36 % |
| lane vs bf16 | 0.18 % / 0.30 % |

The minimum cosine is 0.99999. The lane arithmetic adds nothing visible beyond the bf16 interface.
WS-A independently measured the tables alone on the fp32 model: 1.7e-6 relative.

## RTL and verification

- `paper/rtl/vector_unit/OPS.md` covers:
  - every op, with its unit composition, row-reduction handling and model-op mapping;
  - elements per chunk: 1,262,947,824 vector-unit visits with INT8 attention, plus 846,451,248 chain-side dequant elements;
  - latency: 32 pipeline cycles plus 2 FIFO cycles.
- `vu_lane.sv` (units `vu_add`, `vu_mul`, `vu_tbl`, `vu_round`, `vu_pkg`, `vu_qtab`) runs all 16 ops on one datapath:
  - the row reductions use fixed-point accumulators with row-end microcode;
  - QUANT row constants come from a 256-entry table.
- `paper/rtl/run_vu_lane_sim.sh all` [M] (`sim_results.txt`):
  - PASS on 47 cases, 226,264 beats, 0 mismatches, with 25 % input gaps and 20 % back-pressure;
  - negative controls fail: GELU table +1 LSB (2,254 mismatches), bf16 ties rounded half-up (662 mismatches).
- Per-row overhead: 1 cycle for element ops, 71 for RMS_STAT, 169 for LN_STAT, 50 each for SMAX_SUM and SMAX_Q8.

## One lane at 250 MHz (AC7t1500 C1, Synplify + ACE) [M]

Flow: `paper/synth/run_vu_lane_synth.sh`, `parse_vu_lane_synth.py`.

| variant | LUT | DFF | ALU8 | BRAM72K | MLP72 | RLB tiles | setup / hold slack | upper limit |
|---|---|---|---|---|---|---|---|---|
| multipliers in DSP | 3,504 | 6,490 | 144 | 7 | 4 | 659 | +1.226 / +0.009 ns | 360.5 MHz |
| multipliers in fabric (`VU_MUL_LOGIC`) | 4,763 | 7,868 | 334 | 7 | 0 | 849 | +0.395 ns | 277.4 MHz |

- Worst paths: the QUANT ROM read → s_row fp32 conversion (DSP variant); the fabric 24×24 multiplier (fabric variant).
- Pitfall: Synplify first packed the QUANT ROM into 12 LRAM2K, which blocked 12 paired MLP72 sites. A module-level `syn_romstyle = "block_rom"` fixed it.

## 32 lanes [P] (`projection.json`, linear scaling)

- DSP variant: LUT 16.2 %, DFF 15.0 %, BRAM72K 8.75 % (6.25 % with shared tables, assumed), MLP72 5.0 %, **RLB tiles 36.6 %**.
- Fabric variant: LUT 22.1 %, DFF 18.2 %, MLP72 0, **RLB tiles 47.2 %**.
- Throughput at 250 MHz:
  - one chunk's vector-unit ops take 5.16 lane-seconds, so 32 lanes need ≥ 0.161 s;
  - including the chain-side dequant, 8.55 lane-seconds, ≥ 0.267 s;
  - with bf16 attention, 0.169 s and 0.275 s.
- Element rate: ≈ 8.0 G elements/s for the element ops; the row reductions lose 2-12 % to microcode.

## Open

- No multi-lane place and route yet; watch RLB occupancy.
- The throughput bound ignores NoC, GDDR6 and sequencer limits.
- Exporter folds are assumed, not built: log2-unit logits (1/ln 2 and 1/√d_head folded into the QKᵀ column scale), and SmoothQuant folded into the RMS gains.
- Not run: the chunk check with W8A8 GEMMs and lane ops together.
- DEQUANT serves V-per-channel PV. The per-key V form does not factor into acc × s_row × s_col, so WS-A's chip candidate uses V per channel.
