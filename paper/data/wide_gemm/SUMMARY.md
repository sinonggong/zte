# o_proj at its real width on one chain node (2026-09-16)

`paper/rtl/run_pi0_wide_gemm_sim.sh all` — generator `paper/sw/pi0_wide_gemm_golden.py`, testbench
`paper/rtl/tb_pi0_wide_gemm.sv`. The whole-layer test (`run_pi0_attn_sim.sh`) is cut down so every image
fits one node; this runs the single widest GEMM of the model at its real shape instead.

## Shape

The expert's `o_proj` is (1024 out, 2048 in) and a stage BRAM72K holds 512 words of 16 int8:

| | |
|---|---:|
| W = K / 16, words per column | 128 |
| P = 512 / W, columns per stage | 4 |
| columns per weight image, 16 P | 64 |
| **column tiles, N / 64** | **16** |
| tokens T | 51 |
| feeder rows per load, M = 512 / W | 4 (12 groups of 4 + one of 3) |
| OUT block beats / gap | 8 / 3,840 bytes |
| commands in the program | 65 |
| row passes = records | 3,264 |

Weights are the real `o_proj` slice of `model.safetensors`, quantised as `pi0_layer_lower.py` quantises it;
activations are the captured `ex.L<n>.o_in` frame quantised per token by the vector node's QUANT reference.
Expected values are an int64 matmul, self-checked against a direct loop on eight sampled entries.

## Results

| run | verdict | expected | written | wrong | extra | rd bursts | wr bursts |
|---|---|---:|---:|---:|---:|---:|---:|
| layer 0, step 0 | **PASS** | 6,528 | 6,528 | 0 | 0 | 7,393 | 816 |
| layer 9, step 1 | **PASS** | 6,528 | 6,528 | 0 | 0 | 7,393 | 816 |
| layer 17, step 0 | **PASS** | 6,528 | 6,528 | 0 | 0 | 7,393 | 816 |
| layer 0, 3 feeder rows per load | **PASS** | 6,528 | 6,528 | 0 | 0 | 7,393 | 816 |
| neg `blockgap` (no OUT block / gap) | FAIL as intended | 6,528 | 528 | 6,400 | 0 | 7,393 | 416 |
| neg `tileorder` (two tiles' images swapped) | FAIL as intended | 6,528 | 6,528 | 816 | 0 | 7,393 | 816 |
| neg `blockbeats` (block beats halved) | FAIL as intended | 6,528 | 6,528 | 6,464 | 3,160 | 7,393 | 1,632 |
| neg `stageorder` (two stage images swapped) | FAIL as intended | 6,528 | 6,528 | 3,264 | 0 | 7,393 | 816 |

No node error and no AXI error in any run. `rd_max_outstanding` 7, `wr_max_outstanding` 1.

## What it measured that the cut-down test could not

- **A column block caps the write burst at the block.** One burst per block: 8 beats = 256 bytes, not the
  16-beat bursts a contiguous result would get. Every wide projection is split this way, so whole-model
  write traffic moves in 256-byte bursts unless a node covers a whole row of the result.
- **Re-read activations cost as much as the weights.** 7,393 read bursts of 512 bytes = 3.78 MB against
  2.00 MB of weights: `o_proj` is read once, but the 104 KB of activations is read once per column tile,
  1.67 MB in all. At the real width the two are the same order — the reason the per-chunk figure in
  `full_chip/CHIP_PROGRAM_SUMMARY.md` is 43.7 GB of activation reads against 5.6 GB of weight traffic.
- **The schedule inside a node is free.** Three feeder rows per load instead of four is bit-identical, so
  the compiler may choose M for bandwidth without affecting results.
