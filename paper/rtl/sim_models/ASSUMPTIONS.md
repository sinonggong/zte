# Behavioural ACX_MLP72 / ACX_BRAM72K models: assumptions

`acx_mlp72_behav.sv` and `acx_bram72k_behav.sv` model only the subset that
`paper/rtl/mlp72_int8_chain.sv` uses. They are written from the vendor documents
and the unencrypted vendor wrappers, not from the encrypted cores.

Line references:

- `ug086:N` and `ug088:N` are line numbers in the pdftotext extraction of
  UG086 (Component Library) and UG088 (Machine Learning Processor).
- `wrapper` is ACE 10.3.1 `libraries/speedster7t/sim/speedster7t_sim_BRAM72K.sv`.
- `GEN_DEEP` is `libraries/speedster7t/macros/ACX_BRAM72K_GEN_DEEP.sv`.

Each row below is a behaviour the documents do not pin down. The model tags it
`[An]`/`[Bn]` in its source. Silicon bring-up has to confirm exactly these rows.

## Assumptions

| ID | Modelled behaviour | Basis | What the chain needs from it | Sensitivity / silicon check |
|---|---|---|---|---|
| A1 | `fwdo_mult*` carries the stage-0 register **output** (`del_mult*`), not the mux output. | UG086 says only "the selection from mux_sel" (ug086:4322, 4533-4547). UG088 says that with `del_mult*` enabled "each MLP72 ... [processes] the data one cycle after the MLP72 below", at ~750 MHz (ug088:772-777). A pre-register tap would make the column one combinational path. | Stage m reads weight address a at cycle +m. | Flipping it (`+define+ACX_MLP72_BEHAV_FWDO_PRE_REG`) makes 32/32 rows wrong (N=4). Silicon test: 2 stages, one-hot activation words, distinct weights per word; the sum shows which word each stage multiplied. |
| A2 | Stage-1 registers sit on the multiplier inputs, after byte selection. | ug086:4897-4899. | Pipeline depth only. | Byte selection is wiring, so the placement is not observable. |
| A3 | `del_rndsubload_ab_reg` / `del_rndsubload_reg` are pure delay lines of N registers. The load pin is driven in the cycle the operands are at the MLP inputs. | `acx_integer.sv:130-131` sets N = number of data registers to the accumulator. `stack_stage_fp.sv:278` adds `del_accum_ab` for the CD accumulator fed from AB. | Accumulator load: N = 4 (stage 0, 1, 2, ACCUM_AB_REG). The load pin is at T_CD−4. | Off by one, a row's sum gains the next row's first word or loses its own. Silicon test: back-to-back rows with distinct sums. |
| A4 | Multiplier NO OP (5'h11) outputs 0. | Not documented. | Nothing any more. The feeder's AB/CD output is unused, stage 1 bypasses CD, and the accumulator gets zero operands on MLP_DIN in SIGNED 8x8 mode. | None. |
| A5 | Adders are 48 bit with no internal saturation. `ADD015 = ADD0_7_REG + ADD8_15_REG` after the stage-2 registers, with no register before the accumulators when `del_fpmult_ab_reg=0`. | ug086:5029-5046 and 5080-5082 (the naming "ADD0_7_REG output ... toward FPMULT_AB_REG"). | A 16-product sum is at most 262,144 (19 bits signed). Row sums stay below 2^28. | The corner rows (all products +16384 or −16256) pass in the model. Silicon: the same corner rows. |
| A6 | `add_accum_ab_bypass=1` / `add_accum_cd_bypass=1` pass the adder input (the add_00_15_sel output, or dina) to the register. dinb, load and sub have no effect. | ug086:5119-5126 says only "bypass". `acx_integer.sv:109,288` uses bypass for "no accumulate". | **Critical.** On multiplying stages `load_ab` = the BRAM's wide write enable (F2). With AB bypassed it must not touch the sum. | The same RTL with AB in circuit makes every phase-2 row wrong (8/8 at 32 words, 11/16 at 3 words), because weights are written during computation; phase-3 rows stay clean. Silicon test: rewrite the other tile while computing; the sums must not change. |
| A7 | `fpadd_cd_dina_sel=1` takes the optionally registered ACCUM_AB_REG output. The CD half's own integer input is the add[15:8] bank sum. | ug086:5095-5098; Table 125 "load ... add[15:8] sum" (ug086:5188). Precedent: `stack_stage_fp.sv:278`. | Per-stage timing: ACCUM_AB_REG +1, OUT_REG +1. The dout cascade stays one cycle per stage for any uniform stage latency. | If CD took the pre-register AB value, a stage would add the stage-below partial sum of the previous word. Silicon test: same as A1. |
| A8 | `sub` / `sub_ab`: dinb − dina; load wins over sub. | Not documented. | Not used (tied 0). | None. |
| A9 | `out_reg_din_sel=011` puts the CD result on OUT_REG[47:0] with [63:48] = 0. `dout_mlp_sel=00` gives dout[71:64] = 0. | ug086:5129-5135, 5142-5167 give no widths. | The chain uses only [47:0]. | None. |
| A10 | Delay-stage registers use synchronous reset with priority over ce, and power up at 0 in simulation. | ug086:4270-4306. | Only the accumulator OUT_REG is reset. After the first row, `load` makes the output independent of the reset state. | Low. |
| A11 | `expb` has no effect in integer mode (block-fp exponent input). | ug086 integer sections do not use it. | The multiplying stages drive `expb` = the upper byte write enables (F2). | Silicon test: as for A6. |
| B1 | 144-bit width codes are 4'h2 = sixteen 9-bit bytes and 4'h3 = eighteen 8-bit bytes. Byte enables: `we[8:0]` for din[71:0], `mlpram_we[8:0]` for din[143:72] (= SDP we[17:9]). With 9-bit bytes, we[8] and mlpram_we[8] are ignored. | SDP we[] mapping (ug086:10764-10778). GEN_DEEP:507-511 routes byte_en[9] to `load_ab` and byte_en[17:10] to `expb` (GEN_DEEP:949, 955). The width-code numbers come from an earlier reading of the macros and are unconfirmed. | The chain drives all 18 enables with the same `i_wen[m]`, so it needs only that `mlpram_we` covers the upper half. | Silicon test: write/read-back of all 144 bits through the MLP path (`outmode_sel=10` shows BRAM_DOUT[143:72] on dout). |
| B2 | `mlpram_dout2mlp` is taken after the output register (outreg_enable=1), i.e. read latency 2, the same as dout. | "Connects BRAM_DOUT[143:0] to MLP" (ug086:4501-4503). Latency 2 with outreg (ug086:11126-11132). | `BRAM_RD_LAT` = 2 in the fabric control pipeline (load, ce, valid). The data alignment between stages does not depend on it. | Tapping before the outreg (`+define+ACX_BRAM72K_BEHAV_DOUT2MLP_PRE_OUTREG`) makes 32/32 rows wrong with BRAM_RD_LAT=2. If silicon differs, set `BRAM_RD_LAT` = 1. |
| B3 | Read-address source: `rdmem_input_sel` 0, 1, 3, 8, >9 take `rdaddrhi` (vendor sim decode). **2, 4, 5, 6, 7, 9 are assumed to take `fwdi_ram_rd_addr`.** The 14-bit cascade address is {hi[9:0], lo[3:0]}; a 144-bit read uses [13:5]. | Decode: `speedster7t_sim_BRAM72K_COMMON.sv:1740-1742`. UG086 does not document `rdmem_input_sel`. GEN_DEEP / `acx_bram_gen_deep_direct.sv` use 4'h1 on the stack's entry BRAM and 4'h2 on the others. **But there the read address travels top-down on the reverse (revi/revo) cascade, so the forward code may be different.** | `ADDR_CASCADE=1` only: feeder BRAM 4'h1 (fabric, drives fwdo), stage BRAMs 4'h2. | Not settled by the model. ACE 10.3.1 prepares, places and routes the ADDR_CASCADE=1 chain with 0 errors (16 stages: 18 MLP72, 17 BRAM72K, 130 DFF; 991.8 MHz upper limit). That shows the configuration is legal, not what it does. Next checks: an ACE `report_timing` through `fwdo_ram_rd_addr → fwdi_ram_rd_addr`, then silicon (the A1 test with ADDR_CASCADE=1). |
| B4 | `del_fwdi_ram_rd_addr=1` is one rdclk register on the cascaded addr/rden/rdmsel, loaded every cycle. `ce_fwdi_ram_rd_addr` is set equal to `del` (as GEN_DEEP does); its own meaning is unknown. | "the read address is also cascaded up the column, with a delay stage enabled between each BRAM" (ug088:792-797). | One cycle per BRAM, matching A1. | Without the register, 32/32 rows are wrong (ADDR_CASCADE=1, N=4). |
| B5 | `fwdo_ram_rd_addr/rden/rdmsel` carry the address this BRAM uses: after its register, or the fabric address on the entry BRAM. | By analogy with A1; not documented. | As B4. | As B4. |
| B6 | A read during a write of the same word returns the old word. | Not documented for different clocks. | Not exercised: the chain never reads a word that is being written. | None. |

## Vendor facts the corrected RTL relies on (unencrypted code or explicit text, not assumptions)

- **F1.** The upper 72 bits of a 144-bit write come from the co-sited MLP72's
  `din` via `mlpram_din`, but only with `enable_wide_fabric_input=1`. Otherwise
  both data and enables are tied Open. Sources: wrapper:4250-4271;
  `syn/speedster7t_user_macros_BRAM72K.sv:3324-3337`. The first chain RTL left
  this parameter at 0, so bytes 8..15 of every word were never written.
- **F2.** `mlpram_we = {expb, load_ab}` (wrapper:5884; syn macro:4956). GEN_DEEP
  drives `load_ab` and `expb` with the wide byte enables. So `load_ab` is a write
  enable and cannot also load the AB accumulator while tiles are being written.
- **F3.** A 144×512 word address is `wraddrhi[9:1]` / `rdaddrhi[9:1]` (ug086:11011;
  GEN_DEEP:558).
- **F4.** `load_ab` loads the AB accumulator with the add_00_15_sel output
  (ug086:5193). `dout_mlp_sel` selects the `fwdo_dout` value (ug086:5142).
  ACCUM_AB_REG/OUT_REG feedback always uses the register (`acx_integer.sv:210-211`).
  The ce/rstn select codes are in Table 101 (ug086:4242-4268).
- **F5.** The LRAM in mode 0 is slaved to the BRAM's `wrmsel`/`rdmsel` (ug086:5247).
  With both tied 0 it stays idle.

## Negative controls (paper/rtl/run_chain_sim.sh, N_STAGE=4, seed 1)

| Run | Rows wrong |
|---|---|
| first chain RTL (no fixes; unused `i_last` added) | every checked row (plus one o_sum_valid per word instead of per row) |
| corrected RTL, A1 flipped | 32/32 |
| corrected RTL, B2 flipped | 32/32 |
| corrected RTL with `enable_wide_fabric_input=0` | 32/32 |
| corrected RTL with the AB-add cascade and `load_ab` as write enable | 8/16 (32 words), 11/32 (3 words); all in phase 2 (writes during computation) |
| corrected RTL, ADDR_CASCADE=1 without `del_fwdi_ram_rd_addr` | 32/32 |
| corrected RTL, all 18 configs (N 1/4/16 × 1/3/32 words × ADDR_CASCADE 0/1) | 0 |
