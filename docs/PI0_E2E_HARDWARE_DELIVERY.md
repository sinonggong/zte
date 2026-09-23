# pi0 端到端硬件交付说明 —— AC7t1500 节点阵列

本文档说明 **pi0 在 VP815 板卡上端到端运行**的硬件交付物：两路相机图像、机器人状态和语言提示进入，输出一个
50 × 32 的动作块（chunk）；**一个 chunk 内的全部 transformer 计算都在 AC7t1500 上完成**（两路相机的 SigLIP、
PaliGemma prefix、10 步动作专家及其动作头和 Euler 更新）。主机只做 I/O：图像预处理、token 嵌入查表、噪声、
把输入写进 GDDR6、启动一次节点、读回动作。

状态用语：**已在硅上验证** = 在 VP815 上实测；**仅仿真验证** = Verilator 位精确仿真通过，尚未上板；
**预测** = 由实测的层级仿真周期数和时钟推算（`paper/sw/pi0_chunk_time.py`）。

---

## 0. 目的与阅读顺序

1. 本文档：系统边界、交付清单、已实现的硬件设计、接口、性能、验证阶段状态和未决项。
2. `paper/rtl/README.md`：RTL 目录说明——每个文件的用途、命令与记录格式、参数、仿真和综合的运行方法。
3. `docs/PI0_DEMO_SYSTEM.md`：演示系统——从 Jetson 上的 `pi0_remote` 到板卡的完整流程、各组件实现和操作步骤。

仓库保持源码树的相对路径布局：`paper/rtl`（节点 RTL 与测试平台）、`paper/sw`（生成器、golden、时间模型、主机
运行时）、`paper/synth`（单节点综合与 PLL 文件）、`src/rtl` + `src/constraints` + `src/acxip` + `src/ace`（芯片顶层、
约束、IP 与 ACE 工程）、`scripts`（构建、时钟、烧写、板卡脚本）、`host`（PCIe 主机工具）、`glue`（策略服务与机器人侧）、
`bitstream/s1s_325`（已部署的半阵列比特流，`tc_ref_design_top.hex.gz`）。

---

## 1. 交付总表

| 项目 | 位置 | 状态 |
|---|---|---|
| 数值方案：W8A8、逐 token 动态尺度、SmoothQuant、INT8 QKᵀ、uint8-P 的 PV | `paper/sw/pi0_layer_lower.py`、`paper/sw/prefix_w8a8_eval.py`、`paper/data/prefix_w8a8/SUMMARY.md` | 离线 46 帧 G3 相对误差 0.978 / 1.571 / 1.718 %（均值 / p95 / 最大）；芯片模型与之逐位一致 |
| 三种节点 IP：int8 链节点、uint8 PV 链节点、向量节点，GDDR6 中统一的程序格式 | `paper/rtl/`、`paper/rtl/vector_unit/` | 已在硅上验证 |
| 芯片顶层：节点阵列 + 主机寄存器窗口 + BAR1 分页桥 + 三个时钟域 | `src/rtl/pi0_s0_top.sv`、`paper/rtl/pi0_chip_top.sv`、`pi0_chip_ctrl.sv`、`pi0_host_gddr_bridge.sv` | 已在硅上验证 |
| 整块 chunk 生成器与 GDDR6 布局 | `paper/sw/pi0_chunk_program.py`、`pi0_chunk_layout.py`、`pi0_chunk_tiler.py`、`pi0_chunk_layers.py` | 已在硅上验证 |
| 主机运行时与策略接入 | `host/pi0_chunk_run.cpp`、`paper/sw/pi0_chip_runtime.py`、`glue/level3/` | 已在硅上验证（策略服务 chip 模式在桌面机上应答 `/predict`） |
| **整个 pi0 在硅上逐位正确** | 18 节点半阵列比特流 `bitstream/s1s_325` | 已在硅上验证：整 chunk 4.76 s，flags 与动作与生成器逐位一致 |
| 策略级精度 | `glue/level3/eval_chip_frames.py` | 已在硅上验证：46 个保留帧，关节误差最大 0.66°、均值 0.072°，7 维动作相对 RMS 均值 0.91 % |
| 半阵列布局布线 | ACE 10.5.2，`scripts/launch_pi0_s0_build.sh <tag> 14 2 4 6` | 已布通：RLB 36.15 %，板上运行时钟 325 / 125 / 117.6 MHz（array / vector / fabric） |
| 全阵列（35 节点） | `launch_pi0_s0_build.sh <tag> 27 3 8 12` | 布局通过（RLB 56.92 %），**布线未通过**（248 轮迭代后仍有 10,423 处溢出）；没有全阵列比特流 |
| chunk 时间模型 | `paper/sw/pi0_chunk_time.py` | 预测：设计时钟下全阵列 0.767 s，加 PDQ 融合 0.727 s；半阵列 1.46 s |
| 时序重定时后的 RTL（向量节点 348 MHz、fabric 356 MHz、array 870 MHz，单节点） | 当前 RTL | 仅仿真验证：节点矩阵 70/70、半阵列 chunk 门禁逐位一致；尚无搭载此 RTL 的比特流 |
| GDDR6 通道条带化 + AXI ID 重排 | `paper/rtl/axi_stripe.sv`、`axi_id_reorder.sv` | 仅仿真验证；s1s 比特流带开关，硅上尚未证实 |
| PDQ（预反量化）融合 | `paper/rtl/vector_unit/vu_pdq.sv` | 层级和 1 prefix + 1 expert 步在硅上逐位一致；整 chunk 动作偏差 ≤ 0.018（未决） |
| 验证阶段 S0–S3 | 第 7 节 | 完成；S4（机械臂经 chip 后端运行）未做 |

---

## 2. 目标系统

```
 桌面主机 (PCIe Gen4 x16, 16 GT/s)
   每个 chunk: 写入  ~1.3 MB  两路相机的 patch 编码、提示词嵌入、状态、噪声
              启动  每个节点一次（主机寄存器窗口，广播写）
              等待  halt 向量
              读回  50 x 32 动作 (fp32, 6.4 KB)
   一次性:     写入  INT8 权重、尺度、偏置、RoPE / 位置向量、节点程序（镜像 2.79-2.8 GB）
        │
 GDDR6: 16 个通道 x 2 GB = 32 GB，通道 k 在 NoC 地址 k << 33；每通道内偏移超过 2 GB 后回绕（bit 31 被忽略）
        │  NoC: 256 位 AXI4，突发 <= 16 拍，每节点最多 8 个未完成请求
        ├── int8 链节点     (16 或 32 级 MLP72，列并行 GEMM；array 时钟 + fabric 时钟)
        ├── uint8 PV 链节点 (注意力 P·V，MLP72 模式 5'h13)
        └── 向量节点        (2 lane：QUANT、DEQUANT、归一化、RoPE、softmax、GELU/GeGLU/SiLU、ADD、Euler；vector 时钟)
```

- **板卡**：VP815，AC7t1500，8 个 GDDR6 控制器（16 个通道）。GDDR6 只有在 PCIe 链路训练到 16 GT/s x16 时才正确。
- **主机链路**：BAR0 → `NOC[3][4]` 的寄存器窗口（每节点一个 32 字节拍）；BAR1（256 MB）→ `NOC[3][5]` 的分页桥
  `pi0_host_gddr_bridge.sv`，页基址在窗口拍 N_NODE+1，因此主机可以到达整个 32 GB。
- **为什么是 PIO 而不是 DMA**：这些比特流的 DBI 网关是压缩布局，SDK 的 DMA 引擎初始化会把网关卡死（之后 BAR3
  读全 1），所以批量路径是经分页桥的 PIO：写 69.5 MB/s（64 位 posted 写），读 5.1 MB/s；整个镜像写入 55-62 s
  （45-51 MB/s，约 50 次换页）。
- **节点只通过 GDDR6 区域交换张量**，字节布局完全对齐：链节点的 int8 输入区就是向量节点 QUANT 的输出区（或带 QUANT
  尾部的融合算子输出），链节点的 int32 和区就是向量节点 DEQUANT / PDQ 的输入区。
- **无主机排序**：每个节点每 chunk 跑一个程序，每个 stage 在 GDDR6 的 flag 字上 WAIT 前一 stage、POST 自己
  （`node_sync.sv`）。当前镜像：4,410 个 stage，半阵列 28,316 个 part、全阵列 54,588 个 part。
- **每个 chunk**：54 个 SigLIP 层镜像（27 层 × 2 相机）加 patch 嵌入与投影、18 个 prefix 层（525 token，K/V 留在
  GDDR6）、180 个专家层步（18 层 × 10 步）加动作头与 Euler 更新。GDDR6 布局（16 级链的干跑）：PROG 28.6 MB、STATIC
  2.78 GB、IO 6.3 MB、FLAGS 1.2 MB、SCRATCH 7.75 GB；每 chunk 读 41.5 GB、写 7.7 GB。
- **机器人侧**：Jetson Thor 运行 LeRobot 与 `pi0_remote` 策略插件，只做观测打包和 HTTP 调用，不做模型计算；
  Thor 自身 GPU 推理约 1.0 s/chunk，是阵列的对比基线。

---

## 3. 交付内容

### 3.1 RTL（`paper/rtl/`、`src/rtl/`）

| 模块 | 文件 | 单节点独立综合布线（ACE 10.5.2） |
|---|---|---|
| int8 链节点，16 级 | `colpar_chain_node.sv`、`colpar_node_ctrl.sv`、`colpar_tile_loader.sv`、`colpar_row_sequencer.sv`、`colpar_result_port.sv`、`colpar_result_writer.sv`、`colpar_prog_fetch.sv`、`mlp72_int8_colpar_chain.sv`、`node_sync.sv` | 约 440 RLB tile，17 MLP72，17 + 8 BRAM72K |
| int8 链节点，32 级 | 同上，`N_STAGE=32` | 约 744 RLB tile，33 MLP72，33 + 15 BRAM72K |
| uint8 PV 链节点 | 同上，`MULT_MODE=5'h13` | 随级数 |
| 向量节点，2 lane、2 个加载器、SLOT_BITS 12 | `vector_unit/vu_node_ml.sv`、`vu_lane.sv`、`vu_pdq.sv`、`vu_word_loader.sv`、`vu_rd_fanout.sv`、`vu_beat_writer.sv`、`vu_wr_merge.sv` 等 | 带 QUANT 尾部 2,137 RLB tile、72 BRAM72K、6 MLP72；重定时后带 PDQ 2,570 RLB tile，348.4 MHz |
| 节点与 NAP 之间 | `axi_stripe.sv`（4 KB 通道条带化，运行时开关）、`axi_id_reorder.sv`（逐突发 ID + 重排 RAM，16 个在途突发） | 随节点 |
| 芯片级 | `pi0_chip_top.sv`（`N_DEEP` 混合深度、`NAPS_PER_NODE=2`、`STRIPE`）、`pi0_chip_ctrl.sv`、`pi0_host_gddr_bridge.sv`、`nap_axi_ports.sv`；`src/rtl/pi0_s0_top.sv`（顶层 `tc_ref_design_top`：Device Manager、三个节点时钟、按域的上电 / 软复位） | 见 3.3 |

单节点独立时序（控制路径重定时之后）：array 870 MHz、fabric 355.8 MHz、向量 348.4 MHz（2.0 ns 目标）。重定时之前
为 699 / 315 / 319-350 MHz。

### 3.2 仿真门禁（Verilator，行为级 MLP72 / BRAM72K 模型，GDDR6 AXI 模型）

| 计算 | 真实尺寸 | 期望拍数，全部逐位一致 |
|---|---|---:|
| SigLIP 编码层（L0 相机 0、L13、L26 相机 1） | 256 token × 1152，16 头，FF 4304 | 1,197,056 |
| patch 嵌入 + 位置表；post-LN + 投影（两相机） | 256 × 588 → 1152 → 2048 | 205,024 |
| PaliGemma prefix 层（L0、L17） | 525 token × 2048，FF 16384 | 6,912,844 |
| 动作专家层（L0 s0、L9 s1、L17） | 51 token × 1024，867 个 key | 289,483 |
| 动作头 + Euler（步 0 和 9） | 50 × 32 | 41,023 |
| 无主机排序的专家层 / 160 token 的 prefix 层 | 真实尺寸 | 289,497 / 1,999,022 |
| 半阵列节点组合上的融合 smoke chunk（SigLIP 层 × 2、投影、头、Euler） | 18 节点 | 2,116,166 |
| 8 向量节点上的 tiny chunk（+ 525 token prefix 层 + 专家层） | | 6,885,664 |
| 带 PDQ 的 1 prefix 层 + 1 专家层 | | 5,494,033 |

- 覆盖 16 与 32 级链、1 / 2 / 4 lane 向量节点、双缓冲 GEMM 馈送、写合并、QUANT 尾部、PDQ、条带化 + ID 重排
  （`PER_NODE_MEM=1 STRIPE=12`）。负向对照按预期失败（合并顺序、加载器分段步长、5 位级目标、拍序反转、延迟环短一、
  共享地址）。
- 存储器往返模型（`tb_axi_gddr6_model.sv +rd_lat / +wr_lat`、`+stall_pct`）：半阵列节点组合的 smoke chunk 在 64 周期往返
  下 4.76 M fabric 周期、共享 NAP 且 50 % 随机停顿下 5.00 M 周期，均逐位一致。
- 向量节点矩阵（1 / 2 / 4 lane × 全部套件 × 两个种子 × MAX_OUT 1 / 8）70 / 70 逐位一致。
- 一条命令复跑层级门禁：`paper/rtl/run_pi0_full_size_gates.sh quick`（约 10 min）或 `full`（加 525 token prefix 两种
  深度等，约 3 h）。

### 3.3 布局与布线（ACE 10.5.2；器件 57,600 RLB tile、2,560 BRAM72K、2,560 MLP72）

| 构建 | RLB tile | LUT / DFF | BRAM72K | MLP72 | 状态 |
|---|---|---|---|---|---|
| 半阵列，18 节点，未带写合并 | 29.10 % | 14.91 / 11.03 % | 30.63 %（784） | 17.03 %（436） | 已布通、已上板 |
| 半阵列 `s1s`（QUANT 尾部、PDQ、条带化 + 重排、写合并、双缓冲） | **36.15 %** | 18.86 / 14.23 % | 33.44 %（856） | 21.02 %（538） | 已布通、已上板 |
| 全阵列 `s2i`，35 节点 | **56.92 %** | 28.01 / 21.60 % | 59.34 %（1,519） | 37.77 %（967） | 仅布局；布线失败（10,423 处溢出） |

半阵列各比特流的布线后 Fmax（array / fabric / vector）：`s1q` 243.1 / 132.7 / 192.9 MHz，`s1r` 359.3 / 111.0 / 166.4 MHz，
`s1s` 355.6 / 123.4 / 135.8 MHz。这些都用 `timing_driven_routing 0` 布线；同一设计用默认的时序驱动布线器在半阵列上约
20 min 达到 99 % 布通。array 的报告值是下限：MLP72 → 捕获寄存器是 2 周期路径，报告未必认可（一次报告 132 MHz 的
比特流在 250 MHz 下逐位正确）。

### 3.4 硅上事实

- **AXI 事务开销**：链节点流量下 1 个未完成事务时约 140 ns / 事务（读 + 写）；**8 个未完成事务使链程序快 1.85 倍**，
  向量节点快 13-18 %。因此性能规划按 8 个未完成事务算。
- NAP 在 250 MHz（链 fabric）和 333 MHz（向量）下工作；MLP72 / BRAM72K 上的算术与 golden 逐位一致（行为模型的每条假设
  见 `paper/rtl/sim_models/ASSUMPTIONS.md`）；array 时钟 725 MHz 在 S0 测试比特流上运行过。
- 分页桥：写 69.5 MB/s，读 5.1 MB/s；整镜像 55-62 s。每 chunk 主机开销：编码约 10 ms，写 1.3 MB 输入 23 ms，清
  flags 0.4 ms，读动作 1.4 ms。
- 32 级链在 array 325 MHz 下会置起溢出标志（0x04）而数据逐位正确：监视器的误报，已知。

### 3.5 软件

- 生成器：`paper/sw/pi0_chunk_program.py`（整块 chunk）、`pi0_chunk_layers.py`（各层构建）、`pi0_chunk_layout.py`
  （GDDR6 布局 channels / compact / striped）、`pi0_chunk_tiler.py`（混合深度分块、双缓冲）。
- 时间模型与分析：`pi0_chunk_time.py`（按 part 的真实 WAIT 集合做列表调度）、`pi0_timeline_compare.py`（硅上
  `--timeline` 对模型逐段比较）、`program_traffic.py`、`pi0_chunk_wrong_regions.py`、`pi0_chunk_trace_wrong.py`。
- 运行时：`host/pi0_chunk_run.cpp`（selftest / run / serve / probe / fill / peek / verify）、`paper/sw/pi0_chip_runtime.py`
  （常驻 `serve` 子进程）、`pi0_chip_host_inputs.py`（LeRobot 观测 → 芯片输入，与激活捕获逐位相等）。
- 策略接入：`glue/level3/run_pi0_chip_policy_server.sh`、`pi0_policy_rpc_server.py`、`pi0_fpga_policy.py`
  （`PI0_ACTION_EXPERT=chip`）、`eval_chip_frames.py`；机器人侧 `pi0_arm_client.py` 与插件 `lerobot_policy_pi0_remote`。
- 逐位参考：`vector_unit_ref.py`（lane；`--tables` 生成 5 个 ROM）、`vu_node_golden.py`、`colpar_tile_golden.py`、
  `pi0_layer_lower.py`，以及各层 golden（`pi0_attn_golden.py`、`pi0_siglip_golden.py`、`pi0_action_head_golden.py`、
  `pi0_vision_ends_golden.py`、`pi0_linear_golden.py`、`pi0_wide_gemm_golden.py`）。
- 构建与板卡：`scripts/launch_pi0_s0_build.sh`、`run_pi0_s0_ace_flow.tcl`、`run_pi0_s0_reclock_bitstream.sh`、
  `pi0_s0_pick_clocks.py`、`pi0_bundle_s0.sh`、`pi0_board_program_s0.sh`、`pi0_board_recover.sh`、
  `pi0_array_silicon_session.sh`。

---

## 4. 已实现的硬件设计（D1–D5）

### D1.1 时钟方案

三个 PLL 共用 100 MHz 参考（`fpga_fab_clk_7`，CLKIO_SW REFIO_0）。

| 时钟 | 来源 | 分频 | 设计值 MHz | 用途 |
|---|---|---|---:|---|
| `i_clk_array` | PLL_SW_3（`src/acxip/pll_array.acxip`） | ref 2，fb 29（VCO 5800），ODN 8 | 725 | MLP72 链 |
| `i_clk_vec` | PLL_SW_1（`src/acxip/pll_nap.acxip`）clkout0 | ref 1，fb 20（VCO 8000），ODN 24 | 333.333 | 向量节点及其 NAP |
| `i_clk_fabric` | PLL_SW_1 clkout1 | ODN 32 | 250 | 链的 fabric 侧、链 NAP、主机窗口、分页桥 |
| `i_reg_clk`、NoC 200、`i_mcu_clk` 100 | PLL_SW_0（`src/acxip/pll.acxip`），不改动 | | 64.516 / 200 / 100 | Device Manager、PCIe、NoC 参考 |

PLL 硬件规则（由 IO 环干跑确认）：输出分频 ODN 只能是 2 或 4 的倍数；ACE 只在预填的 ref / fb / clkout0 ODN 合法且
精确命中目标时保留它们，否则按 clkout0 ODN 8、最低 VCO 重解，并让 clkout1/2 的频率一起漂移——所以 array 和向量
时钟不能共用一个 VCO（333.33 MHz 与 700-800 MHz 没有公共 VCO），727.3 MHz（8000/11）不存在，725 MHz 是最近的合法值。
公式：f_VCO = 100 MHz / ref × fb × 4；f_out = f_VCO / ODN。

板上实际运行时钟由布线后 Fmax 决定，不重新布线即可换钟：`scripts/pi0_s0_pick_clocks.py <impl> [--margin 0.97]
[--array MHz]` 读取报告并给出合法 PLL 设置；`scripts/run_pi0_s0_reclock_bitstream.sh <impl> <array MHz> <ref> <fb>
[vec ODN] [fabric ODN]` 只重新生成 IO 环并从已布线数据库重写比特流（array f = 400·fb/ref/8；向量与 fabric =
8000/ODN）。例：`s1s` 以 `325 2 13 64 68` 换钟得到 325 / 125 / 117.6 MHz。换钟会修改 `src/acxip/pll_*.acxip` 与
`src/ace/ioring_design`，下次构建前 `git checkout --` 它们。

### D1.2 顶层

- IO 环：PCIe Gen4 x16 端点（BAR0 窗口、BAR1 256 MB 分页桥）、8 个 GDDR6 控制器、上述三个 PLL、PERST# 与参考时钟。
  IO 环文件由 ACE 从 `src/acxip/*.acxip` 生成，随仓库交付在 `src/ace/ioring_design/`（S0 流程不重新生成它们）。
- `pi0_chip_top`：链节点在前（`N_DEEP` 个 32 级，然后 16 级，最后 `N_PV` 个为 uint8 PV），向量节点在后；每节点 2 个
  NAP；`STRIPE != 0` 时每个节点与 NAP 之间插入 `axi_rd_stripe` / `axi_wr_stripe` + `axi_id_reorder`，开关经两级同步进入
  各节点时钟域，主机桥的地址在开关打开时同样按 `stripe_addr` 转换。
- 半阵列 `14 2 4 6` = 18 节点（芯片节点 0-5 int8 32 级，6-11 int8 16 级，12-13 PV，14-17 向量）；全阵列 `27 3 8 12`
  = 35 节点（0-11 32 级，12-23 16 级，24-26 PV，27-34 向量）；3 节点全功能阵列 `2 1 1 0`。
- 每个域有自己的上电复位与软复位（`src/rtl/pi0_s0_top.sv`）；软复位由窗口拍 N_NODE 的 bit 1 触发，会复位整个 fabric
  域，包括窗口本身（页基址回 0、条带化开关关闭）——运行时在每次软复位后重新应用布局。

### D1.3 构建流程

`scripts/launch_pi0_s0_build.sh <tag> [N_CHAIN] [N_PV] [N_VNODE] [N_DEEP]`，在内存受限的用户 scope 中后台运行
ACE 10.5.2（`scripts/run_pi0_s0_ace_flow.tcl`：新建 impl，用 `src/constraints/pi0_s0_ace.{sdc,pdc}` 替换工程里的
旧约束，Synplify 用 `paper/synth/s0/pi0_s0_synth.prj`，PLACE / ROUTE，写比特流）。10.3.1 在多节点、深于 16 级的
阵列上布局崩溃，只能用 10.5.2。

| 环境变量 | 默认 | 含义 |
|---|---|---|
| `PI0_ACE_SEED` | | 布局种子；同一设计不同种子结果差异大，一次失败只是一个样本 |
| `PI0_S0_ID_WORD` | `24'h533001` | 比特流 ID，窗口拍 N_NODE 的 [255:232] 读回 |
| `PI0_S0_NAPS_PER_NODE`、`PI0_S0_MAX_OUT`、`PI0_S0_N_LANE` | 2、8、2 | 每节点 NAP 数、AXI 未完成上限、向量 lane 数 |
| `PI0_S0_STRIPE` | 12 | 4 KB 通道条带化（0 关闭） |
| `PI0_S0_RESET_FP` | 1 | 复位释放路径设为伪路径 |
| `PI0_S0_CAP_MCP` | | 1 = MLP72 → 捕获寄存器的受保护 2 周期多周期约束（尚未用于任何构建） |
| `PI0_S0_REGIONS_PDC` | | 每节点软区域（`paper/synth/dump_placement.tcl` + `gen_node_regions.py` 生成） |
| `PI0_S0_FLOW_MODE` | | `evaluation` = 快速非时序驱动布线（不能写比特流，仅用于评估） |
| `PI0_S0_IMPL_OPTS`、`PI0_S0_RETIMING`、`MEM_HIGH` | | 额外实现选项、Synplify 重定时、内存上限（20G） |

已交付的 `s1s` 用 `PI0_S0_STRIPE=12 PI0_S0_RESET_FP=1 PI0_ACE_SEED=7 PI0_S0_REGIONS_PDC=<regions_half.pdc>`、
`launch_pi0_s0_build.sh s1s 14 2 4 6` 构建，`timing_driven_routing 0`。构建后 `scripts/pi0_bundle_s0.sh <impl> <name>`
把 `tc_ref_design_top.hex` 连同 INFO.txt 和 SHA256SUMS 打包到 `~/pi0_board_bundle/bitstream/<name>/`。

### D2 GDDR6 布局与镜像格式

- 区域：按列块的 INT8 权重级镜像、fp32 尺度 / 偏置 / 增益、RoPE cos / sin、SigLIP 位置表、每 stage 的激活与摘要区、
  prefix K/V 槽、节点程序（链命令 128 位，向量记录 6 × 128 位）、每 stage 一个 flag 字、每 chunk 的输入输出区。
- 向量节点的函数表（gelu、sigmoid、exp、rsqrt、quant）是综合时由 `vector_unit_ref.py --tables` 初始化的 BRAM ROM，
  随比特流交付，不在 GDDR6 中。
- 三种布局（`PI0_CHUNK_MAP`）：`channels`（默认；STATIC / SCRATCH 区在 16 个通道上轮转，每个区域落在一个通道内）、
  `compact`（< 512 MB，用于无分页桥的窗口和仿真）、`striped`（一个逻辑 32 GB 空间，只能配合 `STRIPE` 比特流并打开开关）。
  条带化的 chunk 不能在未条带化的比特流上运行，反之亦然。
- 分配器把整个 chunk 放进布局（静态区一次、每 chunk 区复用），检查所有区域不重叠，输出主机写列表；`--bin` 输出板卡
  格式 `image/data.{idx,bin}`（写入拍的连续段）与 `image/exp.{idx,bin}`（每个 part 的 flag 与动作）。

### D3 整块 chunk 程序生成器

- 把各层生成器拼成一个 chunk：SigLIP × 54、vision ends × 2、prefix × 18（K/V 保留）、专家 × 180（含逐步时间偏置）、
  动作头 × 10、Euler × 10。
- 每个 stage 分配到节点：GEMM 按列块与行片分到各深度的链节点（`--n-deep` 混合深度），向量算子按行分到向量节点
  （最细 4 / 8 / 16 行的拍对齐粒度）；输出每节点带 WAIT / POST 的程序。
- 调度：`--sync deps`（只等真正读到的 part）、`--interleave`（两相机 SigLIP 交错）、`--prefix-row-blocks 3`
  （prefix 层的行局部段切成 3 块，与 GEMM 重叠）。
- 融合：`--fuse-quant`（X; QUANT(X) → 带 w0[118] 的一条记录）、`--fuse-pdq`（DEQUANT、DEQUANT、GEGLU → 带 w0[119]
  的一条 GEGLU 记录）；`PI0_CHUNK_GEMM_DB=1` 给 GEMM 命令置双缓冲位。
- 输出：GDDR6 镜像 + 每节点程序基址 + 选定帧的期望中间区域（golden）；`info.json` 记录布局种类、双缓冲、融合比例
  及芯片方案动作相对 fp32 模型的误差（0.0133）。半阵列整 chunk 镜像生成约 8-12 min。

### D4 主机运行时与策略接入

- 一次：`pi0_chunk_run run <chunk> --map <map>` 写入镜像并校验（或 `--load-only`）。
- 每 chunk：软复位并重新应用布局，清 flags，写入两相机 patch 编码、提示词嵌入、状态与噪声（`--inputs`），广播启动
  （窗口拍 N_NODE：bit 0 + 启动掩码），轮询 halt 向量，读回 50 × 32 fp32 动作。`serve` 模式常驻，设备只打开一次，
  每收到一行 `infer <inputs> <actions>` 跑一个 chunk。
- 策略侧：`PI0_ACTION_EXPERT=chip` 时 `pi0_fpga_policy.py` 经 `pi0_chip_runtime.py` 调用运行时；`/predict` 与
  `pi0_remote` 插件不变，机械臂客户端不变。
- 安全：持有 `/dev/ac7t15xx0` 的进程只能用 SIGINT 结束（驱动的释放路径在异常退出时可能冻结主机）。

### D5 验证

- 整块 chunk 的 RTL 仿真（`paper/rtl/run_pi0_chunk_sim.sh`，`tb_pi0_chunk.sv`）：缩减阵列或半阵列节点组合，逐位对比
  每个区域、flag 与动作；`PER_NODE_MEM=1` 每节点独立存储端口（芯片的形态），`STRIPE=12`，`+rd_lat/+wr_lat` 往返。
- 每次 RTL 改动重跑 `run_pi0_full_size_gates.sh` 与节点矩阵；硅上按第 7 节分阶段验收。

---

## 5. 接口

| 接口 | 位置 | 要点 |
|---|---|---|
| 主机寄存器窗口 | `paper/rtl/pi0_chip_ctrl.sv` 头部 | 拍 n < N_NODE：写 [41:0] 程序基址、bit 64 启动该节点；读 [64] halt、[65] error、[79:72] 错误细节、[127:96] fabric 周期数、[255:128] 调试快照。拍 N_NODE：写 bit 0 按掩码启动、bit 1 软复位、[96 +: N_NODE] 掩码；读 [N_NODE-1:0] halt 向量、[128 +: N_NODE] error 向量、[231:224] N_NODE、[255:232] ID_WORD。拍 N_NODE+1：[41:0] 分页桥页基址，**bit 64 条带化开关**（加载镜像前设置，节点运行时不改） |
| 比特流识别 | `host/pi0_chunk_run.cpp` | 逐拍扫描，接受 ID 0x5330xx-0x533fxx；`s1s` 为 0x53310a |
| 链命令（128 位，[127:124] 操作码） | `paper/rtl/colpar_node_ctrl.sv` 头部、`paper/sw/colpar_tile_golden.py` | 0 END、1 LOAD、2 TILE、3 OUT（块 / 间隔）、4 FLUSH、5 LOADX（分段镜像）、6 GEMM 列组（[41:0] 激活基址、T / wt_base / W / M / P / G，[105:96] 行步长 S，**[106] 双缓冲**：馈送器两个 256 字半区交替，M × W ≤ 256）、7 WAIT、8 POST（[41:0] 地址，[95:64] 值） |
| 向量记录（6 × 128 位） | `paper/rtl/vector_unit/vu_node.sv` 头部、`vu_node_ml.sv`、`paper/rtl/vector_unit/OPS.md` | w0[127:124]：0xA 算子记录、0xB WAIT、0xC POST（值在 w0[39:8]，flag 地址在 w1）、0 END；w0[3:0] 算子 0-15；[4] b_fp32、[5] bias_en、[6] out_fp32、[7] x_rot；**[118] QUANT 尾部**、**[119] PDQ**；[39:8] k、[59:40] 行数、[75:60] 行长、[117:76] 元素输出基址；w1-w4 操作数描述符（形状 E/R/C/K，格式 bf16/fp32/int32/摘要） |
| 结果记录 | `paper/rtl/colpar_result_writer.sv` 头部 | 每行遍 N_STAGE 个 int32，拍对齐，可按列块跨行写入 |
| 数值格式 | `paper/sw/vector_unit_ref.py` | xf 内部格式、bf16、fp32、int8 / uint8 舍入、五张表 |
| GDDR6 地址 | `paper/sw/pi0_chunk_layout.py`、`paper/rtl/axi_stripe.sv` 头部 | 通道 k 在 k << 33，每通道 2 GB；条带化时逻辑地址 L 的通道 = L[15:12]，4 KB 条带 |

---

## 6. 性能

### 6.1 校准的时间模型（预测；`pi0_chunk_time.py`，理想存储器，重叠调度，融合 QUANT，per-channel 布局）

| 时钟 fabric / vector / array（MHz） | 半阵列 | 全阵列 |
|---|---:|---:|
| 114.6 / 166.7 / 250 | 3.44 s | 1.79 s |
| 125 / 133.3 / 200 | 4.28 s | 2.22 s |
| 250 / 333.3 / 725（设计） | 1.46 s | **0.767 s**；加 PDQ 融合 **0.727 s** |
| 250 / 285.7 / 725 | 1.63 s | 0.851 s；加 PDQ 0.783 s |

链 stage 按 min(f_fabric, f_array / 3) 计时（层级仿真在 3:1 比例下测得）。同等 tile 预算下节点配比很平：8 向量 + 24 int8
0.767 s、10 + 20 0.777 s、12 + 18 0.779 s，保持 8 + 24。

### 6.2 硅上实测

| 比特流 | 节点 | array / vector / fabric MHz | chunk 镜像 | 结果 | 整 chunk 时间 |
|---|---|---|---|---|---:|
| `s0tm_250` | 3 | 250 / 166.7 / 166.7 | 融合整 chunk | 0 错，2 / 2 | 43.65 s |
| `s0tn_500` | 3 | 500 / 285.7 / 222.2 | 融合整 chunk | 0 错，3 / 3 | 26.28 s |
| `s1m_250` | 18 | 250 / 166.7 / 114.6 | per-channel | 0 错，× 3 | 7.29 s |
| `s1q_225`（写合并） | 18 | 225 / 181.8 / 125 | 无双缓冲 | 逐位一致，3 / 3 | 6.66 s |
| `s1r_325`（+ 启动修复） | 18 | 325 / 153.8 / 105.3 | 双缓冲 | 逐位一致 | 6.73 s |
| **`s1s_325`**（+ PDQ、条带化开关、ID 重排） | 18 | 325 / 125 / 117.6 | 双缓冲，无 PDQ | **逐位一致，2 / 2** | **4.76 s** |
| `s1s_325` | 18 | 同上 | 带 PDQ 的 SigLIP 1 + prefix 18 + 专家 1 步（92.8 M 拍全部对比） | 逐位一致 | 2.40 s |
| `s1s_325` | 18 | 同上 | 带 PDQ 的整 chunk | flags 正确，200 个动作拍偏差 ≤ 0.018（量程 1.73），确定性 | 4.63 s |

3 节点阵列比模型快 9-22 %（链 stage 并非纯 array 时钟受限）；半阵列比模型慢：`s1m` 1.63 倍、`s1q` 1.41 倍。
策略级精度在 `s0tn_500`、`s1m`、`s1q` 上相同（芯片模型逐位一致）：46 帧关节误差最大 0.66°、均值 0.072°。

### 6.3 差距的原因与修复

硅上按 `--timeline` 逐段对比模型：SigLIP 2.14 倍、prefix 1.62 倍、专家步 1.35 倍（`s1m`）。原因是节点等待的 GDDR6
往返：向量节点的两个写入器在 `colpar_nap_mux` 后面每次只有一个写突发在途，GEMM 馈送器的加载与计算串行。修复：
`vu_wr_merge.sv`（两个写入器的突发同时在途）、双缓冲馈送器（操作码 6 的 bit 106），smoke chunk 在 128 周期往返下
26.43 M → 18.57 M 周期。第二个原因是激活区域各落在一个通道而 12 个链节点同时读（同一 smoke chunk 全放通道 0 为
0.081 s，轮转 16 通道为 0.059 s）——按通道轮转的布局已是默认，4 KB 条带化 + ID 重排是下一步（NoC 会把同 ID 的不同
通道应答乱序返回，无重排的条带化在硅上失败过）。

### 6.4 杠杆

已实现：8 个未完成事务（1.85 倍）、写合并、双缓冲馈送、QUANT 尾部融合（向量 pass 数 -8 % tile 代价）、PDQ 融合、
更细的注意力行片、prefix 行块与 GEMM 重叠、按通道轮转布局。未实现：馈送器 BRAM 每 fabric 周期写 2 字（约 -20 %
链时间，需要 array / fabric 同步关系）、链上 K 片 int32 求和（约 -5 %）、向量 lane 的 2 项操作数预取队列（单节点
约 +10-15 % 频率）、更深的 lane 流水（加法器 6 级 / 乘法器 4 级 / 表 6 级按 333 MHz 定的）。

---

## 7. 启动与验证阶段状态

| 阶段 | 比特流 | 证明 | 状态 |
|---|---|---|---|
| S0 | 1 个 16 级链节点 + 1 个 2 lane 向量节点 + 窗口 + IO 环 | array / vector 时钟上硅；NAP 在节点 fabric 时钟；GDDR6 1-8 个未完成；真实 MLP72 / BRAM72K 上的算术 | 已验收：全部向量集与链集在 725 / 333 / 250 MHz 逐位一致，8 个未完成 1.85 倍 |
| S1 | + PV 链 + 32 级链 | uint8 × int8 模式、深链 | 在 3 节点与半阵列内完成：注意力层、专家层逐位一致 |
| S2 | 布局通过的全阵列 | 全阵列布通并运行 | **未完成**：全阵列布线失败；以 18 节点半阵列代替，已布通并运行 |
| S3 | 整 chunk | 动作等于 golden chunk；46 帧 G3 ≤ 1.8 % | 在半阵列完成：4.76 s 逐位一致；46 帧关节最大 0.66°、相对 RMS 0.91 % |
| S4 | 经 `/predict` 驱动机械臂 | 客户端不变；白天、安全负责人在场 | **未做**（策略服务 chip 模式已在桌面机上应答，机械臂未经 chip 后端运行） |

---

## 8. 风险与未决项

| 风险 / 未决项 | 影响 | 处理 |
|---|---|---|
| 全阵列在 56.92 % RLB 下布线不通（10,423 处溢出） | 没有全阵列比特流，chunk 时间停留在半阵列的 4.76 s | NAP 局部化的布图、更小的节点、时序驱动布线（半阵列上约 20 min 到 99 %）；在此之前交付半阵列 |
| 板上时钟比设计低 2-3 倍（325 / 125 / 117.6 对 725 / 333 / 250） | 链、向量、NAP 数据搬运全部按比例变慢 | 单节点 RTL 已到 870 / 348 / 356 MHz；下一比特流用重定时 RTL + `PI0_S0_CAP_MCP=1` + 时序驱动布线。fabric 目标 340 MHz：NAP 在 1.7 GHz NoC 下 340 MHz 时带宽 100 %，485.7 MHz 反而只有 66.7 % |
| 带 PDQ 的整 chunk 动作偏差 ≤ 0.018 | PDQ 只能作为可选融合 | 层级与 1 prefix + 1 专家步逐位一致，偏差在后续专家层 / 步，正在仿真二分 |
| 条带化 + ID 重排在硅上未证实 | 激活读取仍集中在单通道 | `s1s` 已带开关与重排，运行时修复后重新测试 |
| 无 DMA：批量路径是 PIO | 镜像加载 55-62 s；每 chunk 输入 23 ms | 可接受；DMA 需要非压缩 DBI 网关 |
| 提示词长度固定（525 token prefix） | 换提示词需重新生成 chunk | 生成器参数化 |
| 主机死机后板卡可能从 PCIe 消失 | 需要物理检查（插槽、供电、JTAG USB）后再上电恢复 | `scripts/pi0_board_recover.sh`、`pi0_board_after_warm_reboot.sh` |
| 布局种子彩票 | 同一设计一次失败不能下结论 | `PI0_ACE_SEED` 多种子 |

---

## 9. 操作约束

- 持有 `/dev/ac7t15xx0` 的进程只能用 SIGINT（Ctrl+C）停止；不要 kill / SIGTERM，也不要在它运行时关窗口。
- 板卡恢复：`scripts/pi0_board_recover.sh <bundle hex>`（JTAG 烧写、重训练到 DLActive+、remove + rescan、16 GT/s、
  加载驱动）；冷启动后板卡不在总线上时按 `pi0_board_prepare_warm_reboot.sh` / `pi0_board_after_warm_reboot.sh`。
  PCIe 必须是 16 GT/s x16，否则 GDDR6 数据不正确。
- 烧写用 `scripts/pi0_board_program_s0.sh <bundle-relative hex>`，需要 `build/host/pi0_s0_replay`；用
  `PI0_HOST_BRIDGE=1 PI0_FPGA_DBI_ROUTE=comp build/host/pi0_chunk_run selftest` 核对 ID。
- chunk 镜像必须与比特流匹配：双缓冲位（`PI0_CHUNK_GEMM_DB=1`）对应带双缓冲馈送器的比特流；`striped` 布局只配
  `STRIPE` 比特流且开关打开；`eval_chip_frames.py` 之后不要用 `--no-load`（板上输入已被替换）。
- ACE：只用 10.5.2 构建阵列；换钟后 `git checkout -- src/acxip src/ace/ioring_design`；在同一工程里切换半 / 全阵列
  前恢复 `src/ace/tc_ref_design_top.acxprj`；不要在未查看 `/proc/<pid>/cmdline` 的情况下结束 ACE 进程。
- 机械臂只在白天、安全负责人在场时运行；`pi0_remote` 与 `/predict` 接口保持不变。

---

## 10. 复现命令

```bash
# 依赖：Verilator 5（VERILATOR）、numpy 的 Python（PYTHON，默认 ~/lerobot/.venv/bin/python）、LeRobot 检查点、
# 激活捕获（paper/sw/vector_unit_capture.py）与校准文件（paper/sw/regen_calib.py）；ACE 10.5.2；Achronix SDK 2.1.1

# 层级门禁（每个门禁及其预期结果，约 10 min）
paper/rtl/run_pi0_full_size_gates.sh quick

# 向量节点套件（QUANT 尾部 / PDQ）与链节点程序仿真
SUITE=pdq_q N_LANE=2 N_LD=2 SLOT_BITS=12 paper/rtl/run_vu_node_sim.sh 1 8
paper/rtl/run_colpar_prog_sim.sh

# 半阵列节点组合上的融合 smoke chunk：每节点独立存储端口，64 周期往返，双缓冲 GEMM
export PI0_CHUNK_COMPACT=1 PI0_CHUNK_GEMM_DB=1 N_VN=4 N_CH=12 N_DEEP=6 N_PV=2
PER_NODE_MEM=1 SIM_ARGS="+nostall +rd_lat=64 +wr_lat=64" paper/rtl/run_pi0_chunk_sim.sh half_db_smoke \
  --n-vec 4 --n-chain 12 --n-pv 2 --n-stage 16 --n-deep 6 --slot-bits 12 --hex \
  --siglip-layers 1 --prefix-layers 0 --expert-layers 0 --steps 1 --sync deps --interleave --fuse-quant

# 半阵列整 chunk 板卡镜像（s1s：双缓冲，per-channel 布局）
PI0_CHUNK_GEMM_DB=1 python3 paper/sw/pi0_chunk_program.py --out build/paper_pi0_chunk/chdb2_half_full \
  --n-vec 4 --n-chain 12 --n-deep 6 --n-pv 2 --n-stage 16 --slot-bits 12 --bin --exp-areas IO,FLAGS \
  --sync deps --interleave --prefix-row-blocks 3 --fuse-quant

# 时间模型（默认 250 / 333.33 / 725 MHz；--f-fabric/--f-vec/--f-array 换时钟，--scale 做假设分析）
python3 paper/sw/pi0_chunk_time.py build/paper_pi0_chunk/chdb2_half_full --f-fabric 117.6e6 --f-vec 125e6 --f-array 325e6

# 半阵列比特流：构建、选钟、换钟、打包
PI0_S0_STRIPE=12 PI0_S0_RESET_FP=1 PI0_ACE_SEED=7 PI0_S0_ID_WORD="24'h53310a" \
  PI0_S0_REGIONS_PDC=./../../build/s0/regions_half.pdc scripts/launch_pi0_s0_build.sh s1s 14 2 4 6
python3 scripts/pi0_s0_pick_clocks.py src/ace/impl_s0_s1s_<...>
scripts/run_pi0_s0_reclock_bitstream.sh impl_s0_s1s_<...> 325 2 13 64 68
scripts/pi0_bundle_s0.sh impl_s0_s1s_<...> s1s_325

# 主机工具、烧写、运行、评估
cmake -S host -B build/host && cmake --build build/host
gunzip -k bitstream/s1s_325/tc_ref_design_top.hex.gz && mkdir -p ~/pi0_board_bundle/bitstream/s1s_325 \
  && cp bitstream/s1s_325/{tc_ref_design_top.hex,INFO.txt,SHA256SUMS} ~/pi0_board_bundle/bitstream/s1s_325/
scripts/pi0_board_program_s0.sh bitstream/s1s_325/tc_ref_design_top.hex
PI0_HOST_BRIDGE=1 PI0_FPGA_DBI_ROUTE=comp build/host/pi0_chunk_run selftest
PI0_HOST_BRIDGE=1 PI0_FPGA_DBI_ROUTE=comp build/host/pi0_chunk_run run build/paper_pi0_chunk/chdb2_half_full \
  --map vector:14,int8:0,uint8:12 --repeat 3 --timeline build/s0_vec/timeline_s1s.txt
python3 paper/sw/pi0_timeline_compare.py build/paper_pi0_chunk/chdb2_half_full build/s0_vec/timeline_s1s.txt \
  --f-fabric 117.6e6 --f-vec 125e6 --f-array 325e6
PI0_ACTION_EXPERT=chip PI0_CHIP_CHUNK=build/paper_pi0_chunk/chdb2_half_full PI0_CHIP_MAP=vector:14,int8:0,uint8:12 \
  PI0_CHUNK_RUN=build/host/pi0_chunk_run ~/lerobot/.venv/bin/python glue/level3/eval_chip_frames.py \
  --frames demo1_ep20:all demo1_ep40:all recov_pi0_ep00:all recov_pi0_ep10:all --json results.json
```
