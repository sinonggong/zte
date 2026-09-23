# pi0 全模型 FPGA 部署 —— Achronix Speedster7t AC7t1500（VP815）节点阵列

本仓库是把 π0 视觉-语言-动作模型（SigLIP 视觉编码器 + PaliGemma 前缀 + 动作专家，含 10 步去噪）**完整**放进一块
AC7t1500 FPGA 上运行的全部设计与 demo：

* **RTL**：由三种节点（int8 MLP72 列并行 chain 节点、uint8 PV 节点、多 lane 向量节点）组成的节点阵列，节点之间通过
  GDDR6 中的标志字自行同步，一个 chunk（一次观测 → 50 × 32 的动作块）从头到尾不需要主机介入；
* **软件**：把真实 LeRobot 检查点逐层降到节点指令的 chunk 程序生成器、bit-exact 的 golden 模型、标定过的时间模型、
  主机运行时（`pi0_chunk_run`）；
* **构建流程**：ACE 10.5.2 的综合/布局/布线脚本、三 PLL 时钟方案、按已布线结果重发比特流的重定时钟工具；
* **demo**：台式机上的策略服务（LeRobot `predict_action_chunk` HTTP 接口，`PI0_ACTION_EXPERT=chip`）、机器人侧
  （Jetson）的 `pi0_remote` LeRobot 策略插件与机械臂客户端——机器人侧不做任何模型计算；
* **比特流**：已在硅上验证的半阵列比特流 `bitstream/s1s_325/`。

文档均为中文，代码、路径、信号名、环境变量保持英文。每个目录有自己的 README 说明该目录的文件与用法：

| 文档 | 内容 |
|---|---|
| [`paper/rtl/README.md`](paper/rtl/README.md) | RTL：设计概览、每个文件的用途、命令/记录格式、参数、时钟、仿真与综合方法、验证结果、资源 |
| [`scripts/README.md`](scripts/README.md) | 比特流构建、选时钟与重定时钟、打包、烧写与验收、板卡恢复 |
| [`host/README.md`](host/README.md) | 主机工具：`pi0_chunk_run` 等四个工具的构建与用法 |
| [`glue/level3/README.md`](glue/level3/README.md) | demo：策略服务器、`pi0_remote` 插件、机器人侧客户端，端到端运行步骤 |
| [`docs/PI0_DEMO_SYSTEM.md`](docs/PI0_DEMO_SYSTEM.md) | demo 系统：从观测到动作块的完整流程、各组件的实现、操作步骤与故障处理、结果 |
| [`docs/PI0_E2E_HARDWARE_DELIVERY.md`](docs/PI0_E2E_HARDWARE_DELIVERY.md) | 端到端硬件交付：目标系统、交付内容、已实现的硬件设计、接口、性能、验证阶段、风险 |

## 1. 系统一览

```
  机器人侧 Jetson Thor                 台式机（Linux，PCIe 主机）                         VP815（AC7t1500）
  ┌──────────────────────┐   HTTP    ┌─────────────────────────────────────┐   PCIe BAR0 / BAR1   ┌──────────────────────────┐
  │ LeRobot RobotClient  │ ────────▶ │ pi0_policy_rpc_server.py            │ ───────────────────▶ │ 节点阵列（pi0_chip_top）    │
  │ + pi0_remote 插件     │  观测      │  └ pi0_fpga_policy.py (chip 模式)    │  主机窗口 + 分页桥     │  向量节点 / int8 chain /  │
  │ （不做模型计算）        │ ◀──────── │     └ pi0_chip_runtime.py           │ ◀─────────────────── │  uint8 PV，GDDR6 32 GB    │
  │ UR7e 机械臂           │  50×32 动作 │        └ host/pi0_chunk_run serve    │  标志字 + 动作块        │  存程序/权重/激活          │
  └──────────────────────┘           └─────────────────────────────────────┘                      └──────────────────────────┘
```

* 台式机把观测（图像、状态、语言 token）转成嵌入输入（约 1.3 MB）写进 GDDR6，广播启动，轮询标志字，读回动作块；
  模型的每一层（SigLIP 27 层 × 2 幅图、投影、前缀 18 层、10 步 × 专家 18 层 + 动作头）都在 FPGA 上执行。
* 半阵列（18 节点：6 个 32 级 int8 chain + 6 个 16 级 int8 chain + 2 个 PV + 4 个双 lane 向量节点）已在硅上跑通整个
  模型，与生成器的 golden 逐比特一致。

## 2. 当前状态

| 项目 | 状态 |
|---|---|
| 整个 pi0 chunk 在 18 节点半阵列上逐比特正确（比特流 `s1s_325`） | 已在硅上验证：4.76 s/chunk（array 325 / vector 125 / fabric 117.6 MHz） |
| 46 帧留出集上的策略精度（芯片 vs fp32 LeRobot） | 已在硅上验证：关节最大 0.66°，平均 0.072° |
| 策略服务 chip 模式（台式机侧 `/predict`） | 已在硅上验证 |
| 机械臂通过 chip 后端闭环运行 | 未完成 |
| 全阵列（35 节点：12 × 32 级 + 12 × 16 级 int8、3 PV、8 向量） | 已布局（RLB 56.92 %），未布通 |
| PDQ（预反量化融合）整块镜像 | 仿真逐比特一致；硅上动作偏差 ≤ 0.018，原因待查 |
| GDDR6 4 KB 条带化 + AXI ID 重排 | 仅仿真验证 |
| 控制路径重定时后的 RTL（单节点 array 870 / fabric 356 / vector 348 MHz） | 门禁仿真全部通过，尚无比特流 |
| 同一设计的时序驱动布线（hold 不确定度 0.05 ns 后可用） | 已布线：array 525 / fabric 199 / vector 208 MHz（evaluation 布线为 356 / 123 / 136），比特流待上板 |
| 设计时钟 725 / 333.33 / 250 MHz | 目标 |
| 时间模型（设计时钟，理想存储器） | 预测：半阵列 1.46 s，全阵列 0.767 s（PDQ 0.727 s）/chunk |

## 3. 目录结构

| 目录 | 内容 |
|---|---|
| `paper/rtl/` | 节点阵列 RTL：chain 节点（`colpar_*.sv`、`mlp72_int8_colpar_chain.sv`）、芯片级（`pi0_chip_top.sv`、`pi0_chip_ctrl.sv`、`pi0_host_gddr_bridge.sv`、`nap_axi_ports.sv`）、条带化与 ID 重排（`axi_stripe.sv`、`axi_id_reorder.sv`）、节点同步（`node_sync.sv`）；测试平台 `tb_*.sv` 与 Verilator 运行脚本 `run_*.sh`；说明见其 README |
| `paper/rtl/vector_unit/` | 向量节点：`vu_node_ml.sv`（多 lane 节点）、`vu_lane.sv`（lane，含 QUANT 尾）、`vu_pdq.sv`（预反量化）、`vu_wr_merge.sv`（写合并）、算术/表/加载器/写出器，`OPS.md` 算子表 |
| `paper/rtl/sim_models/` | ACX_MLP72 / ACX_BRAM72K / ACX_FLOAT 的周期精确行为模型及其假设清单 |
| `paper/sw/` | chunk 程序生成器（`pi0_chunk_program.py` 及 layout / tiler / layers）、各层 golden、时间模型（`pi0_chunk_time.py`）、主机运行时 Python 侧（`pi0_chip_runtime.py`、`pi0_chip_host_inputs.py`）、向量单元数值参考（`vector_unit_ref.py`）、诊断工具 |
| `paper/synth/` | 节点单独综合脚本、Synplify 工程 `s0/pi0_s0_synth.prj`、三 PLL 文件、按节点生成布局区域 |
| `paper/data/` | 向量单元、实宽 GEMM、W8A8 数值方案的结果摘要 |
| `src/rtl/`、`src/include/` | FPGA 顶层 `pi0_s0_top.sv`（模块名 `tc_ref_design_top`）、复位处理器、接口头文件 |
| `src/acxip/`、`src/ace/`、`src/constraints/` | ACE IP 配置（PCIe、GDDR6、NoC、PLL）、ACE 工程与生成的 IO 环、SDC/PDC 约束 |
| `scripts/` | 比特流构建、重定时钟、打包与烧写、板级会话与恢复脚本；说明见其 README |
| `host/` | 主机工具 `pi0_chunk_run`、`pi0_s0_replay`、`pi0_atu_tool`、`pi0_dbi_probe`；说明见其 README |
| `glue/level3/`、`glue/level2/` | demo：策略服务、`pi0_remote` LeRobot 插件、机器人侧客户端、评估与采集脚本；运行步骤见 `glue/level3/README.md` |
| `tools/acxpcie_driver/` | Achronix PCIe 驱动的 mmap 补丁 |
| `bitstream/s1s_325/` | 半阵列比特流（`tc_ref_design_top.hex.gz`，烧写前 `gunzip`；`INFO.txt`、`SHA256SUMS`） |
| `docs/` | demo 与端到端硬件交付文档 |

## 4. 从哪里开始

| 想做的事 | 看哪里 |
|---|---|
| 读懂设计、跑 Verilator 仿真 | `paper/rtl/README.md`（设计概览、文件说明、"运行仿真"一节的命令与套件） |
| 构建 / 重定时钟 / 烧写比特流 | `scripts/README.md` |
| 编译主机工具、加载并运行一个 chunk | `host/README.md` |
| 跑机械臂 demo | `glue/level3/README.md`（前提、服务器、Jetson 插件与客户端、精度评估、常见问题） |
| 了解性能、接口与未决项 | `docs/PI0_E2E_HARDWARE_DELIVERY.md`、`docs/PI0_DEMO_SYSTEM.md` |

## 5. 环境假设

脚本中的默认路径都可以用环境变量覆盖：

| 依赖 | 默认位置 / 变量 |
|---|---|
| ACE 10.5.2（Synplify 随附，需许可证） | `ACE_ROOT=/home/sngong/ACE_10.5.2/Achronix-linux`；`launch_pi0_s0_build.sh` 内固定为此路径 |
| Achronix SDK 2.1.1 与 `acxpcie` 驱动 | `/opt/achronix/sdk`（`ACHRONIX_SDK_ROOT`）；驱动补丁见 `tools/acxpcie_driver/` |
| Verilator 5 | `VERILATOR=~/tools/verilator5/bin/verilator` |
| Python（numpy、torch、LeRobot） | `PYTHON=~/lerobot/.venv/bin/python`（demo 的策略服务也在这个环境里运行） |
| LeRobot pi0 检查点（UR7e 微调） | `~/pi0_ckpt/ur7e-demo-2-pi0-fsdp/010000/pretrained_model`，不在仓库中 |
| 采集帧与激活（生成器、golden、评估用） | `~/pi0_glue/captures/<ep>/frame_NN.npz`；`build/paper_vector_unit/acts_*.npz`（`paper/sw/vector_unit_capture.py` 生成）；`build/paper_vector_unit/calib/`（`paper/sw/regen_calib.py` 生成） |
| 板卡 | VP815，PCIe 16 GT/s × 16，JTAG 连接到台式机 |

向量单元的函数表（`vu_tbl_*.mem`）由 `paper/sw/vector_unit_ref.py --tables <dir>` 生成，运行脚本与构建脚本会自动生成，
不需要检查点。

## 6. 硬件

| 部件 | 说明 |
|---|---|
| FPGA 板 | Achronix VP815，AC7t1500：57,600 RLB、2,560 MLP72、2,560 BRAM72K、16 个 GDDR6 通道 × 2 GB = 32 GB、2D NoC、PCIe Gen4 × 16 |
| 台式机 | Linux，PCIe 主机；运行策略服务、主机运行时与 ACE 构建 |
| 机器人侧 | NVIDIA Jetson Thor，运行 LeRobot 与 `pi0_remote` 插件，控制 UR7e 机械臂 |
