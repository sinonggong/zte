# 整个 pi0 模型运行在 Achronix AC7t1500 上的机械臂 demo

本文说明这样一个 demo：UR7e 机械臂由 pi0 视觉-语言-动作策略驱动，**整个模型都在 Achronix Speedster7t AC7t1500 FPGA
（VectorPath VP815 加速卡）上计算**：两个 SigLIP 视觉编码器、PaliGemma 语言模型 prefix，以及 action expert 及其 10 个去噪步、
action head 和 Euler 更新。文中说明一个动作 chunk 如何端到端产生、各部分如何实现，以及如何运行系统。

FPGA 设计就是 [`paper/rtl/README.md`](../paper/rtl/README.md) 中说明的节点阵列；本文沿用其中的术语
（链节点、PV 链节点、矢量节点、主机窗口、分页桥）。硬件交付的整体说明见
[`docs/PI0_E2E_HARDWARE_DELIVERY.md`](PI0_E2E_HARDWARE_DELIVERY.md)。

代码地图：

| 目录 | 内容 |
|---|---|
| `paper/rtl/`、`src/rtl/pi0_s0_top.sv` | 节点阵列 RTL 与 FPGA 顶层 |
| `paper/sw/` | chunk 生成器、主机运行时、数值参考模型、时间模型 |
| `host/` | 板卡工具（`pi0_chunk_run` 等，C++，Achronix SDK） |
| `glue/level3/`、`glue/level2/` | 策略服务器、精度评估、机器人端策略插件与客户端 |
| `scripts/` | bitstream 构建、改时钟、打包、板卡编程与恢复 |
| `bitstream/s1s_325/` | 已交付的半阵列 bitstream（`tc_ref_design_top.hex.gz`、`INFO.txt`、`SHA256SUMS`） |

模型：LeRobot base-pi0 checkpoint `ur7e-demo-2-pi0-fsdp/010000`，针对任务 “Pick up the blue cup and put it in the plate” 微调。

---

## 1. 概览

```
 Jetson Thor（机械臂旁）                                 插有 VP815 卡的台式机
 ┌────────────────────────────────────────┐  以太网    ┌───────────────────────────────────────────────────┐
 │ 2 个相机、UR7e（RTDE）、Robotiq 夹爪     │  直连线    │ 策略 RPC 服务器（Python，端口 8081，/predict）      │
 │ LeRobot 预处理 / 后处理                  │            │   主机输入：patch + 位置嵌入、提示词 token 嵌入、   │
 │ pi0_remote 策略（不做模型计算）          │ ─────────▶ │   状态投影、噪声                                   │
 │ 动作队列 → UR 伺服                       │   观测     │   pi0_chip_runtime → pi0_chunk_run serve：         │
 │                                        │ ◀───────── │   写输入、启动、读动作                             │
 └────────────────────────────────────────┘  50 × 32    │        │ PCIe BAR0（寄存器窗口）、BAR1（分页桥）    │
                                             动作       │        ▼                                           │
                                                        │ VP815：AC7t1500 节点阵列 + GDDR6                   │
                                                        │   SigLIP × 2 个相机 → PaliGemma prefix →           │
                                                        │   10 ×（action head → 18 层 expert → Euler）       │
                                                        └───────────────────────────────────────────────────┘
```

- **FPGA 运行一个 chunk 中的全部 transformer 计算。** 链节点以 INT8 完成所有 GEMM，PV 链节点完成注意力的 P·V 乘积，
  矢量节点完成其余全部运算（量化 / 反量化、norm、RoPE、softmax、GELU / GeGLU / SiLU、残差加、Euler 更新）。
  节点之间只通过 GDDR6 交换张量，并用 GDDR6 中的标志字自行安排各 stage 的先后顺序。
- **主机只做输入和输出。** 每个 chunk，主机计算进入模型的嵌入查表（SigLIP 的 patch 与位置嵌入、提示词 token 嵌入、状态投影），
  生成噪声，向 GDDR6 写入约 1.3 MB，把全部节点同时启动一次，等待 halt 向量，再读回 6.4 KB 的动作。半阵列上一个 chunk 的
  4,410 个 stage、28,316 个 part 之间没有任何主机工作。`pi0_chunk_run serve` 在 chunk 之间保持设备打开。
- **Jetson 不做模型计算。** 它负责相机、机械臂和动作队列，运行 LeRobot 的预处理和后处理，把每次观测发送到台式机。

## 2. 现状

| 部分 | 状态 |
|---|---|
| 节点 RTL：一个 chunk 中的每种计算，真实尺寸、真实权重 | 仿真中逐位一致；每种节点都已在硅片上运行 |
| 无主机参与的完整 chunk（SigLIP 27 × 2 + prefix 18 + 10 步 × expert 18 层）在 3 节点阵列上 | **已在硅片上验证**：标志和动作与生成器逐位一致 |
| 完整 chunk 在 18 节点半阵列上（bitstream `s1s_325`） | **已在硅片上验证**：逐位一致，每个 chunk 4.76 s |
| 46 个保留帧经芯片后端得到的动作与 fp32 模型比较 | **已在硅片上验证**：关节最大偏差 0.66°、平均 0.072° |
| 台式机策略服务器（chip 模式）对 Jetson 的 `/predict` 请求作答 | 已运行（录制帧回放）；机械臂经这条路径运行尚未进行 |
| 由实时 LeRobot 观测计算出的主机输入 | 与生成器使用的采集输入逐字节相同 |
| 融合 PDQ（预反量化）的 chunk 镜像 | 仅仿真验证逐位一致；硅片上动作与期望相差 ≤ 0.018（动作量程 1.73），原因待查 |
| GDDR6 4 KB 通道条带化（bitstream 中的开关 + ID 重排） | 仅仿真验证；硅片上未证实 |
| 35 节点全阵列 bitstream | 布局通过（逻辑 tile 56.9 %），**布线未收敛**；没有全阵列 bitstream |
| 板卡 | 台式机一次死机重启后，VP815 未出现在 PCIe 总线上，需要物理检查后才能继续硅片工作 |

第 3 到 5 节描述已建成的系统。第 6 节的步骤用半阵列 bitstream `s1s_325` 和对应的 chunk 镜像在硅片上运行过，
机械臂运行的步骤已准备好，尚未执行。

## 3. 环境与安装

### 3.1 硬件

| 项目 | 说明 |
|---|---|
| FPGA 卡 | Achronix VectorPath VP815（AC7t1500，32 GB GDDR6，16 个通道、每通道 2 GB），插在台式机的 PCIe Gen4 x16 插槽；USB JTAG 线从卡连到台式机 USB 口 |
| 台式机 | Linux，内核 6.8；Achronix ACE 10.5.2（构建 bitstream）、ACE 10.3.1（JTAG 编程，`PI0_JTAG_ACE`）、SDK 2.1.1 及 `acxpcie` 驱动；Python venv 中装有 LeRobot 和 PyTorch（CPU） |
| Jetson | NVIDIA Jetson Thor；LeRobot（conda 环境 `lerobot`），装有 `lerobot_policy_pi0_remote` 插件；demo 脚本的副本放在 `~/pi0_glue/` |
| 机器人 | UR7e，地址 192.168.1.2（Jetson 一侧）；Robotiq 夹爪；前置相机和腕部相机接在 Jetson 上 |
| 网络 | Jetson ↔ 台式机通过千兆直连线：台式机 192.168.10.1，Jetson 192.168.10.2，MTU 9000 |

### 3.2 软件与数据前提

各脚本的默认路径都可以用环境变量覆盖；下表是脚本假定的位置。

| 项目 | 位置 / 说明 |
|---|---|
| Achronix SDK 2.1.1 | `/opt/achronix/sdk`（`host/CMakeLists.txt` 探测；`ACHRONIX_SDK_ROOT` 覆盖）。SDK 自带的 `acxpcie` 驱动要打上 `tools/acxpcie_driver/` 的补丁（`acxdev_mmap()` 改用 `dma_mmap_coherent()`，否则在开启 IOMMU 的主机上 DMA 缓冲区映射会让进程被内核杀死）；构建出的 `acxpcie.ko` 放在 `~/pi0_board_bundle/sdk/driver/`，板卡脚本从那里 `insmod` |
| ACE 10.5.2 | `/home/sngong/ACE_10.5.2/Achronix-linux`（`scripts/launch_pi0_s0_build.sh`、`run_pi0_s0_reclock_bitstream.sh` 和 `paper/synth/*.sh` 的 `ACE_ROOT` 默认值）；ACE 与 Synplify 的许可证服务必须在运行 |
| ACE 10.3.1 | `~/ACE_10.3.1/Achronix-linux`：JTAG 编程用（`scripts/pi0_board_program_s0.sh` 的 `PI0_JTAG_ACE`） |
| Verilator 5 | `~/tools/verilator5/bin/verilator`（`paper/rtl/run_*.sh` 的 `VERILATOR` 默认值） |
| Python | `~/lerobot/.venv/bin/python`：LeRobot、PyTorch（CPU）、numpy、safetensors；`paper/rtl/run_*.sh` 的 `PYTHON` 默认值 |
| checkpoint | `~/pi0_ckpt/ur7e-demo-2-pi0-fsdp/010000/pretrained_model`（16 GB fp32；`paper/sw/pi0_layer_lower.py`、`prefix_w8a8_eval.py` 和策略服务器的默认路径） |
| 采集帧 | `~/pi0_glue/captures/<episode>/frame_NN.npz`：每帧的 prefix KV、状态、噪声和 fp32 参考动作，由 `glue/level3/capture_pi0_prefix_kv.py` 从回放文件（`glue/level2/export_replay_frames.py`）生成；chunk 生成、精度评估都读它 |
| 激活采集 | `build/paper_vector_unit/acts_<episode>_f<NN>.npz`：`paper/sw/vector_unit_capture.py` 在 CPU 上跑一帧 fp32 模型并记录每个矢量运算位置的输入；生成器和各层 golden 都需要 |
| 校准缓存 | `build/paper_vector_unit/calib/{calib.pt,calib_exp.pt}`：`paper/sw/regen_calib.py`（SmoothQuant / MSE 统计量，约 5 分钟） |
| 板卡 bundle | `~/pi0_board_bundle/bitstream/<name>/tc_ref_design_top.hex`、`~/pi0_board_bundle/sdk/`：编程脚本读这里；`scripts/pi0_bundle_s0.sh` 把一次 ACE 实现的结果放进去 |

## 4. 一个动作 chunk 的产生过程

**首个 chunk 之前，做一次。**

1. chunk 生成器（`paper/sw/pi0_chunk_program.py`）把整个模型 lower 到阵列上。它写出每个节点一个程序、GDDR6 映射，以及一个二进制镜像，
   其中包含 INT8 权重镜像、scale、bias、RoPE 表和位置表、掩码和程序。镜像必须与 bitstream 的特性一致（5.3 节）。
2. 通过 JTAG 给板卡编程阵列 bitstream（6.2 节）。
3. `host/pi0_chunk_run run` 通过分页桥把镜像（约 2.8 GB）写入 GDDR6，按 45–51 MB/s 约需 55–62 秒。随后它用生成镜像时所用的输入把 chunk
   运行一次，把每个 stage 的标志和动作与生成器给出的期望值比对。

**每个 chunk。**

1. **观测（Jetson）。** rollout 循环读取两个相机图像和关节状态。LeRobot 的 pi0 预处理器把图像缩放到 224 × 224、归一化状态、
   对提示词分词。`pi0_remote` 策略把这个 batch 发送到台式机的 `/predict`。
2. **主机输入（台式机 CPU）。** `paper/sw/pi0_chip_host_inputs.py` 计算进入模型的内容：
   - 每个相机的 SigLIP 第 0 层输入：14 × 14 的 patch 卷积加位置表（256 × 1152）；
   - 提示词的 token 嵌入（本任务为 13 × 2048）；
   - 状态 token：`state_proj(state)`（1024）；
   - flow-matching 噪声（50 × 32）。
   `pi0_chip_runtime.py` 按芯片要求的格式编码（bf16 码加每行最大值；噪声为 fp32），写成 `pi0_chunk_run --inputs` 的运行清单。
   服务器检查提示词 token 数与 chunk 生成时一致，否则拒绝该观测。
3. **启动。** `pi0_chunk_run serve` 在会话开始时（以及每次 `run` 时）先软复位阵列并重新写入分页桥的页基址和条带化开关；之后每个 chunk
   通过分页桥（PCIe BAR1）写入输入区域，清零全部 stage 标志（每个 part 一个 beat），再向寄存器窗口（PCIe BAR0）写一次广播启动位，
   同时启动全部节点。
4. **FPGA 上。** 每个节点从 GDDR6 运行自己的程序：
   - SigLIP：每个相机 27 层，然后是 post-layernorm 和 projector，得到 2048 维的图像 token；
   - PaliGemma prefix：对 525 个 token（2 × 256 个图像 token + 13 个提示词 token）运行 18 层。每层的 key 和 value 留在 GDDR6 中，
     供 expert 使用；
   - 10 个去噪步：action head 的输入侧（动作与时间嵌入）；对状态 token 和 50 个动作 token 运行 18 层 expert，关注 prefix 的
     key 和 value；最终 norm 和输出投影；对 50 × 32 的动作状态做 Euler 更新。
   一个 stage 由一个或多个节点程序中的 part 组成。链的 stage 按列 tile 分给各链节点，矢量的 stage 按行分给各矢量节点。
   每个 part 先等待它所依赖的 part 在 GDDR6 中的标志（WAIT），结果写完后再写出自己的标志（POST）。两个相机的 SigLIP stage 交错执行，
   prefix 中注意力之后的部分按行切块，使矢量节点和链节点可以重叠工作。
5. **读回。** 主机轮询 halt 向量，从 IO 区读取 50 × 32 的 fp32 动作，随 HTTP 响应返回。
6. **执行（Jetson）。** 后处理器做反归一化，保留 7 个真实维度（6 个关节 + 夹爪）。动作队列把新 chunk 与队列中剩余的动作合并，
   每个控制周期向 UR 伺服交出一个动作（500 Hz 的 servoJ 保持最新目标）。队列低于阈值时发出下一次请求。

每个 chunk 主机侧的开销只有几十毫秒（3 节点阵列的镜像上实测：编码约 10 ms，写入 1.3 MB 输入 23 ms，清标志 0.4 ms，读动作 1.4 ms；
半阵列镜像的标志数多 7 倍，清标志相应变长）；其余时间都是芯片计算（半阵列 `s1s_325` 上 4.76 s）。

## 5. 实现

### 5.1 FPGA 设计

设计的逐文件说明见 `paper/rtl/README.md`。与 demo 相关的要点：

| 项目 | 取值 |
|---|---|
| 半阵列（已交付，`s1s_325`） | 18 个节点：芯片节点 0–5 为 32 级 int8 链、6–11 为 16 级 int8 链、12–13 为 uint8 PV 链（16 级）、14–17 为矢量节点（2 个 lane、2 个操作数加载器、行长最多 4,096 个元素）；每个节点 2 个 NoC 接入点；`launch_pi0_s0_build.sh <tag> 14 2 4 6` |
| 全阵列（`pi0_chip_top`，只布局未布线） | 35 个节点：0–11 为 32 级 int8 链、12–23 为 16 级、24–26 为 PV 链、27–34 为矢量节点；`launch_pi0_s0_build.sh <tag> 27 3 8 12` |
| FPGA 顶层 | `src/rtl/pi0_s0_top.sv`：IO ring（PCIe Gen4 x16 端点、8 个 GDDR6 控制器、Device Manager）包着 `pi0_chip_top`；参数 `STRIPE`（0 或 12） |
| 时钟 | 设计目标：阵列 725 MHz、矢量节点 333.33 MHz、fabric / NoC 接入点 250 MHz，来自两个 PLL（`PLL_SW_3`、`PLL_SW_1`），NoC 自身的 PLL 不变。已布线的半阵列按 `scripts/pi0_s0_pick_clocks.py` 从时序报告选出的合法 PLL 设置改时钟后运行：`s1s_325` 为阵列 325、矢量 125、fabric 117.6 MHz（布线后 Fmax 355.6 / 135.8 / 123.4 MHz） |
| 主机窗口 | PCIe BAR0 → 位于 `NOC[3][4]` 的 `pi0_chip_ctrl`：每个节点的程序基址和启动位、广播启动、软复位、halt / error 向量、每节点周期计数器、bitstream ID；beat `N_NODE+1` 的 bit 64 是 GDDR6 条带化开关 |
| 主机批量数据通路 | PCIe BAR1（256 MB）→ 位于 `NOC[3][5]` 的 `pi0_host_gddr_bridge` → GDDR6，页基址通过主机窗口设置；全部为程序控制 I/O，没有 DMA |
| 资源占用 | 半阵列（布线后）：逻辑 tile 36.2 %、BRAM72K 33.4 %、MLP72 21.0 %；全阵列（仅布局）：逻辑 tile 56.9 %、BRAM72K 59.3 %、MLP72 37.8 %，布线未收敛 |
| 构建 | ACE 10.5.2：`scripts/launch_pi0_s0_build.sh <tag> <N_CHAIN> <N_PV> <N_VNODE> <N_DEEP>`，环境变量 `PI0_S0_STRIPE=12 PI0_S0_RESET_FP=1`、`PI0_S0_REGIONS_PDC`（每节点软区域）、`PI0_ACE_SEED`、`PI0_S0_ID_WORD`；`scripts/run_pi0_s0_reclock_bitstream.sh` 可以不重新布线就修改已布线实现的三个时钟；`scripts/pi0_bundle_s0.sh` 打包 |

### 5.2 数值方案

三个 transformer 使用同一套 W8A8 方案：

- 权重：每个输出通道一个 INT8 scale，按 MSE 最优裁剪；
- GEMM 输入：每个 token 一个 INT8 scale，运行时在芯片上计算（QUANT）；Gemma 各层的 SmoothQuant（α = 0.5）折叠进 norm 增益和权重；
- GEMM 输出：精确的整数和，在芯片上用行 scale × 列 scale 反量化为 bf16（DEQUANT）；
- 注意力：INT8 的 QKᵀ；softmax 输出为 uint8 概率，在 PV 链节点上与 INT8 的 value 相乘；
- norm、softmax、GELU / GeGLU / SiLU、RoPE、残差、Euler：矢量 lane 内部用 fp32 精度的算术，配 2048 项插值表；各运算之间用 bf16。

两项算子融合不改变算术，只减少矢量节点的遍历次数：**QUANT 尾部**（产生 bf16 输出的运算直接在 lane 内量化，记录标志 `w0[118]`，
生成器选项 `--fuse-quant`）和 **PDQ 预反量化**（两个 DEQUANT 与 GeGLU 合成一次遍历，标志 `w0[119]`，`--fuse-pdq`）。
QUANT 尾部已在硅片上逐位一致；PDQ 在仿真中逐位一致，硅片上的整 chunk 尚有 ≤ 0.018 的偏差（见第 2 节）。

`paper/sw/` 中的 Python 参考模型逐位复现这套算术，因此精度可以离线测量，RTL 也用同一套参考来检查。

### 5.3 chunk 生成器（`paper/sw/`）

- `pi0_chunk_layout.py` 保存一个 chunk 的 GDDR6 映射，分五个区：PROG（节点程序）、STATIC（权重镜像、scale、表；只写一次）、
  IO（每 chunk 的输入和动作）、FLAGS（每个 stage part 一个 32 字节 beat）、SCRATCH（stage 之间的区域，每个 chunk 重复使用）。
  板卡的 GDDR6 是 16 个通道、每个 2 GB，通道内超过 2 GB 的地址会混叠。`PI0_CHUNK_MAP` 选择映射：`channels`（默认，STATIC 和
  SCRATCH 轮流分布在各通道上，使同时工作的节点从不同通道读取）、`compact`（512 MB 以内，用于测试）、`striped`（一个逻辑 32 GB
  地址空间，只配合 `STRIPE=12` 的 bitstream 并打开条带化开关）。每次分配都做重叠检查。
- `pi0_chunk_layers.py` 用各层 RTL 测试所验证的同一套 lowering 构建各层（SigLIP、prefix、expert、视觉两端、action head）。
  `pi0_chunk_tiler.py` 按执行该 GEMM 的节点的链深（16 或 32 级）为每个 GEMM 分块；`PI0_CHUNK_GEMM_DB=1` 时使用双缓冲的
  feeder（opcode 6 的 bit 106，下一组行的权重在当前组计算时装载）。
- `pi0_chunk_program.py` 把各层串成 chunk，把每个 stage 分配给节点（矢量 stage 按 4 / 8 / 16 行的最细粒度切分），插入 WAIT / POST，
  做算子融合，输出程序和镜像。它对每个程序重新检查硬件的 lowering 规则。`--sync deps --interleave --prefix-row-blocks 3` 给出
  stage 重叠的调度。
- `pi0_chunk_time.py` 用各层实测仿真校准的时间模型估计一个 chunk 的时间；`pi0_timeline_compare.py` 把硅片上 `--timeline` 记录的各
  标志时刻与模型逐段比较。

镜像家族必须与 bitstream 匹配：`chnd_*`（无双缓冲，用于早期 bitstream）、`chdb_*`（双缓冲，`s1s_325` 使用）、`chdbp_*`（双缓冲 + PDQ）。
`pi0_chunk_run` 根据镜像的 `info.json` 设置条带化开关。

半阵列的完整 chunk（`chdb2_half_full`）：

| 项目 | 数值 |
|---:|---|
| stage / part 数 | 4,410 / 28,316（全阵列 54,588 个 part） |
| 节点程序 | 18 个（全阵列 35 个） |
| GDDR6 镜像（只写一次） | 约 2.8 GB |
| 每个 chunk 的 GDDR6 流量 | 读 41.8 GB，写 5.7 GB |
| 每个 chunk 的主机流量 | 写入约 1.3 MB，读回 6.4 KB |
| QUANT 融合 | 2,112 个 QUANT 记录中 2,004 个被融合 |
| 生成时间 | 约 10 分钟 |

```bash
~/lerobot/.venv/bin/python paper/sw/regen_calib.py                     # 只需一次：激活统计量（约 5 分钟）
PI0_CHUNK_GEMM_DB=1 ~/lerobot/.venv/bin/python paper/sw/pi0_chunk_program.py --out build/paper_pi0_chunk/chdb2_half_full \
    --n-vec 4 --n-chain 12 --n-deep 6 --n-pv 2 --n-stage 16 --slot-bits 12 \
    --bin --exp-areas IO,FLAGS --sync deps --interleave --prefix-row-blocks 3 --fuse-quant
```

全阵列使用 `--n-vec 8 --n-chain 24 --n-deep 12 --n-pv 3`；PDQ 镜像再加 `--fuse-pdq`。

### 5.4 主机运行时

- `host/pi0_chunk_run`（C++，Achronix SDK）是唯一直接访问板卡的程序。所有数据都通过分页桥以程序控制 I/O 方式传输：
  `PI0_HOST_BRIDGE=1`、`PI0_FPGA_DBI_ROUTE=comp`。
  - `selftest` 检查分页桥的各页互不混叠。
  - `run <chunk 目录>` 加载镜像，让每个节点指向自己的程序，软复位后同时启动，轮询 halt 向量，检查标志和动作。选项：
    `--map`（节点映射）、`--repeat N`（同一镜像上重复运行）、`--timeline <file>`（记录每个标志首次出现的时刻）、`--inputs <prefix>`
    与 `--dump-actions <file>`（每 chunk 模式）、`--no-load --scrub-flags-only --check-io-only`、`--load-only`、`--wrong-list <file>`、
    `--timeout-s`。
  - `serve <chunk 目录>` 在镜像已加载的前提下保持设备打开，从标准输入逐行接收 `infer <inputs prefix> <actions file>`，每行运行一个
    chunk，回复 `OK chip_s= flags_s= inputs_s= actions_s= total_s=`；`quit` 退出。
  - `probe`、`fill`、`peek`、`verify` 是 GDDR6 通路的检查工具（地址混叠、条带化开关、镜像回读）。
  - `--map` 把生成器的节点编号（矢量、int8、uint8）转换为芯片的编号：半阵列 `vector:14,int8:0,uint8:12`，全阵列
    `vector:27,int8:0,uint8:24`。
- `paper/sw/pi0_chip_runtime.py`：`ChunkIO` 在 chunk 的 `regions.json` 中按名称找到输入和输出区域并编码输入；`ChipServer` 启动并
  持有一个 `pi0_chunk_run serve` 子进程（默认，`PI0_CHIP_SERVE=1`），`infer()` 返回动作和芯片报告的时间；`PI0_CHIP_SERVE=0` 时
  每个 chunk 启动一次 `pi0_chunk_run run`（可同时与生成器的期望比对）。子进程只会被要求 `quit`，不会被 kill。
- `paper/sw/pi0_chip_host_inputs.py` 从 LeRobot 策略对象计算四项主机输入；其 `validate` 入口用采集帧证明实时观测送入芯片的字节与
  生成器使用的相同。

### 5.5 策略服务器（`glue/level3/`）

- `pi0_fpga_policy.py`（`PI0FpgaPolicy`）在 `PI0_ACTION_EXPERT=chip` 时把整个模型交给节点阵列：`predict_action_chunk` 计算主机输入，
  调用芯片运行时，返回动作。环境变量：`PI0_CHIP_CHUNK`（已加载到板卡的 chunk 目录）、`PI0_CHIP_MAP`（节点映射）、`PI0_CHIP_SW`
  （`paper/sw` 目录，默认按仓库相对路径找到）、`PI0_CHIP_WORK`（运行清单的工作目录，默认 `<chunk>/runtime`）。
  每个 chunk 的 `last_stats` 含 `chip_s`、`flags_s`、`inputs_s`、`actions_s`、`encode_s`、`wall_s`、`host_inputs_s`、`total_s`，
  以及 `from_hardware=True`、`expert_backend="chip"`。
- `pi0_policy_rpc_server.py` 通过 HTTP（端口 8081）提供 `/predict`（输入预处理后的 batch，输出归一化动作和统计）和 `/health`
  （`expert_backend`、`last_stats`）。在 chip 模式下它以内存映射方式打开 checkpoint（`prefix_w8a8_eval.load_policy`），
  而不是加载 16 GB，因为主机只会用到嵌入表、patch 卷积和 `state_proj`。一次只处理一个请求。
- `run_pi0_chip_policy_server.sh` 是启动脚本：设置 `PI0_ACTION_EXPERT=chip`、`PI0_HOST_BRIDGE=1`、`PI0_FPGA_DBI_ROUTE=comp`，
  检查 `PI0_CHUNK_RUN`（默认 `build/host/pi0_chunk_run`）存在，再通过 `run_pi0_policy_rpc_server.sh`（`ASYNC_POLICY_PATH`、
  `PI0_RPC_PORT`、`OMP_NUM_THREADS`，日志在 `~/pi0_glue/logs/`）启动服务器。
- `eval_chip_frames.py` 让录制的帧走同一个后端，把芯片的动作与 fp32 模型的动作比较（关节角误差、7 维相对 RMS）。

### 5.6 机器人端（Jetson）

- **插件** `glue/level3/plugins/lerobot_policy_pi0_remote/`，作为 LeRobot 第三方策略包安装。`pi0_remote` 策略在 Jetson 上保留
  预处理和后处理，把 `predict_action_chunk`（pickle 的预处理 batch）POST 到 `/predict`，超时 180 s。它不持有任何模型权重；
  `make_pi0_remote_policy_dir.py` 从 checkpoint 复制预处理 / 后处理配置（归一化统计量、tokenizer），生成策略目录。
- **无机械臂的回路测试** `test_pi0_remote_roundtrip.py`：在 Jetson 上对一帧录制观测做预处理 → 远程 chunk → 后处理。
- **机械臂客户端** `pi0_arm_client.py` / `run_pi0_arm_client.sh`：LeRobot 的 `RobotClient`（相机、RTDE、动作队列、伺服）加上
  chunk 看门狗（`--watchdog_s`，超时则 `servoStop` 并保持当前位置）、停止文件（`touch /tmp/pi0_arm_stop`）、`--max_run_s` 上限，
  以及 `report.json` / `queue.png`（观测发送时间、chunk 往返、队列深度）。它连接一个 LeRobot 策略服务器（`ASYNC_SERVER_ADDRESS`），
  该服务器加载 `pi0_remote` 策略，从而把每个 chunk 交给台式机。`glue/level2/pi0_replay_client.py` 是同一客户端的回放版本，
  不接机械臂。
- **控制速率与时延。** chunk 为 50 个动作、往返时延为 L 秒时，只有控制速率不超过 25 / L 个动作每秒，队列才不会断供；
  队列阈值必须至少为 L × 速率 + 1。L ≈ 5 s（半阵列）时约可达到每秒 5 个动作。
- **安全。** 只有 Jetson 上的进程连接机械臂。它负责伺服保活和停止路径（Ctrl+C、停止文件，或看门狗 → 伺服停止）。
  如果台式机停止响应，队列会耗尽，机械臂保持最后的目标位置。UR 急停和保护性停止是独立的硬件机制。

## 6. 运行系统

**操作规则。**

- 机械臂只在白天运行，且安全负责人必须在急停按钮旁。
- 打开了 FPGA 设备的进程（`pi0_chunk_run`、服务器、SDK 工具）只能用 **Ctrl+C**（SIGINT）停止。关闭它的窗口、`kill` 或
  SIGTERM 都可能导致台式机死机（见第 8 节）。
- 不要用 `systemctl --force --force reboot` 重启台式机。
- 不要在同一台机器上让 ACE 构建与板卡操作同时进行（`run_pi0_policy_rpc_server.sh` 会拒绝启动，除非 `ALLOW_ACE_BUILD=1`）。

### 6.1 一次性准备

1. 在台式机上安装 Achronix USB-JTAG 的 udev 规则（ACE 的 `install_acx_bitporter_usb.pl`），把 JTAG 线接到台式机 USB 口。
2. 构建板卡工具：`cmake -S host -B build/host && cmake --build build/host`（目标 `pi0_chunk_run`、`pi0_s0_replay`、`pi0_atu_tool`、
   `pi0_dbi_probe`）。
3. 准备 bitstream：解压仓库中的 `bitstream/s1s_325/tc_ref_design_top.hex.gz` 到 `~/pi0_board_bundle/bitstream/s1s_325/`
   （`sha256sum -c SHA256SUMS` 核对）；或者自己构建并打包：

   ```bash
   PI0_S0_STRIPE=12 PI0_S0_RESET_FP=1 scripts/launch_pi0_s0_build.sh half 14 2 4 6   # 日志在 build/s0/，实现目录为 src/ace/impl_s0_half_*
   scripts/pi0_s0_pick_clocks.py src/ace/<impl 目录名>                                # 从时序报告选出合法时钟，打印改时钟命令
   scripts/run_pi0_s0_reclock_bitstream.sh <impl 目录名> 325 2 13 64 68   # 阵列 400 × 13 / 2 / 8 = 325 MHz；参数取自上一步的输出（s1s_325：矢量 ODN 64 = 125 MHz，fabric ODN 68 = 117.6 MHz）
   scripts/pi0_bundle_s0.sh <impl 目录名> s1s_325                                       # → ~/pi0_board_bundle/bitstream/s1s_325/
   ```
4. 准备数据（3.2 节）并生成 chunk（5.3 节）。
5. 在 Jetson 的 `lerobot` 环境中：`pip install -e glue/level3/plugins/lerobot_policy_pi0_remote`；用
   `glue/level3/make_pi0_remote_policy_dir.py --checkpoint <本地 checkpoint> --server-url http://192.168.10.1:8081
   --out ~/pi0_glue/policies/pi0_remote_ur7e-demo-2-pi0-010000` 创建策略目录；把 `glue/level3/` 和 `glue/level2/` 的脚本复制到
   `~/pi0_glue/`。

### 6.2 台式机：给板卡编程

```bash
cd <仓库目录>
SUDO_ASKPASS=<helper> scripts/pi0_board_program_s0.sh bitstream/s1s_325/tc_ref_design_top.hex   # 路径相对于 ~/pi0_board_bundle
```

该脚本在卸载驱动的状态下通过 JTAG 编程 FPGA，重新训练 PCIe 链路直到 DLActive，移除并重新扫描设备（编程会使 BAR 分配丢失），
把链路强制到 16 GT/s（链路速率较低时 GDDR6 传输不可靠），加载驱动，最后用 `pi0_s0_replay identify` 从主机窗口读出 bitstream ID 和
节点数（`s1s_325` 为 `0x53310a`、18 个节点）。JTAG 需要 ACE 许可证服务处于运行状态。之后用
`PI0_HOST_BRIDGE=1 PI0_FPGA_DBI_ROUTE=comp build/host/pi0_chunk_run selftest` 确认分页桥；6.3 节的第一次 `run` 会再打印
`window: N_NODE=18 id=0x53310a`。

### 6.3 台式机：加载 chunk 并验收

```bash
# 分页桥自检，然后加载 chunk，并对照生成器的期望标志和动作运行一次（约 60 s 加载 + 4.8 s 运行）
scripts/pi0_array_silicon_session.sh half build/paper_pi0_chunk/chdb2_half_full
# 期望：selftest PASS；chunk 的标志和动作全部一致（0 wrong）；没有节点报错
# 或者手动：
PI0_HOST_BRIDGE=1 PI0_FPGA_DBI_ROUTE=comp build/host/pi0_chunk_run run build/paper_pi0_chunk/chdb2_half_full \
    --map vector:14,int8:0,uint8:12 --repeat 3 --timeline build/timeline.txt
```

### 6.4 台式机：启动服务器；Jetson：回路测试与运行

```bash
# 台式机：服务器（持有设备；只能用 Ctrl+C 停止）
PI0_CHIP_CHUNK=$PWD/build/paper_pi0_chunk/chdb2_half_full PI0_CHIP_MAP=vector:14,int8:0,uint8:12 \
ASYNC_POLICY_PATH=~/pi0_ckpt/ur7e-demo-2-pi0-fsdp/010000/pretrained_model glue/level3/run_pi0_chip_policy_server.sh
curl -s http://127.0.0.1:8081/health        # "ok": true, "expert_backend": "chip"
```

```bash
# Jetson（conda env lerobot）：先关闭任何相机预览，再做无机械臂的回路测试
fuser /dev/video0 /dev/video2 || echo "cameras free"
curl -s --noproxy '*' http://192.168.10.1:8081/health; echo
python ~/pi0_glue/test_pi0_remote_roundtrip.py --policy ~/pi0_glue/policies/pi0_remote_ur7e-demo-2-pi0-010000 \
    --replay ~/pi0_glue/replay_ur7e_demo1_ep0.npz
# 机械臂：LeRobot 策略服务器加载 pi0_remote 策略（--policy_type=pi0_remote --pretrained_name_or_path=<策略目录>），然后
ASYNC_SERVER_ADDRESS=<策略服务器 host:port> ASYNC_FPS=1 ~/pi0_glue/run_pi0_arm_client.sh
touch /tmp/pi0_arm_stop                     # 从另一个终端停止机械臂客户端（servoStop + stopScript）
```

运行结束后按 Ctrl+C 或触碰停止文件结束客户端；服务器可以保持运行，供下一次使用。

### 6.5 精度评估（不连接机械臂）

```bash
PI0_ACTION_EXPERT=chip PI0_CHIP_CHUNK=$PWD/build/paper_pi0_chunk/chdb2_half_full PI0_CHIP_MAP=vector:14,int8:0,uint8:12 \
PI0_CHUNK_RUN=$PWD/build/host/pi0_chunk_run ~/lerobot/.venv/bin/python glue/level3/eval_chip_frames.py \
    --frames demo1_ep20:all demo1_ep40:all recov_pi0_ep00:all recov_pi0_ep10:all --json build/eval_chip.json
```

生成镜像所用的那一帧（`demo1_ep20:2`）必须逐位复现生成器的动作；其余帧给出关节角误差和相对 RMS。评估结束后，板上的输入区已被
其他帧覆盖，下一次 `pi0_chunk_run run` 不要带 `--no-load`（否则会拿陈旧输入去比对生成器的期望）。

### 6.6 故障处理

| 现象 | 处理 |
|---|---|
| Jetson 报服务器离线 | 在台式机上检查 `/health`；服务器没有运行就重新启动 |
| `pi0_chunk_run` 提示 “DBI gateway does not answer” | 重新给板卡编程（6.2 节） |
| 某个节点报错或 chunk 超时 | `build/host/pi0_s0_replay status --node N` 显示该节点的错误细节位；`pi0_s0_replay reset` 软复位整个阵列；重新加载 chunk |
| 32 级链节点报 overrun 标志（错误位 0x04）而数据逐位一致 | 阵列时钟 325 MHz 下已知的假标志，不影响结果 |
| 标志正确但动作或中间区域错误、每次不同 | 镜像家族与 bitstream 不匹配（`chnd` / `chdb` / `chdbp`，条带化开关），按 5.3 节重新生成 |
| 编程结束后没有链路 | 再运行一次编程脚本；第一次 JTAG 尝试出现 USB I/O 错误不影响结果 |
| 台式机重启后 `lspci` 看不到 `1b59:` 设备 | 卡未重新枚举：先 `scripts/pi0_board_recover.sh --check`；不重启的恢复用 `scripts/pi0_board_recover.sh <hex>`（唤醒根端口 → JTAG 编程 → 重训练 → 移除 / 重扫 → 16 GT/s → 加载驱动）；仍不行则 JTAG 编程后用 `scripts/pi0_board_prepare_warm_reboot.sh` 有序重启，重启后 `scripts/pi0_board_after_warm_reboot.sh` |
| 提示词长度与 chunk 不一致 | 服务器会拒绝该观测；为这个提示词重新生成 chunk |
| 相机错误 | 关闭相机预览（`pkill -f rerun`）后重新连接 |
| 机械臂出现意外动作 | 按急停，然后触碰停止文件 |
| 台式机死机 | 断电重启；板卡可能不再出现在总线上（上一行），然后重做 6.2 节 |
| 在半阵列和全阵列构建之间切换时布局失败 | `git checkout -- src/ace/tc_ref_design_top.acxprj`（上一次构建把区域写进了工程文件） |

## 7. 结果

**硅片上的完整 chunk**（VP815，标志和 50 × 32 动作与生成器比较；时钟为阵列 / 矢量 / fabric MHz）：

| bitstream | 阵列 | 时钟 | 结果 | 每 chunk |
|---|---|---|---|---:|
| `s0tm_250` / `s0tn_500` | 3 节点（1 个 16 级 int8 链、1 个 PV 链、1 个矢量节点） | 250 / 166.7 / 166.7；500 / 285.7 / 222.2 | 逐位一致 | 43.65 s / 26.28 s |
| `s1m_250` | 半阵列，18 节点 | 250 / 166.7 / 114.6 | 逐位一致 | 7.29 s |
| `s1q_225` | 半阵列 + 写合并 | 225 / 181.8 / 125 | 逐位一致 | 6.66 s |
| `s1r_325` | 半阵列 + 双缓冲 feeder | 325 / 153.8 / 105.3 | 逐位一致 | 6.73 s |
| **`s1s_325`**（已交付） | 半阵列 + PDQ、条带化开关、ID 重排 | 325 / 125 / 117.6 | 逐位一致（`chdb2_half_full`） | **4.76 s** |
| `s1s_325`，PDQ 镜像 | 同上 | 同上 | 标志正确，动作偏差 ≤ 0.018 | 4.63 s |

芯片时间与校准模型的比值从 1.63（`s1m`）降到 1.41（`s1q`）：差距来自节点等待 GDDR6 往返，写合并和双缓冲 feeder 就是为此加入的。

**精度**（硅片，`eval_chip_frames.py`，芯片动作与相同输入和噪声下的 fp32 PyTorch 模型比较）：

- 46 个保留帧：关节最大偏差 0.66°，平均 0.072°；7 维动作的相对 RMS 平均 0.91 %。
- 生成镜像的那一帧：芯片动作逐位等于生成器的期望；与 fp32 模型相差 1.33 %（相对 RMS）。
- 由实时观测计算出的主机输入，与生成并验证 chunk 时使用的输入完全相同。

**仿真中**（Verilator，真实权重和采集的激活，检查每一个写出的 beat）：

| 测试 | 结果 |
|---|---|
| 每种层的真实尺寸：SigLIP 层、525 token 的 prefix 层、expert 层、action head + Euler、视觉两端 | 逐位一致 |
| 半阵列 18 个节点、混合链深、融合 QUANT、双缓冲 feeder 的 chunk，每节点独立存储端口、64 周期往返 | 逐位一致（2,116,166 beat） |
| 1 个 prefix 层 + 1 个 expert 层、PDQ 融合 | 逐位一致（5,494,033 beat） |
| 条带化 + ID 重排的 chunk（乱序响应的存储模型） | 逐位一致 |

**chunk 时间的预测**（`pi0_chunk_time.py`，理想存储，按各层实测仿真校准；不含主机每 chunk 的几十毫秒）：

| 阵列 | 设计时钟 250 / 333.3 / 725 MHz | 矢量 285.7 MHz |
|---|---:|---:|
| 全阵列，35 个节点 | 0.77 s（PDQ 0.73 s） | 0.85 s |
| 半阵列，18 个节点 | 1.46 s | 1.63 s |

这些是预测值：全阵列尚未布线，半阵列的布线时钟（325 / 125 / 117.6 MHz）远低于设计时钟。作为对比，同一个模型在 Jetson Thor GPU 上
每个 chunk 约 1 s。

## 8. 局限与注意事项

- **时钟。** 已布线的半阵列只能运行在阵列 325、矢量 125、fabric 117.6 MHz，比设计目标低 2–3 倍，是当前 4.76 s 的主要原因。
  单个节点独立布线可达阵列 870、矢量 348、fabric 356 MHz；差距在阵列级布线。
- **全阵列。** 35 节点全阵列布局通过但布线未收敛，因此只有半阵列 bitstream；chunk 时间是全阵列预测值的 6 倍。
- **PDQ 镜像。** 硅片上有 ≤ 0.018 的动作偏差，原因待查；交付的镜像配方不含 `--fuse-pdq`。
- **条带化。** bitstream 含条带化开关和 ID 重排，`striped` 映射只在仿真中验证。
- **机械臂。** 机械臂经芯片后端运行尚未进行；已完成的是录制帧回放和策略服务器。
- **chunk 形状固定。** 生成的 chunk 固定为两个相机和一种提示词长度（本任务为 13 个 token）。换提示词需要重新生成 chunk（约 10 分钟）
  并重新加载（约 60 秒）。
- **嵌入在主机上。** patch 卷积、token 嵌入查表和状态投影在台式机 CPU 上运行。
- **没有实时分块（RTC）引导。** 10 个去噪步在芯片上连续运行，主机不介入，因此 LeRobot 的逐步 RTC 引导不起作用；
  动作队列仍会合并前后两个 chunk。
- **没有 DMA。** 这些 bitstream 上 PCIe DMA 引擎不可用（SDK 的 DMA 初始化会卡死压缩 DBI 网关）；主机的全部数据都通过分页桥以程序控制
  I/O 方式传输。代价是加载时约 60 秒，每个 chunk 几十毫秒。
- **驱动。** `acxpcie` 的 release 路径在持有自旋锁时释放 DMA 内存，并重复解锁两次。持有设备的进程如果没有正常关闭就退出，
  台式机可能死机。Ctrl+C 是安全的；各工具会捕获异常并关闭设备。
- **链路速率与重新编程。** PCIe 链路必须运行在 16 GT/s x16。每次编程 FPGA 后都必须对 PCIe 设备做移除和重新扫描；台式机重启会让板卡回到
  flash 中的镜像，因此 bitstream 必须重新通过 JTAG 加载；主机死机重启后卡可能不再枚举（6.6 节）。编程脚本会处理常规步骤。

## 9. 文件索引

| 路径 | 内容 |
|---|---|
| `paper/rtl/`（见其 README） | 节点阵列 RTL、testbench、仿真模型 |
| `src/rtl/pi0_s0_top.sv`, `src/constraints/pi0_s0_*.{sdc,pdc}`, `src/acxip/`, `src/ace/`, `paper/synth/s0/` | FPGA 顶层、约束、IO ring IP（PCIe、GDDR6、PLL）、ACE 工程、Synplify 工程 |
| `scripts/launch_pi0_s0_build.sh`, `run_pi0_s0_ace_flow.tcl`, `run_pi0_s0_reclock_bitstream.sh`, `pi0_s0_pick_clocks.py`, `pi0_bundle_s0.sh` | bitstream 构建、选时钟、改时钟和打包 |
| `scripts/pi0_board_program_s0.sh`, `pi0_board_recover.sh`, `pi0_board_prepare_warm_reboot.sh`, `pi0_board_after_warm_reboot.sh`, `pi0_board_bringup_from_fics.sh` | 板卡编程与恢复 |
| `scripts/pi0_array_silicon_session.sh`, `pi0_s0t_silicon_session.sh`, `pi0_s0_silicon_matrix.sh` | 板卡验收：半 / 全阵列 chunk、3 节点阵列、单节点测试集 |
| `paper/sw/pi0_chunk_layout.py`, `pi0_chunk_layers.py`, `pi0_chunk_tiler.py`, `pi0_chunk_program.py` | chunk 生成器与 GDDR6 映射 |
| `paper/sw/pi0_chunk_time.py`, `pi0_timeline_compare.py`, `pi0_chunk_wrong_regions.py`, `pi0_chunk_trace_wrong.py` | 时间模型、硅片时间线比较、错误 beat 定位 |
| `paper/sw/pi0_chip_runtime.py`, `pi0_chip_host_inputs.py` | 主机运行时和主机输入 |
| `paper/sw/vector_unit_capture.py`, `regen_calib.py`, `prefix_w8a8_eval.py`, `pi0_layer_lower.py` | 激活采集、校准、数值方案与逐层 lowering |
| `host/pi0_chunk_run.cpp`, `pi0_s0_replay.cpp`, `pi0_atu_tool.cpp`, `pi0_dbi_probe.cpp`, `CMakeLists.txt` | 板卡工具 |
| `glue/level3/pi0_fpga_policy.py`, `pi0_policy_rpc_server.py`, `run_pi0_chip_policy_server.sh`, `run_pi0_policy_rpc_server.sh`, `eval_chip_frames.py` | 策略后端、服务器、启动脚本、精度评估 |
| `glue/level3/plugins/lerobot_policy_pi0_remote/`, `make_pi0_remote_policy_dir.py`, `test_pi0_remote_roundtrip.py` | 机器人端策略 `pi0_remote`、策略目录、回路测试 |
| `glue/level3/pi0_arm_client.py`, `run_pi0_arm_client.sh`, `capture_pi0_prefix_kv.py`; `glue/level2/pi0_replay_client.py`, `export_replay_frames.py` | 机械臂客户端、采集帧、回放客户端、回放文件导出 |
| `tools/acxpcie_driver/` | 驱动补丁及说明 |
| `bitstream/s1s_325/` | 已交付的半阵列 bitstream、INFO、校验和 |
| `~/pi0_board_bundle/`（仓库外） | 编程脚本使用的 bitstream 目录、SDK 工具和驱动 |
