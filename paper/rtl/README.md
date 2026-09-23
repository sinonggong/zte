# pi0 节点阵列 — RTL

本目录是在 Achronix Speedster7t **AC7t1500**（VectorPath VP815 加速卡）上运行完整 pi0 视觉-语言-动作模型的 RTL：
SigLIP 视觉编码器、PaliGemma 语言模型 prefix，以及带 10 步 flow-matching 循环的 action expert，全部在芯片上完成，
主机只负责写入每个 chunk 的输入、启动各节点、读回动作。

## 设计概览

设计由三种节点组成阵列，节点之间只通过 GDDR6 交换数据：

| 节点 | 顶层模块 | 功能 |
|---|---|---|
| int8 链节点 | `colpar_chain_node.sv`（`MULT_MODE = 5'h00`） | 在一列 16 或 32 个 MLP72 硬核上做 INT8 GEMM |
| uint8 PV 链节点 | `colpar_chain_node.sv`（`MULT_MODE = 5'h13`） | 注意力的 P·V：uint8 概率 × int8 value |
| 矢量节点 | `vector_unit/vu_node_ml.sv` | GEMM 以外的全部运算：量化 / 反量化、RMS norm 与 layer norm、RoPE、softmax、GELU / GeGLU / SiLU、残差加、Euler 更新；量化尾（QUANT tail）和预反量化（PDQ）两种算子融合 |

每个节点都是器件片上网络（NoC）上的一个 AXI 主设备。节点自己从 GDDR6 取程序、读取权重和激活、把结果写回，并通过
GDDR6 中的标志字与其他节点同步（WAIT / POST）。主机通过分页桥写 GDDR6，通过一个小的寄存器窗口把每个节点各启动一次，
然后等待 halt 向量；一个 chunk 运行期间主机不做任何其他事情。

```
 主机 (PCIe) ──BAR0─▶ pi0_chip_ctrl（寄存器窗口，一个 NAP）
             │        │ 每个节点的程序基址 + 启动位；返回 halt / error / 周期数；分页桥页基址；条带化开关
             └─BAR1─▶ pi0_host_gddr_bridge（一进一出两个 NAP）══▶ GDDR6   权重、程序、输入、标志、动作
 GDDR6 ◀══ NoC ◀══ [axi_rd_stripe / axi_wr_stripe ─▶ axi_id_reorder]* ◀══ 链节点 × N_CHAIN（int8 或 uint8 PV）
                ◀══ [同上]* ◀══ 矢量节点 × N_VNODE                        每个节点一个或两个 NAP
                                       * 仅 STRIPE != 0 的构建：4 KB 通道条带化 + 每 burst 独立 ID 的响应重排
 链节点内部：  取指 ─▶ 命令执行器 ─▶ tile 加载器（双缓冲 feeder）─▶ MLP72 链 ─▶ 结果端口 ─▶ 结果写出器
 矢量节点内部：取指 ─▶ 控制 ─▶ N_LD 个操作数加载器（vu_rd_fanout）─▶ [vu_pdq] ─▶ lane（含 QUANT tail）
                      ─▶ 元素写出器 + 摘要写出器 ─▶ vu_wr_merge ─▶ 一个 AXI 写口
```

三个时钟域：`i_clk_array`（MLP72 链和行序列器）、`i_clk_fabric`（链节点的其余逻辑、链的 NAP、主机窗口、分页桥）、
`i_clk_vec`（矢量节点全部逻辑及其 NAP）。链节点内部用 `colpar_result_port.sv` 在阵列时钟和 fabric 时钟之间跨域；
节点之间只在 GDDR6 相遇，芯片级不再有跨时钟域路径。

## 当前状态

| 项目 | 状态 |
|---|---|
| 完整 pi0 chunk（SigLIP × 2 相机 + prefix 18 层 + expert 18 层 × 10 步 + action head）在 18 节点半阵列上 | **已在硅上验证**：标志和 50 × 32 动作与生成器逐位一致，**4.76 s / chunk**（阵列 325 / 矢量 125 / fabric 117.6 MHz，bitstream `s1s_325`）；46 帧策略评估相对 fp32 关节角最大 0.66°、平均 0.072° |
| 完整 chunk 在 3 节点完备阵列（1 int8 链 + 1 PV 链 + 1 矢量节点）上 | 已在硅上验证：逐位一致，26.28 s（500 / 285.7 / 222.2 MHz） |
| 写合并（`vu_wr_merge`）、双缓冲 GEMM feeder | 已在硅上验证（整 chunk 逐位一致） |
| 预反量化融合（PDQ） | 仿真逐位一致（1 prefix 层 + 1 expert 层、2 步去噪）；硅上整 chunk 4.63 s，动作偏差 ≤ 0.018（动作范围 1.73），原因待查 |
| GDDR6 通道条带化 + AXI ID 重排 | 仅仿真验证（半阵列节点组合、条带化 + 双缓冲 smoke chunk，逐位一致） |
| 控制路径重定时后的 RTL（本目录当前版本） | 仅仿真验证：矢量节点矩阵 70 / 70、链节点程序 / 节点仿真、半阵列 18 节点 chunk 门禁全部逐位一致；单节点布线 Fmax 阵列 870 / fabric 356 / 矢量 348 MHz；尚未构建 bitstream |
| 35 节点全阵列 bitstream | 布局占 56.92 % RLB tile，**布线未通过**（248 轮迭代后仍有 10,423 处溢出）；目前可交付的最大阵列是 18 节点半阵列 |
| 板卡时钟 | 半阵列 bitstream 按布线后 Fmax 降频运行（325 / 125 / 117.6 MHz）；设计目标 725 / 333.33 / 250 MHz |

---

## 目录结构

```
paper/rtl/
├── colpar_*.sv, mlp72_int8_colpar_chain.sv, node_sync.sv     链节点
├── vector_unit/                                              矢量节点及其 lane（含 vu_wr_merge、vu_pdq）
├── pi0_chip_top.sv, pi0_chip_ctrl.sv, nap_axi_ports.sv       芯片级阵列与主机窗口
├── pi0_host_gddr_bridge.sv                                   主机 → GDDR6 分页桥
├── axi_stripe.sv, axi_id_reorder.sv                          GDDR6 通道条带化、乱序响应重排
├── sim_models/                                               MLP72 / BRAM72K 行为模型及其假设说明
├── tb_*.sv, vector_unit/tb_*.sv                              testbench
├── run_*.sh                                                  Verilator 运行脚本
├── mlp72_int8_chain.sv, tb_mlp72_int8_chain.sv, run_chain_sim.sh   第一版链，保留供参考（见文末）
└── README.md
paper/sw/      逐位精确参考、golden 生成器、整 chunk 程序生成器、时间模型和主机运行时
paper/synth/   节点和阵列的 ACE 综合 / 布局 / 布线脚本；PLL 文件；每节点布局区域生成
paper/data/    矢量单元、真实宽度 GEMM、W8A8 数值方案的结果汇总
src/rtl/pi0_s0_top.sv     节点阵列 bitstream 的 FPGA 顶层（IO ring + pi0_chip_top）
src/constraints/, src/acxip/, src/ace/   约束、IP 配置、ACE 工程和 ACE 生成的 IO ring
host/, scripts/           板卡工具：chunk 运行器、向量回放、bitstream 构建 / 降频 / 打包 / 编程脚本
bitstream/s1s_325/        已在硅上验证的半阵列 bitstream（INFO.txt、SHA256SUMS、tc_ref_design_top.hex.gz，编程前先 gunzip）
```

## 文件说明

### 链节点

| 文件 | 用途 |
|---|---|
| `colpar_chain_node.sv` | 链节点顶层。例化取指、命令执行器、tile 加载器、行序列器、MLP72 链、结果端口、结果写出器和 `node_sync`，并把它们的 AXI 通道复用到一个读主口和一个写主口上。两个时钟：`i_clk_array`（链、序列器、结果寄存组）和 `i_clk_fabric`（其余逻辑、BRAM 写、NAP）。参数：`N_STAGE`、`MULT_MODE`、`WR_PIPE_EVERY`、`VALID_COPIES`、`WR_SLIM`，以及 burst 长度和在途请求数上限。输出 `o_halt`、`o_error` 和 `o_error_bits`（{sync, 取指, 溢出 0x04, 写出, 加载}，指出是哪个子模块报错）。 |
| `mlp72_int8_colpar_chain.sv` | 计算列：一个 feeder MLP72 + BRAM72K 存放激活字；`N_STAGE` 个乘法 MLP72，各自的 BRAM72K 存放权重列。feeder 每周期把一个 16 字节激活字沿硬连线 `fwd_multa` 级联向上送出，每一级把它与不同的输出列相乘（16 个 INT8 MAC）并在自己的累加器里对整行累加，因此对一行遍历一次就得到 `N_STAGE` 个列和。该模块还负责选择乘法模式（int8 × int8 或 uint8 × int8），为深链流水化权重写总线，并为每份结果有效信号保留一小段私有寄存器尾（`VALID_TAIL`）。 |
| `colpar_row_sequencer.sv` | 阵列时钟下的序列器：把一个 tile（`M` 行 × 每级 `P` 列）拆成 `M·P` 次行遍历执行，生成 feeder（行）和各级（列）的读地址以及遍历之间的间隔；遇到反压时拉长间隔。 |
| `colpar_tile_loader.sv` | 用流水化 AXI burst（每个 ≤ 16 拍，最多 8 个在途请求）从 GDDR6 读取 BRAM 镜像，写入 feeder 或各级 BRAM72K。支持连续镜像、矩阵的交错列（LOADX）和带步长的 feeder 行；`i_wbase` 指定目标 BRAM 的起始字（双缓冲 feeder 的两个 256 字半区）。 |
| `colpar_node_ctrl.sv` | 命令执行器。解码 128 位命令（LOAD、TILE、OUT、FLUSH、LOADX、GEMM column group、WAIT、POST、END），驱动加载器、序列器、写出器和同步主口。GEMM column group 的双缓冲模式（命令位 `[106]`）让下一组行在当前组计算时装入 feeder 的另一半；序列器字段先寄存、`o_seq_start` 晚 3 个 fabric 周期拉高，保证跨到阵列时钟时字段已稳定。**它的文件头就是链命令格式的规范。** |
| `colpar_prog_fetch.sv` | 在执行器等待命令时从 GDDR6 取节点程序（每个 256 位 beat 含两条 128 位命令）。两种节点共用（记录长度分别为 1 字和 6 字）。 |
| `colpar_result_port.sv` | 每次行遍历的列和从阵列时钟跨到 fabric 时钟：一个寄存组加 toggle 握手，结果计数用 Gray 码跨时钟域，反压回送给序列器。溢出检查用带符号的计数差（跨域路径不受约束时 fabric 侧可能短暂领先）。文件头给出了编译器必须遵守的时序规则。 |
| `colpar_result_writer.sv` | 把每次行遍历打包成每列 32 位的记录，以 AXI burst 写入 GDDR6。可选按列块跨步写（该节点的列作为一块落在更宽的行主序矩阵中）。`o_drained` 表示所有 beat 已发出（B 响应可以仍未返回），节点重启不必等 B；AW 决策比 42 位地址步进早一个周期。 |
| `node_sync.sv` | 在 GDDR6 的 32 位标志字上执行 WAIT / POST，轮询间隔指数退避到 `POLL_MAX`（512 个 fabric 周期）。节点之间因此可以互相排序，不需要主机参与。 |
| `colpar_nap_mux.sv` | 以 burst 为粒度、轮询方式让多个节点共享一个 AXI 端口。用于 `PER_NODE_MEM=0` 的多节点 testbench（芯片上每个节点有自己的 NAP，矢量节点的两个写出器改由 `vu_wr_merge` 合并）。 |

### 矢量节点（`vector_unit/`）

| 文件 | 用途 |
|---|---|
| `vu_node_ml.sv` | 矢量节点顶层：取指（6 字的 op 记录）；`N_LD` 个操作数加载器把操作数装入每个 lane 的 slot 存储器；`N_LANE` 个 lane 按行块同步运行；每个 lane 的输出缓冲按顺序排空到元素写出器和摘要写出器，两者经 `vu_wr_merge` 共用一个写口；WAIT / POST。参数 `N_LANE`、`N_LD`、`SLOT_BITS`（最大行长 2^SLOT_BITS）、`RD_FIFO_LOG2`（每个加载器可开 2^(RD_FIFO_LOG2−4) 个 burst）、`QTAIL`、`PDQ`。输出 `o_error_bits` 和 `o_dbg`（主机可读的 op 结束状态快照）。 |
| `vu_node.sv` | 单 lane 参考实现（`QTAIL = 0`），程序格式相同，输出逐字节一致。**它的文件头就是矢量 op 记录和操作数描述符的规范。** |
| `vu_lane.sv` | lane 本身：对一串行执行一个 op，每周期一个元素。流水为解包 → 加 → 查表 → 两次乘 → 加 → 舍入，行归约用微码步完成。`QTAIL = 1` 时带融合的 QUANT 尾：记录标志 `w0[118]` 置位的 bf16 输出 op 不输出 bf16 行，而是把行存入行 FIFO，行结束后用该行自己的 amax 求出 R / s_row 并量化成 int8，输出与紧随其后的 `QUANT` 记录完全相同的 int8 元素 beat 和每行一条 `{0, 0, s_row}` 摘要。 |
| `vu_pdq.sv` | 预反量化（PDQ，算子融合 2）：记录标志 `w0[119]` 置位时，把 int32 累加和操作数 x、d 先按 `x' = bf16((norm64(x)·c)·e)`、`d' = bf16((norm64(d)·c)·b)` 反量化（与两条 DEQUANT 记录逐位相同），再送入 lane 做 GEGLU；固定 15 级流水加 `2^FD_LOG2` 项 FIFO，关闭时是一根导线。MLP 的 DEQUANT、DEQUANT、GEGLU 三次 lane 遍历因此合并为一次。 |
| `vu_pkg.sv` | op 编码、数据格式、内部 `xf` 数值格式及辅助函数。 |
| `vu_add.sv`, `vu_mul.sv` | `xf` 加法器（6 级）和乘法器（4 级）。 |
| `vu_tbl.sv` | 四张 2048 项 BRAM 函数表（GELU 门、sigmoid、2^−x、rsqrt），线性插值。 |
| `vu_qtab.sv` | QUANT 每行常数（127 / amax 和行 scale）的 256 项表；QUANT 尾共用它的一个端口。 |
| `vu_round.sv` | 输出舍入到 bf16、fp32、int8 或 uint8（就近偶数舍入，饱和）。 |
| `vu_word_loader.sv` | `vu_node_ml` 的操作数加载器：读取张量的一段元素，每周期输出 4 个元素。 |
| `vu_slot_loader.sv` | 单 lane `vu_node` 的操作数加载器（每周期一个元素）。 |
| `vu_rd_fanout.sv` | 让多个操作数加载器共享一个 AXI 读端口，且不把它们的 burst 串行化；队列状态寄存一拍。 |
| `vu_beat_writer.sv` | beat 流 → AXI 写 burst，可选按行块跨步写。用于元素输出和摘要输出；`o_drained` 后即可重启。 |
| `vu_wr_merge.sv` | 两个写出器到一个 AXI 写口的合并：AW 轮询，W 按 AW 顺序、B 按 AW 顺序用队列（`Q_LOG2`）分派，两个写出器的 burst 同时在途。取代了原来的 `colpar_nap_mux`（它一次只放行一个 burst 并等到 B 才切换，使节点在 GDDR6 往返上停顿）。 |
| `OPS.md` | op 表：16 个 lane op 各自计算什么、用哪些操作数端口、格式、时延，以及 pi0 三个模型如何映射到这些 op。 |

### 芯片级

| 文件 | 用途 |
|---|---|
| `pi0_chip_top.sv` | 把整个阵列做成一个设计：`N_CHAIN` 个链节点（前 `N_DEEP` 个为 `DEEP_STAGE` 级深链，最后 `N_PV` 个为 uint8 PV 模式）和 `N_VNODE` 个矢量节点，每个节点接一个或两个 NAP（`NAPS_PER_NODE`），另加主机寄存器窗口和分页桥（`HOST_BRIDGE`）。`STRIPE != 0` 时每个节点与 NAP 之间插入 `axi_rd_stripe` / `axi_wr_stripe` 和 `axi_id_reorder`，条带化开关经两级同步器进入每个节点时钟域，分页桥的地址也按同一映射换算。启动和 halt 信号经同步器进出矢量时钟域。`CTRL_NAP_*` / `BRIDGE_NAP_*` 把两个主机 NAP 固定到 PCIe BAR 指向的 NoC 位置；`ID_WORD` 让主机识别 bitstream。 |
| `pi0_chip_ctrl.sv` | PCIe BAR0 后面的主机寄存器窗口，每个节点一个 32 字节 beat。主机的 32 位写按字节 strobe 合并。写：节点的程序基址和启动位；广播 beat 中的启动掩码、go 位和软复位位；分页桥的页基址和条带化开关。读：程序基址、halt、error、错误细节位、从启动到 halt 的 fabric 周期数、节点的调试快照；halt / error 向量、节点数和 ID 字。文件头给出寄存器映射。 |
| `pi0_host_gddr_bridge.sv` | 主机 → GDDR6 分页桥。PCIe BAR1（256 MB）映射到一个 fabric NAP；分页桥把每次访问转发到 GDDR6 的 `页基址 + 偏移`，经由第二个 NAP。主机每 256 MB 写一次页寄存器，因此不用 PCIe DMA 引擎也能访问全部 32 GB GDDR6。 |
| `nap_axi_ports.sv` | 两种 NAP 原语的直连线封装：`nap_axi_initiator`（`ACX_NAP_AXI_SLAVE`，fabric 作为 AXI 主设备，所有节点和分页桥的 GDDR6 侧使用；带 AXI ID 端口供重排模块使用）和 `nap_axi_target`（`ACX_NAP_AXI_MASTER`，NoC 作为主设备，主机窗口和分页桥的主机侧使用）。target 封装会回送每个请求的 AXI ID，这是 PCIe 桥所要求的。 |
| `axi_stripe.sv` | GDDR6 通道条带化。VP815 的 GDDR6 是 16 个通道，每个 2 GB，位于 NoC 地址 `k << 33`；一个区域若只落在一个通道里，读同一激活的 12 个链节点会在该通道上排队。逻辑地址 `L`（32 GB 平坦空间）按 `channel = L[S+3:S]`、`offset = {L[34:S+4], L[S-1:0]}`、`NoC = channel << 33 \| offset` 换算（`S = STRIPE_LOG2 = 12`，4 KB 条带）。`axi_rd_stripe` / `axi_wr_stripe` 把跨越条带边界的 burst 切成两段（隐藏第一段的 RLAST、重新生成 WLAST、合并两段的 B）。`i_en` 是运行时开关。 |
| `axi_id_reorder.sv` | NoC 对同一 AXI ID 来自不同通道的响应可能乱序返回。本模块给每个在途 burst 一个独立 ID（`TAG_LOG2 = 4`：最多 16 个 burst 在途），R 拍写入重排 RAM（16 × 16 × 259 位，两个 BRAM72K），按请求顺序交还；B 按 AW 顺序交还。 |
| `src/rtl/pi0_s0_top.sv` | 节点阵列 bitstream 的 FPGA 顶层（模块名 `tc_ref_design_top`，端口与 IO ring 一致）：Device Manager（GDDR6 训练、PCIe）、`pi0_chip_top`（参数含 `STRIPE`）、由 PLL_SW_3 和 PLL_SW_1 提供的三个节点时钟，以及进入各时钟域的上电复位和主机软复位。节点数量是参数，所以同一个顶层可以构建 2 节点的测试 bitstream、3 节点完备阵列、18 节点半阵列和 35 节点全阵列。 |

### 仿真模型（`sim_models/`）

| 文件 | 用途 |
|---|---|
| `acx_mlp72_behav.sv` | 链所用 `ACX_MLP72` 子集的周期精确行为模型（模块名和端口名与厂商原语一致，厂商核心是加密的）。 |
| `acx_bram72k_behav.sv` | `ACX_BRAM72K` 在 512 × 144 位模式下的行为模型，包括进入 MLP72 的专用读通路。 |
| `acx_float_behav.sv` | 未连接引脚原语 `ACX_FLOAT` 的替身。 |
| `ASSUMPTIONS.md` | 两个模型中厂商文档没有明确规定的每一项行为：依据、链对它的敏感程度、以及在硅片上确认它的测试方法。这些假设已在硅片上得到确认。 |

### testbench 与运行脚本

所有运行脚本都使用 Verilator 5；每个脚本打印 `RESULT PASS` / `RESULT FAIL`，构建产物放在 `build/` 下。

| testbench | 运行脚本 | 检查内容 |
|---|---|---|
| `tb_mlp72_int8_colpar_chain.sv` | `run_colpar_sim.sh` | 单独的 MLP72 链：每一行的列和对照 `paper/sw/mlp72_colpar_golden.py`，包括计算期间同时写入 |
| `tb_colpar_tile.sv` | `run_colpar_tile_sim.sh` | 序列器 + 链 + 结果端口：tile GEMM 对照 int64 矩阵乘，带随机反压；`MULT_MODE=13` 测 PV 模式 |
| `tb_colpar_node.sv` | `run_colpar_node_sim.sh` | 节点从 GDDR6 到 GDDR6 的数据通路 |
| `tb_colpar_node_prog.sv` | `run_colpar_prog_sim.sh` | 完整链节点运行命令程序（来自命令流或 GDDR6），含双缓冲 GEMM column group |
| `tb_colpar_nap_share.sv` | `run_colpar_nap_share_sim.sh` | 多个链节点共享一个 AXI 端口 |
| `vector_unit/tb_vu_lane.sv` | `run_vu_lane_sim.sh` | lane 对照 `paper/sw/vector_unit_ref.py`，覆盖所有 op，带随机间隙和反压；另有吞吐量模式 |
| `vector_unit/tb_vu_node.sv` | `run_vu_node_sim.sh` | 矢量节点（`vu_node` 或 `vu_node_ml`）端到端：测试集 `base`、`ml`、`attn`、`wide`；`base_q`、`ml_q`、`attn_q`、`wide_q`（同一测试集，每个 bf16 输出 op 都带 QUANT 尾）；`pdq`、`pdq_q`、`pdq_small`（预反量化）；`+debug` 打印调试快照 |
| `tb_pi0_linear.sv` | `run_pi0_linear_sim.sh` | 跨节点类型的一个线性层：QUANT → 链 GEMM → DEQUANT |
| `tb_pi0_wide_gemm.sv` | `run_pi0_wide_gemm_sim.sh` | 真实宽度的一个 GEMM（expert `o_proj`，16 个列 tile），拆分到 1–8 个链节点 |
| `tb_pi0_chunk.sv` | `run_pi0_chunk_sim.sh` | 由 `paper/sw/pi0_chunk_program.py` 生成的完整 chunk（或缩小的 chunk），运行在由矢量节点、int8 链（16 级和 32 级混合，`N_DEEP`）和 PV 链组成的缩小阵列上，无主机参与：每个节点只启动一次，检查每个中间区域、标志和动作的 beat。`PER_NODE_MEM=1` 时每个节点有自己的存储端口（与芯片一致），`STRIPE=12` 时在每个节点前插入条带化和重排 |
| `tb_pi0_chip_ctrl.sv` | `run_pi0_chip_ctrl_sim.sh` | 按主机实际访问方式测试主机寄存器窗口：带字节 strobe 的 32 位访问、启动掩码、go 和软复位脉冲、周期计数器、页寄存器 |
| `tb_pi0_host_gddr_bridge.sv` | （Verilator，无运行脚本） | 分页桥对照带字节 strobe 的 GDDR6 模型：64 位 PIO 写、32 位读、一个 burst、相距 9 GB 的两个页 |
| `tb_axi_stripe.sv`, `tb_axi_id_reorder.sv` | （Verilator，无运行脚本） | 随机 burst 下的条带化切分和乱序响应重排，各带负对照 |
| `tb_pi0_attn.sv` | `run_pi0_attn_sim.sh` | 由一个矢量节点、一个 int8 链和一个 PV 链运行整层：expert 层（缩小尺寸或 `FULL=1`）、`PART=lm` PaliGemma prefix 层、`PART=siglip` SigLIP 层、`PART=head` action head + Euler 更新、`PART=vision` patch embedding 和 projector；`BARRIER=1` 时每个节点只跑一个程序，由 GDDR6 标志排序 |
| `tb_axi_gddr6_model.sv` | （被上述 testbench 使用） | AXI4 后面的 GDDR6 模型：在途请求、随机握手、协议检查；`+nostall` 为理想存储器；`+rd_lat=N` / `+wr_lat=N` 加入 GDDR6 + NoC 的往返时延；`+stall_pct=N` 随机停顿 |
| — | `run_pi0_full_size_gates.sh quick\|full` | 上面所有检查项及其要求结果，包括负对照；汇总写入 `build/paper_gates/summary_<mode>.txt` |

负对照是编译期宏（`+define+COLPAR_LD_NEG_SEG_STEP`、`VU_NODE_ML_NEG_MERGE_ORDER` 等，通过 `DEFS=` 传入），每个宏故意破坏一处机制；使用它们的运行必须失败。

### 早期设计（不属于节点阵列）

| 文件 | 用途 |
|---|---|
| `mlp72_int8_chain.sv`, `tb_mlp72_int8_chain.sv`, `run_chain_sim.sh`, `paper/sw/mlp72_chain_golden.py`, `paper/synth/run_chain_synth.sh` | 第一版链：对同一个激活字把所有级的乘积相加（每次遍历只得到一列）。已被列并行链取代；保留它是因为它首先确立了行为模型所检查的 MLP72 / BRAM72K 配置，且 `mlp72_chain_golden.py` 仍被列并行链的 golden 引用。 |

## 本目录之外的相关文件

**`paper/sw/` — 参考模型、生成器与运行时**（Python 3 + numpy；使用真实权重的脚本还需要 torch 和 safetensors）：

| 文件 | 用途 |
|---|---|
| `vector_unit_ref.py` | 矢量 lane 的逐位精确参考；`--tables` 生成 lane 加载的五个 `.mem` ROM 文件（综合和仿真脚本自动调用）。 |
| `vector_unit_vectors.py` | lane 测试向量（随机向量和采集的激活）。 |
| `vector_unit_capture.py` | 在 CPU 上跑一帧真实观测的 fp32 模型，用 hook 采集每个矢量 op 位置的激活（`build/paper_vector_unit/acts_*.npz`），是各层 golden 的输入。 |
| `mlp72_colpar_golden.py`, `colpar_tile_golden.py` | 链的测试向量；`colpar_tile_golden.py` 同时包含链命令的 Python 编码函数（`op_load`、`op_colgroup` 等）。 |
| `vu_node_golden.py` | 矢量节点测试集（含 `*_q` 和 `pdq*`）；包含 op 记录和操作数描述符的编码函数（`w0[118]`、`w0[119]` 标志）。 |
| `pi0_linear_golden.py`, `pi0_wide_gemm_golden.py` | 跨节点线性层和真实宽度 GEMM。 |
| `pi0_layer_lower.py` | 把一个真实层降级（lower）到节点 op 上，并在参考模型上运行（芯片数值方案）；checkpoint、采集激活和校准统计量的路径在文件开头设置。 |
| `pi0_attn_golden.py` | expert 层和 prefix 层的 GDDR6 镜像、节点程序和期望 beat（`Tiler`：列 tile、行切片、K 拆分、双缓冲组）。 |
| `pi0_siglip_golden.py`, `pi0_action_head_golden.py`, `pi0_vision_ends_golden.py` | 同上，分别用于 SigLIP 层、action head + Euler 更新、视觉通路两端。 |
| `program_traffic.py` | 从一组节点程序解码出其读写的 GDDR6 字节数（与 testbench 计数器一致）。 |
| `pi0_chunk_layout.py` | 整 chunk 的 GDDR6 映射：PROG、STATIC、IO、FLAGS、SCRATCH 五个区，逐次分配都做重叠检查；`PI0_CHUNK_MAP=channels`（默认，STATIC / SCRATCH 在 16 个通道上轮流分配）、`compact`（512 MB 以内）、`striped`（条带化 bitstream 用的 32 GB 逻辑空间）。 |
| `pi0_chunk_layers.py`, `pi0_chunk_tiler.py` | chunk 的层构建器（SigLIP、Gemma prefix / expert、视觉两端、action head），由各层 golden 转写而来；针对混合链深的 GEMM 分块，`PI0_CHUNK_GEMM_DB=1` 时生成双缓冲 column group（`M = 256 / W`）。 |
| `pi0_chunk_program.py` | **整 chunk 程序生成器**：SigLIP × 27 × 2 个相机 → projector → prefix × 18 → 10 ×（head 输入侧 → expert × 18 → head 输出侧 + Euler），把每个 stage 分给阵列的各节点，在有依赖的 part 之间插入 WAIT / POST，每个节点一个程序。`--fuse-quant` 把只有一个读者的 `X; QUANT(X)` 合并成带 QUANT 尾的一条记录，`--fuse-pdq` 把 DEQUANT、DEQUANT、GEGLU 合并成一条 PDQ 记录；矢量 stage 按 4 / 8 / 16 行的最细粒度切分到各节点。可写出 testbench 镜像（`--hex`）或板卡用的二进制镜像（`--bin`）；`PI0_CHUNK_STAGE_IO=1` 另写 `stage_io.json`。 |
| `pi0_chunk_time.py` | 根据各层实测仿真，计算生成的 chunk 在其真实节点分配下的运行时间（链 stage 取 `min(f_fabric, f_array / 3)`）；`--scale name=factor` 做假设分析。 |
| `chunk_latency_calibrated.py` | 早期的 chunk 时间模型，由实测的真实尺寸层仿真、节点面积和时钟构建；已被 `pi0_chunk_time.py` 取代，保留供对照。 |
| `pi0_timeline_compare.py` | 把板上运行（`pi0_chunk_run --timeline` 记录每个标志的置位时刻）与模型的调度按 chunk 段（SigLIP / 视觉两端 / prefix / 每步 expert / head）逐段对比。 |
| `pi0_chunk_wrong_regions.py`, `pi0_chunk_trace_wrong.py` | 板上运行出错时：把错误 beat 归到区域；沿 `stage_io.json` 的数据流找出只读到正确数据却写出错误 beat 的根源 stage。 |
| `pi0_chip_runtime.py`, `pi0_chip_host_inputs.py` | 阵列的主机运行时：把 LeRobot 观测的每 chunk 输入（patch embedding + 位置表、语言 token embedding、状态、噪声）编码到对应的 GDDR6 区域，通过常驻的 `host/pi0_chunk_run serve` 子进程运行一个 chunk，返回 50 × 32 的动作。 |
| `prefix_w8a8_eval.py` | W8A8 数值方案的研究脚本；它的 `load_policy()`（内存映射加载策略）、`get_capture()` 被主机运行时和评估脚本调用，校准阶段产生 layer golden 使用的激活统计量。 |
| `pi0_deploy_model.py`, `pi0_ae_model.py` | `prefix_w8a8_eval.py` 在模块级导入的两个早期数值模型（`pi0_deploy_model.py` 又导入 `scripts/export_pi0_int8_quant_bank_pages.py`）；只作为依赖保留。 |
| `regen_calib.py` | 重新生成校准统计量（`calib.pt`、`calib_exp.pt`），写入 `build/paper_vector_unit/calib`。 |
| `s0_vectors.py`, `vu_dbg_decode.py` | 把 golden 的向量集转换成板卡回放格式；解码矢量节点的调试快照。 |

**`paper/synth/` — ACE 脚本**（ACE 10.5.2；通过 `ACE_ROOT` 找到安装目录；结果在 `build/paper_*_synth/<name>/rev_1/pnr/reports/`）：

| 文件 | 用途 |
|---|---|
| `run_colpar_synth.sh` | 在给定阵列时钟周期下综合、布局、布线单独的链或完整链节点（`WITH_SEQ=3`），fabric 周期由 `FCLK_NS` 指定。 |
| `run_vu_lane_synth.sh`, `run_vu_node_synth.sh` | 同上，用于单个 lane 和矢量节点（`QTAIL=0\|1`）。 |
| `parse_vu_lane_synth.py`, `sweep_vu_packing.sh` | 解析一次 lane / 节点综合的资源与时序为 `result.json`；矢量节点 RLB 装填率的综合选项扫描。 |
| `run_chip_fit_synth.sh` | 为给定阵列综合并布局 `pi0_chip_top`（`ROUTE=1` 时同时布线）。 |
| `set_node_pll.py`, `run_node_pll_dryrun.sh` | 节点时钟的 PLL 配置（改写 `.acxip`），以及通过 ACE IO ring 生成做的试运行。 |
| `pll/README.md`, `pll/pll_array_725.acxip`, `pll/pll_nap_vec333_fab250.acxip`, `pll/port_list_3pll.svh` | ACE 重新求解后的两个节点 PLL 文件（PLL_SW_3 阵列 725 MHz；PLL_SW_1 矢量 333.33 + fabric 250 MHz）和 IO ring 期望的顶层端口列表。 |
| `s0/pi0_s0_synth.prj` | 节点阵列 bitstream 的 Synplify 工程（顶层 `src/rtl/pi0_s0_top.sv`，源文件清单，参数取自 `PI0_S0_*` 环境变量）。 |
| `dump_placement.tcl`, `gen_node_regions.py` | 两遍布局：读取第一遍布局中硬核的位置，为每个节点生成一个软布局区域（`PI0_S0_REGIONS_PDC`）。 |

**`paper/data/` — 结果汇总**：

| 文件 | 内容 |
|---|---|
| `vector_unit/SUMMARY.md` | 矢量单元的算术、精度、资源和投影汇总 |
| `vector_unit/sim_results.txt`, `errors.json`, `inventory.json` | lane 仿真结果；各 op 相对 fp32 的误差；模型中矢量 op 的清单 |
| `vector_unit/e2e_chunk.json`, `projection.json` | 端到端 chunk 的矢量算子统计；矢量节点吞吐量投影 |
| `vector_unit/synth_one_lane_250mhz.json`, `synth_one_lane_250mhz_fabric_mult.json` | 单 lane 在 250 MHz 目标下的综合结果（两种乘法器实现） |
| `wide_gemm/SUMMARY.md` | 真实宽度 `o_proj` 在一个链节点上的结果 |
| `prefix_w8a8/SUMMARY.md` | 全模型 W8A8 数值方案的评估结果 |
| `pi0_layer_lower/layers_alpha0.json`, `layers_alpha0.5.json` | 逐层 lowering 的数值结果（SmoothQuant alpha 0 和 0.5） |

**`src/` — FPGA 顶层、约束与 ACE 工程**：

| 文件 | 用途 |
|---|---|
| `src/rtl/pi0_s0_top.sv` | 见上文"芯片级"。 |
| `src/rtl/reset_processor_v2.sv`, `src/include/7t_interfaces.svh` | 厂商提供的复位处理器（顶层例化三次）和接口定义。 |
| `src/constraints/pi0_s0_synplify.sdc` | 综合时钟：`i_clk_array` 1.379 ns、`i_clk_vec` 3.0 ns、`i_clk_fabric` 4.0 ns、`i_mcu_clk` 10 ns、`i_reg_clk` 15.5 ns，五者异步。 |
| `src/constraints/pi0_s0_ace.sdc` | ACE 约束：异步时钟组、时钟不确定度；可选的复位释放伪路径（`PI0_S0_RESET_FP=1`）和 MLP72 → 捕获寄存器的 2 周期多周期路径（`PI0_S0_CAP_MCP=1`，带保护，日志打印匹配到的捕获寄存器数）。 |
| `src/constraints/pi0_s0_ace.pdc` | 把主机窗口的 NAP 固定到 `NOC[3][4]`（BAR0）、分页桥的 target NAP 固定到 `NOC[3][5]`（BAR1）。 |
| `src/constraints/ace_constraints.sdc`, `ace_placements.pdc`, `synplify_constraints.sdc`, `synplify_constraints.fdc` | 早期设计的约束，仅因 ACE 工程文件登记了它们而保留；`run_pi0_s0_ace_flow.tcl` 在每个 impl 中停用它们。 |
| `src/acxip/*.acxip` | IO ring 的 IP 配置：`pci_express.acxip`（PCIe 端点，BAR0 → `NOC[3][4]`，BAR1 256 MB → `NOC[3][5]`）、`gddr6_0.acxip` … `gddr6_7.acxip`（8 个 GDDR6 控制器）、`noc.acxip`、`device_manager.acxip`、`pll.acxip`（PLL_SW_0）、`pll_nap.acxip`（PLL_SW_1）、`pll_array.acxip`（PLL_SW_3）、`pll_pcie.acxip`、`vp_clkio_ne.acxip` / `vp_clkio_se.acxip` / `vp_clkio_sw.acxip`（时钟 IO）、`vp_pll_se_2.acxip` / `vp_pll_sw_2.acxip`。 |
| `src/ace/tc_ref_design_top.acxprj` | ACE 工程文件（`run_pi0_s0_ace_flow.tcl` 用 `restore_project` 打开）。其中登记的早期 `impl_*` 条目对应的文件不在仓库中，`restore_project -no_db` 会忽略它们。 |
| `src/ace/ioring_design/` | ACE 由 `src/acxip` 生成的 IO ring：时钟约束、引脚约束、各硬核的配置 bitstream 片段。构建流程直接使用它们，不重新生成；`run_pi0_s0_reclock_bitstream.sh` 会重新生成。 |

**`host/` 和 `scripts/` — 板卡工具**（Achronix SDK，默认在 `/opt/achronix/sdk`，可用 `ACHRONIX_SDK_ROOT` 指定；`host/CMakeLists.txt` 只定义四个目标 `pi0_chunk_run`、`pi0_s0_replay`、`pi0_atu_tool`、`pi0_dbi_probe`，用 `cmake -S host -B build/host && cmake --build build/host` 构建）：

| 文件 | 用途 |
|---|---|
| `host/pi0_chunk_run.cpp` | chunk 运行器：`selftest`（分页桥自检）；`run <dir> --map <map>`（把生成的镜像装入 GDDR6，让每个节点指向自己的程序，同时启动，轮询 halt 向量，检查标志和动作；`--repeat N`、`--inputs`、`--dump-actions`、`--timeline`、`--wrong-list`、`--load-only`、`--no-load`）；`serve <dir>`（常驻，按 stdin 的 `infer <inputs> <actions>` 行每次运行一个 chunk）；`probe`、`fill`、`peek`、`verify`（GDDR6 诊断）。从 chunk 的 `info.json` 设置条带化开关；首次启动前做软复位并重新写入映射。环境变量 `PI0_HOST_BRIDGE=1`（BAR1 为分页桥）、`PI0_FPGA_DBI_ROUTE=comp\|full`。 |
| `host/pi0_s0_replay.cpp` | 在板上回放单个节点的测试向量集（`identify`、`reset`、`run --node N`、`status`），比对每个期望 beat。 |
| `host/pi0_atu_tool.cpp` | 列出并移动 PCIe 端点的 BAR → NoC 地址转换区域。 |
| `host/pi0_dbi_probe.cpp` | 只读探测 PCIe 控制器的 DBI 访问路径（SDK 选了哪条路径、DMA 寄存器在哪）。 |
| `host/pi0_fpga_backend.h`, `pi0_fpga_sdk_backend.h`, `pi0_fpga_sdk_backend.cpp`, `pi0_fpga_mock_backend.h`, `pi0_fpga_mock_backend.cpp` | `pi0_s0_replay` 使用的寄存器 / 批量访问后端（SDK 实现和无板卡的软件模型）；`pi0_fpga_mock_backend.cpp` 引用 `scripts/llama3_8b_runtime_host_regs.h`，该头文件只作为依赖保留。 |
| `scripts/launch_pi0_s0_build.sh`, `run_pi0_s0_ace_flow.tcl` | 构建节点阵列 bitstream：`<tag> <N_CHAIN> <N_PV> <N_VNODE> <N_DEEP>`，在内存受限的用户 scope 中后台运行 ACE 10.5.2；impl 命名 `impl_s0_<tag>_c<N_CHAIN>p<N_PV>v<N_VNODE>d<N_DEEP>o<MAX_OUT>[s<STRIPE>]_<commit>_<UTC 时间>`。 |
| `scripts/run_pi0_s0_reclock_bitstream.sh` | 不重新布线，用不同的阵列 / 矢量 / fabric 时钟重新生成已布线 impl 的 bitstream（时钟只在 IO ring 中）：`<impl> <array MHz> <ref> <fb> [vec ODN] [fabric ODN]`。 |
| `scripts/pi0_s0_pick_clocks.py` | 从已布线 impl 的时序报告读三个时钟的 Fmax，给出合法的 PLL 设置并打印 reclock 命令（`--margin 0.97`；阵列 Fmax 是下界，可用 `--array` 覆盖）。 |
| `scripts/pi0_bundle_s0.sh`, `pi0_board_program_s0.sh` | 把 bitstream 连同时序报告和校验和打包到 `~/pi0_board_bundle/bitstream/<name>/`；通过 JTAG 给板卡编程并恢复 PCIe 端点（重训练到 16 GT/s、remove + rescan、加载驱动、`pi0_s0_replay identify`）。 |
| `scripts/pi0_board_recover.sh`, `pi0_board_after_warm_reboot.sh`, `pi0_board_prepare_warm_reboot.sh`, `pi0_board_bringup_from_fics.sh` | 板卡恢复流程：不重启的恢复、热重启前后的步骤、冷启动后的完整 bring-up。 |
| `scripts/pi0_s0_silicon_matrix.sh`, `pi0_s0t_silicon_session.sh`, `pi0_array_silicon_session.sh` | 板上验收运行：测试 bitstream 上的每一组向量；3 节点完备阵列的自检、smoke / tiny / 整 chunk；半阵列 / 全阵列上的分页桥自检和生成的 chunk。 |
| `scripts/export_pi0_int8_quant_bank_pages.py` | 早期量化元数据导出器，仅作为 `paper/sw/pi0_deploy_model.py` 的依赖保留。 |

## 命令与记录格式

**链命令**（128 位，操作码在 `[127:124]`；位域见 `colpar_node_ctrl.sv` 文件头）：

| 操作码 | 命令 | 含义 |
|---:|---|---|
| 0 | END | 停机；再次启动时从程序基址重新开始 |
| 1 | LOAD | 从 GDDR6 填充 feeder / 各级 BRAM 镜像 |
| 2 | TILE | 用已加载的镜像运行 `M` 行 × 每级 `P` 列 |
| 3 | OUT | 设置结果地址（可选：作为更宽矩阵中的一个列块） |
| 4 | FLUSH | 把写出器中的全部数据写出 |
| 5 | LOADX | 从 GDDR6 中矩阵的交错列填充各级镜像 |
| 6 | GEMM column group | `T` 行：节点自己每次重新加载 `M` 行 feeder，对所有行运行 tile。`[105:96]` `S` 为 feeder 行步长（字；0 = 行连续），`[106]` 为双缓冲（feeder 的两个 256 字半区交替，下一组在当前组计算时加载；要求 `M × W ≤ 256`） |
| 7 | WAIT | 轮询 GDDR6 标志 `[41:0]`，直到 ≥ `[95:64]` 的值 |
| 8 | POST | 所有结果写完后把 `[95:64]` 写入 GDDR6 标志 `[41:0]` |

**矢量记录**（6 × 128 位；布局见 `vu_node.sv` 文件头）：`w0[127:124]` = `0xA` op 记录、`0xB` WAIT、`0xC` POST（值在 `w0[39:8]`，标志地址在 `w1`）、`0` END。
op 记录：`w0[3:0]` op（0 ADD、1 RMS_STAT、2 RMS_APPLY、3 LN_STAT、4 LN_APPLY、5 ROPE_A、6 ROPE_B、7 GELU、8 GEGLU、9 SILU、
10 SMAX_SUM、11 SMAX_OUT、12 SMAX_Q8、13 QUANT、14 DEQUANT、15 EULER）；标志 `w0[4]` b 为 fp32、`[5]` 加偏置、`[6]` 输出 fp32、
`[7]` x 旋转半行（RoPE）、**`[118]` QUANT 尾**、**`[119]` PDQ**；`w0[39:8]` 常数 k、`[59:40]` 行数、`[75:60]` 行长、`[117:76]` 元素输出基址；
`w1[41:0]` 摘要输出基址；`w1` 到 `w4` 为七个 48 位操作数描述符（`x b c d e rs mask`），各含基址、形状（逐元素、每行一个、每列一个、常数）和格式
（bf16、fp32、int32、64 位摘要记录）。op 列表见 `vector_unit/OPS.md`。

**GDDR6 中的数据。** 所有数据以 256 位 beat 传输，每个区域从 beat 边界开始。INT8 张量是原始字节，每个 BRAM 字 16 字节。
链每次行遍历写一条记录（`N_STAGE` 个 int32 和，补齐到整 beat），这些记录合起来构成行主序的 int32 矩阵。矢量节点的 DEQUANT（或 PDQ）直接读取
该区域，而它的 QUANT（或 QUANT 尾）写出的正是链读取的 int8 布局。矢量节点的函数表是 bitstream 中的 ROM，不在 GDDR6 里。

**一个 chunk 的 GDDR6 映射**（`paper/sw/pi0_chunk_layout.py`）：PROG（节点程序）、STATIC（权重镜像、scale、表；只写一次）、
IO（每 chunk 的输入和动作）、FLAGS（每个 stage part 一个 beat）、SCRATCH（stage 之间的区域）。VP815 的 GDDR6 是 16 个通道，
每个 2 GB，位于 NoC 地址 `k << 33`，通道内超过 2 GB 的偏移会回绕（不是 32 GB 平坦空间）。三种映射：`channels`（默认，STATIC 和 SCRATCH
轮流分布在各通道上）、`compact`（全部在 512 MB 以内）、`striped`（`STRIPE` bitstream 打开条带化开关后使用的 32 GB 逻辑空间）。
镜像和 bitstream 的映射必须一致：`pi0_chunk_run` 按 chunk 的 `info.json` 设置开关。

**主机寄存器窗口**（`pi0_chip_ctrl.sv`，PCIe BAR0）：beat `n` 对应节点 `n`（写：`[41:0]` 程序基址，`[64]` 启动；
读：halt `[64]`、error `[65]`、错误细节 `[79:72]`、周期数 `[127:96]`、调试快照 `[255:128]`）；beat `N_NODE`：
写时为启动掩码 `[96 +: N_NODE]`、go `[0]`、软复位 `[1]`，读时为 halt 向量 `[N_NODE-1:0]`、error 向量 `[128 +: N_NODE]`、
节点数 `[231:224]` 和 ID 字 `[255:232]`；beat `N_NODE + 1`：`[41:0]` 分页桥的页基址，`[64]` 条带化开关（装入镜像前设置，节点运行时不得改动）。
软复位会复位整个 fabric 域，包括窗口本身（页寄存器归零、开关关闭），所以主机在软复位后要重新写入映射。

**主机批量数据通路**（`pi0_host_gddr_bridge.sv`，PCIe BAR1）：在页基址处看到 GDDR6 的一个 256 MB 窗口。这些 bitstream 上 PCIe DMA 引擎不可用，
批量数据以程序控制 I/O 方式传输。

## 主要参数

| 参数 | 所在模块 | 取值 |
|---|---|---|
| `N_STAGE` | 链节点 | 16 或 32 个乘法级 |
| `MULT_MODE` | 链节点 | `5'h00` int8 × int8，`5'h13` uint8 × int8（PV） |
| `WR_PIPE_EVERY`, `VALID_COPIES` | 链节点 | 16 级时为 0 / 1；32 级时为 4 / 4（布线通过和满足时序所必需） |
| `WR_SLIM`, `VALID_TAIL` | 链节点 | 1（记录 FIFO 直接从结果寄存组写入）；3（每份结果有效信号的私有寄存器尾） |
| `POLL_CYCLES`, `POLL_MAX` | `node_sync` | 64；512（最长轮询间隔） |
| `N_LANE`, `N_LD` | 矢量节点 | 2 个 lane、2 个加载器（1 / 1、4 / 2 和 4 / 4 也已验证） |
| `SLOT_BITS` | 矢量节点 | 12（行长最多 4,096 个元素） |
| `RD_FIFO_LOG2` | 矢量节点 | 7（每个加载器 8 个 burst 在途；5 时节点受 GDDR6 往返时延限制） |
| `QTAIL`, `ROW_AW` | `vu_node_ml`, `vu_lane` | 1（`vu_node` 为 0）；12 |
| `PDQ`, `FD_LOG2` | `vu_node_ml`, `vu_pdq` | 1；5 |
| `Q_LOG2` | `vu_wr_merge` | 4 |
| `STRIPE`, `STRIPE_LOG2`, `TAG_LOG2` | 芯片顶层, `axi_stripe`, `axi_id_reorder` | 0（关闭）或 12（4 KB 条带）；12；4（16 个 burst 在途） |
| `N_CHAIN`, `N_DEEP`, `DEEP_STAGE`, `N_PV`, `N_VNODE`, `NAPS_PER_NODE` | 芯片顶层 | 全阵列（35 节点）：27、12、32、3、8、2；半阵列（18 节点）：14、6、32、2、4、2；3 节点完备阵列：2、0、–、1、1、2；测试 bitstream：1、0、–、0、1、2 |
| `HOST_BRIDGE`, `CTRL_NAP_*`, `BRIDGE_NAP_*`, `ID_WORD`, `MAX_OUT` | 芯片顶层 | 1；窗口位于 `NOC[3][4]`（BAR0），分页桥位于 `NOC[3][5]`（BAR1）；构建默认 `24'h533001`（`PI0_S0_ID_WORD`），主机从 beat `N_NODE` 读回；每个节点最多 8 个在途 AXI 请求 |

芯片上的节点编号：先是链节点（前 `N_DEEP` 个 32 级，然后 16 级，最后 `N_PV` 个 uint8 PV），再是矢量节点。半阵列：0–5 为 32 级 int8、
6–11 为 16 级 int8、12–13 为 PV、14–17 为矢量节点（`pi0_chunk_run --map vector:14,int8:0,uint8:12`）；全阵列：0–11、12–23、24–26、27–34
（`--map vector:27,int8:0,uint8:24`）。

## 时钟

设计目标（三 PLL 时钟方案，`src/acxip/pll_array.acxip` 和 `pll_nap.acxip`）：`i_clk_array` 725 MHz 来自 PLL_SW_3（参考分频 2、反馈 29、
VCO 5800、输出分频 8）；`i_clk_vec` 333.33 MHz 和 `i_clk_fabric` 250 MHz 来自 PLL_SW_1（VCO 8000，输出分频 24 和 32）；PLL_SW_0（NoC 的
200 MHz 参考、`i_mcu_clk` 100 MHz、`i_reg_clk` 64.5 MHz）保持不变。阵列时钟与 fabric 时钟之间是异步跨时钟域，不要求整数时钟比。

实际运行的频率由布线后的时序决定：`scripts/pi0_s0_pick_clocks.py <impl>` 读出三个时钟的 Fmax 并给出合法的 PLL 设置（阵列
`400 × fb / ref / 8`，不超过 725；矢量和 fabric 为 `8000 / ODN`，ODN 为 2 或 4 的倍数），`scripts/run_pi0_s0_reclock_bitstream.sh` 不重新布线
就按这些设置重新生成 bitstream。半阵列 bitstream `s1s_325` 的布线 Fmax 为阵列 355.6 / fabric 123.4 / 矢量 135.8 MHz，运行在 325 / 117.6 / 125 MHz（同一顺序）。
单节点单独布线的 Fmax 为阵列 870 / fabric 356 / 矢量 348 MHz，所以频率主要损失在多节点阵列的布线上。

两个硬限制：MLP72 和 BRAM72K 的上限是 750 MHz；NAP 与 NoC 之间的带宽在 NAP 时钟 340 MHz（NoC 1.7 GHz）时达到 100 %，再提高 fabric
时钟不会增加 NoC 带宽（485.7 MHz 时只有 66.7 %），所以 fabric 时钟的目标是 340 MHz，矢量时钟则按纯计算线性受益。

## 运行仿真

依赖：Verilator 5（`VERILATOR`，默认 `~/tools/verilator5/bin/verilator`）、带 numpy 的 Python 3（`PYTHON`，默认 `~/lerobot/.venv/bin/python`，
不存在时用 `python3`）；真实权重测试还需要 torch 和 safetensors，以及三项数据输入，其位置在 `paper/sw/pi0_layer_lower.py` 开头设置：

- `CKPT`：pi0 checkpoint（LeRobot pi0 模型的 `model.safetensors`，默认 `~/pi0_ckpt/ur7e-demo-2-pi0-fsdp/010000/pretrained_model/`）；
- `CAPTURE`：一帧的采集激活（`build/paper_vector_unit/acts_*.npz`，由 `paper/sw/vector_unit_capture.py` 生成）；
- `CALIB`：每通道激活统计量 `calib.pt` 和 `calib_exp.pt`（默认在 `build/paper_vector_unit/calib`，或由 `PI0_CALIB_DIR` 指定）；
  用 `paper/sw/regen_calib.py` 生成一次即可（约 5 分钟）。

lane 的五个 `.mem` 函数表由各运行脚本自动调用 `paper/sw/vector_unit_ref.py --tables` 生成。

```bash
# 全部检查项及其要求结果（quick 约 25 分钟；full 会加上 525 token 的 prefix 层等，约 2.5 小时）
paper/rtl/run_pi0_full_size_gates.sh quick

# 单独的测试集
paper/rtl/run_colpar_prog_sim.sh all                                       # 链节点程序（含双缓冲 column group）
SUITE=ml_q N_LANE=2 N_LD=2 SLOT_BITS=12 paper/rtl/run_vu_node_sim.sh 1 8   # 矢量节点，QUANT 尾；参数：种子、在途请求数
SUITE=pdq  N_LANE=2 N_LD=2 SLOT_BITS=12 paper/rtl/run_vu_node_sim.sh 2 1   # 预反量化
paper/rtl/run_pi0_linear_sim.sh all                                        # 跨节点线性层

# 一个真实尺寸层（expert 第 0 层、去噪第 0 步），32 级链，2-lane 矢量节点，理想存储器
N_STAGE=32 FULL=1 N_LANE=2 N_LD=2 SIM_ARGS=+nostall paper/rtl/run_pi0_attn_sim.sh 0 0
# 525 个 token 的 PaliGemma prefix 层（约 30 分钟，约 5 GB 内存）
N_STAGE=32 PART=lm N_LANE=2 N_LD=2 SIM_ARGS=+nostall TIMEOUT_PS=2000000000000000 paper/rtl/run_pi0_attn_sim.sh 0

# 完整 chunk，无主机参与，在缩小阵列上运行（默认 2 个矢量节点 + 2 个 int8 链 + 1 个 PV 链，共享一个 NAP）
SIM_ARGS=+nostall paper/rtl/run_pi0_chunk_sim.sh smoke --siglip-layers 1 --prefix-layers 0 --expert-layers 0 --steps 1 --fuse-quant
SIM_ARGS=+nostall paper/rtl/run_pi0_chunk_sim.sh tiny  --siglip-layers 1 --prefix-layers 1 --expert-layers 1 --steps 1 --fuse-quant --fuse-pdq

# 半阵列的 18 节点组合（4 矢量 + 6 × 32 级 + 6 × 16 级 + 2 PV），双缓冲 GEMM，先生成再以三种存储器条件运行
export PI0_CHUNK_COMPACT=1 PI0_CHUNK_GEMM_DB=1 N_VN=4 N_CH=12 N_DEEP=6 N_PV=2 TIMEOUT_PS=40000000000000000
python3 paper/sw/pi0_chunk_program.py --out build/paper_pi0_chunk/half_db_smoke --n-vec 4 --n-chain 12 --n-pv 2 --n-stage 16 \
    --n-deep 6 --slot-bits 12 --hex --siglip-layers 1 --prefix-layers 0 --expert-layers 0 --steps 1 --sync deps --interleave --fuse-quant
GEN=0 PER_NODE_MEM=1 SIM_ARGS="+nostall +rd_lat=64 +wr_lat=64" paper/rtl/run_pi0_chunk_sim.sh half_db_smoke   # 每节点独立端口，64 周期往返
GEN=0 SIM_ARGS="+stall_pct=50" paper/rtl/run_pi0_chunk_sim.sh half_db_smoke                                    # 共享 NAP，50 % 随机停顿
GEN=0 PER_NODE_MEM=1 STRIPE=12 SIM_ARGS=+nostall paper/rtl/run_pi0_chunk_sim.sh half_db_smoke                   # 条带化 + ID 重排

paper/rtl/run_pi0_chip_ctrl_sim.sh                                         # 主机寄存器窗口
```

`run_pi0_chunk_sim.sh` 的环境变量：`N_VN` / `N_CH` / `N_PV` / `N_DEEP`（须与生成器的 `--n-vec` / `--n-chain` / `--n-pv` / `--n-deep` 一致）、
`PER_NODE_MEM`、`STRIPE`、`N_LANE`、`N_LD`、`SLOT_BITS`、`N_STAGE`、`MAX_OUT`、`VN_RD_FIFO_LOG2`、`DEFS`、`SIM_ARGS`、`TIMEOUT_PS`、`GEN=0`（复用已生成的目录）。
生成 board 镜像时把 `--hex` 换成 `--bin`，并按 bitstream 决定是否设置 `PI0_CHUNK_GEMM_DB=1`（双缓冲 bitstream）和 `PI0_CHUNK_MAP`。

## 运行综合与构建 bitstream

```bash
export ACE_ROOT=<ACE 10.5.2>/Achronix-linux                                    # 脚本默认 /home/sngong/ACE_10.5.2/Achronix-linux
paper/synth/run_colpar_synth.sh node16 16 1.333 0 0 3                          # 16 级链节点，750 / 250 MHz
WR_PIPE=4 VALID_COPIES=4 paper/synth/run_colpar_synth.sh node32 32 1.333 0 0 3  # 32 级链节点；FCLK_NS 指定 fabric 周期
paper/synth/run_vu_node_synth.sh vn2 2.0 12 6 2 2                                # 2-lane / 2 加载器矢量节点（QUANT 尾 + PDQ），2.0 ns 目标
N_DEEP=12 SLOT_BITS=12 N_LD=2 NAPS_PER_NODE=2 paper/synth/run_chip_fit_synth.sh m27v8x2 27 3 8 2   # 阵列布局

# 带 IO ring（PCIe、GDDR6、PLL）的 bitstream：<tag> <N_CHAIN> <N_PV> <N_VNODE> <N_DEEP>；半阵列的构建环境如下
PI0_S0_STRIPE=12 PI0_S0_RESET_FP=1 PI0_ACE_SEED=7 PI0_S0_ID_WORD="24'h53310a" \
    PI0_S0_REGIONS_PDC=./../../build/s0/regions_half.pdc scripts/launch_pi0_s0_build.sh s1s 14 2 4 6
scripts/launch_pi0_s0_build.sh s0t 2 1 1 0        # 3 节点完备阵列
scripts/launch_pi0_s0_build.sh full 27 3 8 12     # 全阵列，35 个节点（布局通过，布线未通过）

# 布线后：选时钟、降频重出 bitstream、打包、编程
scripts/pi0_s0_pick_clocks.py src/ace/<impl>
scripts/run_pi0_s0_reclock_bitstream.sh <impl> 325 2 13 64 68   # 阵列 325 MHz（ref 2, fb 13）、矢量 ODN 64 = 125、fabric ODN 68 = 117.6
scripts/pi0_bundle_s0.sh <impl> s1s_325
scripts/pi0_board_program_s0.sh bitstream/s1s_325/tc_ref_design_top.hex
```

说明：

- 链深超过 16 级的阵列必须用 ACE 10.5.2（10.3.1 在布局阶段会崩溃）。构建日志在 `build/s0/<impl>.log`，`PI0_S0_ACE_FLOW_PASS` 表示完成。
- `PI0_S0_REGIONS_PDC` 是两遍布局的第二遍：先不带它布局一次，用 `paper/synth/dump_placement.tcl` 导出各节点硬核的位置，
  `paper/synth/gen_node_regions.py` 生成每个节点的软区域，再带它重新构建。
- `PI0_S0_CAP_MCP=1` 给 MLP72 → 捕获寄存器的 2 周期路径加多周期约束（RTL 保证该路径有两个阵列周期），使阵列时钟的报告和布线针对真实的关键路径；
  `PI0_S0_FLOW_MODE=evaluation` 是快速非时序驱动路由，但不能写出 bitstream。默认（时序驱动）路由在半阵列上约 20 分钟到 99 %。
- 在半阵列和全阵列之间切换构建前，恢复 `src/ace/tc_ref_design_top.acxprj`（`git checkout -- src/ace/tc_ref_design_top.acxprj`），
  否则上一次构建留下的布局区域会使布局失败。
- 编程脚本只认 `0x5330xx` 系列 ID；也可以用 `PI0_HOST_BRIDGE=1 PI0_FPGA_DBI_ROUTE=comp build/host/pi0_chunk_run selftest` 读回 ID 字。

## 验证结果

**仿真**（每个中间区域和输出区域的每一个期望 beat 都正确，且没有写入任何其他位置）：

| 测试 | 规模 | 结果 |
|---|---|---|
| 链的行 / tile / 节点 / 程序 / 共享端口 | 单元测试集 | 全部通过；负对照按预期失败 |
| 矢量 lane 和节点，1 / 2 / 4 lane，含 `*_q` 和 `pdq*` 测试集 | 所有 op、形状和格式，行长到 4,096；节点矩阵 70 / 70 | 全部通过 |
| 条带化切分、乱序响应重排 | 随机 burst × 3 种子 | 通过；负对照失败 |
| 跨节点线性层 | 最大 7 × 1024 → 128 | 通过 |
| 真实宽度的 expert `o_proj`，1–8 个节点 | 51 × 2048 → 1024 | 通过，6,528 beat |
| SigLIP 编码器层（第 0、13、26 层；两个相机） | 256 token × 1152，16 头，FF 4304 | 通过 |
| patch embedding 和 projector，两个相机 | 256 × 588 → 1152 → 2048 | 通过，205,024 beat |
| PaliGemma prefix 层（第 0、17 层） | 525 token × 2048，FF 16384 | 通过，6,912,844 beat |
| action expert 层（第 0、9、17 层） | 51 token × 1024，FF 4096，608 个 key slot | 通过，289,483 beat |
| action head + Euler 更新（第 0、9 步） | 50 × 32 动作 | 通过，41,023 beat |
| 无主机参与的 expert 层 / prefix 层（WAIT / POST，每节点一个程序） | 真实尺寸 | 通过 |
| chunk：1 个 SigLIP 层（两个相机）+ projector + action head + Euler，5 个节点 | 2,438,488 beat | 通过 |
| chunk：1 个 SigLIP + 1 个 prefix（525 token）+ 1 个 expert 层 + head + Euler，5 个节点 | 9,906,625 beat | 通过 |
| chunk：3 个 SigLIP + 2 个 prefix + 2 个 expert 层，2 个去噪步，5 个节点 | 2,250 万 beat | 通过 |
| 半阵列 18 个节点上的 chunk，32 级 / 16 级链混合，QUANT 尾融合，双缓冲 GEMM | 2,116,166 beat | 通过；每节点独立端口、64 周期往返，以及共享 NAP、50 % 随机停顿两种条件下均通过 |
| 半阵列 18 个节点上的 chunk，条带化 + ID 重排（`STRIPE=12`） | 2,116,166 beat | 通过 |
| 8 个矢量节点的 tiny chunk | 6,885,664 beat | 通过 |
| chunk：1 个 prefix 层 + 1 个 expert 层，PDQ 融合 | 5,494,033 beat | 通过 |
| 主机寄存器窗口；分页桥 | 单元测试 | 通过 |
| `run_pi0_full_size_gates.sh quick` | 全部检查项 | 全部符合要求结果 |

**硅片上**（VP815）：

| bitstream | 阵列 | 时钟（阵列 / 矢量 / fabric，MHz） | 结果 |
|---|---|---|---|
| 测试 bitstream | 1 个 16 级 int8 链节点 + 1 个 2-lane 矢量节点 | 725 / 333.33 / 250 | 链程序（7 个 tile、171 次行遍历、2,426 个结果 beat）和矢量测试集 `base` / `attn` / `ml` / `wide` 全部逐位一致，无错误标志；8 个在途 AXI 请求比 1 个快 1.85 倍（链）、13–18 %（矢量）；分页桥自检通过（页位于 0、256 MB、1 GB、9 GB、15.7 GB），主机写 69.5 MB/s、读 5.1 MB/s |
| `s0tn_500` | 3 节点完备阵列 | 500 / 285.7 / 222.2 | 完整 pi0 chunk 逐位一致，26.28 s（250 / 166.7 / 166.7 MHz 时 43.65 s） |
| `s1m_250` | 半阵列，18 节点 | 250 / 166.7 / 114.6 | 完整 chunk 逐位一致（3 次），7.29 s；46 帧策略评估相对 fp32 关节角最大 0.66°、平均 0.072° |
| `s1q_225` | 半阵列 + 写合并 | 225 / 181.8 / 125 | 完整 chunk 逐位一致，6.66 s；46 帧精度相同 |
| `s1s_325` | 半阵列 + 写合并 + 双缓冲 GEMM + 条带化开关 + ID 重排 | 325 / 125 / 117.6 | **完整 chunk 逐位一致，4.76 s**；PDQ 镜像 4.63 s，动作偏差 ≤ 0.018 |

完整 chunk 镜像约 2.8 GB，4,410 个 stage，半阵列上 28,316 个 part（全阵列 54,588）；装载一次约 55–62 s。每 chunk 主机开销约
10 ms 编码、23 ms 输入写入（1.3 MB）、0.4 ms 标志、1.4 ms 动作读回。

**时间模型**（`paper/sw/pi0_chunk_time.py`，理想存储器，预测值）：半阵列在设计时钟 250 / 333.33 / 725 MHz 下 1.46 s；全阵列 0.767 s，
带 PDQ 0.727 s。硅上时间与模型的比值从 1.63（`s1m`）降到 1.41（`s1q`）；差值是节点等待 GDDR6 往返的时间。

## 资源与时序（ACE 10.5.2）

| 模块 | RLB tile | MLP72 | BRAM72K | 满足的时钟 |
|---|---:|---:|---:|---|
| int8 链节点，16 级 | 440 | 17 | 25 | 阵列 750 MHz / fabric 250 MHz |
| int8 链节点，32 级 | 744 | 33 | 48 | 750 / 250 MHz |
| 链节点，32 级，控制路径重定时后 | — | — | — | 阵列 870 MHz / fabric 356 MHz |
| 矢量节点，2 lane，2 加载器，无融合 | 1,978 | 4 | 68 | 333 MHz |
| 矢量节点，2 lane，2 加载器，QUANT 尾 | 2,137 | 6 | 72 | 333 MHz |
| 矢量节点，2 lane，2 加载器，QUANT 尾 + PDQ，控制路径重定时后 | 2,570 | | | 348 MHz |
| 矢量节点，4 lane，4 加载器 | 3,624 | 8 | 120 | 303 MHz |
| 半阵列 bitstream `s1m`（已布线） | 29.10 % | 17.03 %（436） | 30.63 %（784） | 布线 Fmax 阵列 132 / fabric 81 / 矢量 179 MHz（报告受复位扇出支配） |
| 半阵列 bitstream `s1s`（已布线） | 36.15 % | 21.02 %（538） | 33.44 %（856） | 布线 Fmax 阵列 355.6 / fabric 123.4 / 矢量 135.8 MHz |
| 全阵列 bitstream `s2i`（已布局，布线未通过） | 56.92 % | 37.77 %（967） | 59.34 %（1,519） | — |

AC7t1500 共有 57,600 个 RLB tile、2,560 个 MLP72、2,560 个 BRAM72K 和 80 个 NoC 接入点。PV 链节点与 int8 链节点只在乘法模式上不同。
半阵列 bitstream 到目前为止都用非时序驱动路由构建，阵列的布线 Fmax 报告还未包含捕获路径的多周期约束，所以是下界。

## 限制与未决项

- **全阵列没有 bitstream。** 35 节点阵列布局通过（56.92 % RLB）但布线不收敛；目前可交付的最大阵列是 18 节点半阵列（4.76 s / chunk）。
- **板卡时钟比设计目标低 2–3 倍。** 频率损失主要在多节点阵列的布线（单节点 870 / 356 / 348 MHz，阵列布线后 355.6 / 123.4 / 135.8 MHz）；
  本目录的 RTL 已完成控制路径重定时并通过全部仿真门禁，但尚未构建 bitstream。
- **PDQ 融合在硅上有偏差。** 仿真逐位一致，硅上整 chunk 的动作偏差 ≤ 0.018（前 18 个 prefix 层和 1 个 expert 步逐位一致，偏差出现在之后的 expert 层 / 步），原因待查；
  不带 PDQ 的镜像逐位一致。
- **条带化 + ID 重排只在仿真中验证过。** 没有重排的条带化在硅上失败（NoC 对同一 ID 跨通道的响应乱序），带重排的版本尚未在硅上完成有效测试。
- 32 级链在 325 MHz 阵列时钟下会报出假的溢出标志（`o_error_bits` 0x04），数据仍逐位正确。
- 这些 bitstream 上 PCIe DMA 引擎不可用（SDK 的 DMA 初始化会锁死压缩 DBI 网关）；批量数据通过分页桥以程序控制 I/O 方式传输，装载一次镜像约 1 分钟。
- 生成的 chunk 固定为一种提示词长度和两个相机；提示词长度不同需要重新生成 chunk。
- 完整 chunk（4,410 个 stage、2.8 GB 镜像）太大，无法整体仿真；最大的整块仿真是 1 个 prefix 层 + 1 个 expert 层带 PDQ，完整 chunk 只在硅上验证。
