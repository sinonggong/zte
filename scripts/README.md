# scripts —— 比特流构建、时钟、打包与板卡操作

节点阵列比特流从综合到上板的全部脚本都在这里。Verilator 仿真的脚本在 `paper/rtl/`，主机工具在 `host/`，demo 在
`glue/level3/`。

## 文件

| 文件 | 用途 |
|---|---|
| `launch_pi0_s0_build.sh` | 启动一次 ACE 10.5.2 构建（后台、内存受限的 systemd scope）：`<tag> <N_CHAIN> <N_PV> <N_VNODE> <N_DEEP>`；先用 `paper/sw/vector_unit_ref.py --tables` 生成向量单元的函数表，把本次的 SDC 开关写到 `build/s0/<impl>.env`，日志在 `build/s0/<impl>.log` |
| `run_pi0_s0_ace_flow.tcl` | ACE 流程本身：建 impl、换上 `src/constraints/pi0_s0_ace.{sdc,pdc}`、Synplify 综合（`paper/synth/s0/pi0_s0_synth.prj`）、布局、布线、报告、写比特流 |
| `pi0_s0_pick_clocks.py` | 读已布线 impl 的时序报告，给出三个时钟合法的 PLL 设置（阵列 `400 × fb / ref / 8`，矢量与 fabric `8000 / ODN`），并打印重定时钟命令 |
| `run_pi0_s0_reclock_bitstream.sh` | 不重新布线，用新的时钟重新生成 IO 环和比特流：`<impl> <阵列 MHz> <ref> <fb> [矢量 ODN] [fabric ODN]`；从 `build/s0/<impl>.env` 恢复构建时的 SDC 开关 |
| `pi0_bundle_s0.sh` | 把 impl 的 `tc_ref_design_top.hex`、时序摘要打包到 `~/pi0_board_bundle/bitstream/<name>/`（附 `INFO.txt`、`SHA256SUMS`） |
| `pi0_board_program_s0.sh` | JTAG 烧写一个 bundle 里的比特流：卸载驱动、编程、重训 PCIe 链路到 DLActive、移除并重新扫描设备、强制 16 GT/s、加载驱动、`pi0_s0_replay identify` 读回 ID |
| `pi0_array_silicon_session.sh` | 板上验收：分页桥自检，加载一个 chunk 镜像并与生成器的期望比对（`half` / `full` 选节点映射） |
| `pi0_s0_silicon_matrix.sh`, `pi0_s0t_silicon_session.sh` | 节点级验收（每一组测试向量在 S0 比特流上回放）、3 节点阵列会话 |
| `pi0_board_recover.sh`, `pi0_board_after_warm_reboot.sh`, `pi0_board_prepare_warm_reboot.sh`, `pi0_board_bringup_from_fics.sh` | 板卡从死机、冷启动、卡不在总线上等状态恢复的步骤脚本 |
| `llama3_8b_runtime_host_regs.h`, `export_pi0_int8_quant_bank_pages.py` | 只是 `host/pi0_fpga_mock_backend.cpp` 与 `paper/sw/pi0_deploy_model.py` 的依赖，不单独使用 |

## 构建半阵列比特流

```bash
# 18 节点半阵列：14 个 chain 节点（其中 2 个 PV、6 个 32 级）+ 4 个矢量节点；约 1.5-3 小时，日志 build/s0/<impl>.log
PI0_S0_STRIPE=12 PI0_S0_RESET_FP=1 PI0_S0_CAP_MCP=1 PI0_ACE_SEED=7 \
PI0_S0_REGIONS_PDC=./../../build/s0/regions_half.pdc scripts/launch_pi0_s0_build.sh half 14 2 4 6
```

| 环境变量 | 含义 |
|---|---|
| `PI0_S0_STRIPE` | 12 = 4 KB GDDR6 通道条带化（运行时开关在主机窗口 beat N_NODE+1 bit 64）；0 = 关闭 |
| `PI0_S0_RESET_FP` | 1 = 复位释放作为假路径（否则每个时钟的最差路径都是复位扇出，报告看不到真实路径） |
| `PI0_S0_CAP_MCP` | 1 = MLP72 → 捕获寄存器的 2 周期多周期约束（RTL 保证 2 周期窗口）；日志打印 `PI0_S0_CAP_MCP: N capture registers` |
| `PI0_S0_FLOW_MODE` | 空 = 时序驱动布线（默认）；`evaluation` = 快速的非时序驱动布线，比特流用 normal 模式补写 |
| `PI0_S0_REGIONS_PDC` | 按节点划分的布局区域（`paper/synth/gen_node_regions.py` 生成；相对 `src/ace` 的路径） |
| `PI0_ACE_SEED`, `PI0_S0_ID_WORD` | 布局种子；比特流 ID 字（24 位，主机窗口可读回） |
| `PI0_S0_IMPL_OPTS` | 额外的 ACE 实现选项，如 `"clock_skew_opt 0"` |
| `PI0_S0_MAX_OUT`, `PI0_S0_NAPS_PER_NODE`, `PI0_S0_N_LANE`, `MEM_HIGH` | 节点 AXI 未完成事务上限（8）、每节点 NAP 数（2）、矢量节点 lane 数（2）、scope 内存上限（20G） |

`src/constraints/pi0_s0_ace.sdc` 里的 hold 不确定度是 0.050 ns：0.150 ns 会让每个 MLP72 / BRAM72K 内部的零连线寄存器环路
都成为修不掉的 hold 违例，时序驱动流程的全温度 hold 修复会跑几小时后在 ACE 内部断言退出；0.050 ns 下同一布局 13 分钟布线完成。
时序驱动布线与 evaluation 布线的差别（同一设计、同一布局）：array 525 / fabric 199 / vector 208 MHz 对 356 / 123 / 136 MHz。

## 选时钟、重定时钟、打包

```bash
scripts/pi0_s0_pick_clocks.py src/ace/<impl>                        # 例：array 500 (ref 1 fb 10), vector 200 (ODN 40), fabric 181.8 (ODN 44)
scripts/run_pi0_s0_reclock_bitstream.sh <impl> 500 1 10 40 44         # 只改 PLL 与 IO 环，几分钟
scripts/pi0_bundle_s0.sh <impl> <name>                                # → ~/pi0_board_bundle/bitstream/<name>/
```

阵列时钟的报告值偏保守（捕获路径按单周期分析时），以 `PI0_S0_CAP_MCP=1` 构建的报告为准。

## 烧写与验收

```bash
SUDO_ASKPASS=<helper> scripts/pi0_board_program_s0.sh bitstream/<name>/tc_ref_design_top.hex   # 路径相对于 ~/pi0_board_bundle
PI0_HOST_BRIDGE=1 PI0_FPGA_DBI_ROUTE=comp build/host/pi0_chunk_run selftest                    # 分页桥自检
scripts/pi0_array_silicon_session.sh half build/paper_pi0_chunk/<chunk>                           # 加载 + 整块比对（0 wrong）
```

烧写需要 ACE 的许可证服务在运行，JTAG 线接在台式机 USB 口；烧写后 BAR 分配丢失，脚本会移除并重新扫描设备。链路必须是
16 GT/s × 16，否则 GDDR6 传输不可靠。

## 板卡恢复

| 情况 | 脚本 |
|---|---|
| 台式机重启后 `lspci` 看不到 `1b59:` 设备 | `pi0_board_recover.sh --check`；不重启的恢复 `pi0_board_recover.sh <hex>`（唤醒根端口、JTAG 编程、重训链路、重新扫描） |
| 冷启动后卡不响应 | `pi0_board_prepare_warm_reboot.sh` 之后普通重启（不要 `systemctl --force --force reboot`），再 `pi0_board_after_warm_reboot.sh` |
| 从零开始的完整流程 | `pi0_board_bringup_from_fics.sh` |

持有 `/dev/ac7t15xx0` 的进程只能用 Ctrl+C 停止；`kill`、SIGTERM 或关闭窗口可能让台式机死机。
