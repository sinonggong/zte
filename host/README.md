# host —— 主机工具

四个 C++ 工具，通过 Achronix SDK 访问 PCIe 上的节点阵列比特流。构建：

```bash
cmake -S host -B build/host && cmake --build build/host      # 需要 Achronix SDK 2.1.1：/opt/achronix/sdk 或 ACHRONIX_SDK_ROOT
```

| 工具 | 用途 |
|---|---|
| `pi0_chunk_run` | chunk 运行器：把生成器的镜像经 BAR1 分页桥写进 GDDR6，设置条带化开关，广播启动，轮询标志字，读回动作并与期望比对；`serve` 模式常驻，供策略服务器逐 chunk 调用 |
| `pi0_s0_replay` | 节点测试向量回放与板卡识别：`identify`（读 ID 字与节点数）、`reset`（软复位阵列）、`run --node N`、`status`（节点错误位） |
| `pi0_atu_tool` | 列出 / 移动 PCIe 端点的 BAR → NoC 地址翻译窗口 |
| `pi0_dbi_probe` | 只读探测 PCIe 控制器的 DBI 访问路径 |
| `pi0_fpga_backend.h`, `pi0_fpga_sdk_backend.*`, `pi0_fpga_mock_backend.*` | `pi0_s0_replay` 用的寄存器 / 批量访问后端（SDK 实现与软件模型） |

## pi0_chunk_run

```
pi0_chunk_run selftest                                   # 分页桥自检（需要 PI0_HOST_BRIDGE=1 PI0_FPGA_DBI_ROUTE=comp）
pi0_chunk_run run   <chunk 目录> --map <映射> [选项]       # 加载并运行一次或多次，比对标志与动作
pi0_chunk_run serve <chunk 目录> --map <映射>              # 常驻：stdin 收 "infer <inputs> <actions>" / "quit"
pi0_chunk_run probe <addr...>                            # GDDR6 通道别名探测
pi0_chunk_run fill <addr> <beats> [stripe] [check] | peek <addr> [dir] | verify <chunk 目录> [lo hi [write [wlo whi]]]
```

| 选项 / 变量 | 含义 |
|---|---|
| `--map vector:14,int8:0,uint8:12` | 生成器节点编号到芯片节点编号的映射（半阵列；全阵列 `vector:27,int8:0,uint8:24`） |
| `--repeat N`, `--timeline <file>` | 重复运行；记录每个标志字的完成时间，供 `paper/sw/pi0_timeline_compare.py` 与时间模型对比 |
| `--inputs <file>`, `--dump-actions <file>` | 用另一帧的主机输入覆盖镜像；把动作块写到文件 |
| `--no-load`, `--load-only`, `--no-check`, `--check-io-only`, `--wrong-list`, `--timeout-s` | 跳过加载（镜像已在板上）、只加载、不比对、只比对 IO 区、列出错误 beat、超时 |
| `PI0_HOST_BRIDGE=1` | 走 BAR1 分页桥（本设计唯一的批量路径，无 DMA） |
| `PI0_FPGA_DBI_ROUTE=comp` | 压缩的 DBI 网关（SDK 的 DMA 初始化会卡住该网关，因此不用 DMA） |
| `PI0_ATU_SLIDE`, `PI0_VERIFY_CANARY` | ATU 窗口滑动策略；`verify` 时写入金丝雀值 |

第一次 `run` 会打印 `window: N_NODE=18 id=0x53310a`，用它确认板上的比特流与镜像家族一致。`run` 与 `serve` 在第一次启动前
软复位阵列并重新应用 GDDR6 映射；`eval_chip_frames.py` 之后板上的输入区已被覆盖，下一次 `run` 不要带 `--no-load`。
