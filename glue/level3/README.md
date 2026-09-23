# glue —— demo：策略服务器与机器人端（整个 pi0 在 FPGA 上）

这个目录把节点阵列接到 LeRobot 的机械臂 rollout 上。台式机一侧是 HTTP 策略服务器：收到 Jetson 发来的预处理 batch，
计算进入模型的嵌入（patch 卷积、token 嵌入、状态投影、噪声），把它们写进板卡，等芯片跑完整个 chunk，再把 50 × 32 的动作返回。
Jetson 一侧是 `pi0_remote` 策略插件和机械臂客户端：预处理、后处理、动作队列和伺服都留在机械臂旁，不做任何模型计算。
系统的原理、每个组件的实现和完整的故障处理见 [`docs/PI0_DEMO_SYSTEM.md`](../../docs/PI0_DEMO_SYSTEM.md)。

## 文件

| 文件 | 用途 |
|---|---|
| `run_pi0_chip_policy_server.sh` | 台式机启动脚本：`PI0_ACTION_EXPERT=chip`，设置 `PI0_HOST_BRIDGE=1`、`PI0_FPGA_DBI_ROUTE=comp`，检查 `PI0_CHUNK_RUN`（默认 `build/host/pi0_chunk_run`），然后调用 `run_pi0_policy_rpc_server.sh` |
| `run_pi0_policy_rpc_server.sh` | 通用启动脚本：LeRobot venv、`ASYNC_POLICY_PATH`（checkpoint）、`PI0_RPC_PORT`（8081）、`OMP_NUM_THREADS`；日志写到 `~/pi0_glue/logs/`；有 ACE 构建在跑时拒绝启动（`ALLOW_ACE_BUILD=1` 覆盖） |
| `pi0_policy_rpc_server.py` | HTTP 服务器：`/predict`（pickle 的预处理 batch → 归一化动作 chunk + 统计）、`/health`（`expert_backend`、`last_stats`）。chip 模式下以内存映射方式打开 checkpoint，只触碰嵌入表、patch 卷积和 `state_proj` |
| `pi0_fpga_policy.py` | `PI0FpgaPolicy`：LeRobot pi0 加可切换的后端；`PI0_ACTION_EXPERT=chip` 时 `_open_chip` 打开 `paper/sw/pi0_chip_runtime.py` 的 `ChipServer`，`_predict_chip` 用 `paper/sw/pi0_chip_host_inputs.py` 算主机输入并运行一个 chunk。其余后端（`torch`、`fpga`、`fpga_torch`）属于早期的分离式部署，本交付不使用 |
| `eval_chip_frames.py` | 用录制的帧走同一个 chip 后端，把芯片动作与 fp32 模型的采集动作比较（关节角误差、相对 RMS），输出 JSON |
| `capture_pi0_prefix_kv.py` | 在台式机 CPU 上用 fp32 模型为回放文件的每一帧生成采集（prefix KV、状态、噪声、fp32 参考动作）到 `~/pi0_glue/captures/<episode>/frame_NN.npz`；生成器、数值工作和评估都读它 |
| `make_pi0_remote_policy_dir.py` | 从 checkpoint 生成 Jetson 用的 `pi0_remote` 策略目录（复制预处理 / 后处理配置和归一化统计量；不带权重） |
| `plugins/lerobot_policy_pi0_remote/` | LeRobot 第三方策略包 `pi0_remote`：`configuration_pi0_remote.py`（与 pi0 相同的特征布局、`server_url`、超时）、`modeling_pi0_remote.py`（`predict_action_chunk` POST 到 `/predict`） |
| `test_pi0_remote_roundtrip.py` | Jetson 上不接机械臂的回路测试：一帧回放观测 → 预处理 → 远程 chunk → 后处理 |
| `pi0_arm_client.py`, `run_pi0_arm_client.sh` | Jetson 机械臂客户端：LeRobot `RobotClient` + chunk 看门狗、停止文件 `/tmp/pi0_arm_stop`、运行时长上限、`report.json` / `queue.png` |
| `../level2/pi0_replay_client.py`, `../level2/export_replay_frames.py` | 回放版客户端（不接机械臂）和从 LeRobot 数据集导出回放 `.npz` 的工具 |

## 运行 demo

台式机上打开了 FPGA 设备的进程只能用 Ctrl+C 停止；机械臂只在有人守在急停旁时运行。

### 0. 前提

| 项 | 怎么做 |
|---|---|
| 主机工具 | `cmake -S host -B build/host && cmake --build build/host`（见 `host/README.md`） |
| 比特流已烧写并验收 | `bitstream/s1s_325/` 解压后按 `scripts/README.md` 烧写；`pi0_chunk_run selftest` PASS，ID `0x53310a`、18 节点 |
| chunk 镜像已生成 | 需要 checkpoint（`~/pi0_ckpt/ur7e-demo-2-pi0-fsdp/010000/pretrained_model`）、采集帧（`~/pi0_glue/captures/`，由 `capture_pi0_prefix_kv.py` 生成）、标定文件（`build/paper_vector_unit/calib/`，由 `paper/sw/regen_calib.py` 生成）；约 10 分钟，约 2.8 GB |
| 镜像已加载并验收 | `scripts/pi0_array_silicon_session.sh half build/paper_pi0_chunk/chdb2_half_full`：标志与动作全部一致（0 wrong） |
| 网络 | Jetson 与台式机直连：台式机 192.168.10.1，Jetson 192.168.10.2 |

```bash
# 生成半阵列的 chunk 镜像（台式机，~/lerobot/.venv）
PI0_CHUNK_GEMM_DB=1 ~/lerobot/.venv/bin/python paper/sw/pi0_chunk_program.py --out build/paper_pi0_chunk/chdb2_half_full \
    --n-vec 4 --n-chain 12 --n-deep 6 --n-pv 2 --n-stage 16 --slot-bits 12 --bin --exp-areas IO,FLAGS \
    --sync deps --interleave --prefix-row-blocks 3 --fuse-quant
```

镜像家族必须与比特流匹配：`s1s_325` 用双缓冲 GEMM 的 `chdb*` 镜像（`PI0_CHUNK_GEMM_DB=1`）。

### 1. 台式机：启动策略服务器

```bash
PI0_CHIP_CHUNK=$PWD/build/paper_pi0_chunk/chdb2_half_full PI0_CHIP_MAP=vector:14,int8:0,uint8:12 \
ASYNC_POLICY_PATH=~/pi0_ckpt/ur7e-demo-2-pi0-fsdp/010000/pretrained_model glue/level3/run_pi0_chip_policy_server.sh
curl -s http://127.0.0.1:8081/health          # "ok": true, "expert_backend": "chip"
```

服务器持有设备（其子进程 `pi0_chunk_run serve` 常驻），一次处理一个请求；运行结束后可以保持运行。

### 2. Jetson：安装插件、生成策略目录（一次性）

```bash
# conda env lerobot
pip install -e glue/level3/plugins/lerobot_policy_pi0_remote
python glue/level3/make_pi0_remote_policy_dir.py --checkpoint <本地 checkpoint> --server-url http://192.168.10.1:8081 \
    --out ~/pi0_glue/policies/pi0_remote_ur7e-demo-2-pi0-010000
cp glue/level3/*.py glue/level3/*.sh glue/level2/*.py ~/pi0_glue/                  # 客户端脚本
```

### 3. Jetson：无机械臂回路测试

```bash
fuser /dev/video0 /dev/video2 || echo "cameras free"                             # 先关闭相机预览
curl -s --noproxy '*' http://192.168.10.1:8081/health; echo
python ~/pi0_glue/test_pi0_remote_roundtrip.py --policy ~/pi0_glue/policies/pi0_remote_ur7e-demo-2-pi0-010000 \
    --replay ~/pi0_glue/replay_ur7e_demo1_ep0.npz                                # 回放帧由 ../level2/export_replay_frames.py 导出
```

### 4. Jetson：机械臂运行

LeRobot 的策略服务器加载 `pi0_remote` 策略（`--policy_type=pi0_remote --pretrained_name_or_path=<策略目录>`），然后：

```bash
ASYNC_SERVER_ADDRESS=<策略服务器 host:port> ASYNC_FPS=1 ~/pi0_glue/run_pi0_arm_client.sh
touch /tmp/pi0_arm_stop                     # 从另一个终端停止（servoStop + stopScript）
```

### 5. 精度评估（不接机械臂）

```bash
PI0_ACTION_EXPERT=chip PI0_CHIP_CHUNK=$PWD/build/paper_pi0_chunk/chdb2_half_full PI0_CHIP_MAP=vector:14,int8:0,uint8:12 \
PI0_CHUNK_RUN=$PWD/build/host/pi0_chunk_run ~/lerobot/.venv/bin/python glue/level3/eval_chip_frames.py \
    --frames demo1_ep20:all demo1_ep40:all recov_pi0_ep00:all recov_pi0_ep10:all --json build/eval_chip.json
```

生成镜像所用的那一帧必须逐位复现生成器的动作；其余帧给出关节角误差与相对 RMS（46 帧留出集：关节最大 0.66°，平均 0.072°）。

### 常见问题

| 现象 | 处理 |
|---|---|
| Jetson 报服务器离线 | 台式机上检查 `/health`；服务器没有运行就重新启动 |
| `pi0_chunk_run` 提示 "DBI gateway does not answer" | 重新烧写比特流（`scripts/README.md`） |
| 某个节点报错或 chunk 超时 | `build/host/pi0_s0_replay status --node N` 看错误位；`pi0_s0_replay reset` 软复位；重新加载镜像 |
| 标志正确但动作错误、每次不同 | 镜像家族与比特流不匹配（`chnd` / `chdb` / `chdbp`、条带化开关），重新生成 |
| 提示词长度与 chunk 不一致 | 服务器拒绝该观测；为这个提示词重新生成镜像 |
| 相机错误 | 关闭相机预览（`pkill -f rerun`）后重新连接 |
| 机械臂出现意外动作 | 按急停，然后触碰停止文件 |

更完整的表见 `docs/PI0_DEMO_SYSTEM.md` §6.6。

## 环境变量（chip 模式）

| 变量 | 含义 |
|---|---|
| `PI0_ACTION_EXPERT=chip` | 整个模型交给节点阵列 |
| `PI0_CHIP_CHUNK` | 已加载到板卡的 chunk 目录（`paper/sw/pi0_chunk_program.py --bin` 的输出） |
| `PI0_CHIP_MAP` | 生成器节点编号到芯片节点编号的映射；半阵列 `vector:14,int8:0,uint8:12`，全阵列 `vector:27,int8:0,uint8:24` |
| `PI0_CHUNK_RUN` | 板卡运行器二进制，默认 `build/host/pi0_chunk_run` |
| `PI0_CHIP_SERVE` | 1（默认）：一个常驻的 `pi0_chunk_run serve` 子进程；0：每个 chunk 启动一次 `pi0_chunk_run run` |
| `PI0_CHIP_SW`, `PI0_CHIP_WORK` | `paper/sw` 目录（默认按仓库相对路径）；运行清单的工作目录（默认 `<chunk>/runtime`） |
| `PI0_HOST_BRIDGE=1`, `PI0_FPGA_DBI_ROUTE=comp` | 分页桥数据通路和 DBI 访问路径，启动脚本已设置 |
| `ASYNC_POLICY_PATH`, `PI0_RPC_HOST`, `PI0_RPC_PORT`, `OMP_NUM_THREADS` | checkpoint 目录、监听地址（默认 0.0.0.0）、端口（8081）、主机线程数 |

每个 chunk 的统计（`/health` 的 `last_stats`、服务器日志的 `PI0_STAGE` 行）：`chip_s`（芯片运行）、`inputs_s`、`flags_s`、
`actions_s`（主机 I/O）、`encode_s`、`host_inputs_s`、`wall_s`、`total_s`、`server_wall_s`，以及 `from_hardware=True`。

## 注意

- 持有 `/dev/ac7t15xx0` 的进程（服务器及其 `pi0_chunk_run serve` 子进程）只能用 Ctrl+C 停止；子进程只会被要求 `quit`。
- 服务器一次只处理一个请求；提示词 token 数必须与 chunk 生成时一致（本任务 13 个），否则该观测被拒绝。
- `eval_chip_frames.py` 之后板上的输入区已被覆盖，下一次 `pi0_chunk_run run` 不要带 `--no-load`。
