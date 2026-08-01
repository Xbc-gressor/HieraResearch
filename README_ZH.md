# HieraResearch

HieraResearch 是一个由确定性 Python 控制流程、现有 `tools/` 工具链和
受限 Claude 能力组成的多任务实验框架。Claude 只负责有界语义判断、背景
研究和候选代码编辑；生命周期、预算、评估、调优、账本和停止条件都由
Python 决定。

## 快速开始

从仓库根目录运行：

```bash
python -m hieraresearch <task-name> <tag> --model <claude-model>
```

常用模式：

```bash
python -m hieraresearch <task> <tag> --recordings <dir>  # 录制结果回放/消融
python -m hieraresearch <task> <tag> --preflight-only     # 只验证环境，不调用模型或评分
python -m hieraresearch <task> <tag> --resume-blocked --model <model>
```

也可使用安装后的 `hieraresearch` 命令。`--recordings` 是无在线模型的
可替换后端。`autoresearch-hillclimb` 仍是独立的对照基线，不是本框架的
另一条 coordinator 路径。

## 执行流程

`ExperimentCoordinator` 从持久化状态和紧凑 ledger view 推导下一步：

```text
环境初始化/预检
  -> 冻结背景工件
  -> baseline 或图行动 admission
  -> 候选 materialize 与 bounded code edit
  -> tuning contract
  -> no-score candidate preflight
  -> Phase A warm evaluation
  -> 每轮至多一个确定性 Phase C deep-tune
  -> 幂等 finalization
  -> experience refresh 或停止
```

纯决策位于 `src/hieraresearch/state_machine.py`，副作用由 coordinator、
`Toolchain`、过程执行器和各阶段服务显式执行。重启时 Python 从 ledger、
候选收据和报告恢复阶段，不依赖聊天上下文。无法证明继续安全时会
`blocked`，不会猜测分数或无限重试。

## 持久化与预算

- `runs/<task>/<tag>/ledger.json` 只能通过现有 `tools/ledger.py` 变更；
- `evaluation_attempts.jsonl` 在进入 `score_fn` 前立即保留 objective slot，
  后续 crash 也计入预算；
- no-score preflight 不保留 objective slot；
- `.orchestrator/state.json` 是 coordinator 的重启 bookkeeping，不取代
  ledger、evaluation report 或 attempt log；
- completed round receipts 和 invocation journal 记录不可逆操作及模型输入、
  schema、revision、结果和 outcome。

## Python/Claude 边界

| 模块 | Claude 工作 | Python 后置验证 |
|---|---|---|
| `background.py` | 有界研究和写入冻结背景输出 | background contract |
| `semantic.py` | 结构化 proposal prediction 和 idea | semantic helper selection、admission |
| `candidate.py` | 候选/contract bounded edit、失败诊断 | syntax、contract、search-space、preflight |
| `experience.py` | 有界证据综合 | experience validation、space-state application |
| `tuning.py` | 无模型判断 | candidate selection、worker、finalization |

Messages/API 用于固定输入的结构化判断；Agent SDK 只用于有界研究或编辑。
编辑请求有明确的 read roots、exact write paths、turn limit 和禁止 shell 的
tool policy。失败诊断只能返回 `config_invalid`、`code_incompatible` 或
`abandon`；接受 repair 后必须重新做 Python 校验和 no-score preflight，才可
重试 objective worker。`RecordedBackend` 支持回放和 model-free ablation。

详细协议见 [`docs/execution-layer.md`](docs/execution-layer.md)。

## 项目结构

```text
src/hieraresearch/       coordinator、状态、模型边界、过程和阶段服务
tools/                   ledger、图搜索、语义选择、评估和调优 helper
tasks/<task>/            独立 uv task project、TASK.md、task.toml、prepare.py
docs/                    执行层、搜索空间和背景证据契约
runs/<task>/<tag>/       本地实验工件，已加入 gitignore
```

任务的 `TASK.md` 是行为和评估语义合约，`task.toml` 是机器可读配置。
实验只修改 run-local candidate；task-owned `prepare.py` 保持不变。

## 验证

```bash
python -m pytest tests -q
python tools/validate_tasks.py
python tools/validate_background.py
python tools/validate_got.py
python tools/validate_search_backends.py
```

测试重点是 transition/receipt 边界、stale 或 malformed model output、编辑
路径限制、objective accounting 和 subprocess termination，而不是 prompt
措辞或重复 helper 内部实现。
