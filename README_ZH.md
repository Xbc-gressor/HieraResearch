# autoresearch-automl

由确定性 Python driver 驱动的自主 AutoML 实验框架。`driver/` 包编排 Claude Agent SDK 角色会话——driver 拥有排序、预算与升级链，LLM 会话负责生成与判断；确定性的状态、图搜索、评估和调优由共享的 `tools/` 实现。原 `.claude/` Claude Code runtime 已退役并删除。

## 1. 框架功能

该项目从单任务脚本演化为多任务实验框架，采用两层搜索架构：

- **外层搜索（S-GoT）**：候选方案形成开发 DAG。确定性图搜索（PUCB + 结构互补性 `c̃_dag` + bootstrap/stall fresh 规则）决定每轮采取的行动（`fresh` / `improve` / `crossover`）。LLM 智能体将每个行动转化为具体想法。
- **内层搜索（解耦调优）**：超参数调优器在每个候选方案的结构内搜索，但**与外层搜索解耦**。框架不会内联调优每个候选方案，而是每轮选择一个有前景的候选方案进行深度调优。
- 每个候选方案位于独立目录；过往结果永不覆盖。
- 每个任务是独立的 uv 项目，拥有各自的依赖。

## 2. 项目结构概览

框架包含：

- **`CLAUDE.md`**：开发 agent 打开项目时首先读取的入口文档。
- **`driver/`**：唯一 runtime。确定性 Python 循环（`driver/loops/experiment.py`、`driver/loops/hillclimb.py`）编排 Claude Agent SDK 角色会话；角色 prompt 位于 `driver/prompts/`；每次角色调用 = 一个 SDK 会话 + 一份持久化 receipt。
- **`tools/`**：候选方案创建、账本管理、图搜索和调优的确定性脚本。
- **`tasks/`**：独立任务包。每个都是独立的 uv 项目。
- **`runs/`**：本地实验状态（候选方案、日志、账本、循环状态）。被 git 忽略；永不提交。

### 2.1 主要验证任务

**`tasks/tabular-model-search/`**：CPU 友好的表格分类模型搜索任务。用于验证框架能否自主探索 XGBoost、CatBoost、SVM、随机森林、特征选择、PCA、集成等方法。

## 3. 快速开始

### 3.1 开发

要继续开发框架，在此目录打开常规 Claude Code 会话并说：

```text
阅读当前项目并继续开发 autoresearch-automl。
```

### 3.2 运行实验

先在仓库根同步框架环境；这一次同步同时安装 HEBO policy 所需的
HEBO MACE 与 CPU-only torch，不需要再为 HEBO 单独执行 `uv sync`：

```bash
uv sync --frozen
```

任务仍是独立 uv project，所以首次运行某个 `tasks/<task>/` 前仍需同步该任务的固定评估环境。

用 driver 启动完整实验（真实 SDK 会话，消耗 API 额度）：

```bash
uv run python -m driver run tabular-model-search <tag> \
  --loop experiment --model <model-id> [--max-evaluations N] [--timeout SECONDS] \
  [--semantic-policy POLICY] [--scheduler-policy POLICY]
  [--inner-tuner-policy POLICY]
```

新 experiment run 默认启用 `coverage_attempt` 语义策略、scheduler `anchor_challenger_v1` 和 inner-tuner `hebo24-hebo20`，即每个入选候选的 24+10+10 三段 `pool_hebo_mace` 合约。无需预建目录或手改 JSON。对照臂可通过 `--scheduler-policy` 和 `--inner-tuner-policy` 显式选择；所有 24+20 inner policy 都必须与 `anchor_challenger_v1` 配对。解析后的选择会持久化到 run-local `framework_cfg.json`，恢复已有 run 时不改写已冻结策略。

`--loop hillclimb` 是 edit→run→keep/revert 对照基线，启动方式相同。`--loop baseline-tune` 是强调优基线：task 提供的 baseline（需要 `[seed].provided`）在 step 0+1 之后，由 driver 确定性地用 ONE 个 HEBO MACE bout 花完整个 `--max-evaluations` 预算（相当于把 INITIAL BOUT 拉长到整个 run；无 ideation、无 scheduler、无 tuner-orchestrator 会话），冻结 `inner_policy=baseline-hebo-full-v1` + `scheduler_policy=legacy`。`--model` 仅新运行必需；恢复运行时以 `run_metadata.json` 为准。`--max-evaluations`、`--timeout`、`--semantic-policy`、`--scheduler-policy` 与 `--inner-tuner-policy` 经 `tools/init_run.py` 持久化到 `framework_cfg.json`；`--timeout` 是单次评估时限的别名，不是会话看门狗。`anchor_challenger_v1` 和 `v3_2` 都需要有限的 `max_evaluations`；新 run 模板默认提供 200。

对于并行实验，使用不同的 `tag` 值启动多个进程。
CPU task 可以真正并行。声明 `[resources].accelerator = "cuda"` 的 task
会在 warm screening、Phase-C bout 和 hillclimb objective 的整个执行期间
持有同一个 host-local CUDA 独占租约；多个本地 driver 进程会排队而不是
争抢 GPU。长 objective 由 driver 前台持有并同步等待，Agent 不再通过
`nohup`、后台 task 或 PID 轮询启动训练。

## 4. 框架工作原理

### 4.1 核心文档

**`CLAUDE.md`**：项目入口。完整实验协议以确定性 Python 编码在 `driver/loops/experiment.py` 中，各角色行为由 `driver/prompts/*.md` 承载；任务语义来自 `TASK.md` 和 `task.toml`。

**`tasks/<task-name>/TASK.md`**：供人和 LLM 阅读的任务描述。包含 `## Evaluation Contract` 部分，描述：
- 训练表面：候选方案在训练期间做什么
- 官方评分表面：候选方案如何产生分数
- 报告表面：候选方案记录什么
- 调优器评估表面：单一 `config → score` 函数（如果支持调优）
- 行为规则

每个角色会话在工作前读取此合约。主循环在每轮开始时重新读取以避免长时间运行中的漂移。

**`tasks/<task-name>/task.toml`**：机器可读配置。包含：
- uv 环境和超时
- `[evaluation].score_fn`：单一 `config → score` 函数名（官方 = 调优表面；warm-start 和 Phase C 都调用它；没有单独的官方运行）
- 指标字段名
- 可编辑/只读文件约束
- 依赖规则
- 可选的 `[candidate]` 覆盖（仅在需要非默认目录结构时）

**拆分原则**：函数*名称*放在 toml（配置）；函数*语义*放在 TASK.md（散文描述）。

**`tasks/<task-name>/prepare.py`**：固定评估表面。正常实验期间不修改。

**`tasks/<task-name>/train.py`**：可选的用户提供基线。若由 `[seed].provided` 声明，框架会先把它作为 all-baselines 点的首个观测根节点原样复制并评估；多数任务省略它，此时循环仍通过 `fresh` 自举。

### 4.2 实验流程

driver 的实验循环（`driver/loops/experiment.py`）以确定性 Python 拥有排序、预算与升级链；LLM 只通过有界角色会话进入。**一轮 = ① 一代（≤B 个想法，每个经过 step 0+1）+ ② 一次解耦深度调优步骤**：

```text
driver 读取 task.toml / TASK.md（角色 prompt 位于 driver/prompts/）
        ↓
background-researcher: 解析维度策略 → 多后端知识侦察 → background.md + background_retrieval.json
    (llm_induced 额外先生成 dimension_catalog.json；随后冻结显式基线、hyp-* 与关系)
        ↓
(若声明 provided entrypoint) 作为 all-baselines 根节点原样复制并评估一次默认配置
        ↓
(每个已完成的非空轮次后) experience-extractor: 提炼全局经验 → ledger.json experience 块
        ↓
idea-generator:
    SELECT-1: got_select.py decide → 获取图行动与数字父代
            (bootstrap/stall → fresh; 否则在前沿上 PUCB → ≤B improve/crossover)
    SELECT-2: semantic_search.py → 合法点集 → coverage_experience/coverage/gain/gain_uncertainty/gain_uncertainty_nocost 选点
    IDEATE: 将点落成完整方案 → ledger.py add-record（祖先、点、策略收据分开）
        ↓
对每个行动: tools/new_candidate.py --skip-entrypoint → 创建候选目录(prepare.py + 精简 _candidate_brief.json)
        ↓
candidate-writer: 读取候选简报 (idea + 数字父代 + semantic_point) → 写 candidate/train.py
        ↓
step 0+1: tunable-contract-extractor
    ① 制作 PARAM_SCHEMA + 重构 make_model
    ② 提出 K 个热启动配置 + SEARCH_SPACE
    ③ 评估 K 个配置(一个 config→score 函数; 没有单独的官方运行)
       → 对每次崩溃调用 crash-diagnosis 角色(修复配置或修复代码)
       → 构建 BASE_PARAMS, 记录 best_warm_score
       → extractor 调用 record-run (final_best_score=best_warm_score + keep/discard status)
         + set-tuning (元数据, 无 --mark-tuned) 到 ledger.json
        ↓
(每轮一次) tuner-orchestrator:
    tune_tools.py select-candidate → 从整个种群中选择一个符合条件的候选方案
    → Phase C 搜索
    → finalize_tuning.py 验证终态后统一应用参数并原地关闭 report + ledger (无重新运行)
```

上述协议原先是单一主会话内的约定，现已全部落入 driver 代码：循环按顺序为每个角色调用一个 SDK 会话，会话只持路径与紧凑 id，通过 `mcp__receipts__submit_receipt` 返回 receipt；driver 校验 receipt 与后置条件，不满足则在同一会话内纠正跟进，直至升级。角色工具能力由 fail-closed 的 PreToolUse hook 强制（`Agent`/`Task`/`Skill` 一律拒绝），崩溃诊断由 `crash-diagnosis` 角色完成。

### 4.3 两层搜索架构

**外层搜索（图行动 + 语义选点）**：候选方案形成开发 DAG。`got_select.py decide` 使用 PUCB + 结构互补性 `c̃_dag` + bootstrap/stall 规则，只决定行动与父代：
- 空图或停滞 → `fresh`
- 否则 → 在前沿叶子上 PUCB → ≤B `improve`（单亲）/ `crossover`（多亲）

随后 `semantic_search.py` 在冻结的层级空间中生成有界合法点：新 run 默认使用 `coverage_attempt`（确定性 coverage 加上同一 `(point, op)` 历史尝试的 policy-conditioned downside）；`coverage` 是仅用覆盖度的确定性基线，`coverage_experience` 加入按账本边统计的 carrier 先验。`gain` 与 `gain_uncertainty` 先用 `gain-context` 固定当前 experience revision，再把背景先验、带 run/semantic-edge 引用的 experience 调整、最终收益/不确定性、成本、覆盖分别保存并组合；`gain_uncertainty_nocost` 与 `gain_uncertainty` 相同但不预测成本（实现前的成本估计通常是噪声）。`llm_intelligence_score` 是运行前固定的 `[0,100]` 启发式可信度先验：以 `score/100` 缩放完整的 LLM 判断项，不缩放确定性的 coverage，也不改写原始预测；它不是校准概率。helper 校验最终值等于先验加调整，并拒绝只在文字中提及历史却不改变 gain 或 uncertainty 的预测；若合法 snapshot 没有任何被引用的 run/edge，则仍固定 revision，但 citations 与调整均为零。LLM 再把选定点落成完整方案。图策略与语义采集策略互不混写。

**内层搜索（解耦调优）**：每个候选方案结构内的超参数搜索，分为两个阶段，**与外层搜索解耦**：

- **Step 0+1**（tunable-contract-extractor；一个角色会话完成；对每个候选方案运行）：
  ① 行为保持地重构构造逻辑为 `make_model(<task-input>, params)`（首个参数与返回对象的接口由任务的 Evaluation Contract 定义）+ 声明 `PARAM_SCHEMA`
  ② provided entrypoint 仅使用一个原始默认配置；其他候选结合**血统证据**与数据提出 K=5 个热启动配置 + 数据驱动的 `SEARCH_SPACE`；一致性预检 + `check-search-space` + `apply_search_space`
  ③ **评估 K 个配置**（warmstart_eval；顺序/可恢复）；**对每次崩溃调用 crash-diagnosis 角色**（config-invalid → 修复配置；code-incompatible → 最小化修复代码 ≤10 次）；全部通过 → 写 `BASE_PARAMS`=最佳-K′ + `phase_a`，记录 `best_warm_score`；无法修复 → 记录 `status:crash`
  深度调优（step 2）被解耦；所有候选方案在此停在 step 0+1。

- **Step 2（解耦渐进式深度调优）**（tuner-orchestrator；**每轮在整个运行上运行一次**，而非每个候选方案）：
  - 默认 scheduler `anchor_challenger_v1`：seed roots 到齐后立即给当时最优候选一个 INITIAL bout，在保留 challenger 与两个 DEEP segment 硬预算的前提下继续生成，然后初始化 late challenger 并花两个 DEEP segment。第一段 DEEP 有正收益则继续同一候选，否则切换到另一个已初始化候选。`v3_2` 和 legacy percentile/alternation 调度器作为显式对照臂保留。
  - 默认 inner policy `hebo24-hebo20`：INITIAL=24，之后至多两个 DEEP=10 segment；三段都调用同一个 `pool_hebo_mace` 协议（LLM pool POOL=5 + 官方 HEBO MACE），以全部历史 trial 为先验，无 TuRBO 数值维门槛。当前 policy 没有 CONTINUE regime；两个 DEEP segment 的边界只供 scheduler 重新准入或换 candidate。`deferred-random8-hebo10-spsa10-v1`、local-TR 和 TuRBO 等策略作为显式对照臂保留。
  - Finalize：`finalize_tuning.py` 只接受终态 Phase C；随后在 warm incumbent 与**所有 bout 的全部 trial** 上取全局最佳、原子写回 `BASE_PARAMS`，并一次性更新 ledger 中的分数、状态、调优元数据与分级 `evaluation_depth`（**无重新运行**）。被杀死或非终态搜索只保留为部分证据，不得进入下游。
  详见 §5.7。

**关键洞察**：没有单独的官方运行。有**一个全局 `config → score` 函数**（task.toml `[evaluation].score_fn`）。热启动评估和 Phase C 调优都调用它。调优后的最佳值就是候选方案的新分数。

## 5. 角色

原 `.claude/agents/` 子智能体已随 Claude Code runtime 一起退役，迁移为 driver 角色：prompt 位于 `driver/prompts/`，由 `driver/roles.py` 注册（prompt 文件、正向工具能力集、receipt schema、driver 侧后置条件），每次调用是一个独立的 Claude Agent SDK 会话。会话不返回自由文本——通过进程内 MCP 工具 `mcp__receipts__submit_receipt` 提交 schema 校验的 receipt，持久化在 `<run_dir>/receipts/` 下并按 `invocation_id` 关联；后置条件不满足时 driver 在同一会话内发起纠正跟进（至多 `role.corrective_attempts` 次），再失败则抛 `InvocationFailed`，由循环按角色升级。会话 id 在 init 时即持久化，被杀的会话可用 `resume=<session_id>` 恢复。

当前角色：`background-researcher`、`idea-generator`、`experience-extractor`、`candidate-writer`、`tunable-contract-extractor`、`tuner-orchestrator`、`crash-diagnosis`、`hillclimb-editor`。原 `autoresearch-experiment` 主代理的编排职责已确定性化为 `driver/loops/experiment.py`（hillclimb 基线为 `driver/loops/hillclimb.py` + `hillclimb-editor` 角色）。

### 5.1 driver 实验循环

`driver/loops/experiment.py` 管理一个完整的实验运行（一个 `task_name + tag + run_dir`）。

职责：
- 初始化新的 `runs/<task>/<tag>/`（先调用 `background-researcher`；若任务声明 provided entrypoint，则先登记并评估该基线，否则循环通过 `fresh` 自举）
- 写入 `run_metadata.json`（模型、SDK/CLI 版本、权限策略、prompt 哈希）；恢复运行时对漂移发出 `metadata_mismatch` 警告——记录并警告，绝不拒绝
- 按**轮次**推进循环：一代 ≤B 个想法经过 step 0+1，然后一次解耦深度调优步骤
- 依次调用 `idea-generator`（SELECT + IDEATE）、`candidate-writer`、`tunable-contract-extractor`、`tuner-orchestrator` 角色会话
- 用 `crash-diagnosis` 角色进行崩溃分析
- 无单独的候选运行或日志解析阶段；extractor/tuner 各自使用 `record-run` + `set-tuning` 记录分数
- 通过 `tools/ledger.py` 维护 `ledger.json` 和派生的 `loop_state.md`
- 拥有预算检查、升级链与崩溃恢复；持续直到硬停止条件（`blocked` 事件带具体原因）

### 5.2 background-researcher

证据感知的文献侦察员，**在设置期间必需一次**（在循环之前；唯一的设置子代理）。读取 `TASK.md` / `task.toml` / `prepare.py` 的候选可见接口，并先解析 `space_initialization.dimension_strategy`。默认 `catalog_subset` 从内置目录选取任务相关维度；`llm_induced` 则在检索文献前按需读取 `docs/dimension-induction.md`，直接生成最终的任务维度集 `<run_dir>/dimension_catalog.json`。维度集冻结后，声明的 provided entrypoint 只用于把各维度的 baseline hypothesis 对齐到具体基线，不反过来决定维度划分。随后才分解研究问题并检索证据。可复现主条件使用 pinned JSON corpus 的本地 `frozen` backend；DeepXiv 是显式选择的 open-world 学术条件，Jina 仅作为显式 live-web fallback/ablation。外部 backend 全部模块化且可选，失效或缺失只改变覆盖，不影响本地合约、去重、验证或 frozen replay。结果按 canonical URL / arXiv work 去重，以不同 query 的支持数排序，再轮询补齐各 query 的覆盖。随后按 grounding lane（6000 tokens）渐进阅读，并同时寻找反证、复现和官方工件。它总是产生 `<run_dir>/background.md` 与访问轨迹 `<run_dir>/background_retrieval.json`，在 `llm_induced` 下另加维度目录。

`background.md` 现在是 schema-3 的语义搜索空间：每个维度复制已解析目录的定义/边界/来源，登记显式任务基线与稳定 `hyp-*` 值；`catalog_subset` 可使用内置目录的子集，`llm_induced` 必须完整、按序使用 run-local 目录。`activates` / `requires` / `excludes` 关系表示条件激活与不兼容组合。标量设置仍属于内层 HPO。人类可读的 Dimension coverage / Dimensions / Relations 与 JSON 层级必须一致。

来源、结构化负面指导 `g-*` 与每个假设继续使用同一组五轴范围：模型家族、数据情境、指标、干预机制、评估协议。只有直接覆盖假设的指导可影响资格；`unverified` / `contested` 负面只能提示。每个非基线假设保存 claim、比较项、重开条件、来源关系与独立文献可信度标签。绑定负面仍必须保留范围外 `scope_probe`，不会删除邻近机制。

Search space registry 中的每个来源必须在 retrieval manifest 中存在成功且包含正文的 grounding visit（`section`、`preview`、`full_text` 或原样抓取的 `page`）；DeepXiv 的 `head` / `brief` 只用于筛选，不能作为证据。DeepXiv 的 `auto` 会先读取 `head`，再按检索问题选择并读取最多三个正文 section，必要时回退到 preview。只出现在搜索摘要或 novelty lane（2048 tokens）中不算访问。Claude 的原生 web 工具仍可作为本地 backend 全部失败时的 fallback，但成功访问必须通过 `record-visit` 写入同一 manifest。

冻结语料的约定路径是 `tasks/<task>/background_corpus.json`；普通 open-world 开发可不提供，但 frozen / network-disabled 评测必须提供该文件或显式等价路径。

层级语义空间、证据与作用域语义、fallback 和验证契约详见 `docs/background-research.md`。

- 假设是可复用的空间取值，不会在首次使用后“消耗”；每个候选保存完整、带版本的 `semantic_point`
- `source_run_ids` 只保存数字父代，假设归因、策略预测与分数观测分别存放
- 证据为基础（每个声明可追溯；无虚构论文）；对任务/账本只读；不运行实验
- schema-3 空间在 setup 后冻结；运行时剪枝只发生在账本 `search_space_state` 覆盖层，不修改冻结 registry；空间扩展留到 P4
- 模型由运行的 `run_metadata.json` 固定（新运行时 `--model` 指定）

### 5.3 idea-generator

外层搜索的 LLM 着陆点，通过三步产生下一代：

- **SELECT-1（图）**：`got_select.py decide` 确定 `fresh` / `improve` / `crossover` 与数字父代。**不通过目测适应度改选父代。**
- **SELECT-2（语义点）**：`semantic_search.py` 为该行动生成有界合法点集（按账本当前 `search_space_state` revision 过滤/排序）；新 run 默认应用 `coverage_attempt`，也可显式选择 `coverage` / `coverage_experience` / `coverage_carrier_attempt` / `gain` / `gain_uncertainty` / `gain_uncertainty_nocost`。gain 系列用 `[0,1]` rubric 先给出背景先验，再通过当前 bounded experience 的门控 adjustment 得到最终 predicted gain / uncertainty；自由文本经验不会进入 acquisition。schema-7 `policy_receipt` 固定 experience revision、目标与 proposal relation、比较覆盖、证据 id、机械 gain direction、配置的 LLM intelligence score 及实际权重；零调整始终合法，非零 gain 必须来自至少两个方向一致、同一份子代代码内只改变语义开关的 control/treatment 配对。仅继承父代超参数的 config 0 不足以隔离代码语义变化，只能增加 uncertainty，不能制造 signed gain。
- **IDEATE**：把选定点转成自包含的完整具体方案；用 `ledger.py add-record` 同时保存数字祖先、完整 `semantic_point` 与独立策略收据。映射是归因，不是完整代码规格；同一点可有不同实现。

替换旧的 `idea-proposer` skill 和固定的"一个 crossover + 一个 mutation"代数——行动计数和 op 混合由 `decide` 决定（PUCB 代产生 B 个行动；fresh 代每轮自举 1 个，stall 注入 B 个——fresh 计数折叠到 B 中，无单独的 m_fresh）。

### 5.4 experience-extractor

每个已完成的非空轮次后运行一次，从 DAG 增量、Top/Bottom 锚点、机械语义点差异中提炼有界全局经验（schema 3）。除通用 promising regions / lessons / bottlenecks 外，还生成双层 `dimension_evidence` / `hypothesis_evidence` 信念；其引用边 id、逐边观测、评估状态与比较计数只取自 `background_contract.py target-evidence`，绝不从 Top/Bottom 窗口重建。信念只"建议"运行时状态：`set-experience` 成功后调用一次 `ledger.py apply-space-state`，由确定性 helper 拥有所有 append-only `search_space_state` 转移（两阶段剪枝、基线保护、重开即追加）。不把点成员关系当因果，永不改写 background、映射、策略收据或原始观测。

### 5.5 candidate-writer

将账本记录的想法实现到候选目录的 `train.py` 中。**输入仅是候选目录**——它通过 `ledger.py show` 读取自己的账本记录以获取 `idea` + `source_run_ids`，派生父 `train.py` 引用，编写行为取决于文件系统状态，适应三种情况：

- 目标目录已有 `train.py`（提供的基线）：不写；按原样保留；返回 `wrote: false`
- `source_run_ids` 为空（`fresh`）：若任务声明了 provided baseline，先阅读它作为评估表面与文件约定的参考（不是父代、不是可编辑快照），再按记录的完整语义点从头编写；不得复制 baseline，也不得偷偷换成另一个更简单的点
- `source_run_ids` 是数字父 run ids（`improve`/`crossover`）：使用父代代码实现想法，同时保持与 `semantic_point` 一致

职责：
- 通过 `ledger.py show` 读取 idea + 数字父代 + semantic_point + policy_receipt
- 读取从 source_run_ids 派生的父 `train.py`
- `fresh` 时，若任务声明了 provided baseline，先阅读任务根上的该实现作为参考，再从头编写
- 读取只读 `prepare.py` 以理解评估 API
- 仅写目标候选的 `train.py`

不做：
- 提出想法
- 提取调优器合约（之后由 `tunable-contract-extractor` 完成）
- 运行实验
- 解析结果
- 修改 `prepare.py`

### 5.6 tunable-contract-extractor

Step 0+1：在候选 `train.py` 准备好后运行，在一个角色会话中完成所有事情：

① 行为保持地将构造逻辑重构为 `make_model(<task-input>, params)`（首个参数与返回对象的接口由任务的 Evaluation Contract 定义）并声明 `PARAM_SCHEMA`（仅列出可调参数 + 类型，无范围/默认值）
② provided entrypoint 仅使用一个原始默认配置；非 fresh 候选从 primary parent 的完整代码快照开始，并把其已应用 incumbent 精确投影为强制 warm config 0（记录 copied/reset/new/dropped、父代 durable applied snapshot 与 hash）；该控制保证 tuning win 可继承，但 receipt 明确标为 semantic `unverified`，不冒充语义因果比较；其余候选结合**血统证据**（`lineage-evidence`）与数据提出热启动配置及 `SEARCH_SPACE`
③ 评估所选配置（`warmstart_eval`；强制先保留 inherited control、顺序/可恢复/崩溃时停止）；**对每次崩溃调用 `crash-diagnosis` 角色**（config-invalid → 修复配置 / code-incompatible → 最小化修复代码 ≤10 次）直到通过 → 写 `BASE_PARAMS`=全部有限 warm row（包括 inherited control）中的最优项 + `phase_a`，并把完整 transfer/control/score receipt 写入 ledger；control 的 role 只约束语义归因，不剥夺优化资格；无法修复 → 记录 `status:crash`

主循环在 `candidate-writer` 返回后对每个新候选方案运行一次此操作。深度调优（step 2）被解耦；所有候选方案在此停在 step 0+1。

### 5.7 tuner-orchestrator

Step 2（解耦渐进式调优，设计 §15）：**每轮在整个运行上运行一次**，每次至多运行**一个调优 bout**，其 objective 评估数由 inner policy 决定（默认 `hebo24-hebo20` 为 24/10/10）。**无热启动**——`phase_a` 是 step 0+1（extractor）输出用作输入。

流程：

1. 选择候选方案：运行 `tools/tuners/tune_tools.py select-candidate`，并严格执行返回的 exact target 和 complete-bout 预算。默认 `anchor_challenger_v1` 按 early anchor → 保留硬预算继续生成 → late challenger → 两个 DEEP segment 的确定性赛程选择；`v3_2` 用全剩余预算 rollout 在 TUNE/DEFER 间决策。仅 legacy/legacy_wide 使用 `N_min` + `best_warm_score` 百分位门控与 FIRST/CONTINUE 交替规则。
2. Phase R：冻结策略下从不接受 orchestrator 再热提案（INITIAL 消耗 step 0+1 的 deferred 配置；HEBO bout 自生成 pool，`hebo_bout_has_no_rewarm`；SPSA DEEP bout 拒绝提案，`deep_bout_has_no_rewarm`——pair 必须完整）。仅 `tuner.inner_policy=legacy` 的 continuation action 仍走旧的至多 `tuner.rewarm_proposals`（默认 3）条提案路径
3. Phase C：按 bout regime 选方法（默认 `hebo24-hebo20`：INITIAL=hebo 24 槽，之后两个 DEEP=hebo 10 槽 segment；没有 CONTINUE regime。`deferred-random8-hebo10-spsa10-v1` 为历史 FIRST/CONTINUE/DEEP 三段的 bo+RandomSampler 8 / hebo 10 / spsa 10 对照臂；其他策略见 `tools/tuners/inner_policy.py`），以全部历史 trial 为先验续搜
4. Finalize：运行 `tools/finalize_tuning.py`；它验证当前 bout 的 Phase C 已终止，在 warm  incumbent 与**所有 bout 的全部 trial** 上取全局最佳、写回 `BASE_PARAMS`、关闭 report，并一次性更新 ledger（`tuning_bouts`、`last_bout_improved`、分级 `evaluation_depth`；**无重新运行**；可按 bout 安全重试）。若搜索进程被杀死或 report 非终态，则不应用参数且不更新 ledger。

当当前 scheduler 返回 DEFER/STOP，或 legacy 臂没有合格候选时，`run_id` 为 `none`——这是有效的无操作。

## 6. 崩溃诊断

原 `.claude/skills/` 已随 runtime 退役；`crash-diagnosis` 从 skill 迁移为 driver 角色（`driver/prompts/crash-diagnosis.md`，只读工具集：Read/Bash/Glob/Grep），由循环在候选 preflight 或客观评估崩溃时调用。三向判决：

- `config_invalid`：配置值本身无法挽救 → 修复配置
- `code_incompatible`：配置合理，代码不兼容 → **最小化修复代码**以适应（首选）
- `abandon`：需要编辑只读 / 添加禁止的依赖 / 根本不兼容 → 放弃

## 7. 工具

确定性 Python 脚本位于：

```text
tools/
```

关键工具：

### 7.1 new_candidate.py

创建新候选目录并复制任务声明的文件。

常用命令：

```bash
python tools/new_candidate.py <task-name> <tag> <run_id>
python tools/new_candidate.py <task-name> <tag> 000 --provided-baseline
python tools/new_candidate.py <task-name> <tag> <run_id> --skip-entrypoint
python tools/new_candidate.py <task-name> <tag> <run_id> --from-candidate <best_run_id>
```

- `--provided-baseline`：仅用于已登记的首个 all-baselines 记录；验证 `[seed].provided`，复制任务根 entrypoint，并在 `_candidate_brief.json` 保存来源路径与内容哈希
- 默认：复制 `prepare.py` 和任务根的 `train.py`（兼容性保留）
- `--skip-entrypoint`：要求对应账本记录已存在，复制 `prepare.py` 并从该记录生成精简 `_candidate_brief.json`；`train.py` 稍后由 candidate-writer 编写（`fresh` 和所有 `improve`/`crossover` 候选方案的标准）
- `--from-candidate`：从历史候选复制 `train.py`（兼容性保留；正常流程不再使用——`improve`/`crossover` 候选方案引用代码由 candidate-writer 自己从 `source_run_ids` 派生，而非预复制）

### 7.2 ledger.py

`ledger.json` 的唯一写入者，以及实验状态的结构化真相源（每个候选一个 JSON 记录，合并原始 `idea_log.md` 想法字段与 `results.tsv` 分数/状态）。**永不手动编辑**；仅通过子命令变更，确保长时间运行中的模式稳定性（见 `driver/prompts/rules/ledger.md`）：

```bash
python tools/ledger.py add-record      ...   # idea-generator 创建记录 (带 --op)
python tools/ledger.py set-tuning      ...   # extractor 填充 Phase-A 调优元数据（无 --mark-tuned）
python tools/finalize_tuning.py        ...   # 终态 Phase C 的唯一正常关闭路径：应用参数并统一更新 ledger
python tools/ledger.py set-experience --background <background.md> ...   # 校验后写全局 experience 块
python tools/ledger.py record-run      ...   # extractor 用 warm config-eval 最佳调用: 写 final_best_score + 计算 keep/discard/crash
python tools/ledger.py percentile      ...   # 按字段的跨记录百分位 (tuner 门控 / select-candidate 使用; 只读)
python tools/ledger.py evaluations     ...   # 预算检查: 跨记录的 Σ trials_attempted（旧记录回退到 trials_completed）
python tools/ledger.py loop-state      ...   # 从 ledger.json 重新生成 loop_state.md
python tools/ledger.py show            ...   # 读取单个记录或整个账本
```

每个记录携带 `op`；`kind` 始终为 `optimization`。`source_run_ids` 只含数字父代（fresh 为空）；`semantic_point` 保存对精确 background revision 的完整归因；`policy_receipt` 单独保存采集配置、证据与 gain/uncertainty/cost/coverage。首条记录还冻结顶层 `search_space` 收据。缺映射、旧平面记录或 revision 漂移都会被拒绝。MCTS 统计仍由 `got_graph.from_ledger` 重算。

`loop_state.md` 是 `ledger.json` 的派生视图，由 `ledger.py` 重新生成；不单独手动编辑。

当前 experiment run 的持久状态文件为：

```text
runs/<task-name>/<tag>/ledger.json
runs/<task-name>/<tag>/loop_state.md
```

根据 task.toml 指标判断改进（分数始终越小越好）。仅改进结果标记为 `keep`。

### 7.3 got_graph.py / got_cdag.py / got_select.py

外层 S-GoT 图搜索确定性计算层（纯函数；单元可测试）：

- `got_graph.py`：开发 DAG + 反向传播（`V_max`/`V_med`/`N`/`ec`，`cap=⌈C·N^α⌉` 渐进加宽，`select_leaf`，前沿 `F`/`alive`，血统）；`from_ledger(ledger)` 从记录重建图
- `got_cdag.py`：结构互补性 `c̃_dag`——开发 DAG 上的祖先影响扩散向量 + 余弦；计算余弦前排除两个目标节点各自的单位 self 分量，避免人为制造结构正交性
- `got_select.py`：SELECT 层。`idea-generator` 调用 `python tools/got_select.py decide --ledger <path>` 获取本轮的行动：bootstrap/stall fresh 规则，或在前沿叶子上按 op 解耦定额选 ≤B 个动作：op 级收购 `U_op=ḡ_op+c_pucb·√σN/(1+Nop_op)` → `W=softmax(U/τ)` → 最大余数法分配 B 个槽位（确定性，受可用动作数封顶），各 op 内按 `Q` 取顶（`improve: V_max`；`crossover: geomean(V)·(1+c̃_dag)`），两 op 不混排竞争同一排序。所有全局派生量从记录重新计算；无持久状态。

### 7.4 background_contract.py

`background.md` 与候选语义归因之间的确定性合约层：

- `catalog`：校验并输出内置目录或显式 `--path` 目录及内容摘要
- `validate`：检查 schema-3 层级、目录解析、显式基线、关系、五轴范围、来源/指导证据、人类视图以及所有 ledger 点/祖先/策略收据/语义边收据与 `search_space_state` 覆盖层
- `render`：输出有界的维度、假设、关系与覆盖视图
- `validate-point`：检查完整点、条件激活、requires 与 excludes
- `lineage`：并列呈现数字祖先与机械 point diff，不生成因果边
- `validate-experience`：校验 schema-3 快照——通用 belief 的 run-id 证据，以及双层 dimension/hypothesis 信念的引用边、评估状态与比较计数（按持久化收据机械重算，必须一致）
- `target-evidence`：从持久化语义边收据为每个 target 输出有界比较证据（引用边 id、逐边观测、评估状态、引用计数），是 experience-extractor 的权威有界来源

`python tools/validate_background.py` 使用合成 DAG 回归这些合约。

### 7.5 semantic_search.py

图行动之后的语义选点层：`propose` 为 fresh/improve/crossover 生成有界合法点（按账本当前 `search_space_state` revision 过滤/排序：运行时剪枝的假设出局，被剪维度钉在显式基线）；`select` 可替换 `coverage`、`gain`、`gain_uncertainty`、`gain_uncertainty_nocost` 策略并输出 point + policy receipt，提案集与收据都带 revision 戳，过期即拒。策略可替换而不改变 registry、祖先或观测历史。

### 7.6 search_backends.py

受 Arbor 检索层启发的轻量适配器，但不导入 Arbor runtime：

- 本地 frozen-corpus backend 是可复现条件；DeepXiv / Jina 必须显式启用且单个失败不阻塞其他结果
- canonical work/URL 去重，统计不同 query 与 backend 的支持数，并平衡每个 query 的候选覆盖
- `visit` 渐进阅读并自动写 visit receipt；runtime 原生 web fetch 可用 `record-visit` 接入
- 独立的 `grounding=6000` 与 `novelty=2048` token lane
- 保留 raw response、retrieval timestamp、backend/client version、corpus cutoff/hash 和访问内容 hash；token 永不进入运行产物
- `python tools/validate_search_backends.py` 提供完全离线的回归检查

### 7.7 validate_tasks.py

检查 `tasks/` 任务包结构和 `task.toml`。

```bash
python tools/validate_tasks.py
```

### 7.8 tools/tuners/

调优脚本：

- `_common.py`：共享加载、搜索空间、试验读取逻辑
- `tune_tools.py`：调优编排的确定性 CLI——`select-candidate`（解耦深度调优的候选选择门控：种群 ≥ N_min 且前 20% 中的最佳未调优候选）、`select-method`（基于维度的 grid/bo/cmaes 选择，即 legacy CONTINUE 规则）、`select-best`（全局最佳试验）、`lineage-evidence`、`check-search-space`（校验 + 按 outlier/贴边 margin 扩箱）等。
- `inner_policy.py`：regime 条件内层策略（当前 `hebo24-hebo20` 为 INITIAL/DEEP 二分、24+10+10 全 HEBO；历史对照臂保留 FIRST/CONTINUE/DEEP）——bout 大小、方法链、sampler 与 rewarm 规则的唯一来源。
- `warmstart_eval.py`：顺序评估热配置（恢复 / 崩溃时停止），构建 `BASE_PARAMS` + 写 `phase_a`
- `grid_search.py`：低维搜索空间
- `bo_search.py`：使用贝叶斯优化（通过 Optuna 的多元 TPE）的中维搜索空间；`--sampler random` 时为 FIRST bout 的显式 RandomSampler 内核
- `hebo_search.py`：prompt-v2 HEBO（当前 policy 的 INITIAL/DEEP，以及历史对照臂的 CONTINUE/HEBO-DEEP；LLM pool + 官方 HEBO MACE；在仓库根环境跑搜索，评估仍走任务 uv 项目）
- `spsa_search.py`：DEEP bout 的两侧 SPSA（5 个完整扰动 pair，pair 状态持久化可精确续跑）
- `local_tr_search.py`：对照臂 `localtr8-hebo10-spsa10-v1` / `localtr8-hebo10-hebo10-v1` 的 FIRST bout（inner-benchmark `local_tr`；deferred 热配置占 8 槽内的前几个）
- `cmaes_search.py`：使用 CMA-ES 的高维搜索空间
- `../finalize_tuning.py`：验证 Phase C 终态并以 fail-closed 方式统一应用全局最佳、关闭报告和 ledger

## 8. 任务

每个任务是独立的 uv 项目。不要将 `tasks/*` 配置为 uv workspace。

标准任务结构：

```text
tasks/<task-name>/
  TASK.md
  task.toml
  pyproject.toml
  uv.lock
  prepare.py
  train.py                  可选的用户提供基线
```

### 8.1 autoresearch-baseline

```text
tasks/autoresearch-baseline/
```

保留的原始 autoresearch 任务。面向 Torch/GPU 的基线，指标为 `val_bpb`（越小越好）。

### 8.2 tabular-model-search

```text
tasks/tabular-model-search/
```

CPU 表格分类模型搜索任务。指标是 `neg_mean_balanced_accuracy`（越小越好；平衡准确率取负；框架在所有地方最小化）。

用于验证：

- 候选目录机制
- 外层 S-GoT 图搜索
- 内层调优器超参数搜索（解耦深度调优）
- 多模型路线比较
- `keep` / `discard` 结果管理

### 8.3 es-optimization-design

```text
tasks/es-optimization-design/
```

CPU 黑盒连续优化器设计任务，用于契约形状多样化：候选不是 estimator，而是在固定评估预算（`budget = 2000 × dim`）内通过 `problem.evaluate` 最小化未知函数（rastrigin/rosenbrock/ackley/schwefel × dim {2,10,30} × 3 种子）的迭代优化算法。指标为 `mean_log10_1p_best_fitness`（越小越好）。

签名按任务自定义（底层 tuner 本就支持）：`make_model(problem, params)` 返回带 `run() -> float` 的优化器对象；`prepare.evaluate_config` 内部跑完全部 36 个问题并返回均值。

## 9. 候选方案合约

当候选方案需要调优时，`train.py` 应暴露：

```python
BASE_PARAMS = {...}
SEARCH_SPACE = {...}

def make_model(<task-input>, params):
    ...
```

`make_model` 这个符号名和 `params` 字典是框架级约定；**首个参数的名称/形状与返回对象的接口由任务的 Evaluation Contract 定义**——tabular 任务为 `make_model(dataset, params)` 返回 sklearn 风格 estimator，`es-optimization-design` 为 `make_model(problem, params)` 返回带 `run() -> float` 的优化器对象。

官方运行应使用（以 tabular 任务为例）：

```python
model = make_model(dataset, BASE_PARAMS)
```

`prepare.py` 处理固定的问题实例和评估函数。候选的 `train.py` 仅通过 `prepare.py` 暴露的接口访问问题、构造候选、计算分数。

## 10. 新开发者的交接阅读顺序

推荐的阅读顺序：

1. **`README.md`**：理解整体结构
2. **`CLAUDE.md`**：理解 runtime 如何进入项目
3. **`driver/loops/*.py` 和 `driver/prompts/*.md`**：理解循环编排与角色责任边界
4. **`tools/*.py` 和 `tools/tuners/*.py`**：理解确定性执行层
5. **`tasks/tabular-model-search/`**：理解当前主要验证任务

## 11. 继续开发

### 11.1 添加任务

1. 在 `tasks/<task-name>/` 下创建任务目录
2. 添加 `TASK.md`、`task.toml`、`pyproject.toml`、`prepare.py`。可选地添加 `train.py` 如果提供用户基线（作为预设 `fresh` 方向）；可以省略——循环通过 `fresh` 候选方案自举，candidate-writer 从头生成每个 `train.py`
   - `TASK.md` 必须包含 `## Evaluation Contract` 部分（验证器检查）：声明候选训练表面、官方评分表面、报告表面、调优器评估表面（如果有）和行为规则。此合约指导 candidate-writer / tunable-contract-extractor / tuner-orchestrator；更具体 = 自主实验期间更少漂移
3. 在 `task.toml` 中声明：
   - 指标
   - `[evaluation]` 评估函数名：`score_fn`（官方评分函数），对于支持调优的任务，单一 `config → score` 测试函数（`[evaluation].score_fn`；官方 = 调优表面；签名 `score_fn(make_model, params)`；热启动评估和 Phase C 都调用它；没有单独的官方运行）。函数名放在配置中；相应的语义在 TASK.md Evaluation Contract 中
   - uv 环境和超时
   - 如果覆盖默认候选目录模板或需要复制文件，声明 `[candidate]`
   - 可编辑文件
   - 只读文件
   - 是否允许添加依赖
4. 运行：

```bash
uv --directory tasks/<task-name> sync
python tools/validate_tasks.py
```

### 11.2 添加角色

1. 创建 `driver/prompts/<role-name>.md` 并在 `driver/roles.py` 注册：prompt 文件、正向工具能力集、receipt schema、后置条件
2. 使用小写连字符命名
3. 明确其输入、输出、边界和禁止的行动；保持角色范围集中
4. prompt 只承载生成与判断；确定性逻辑写到 `tools/`，排序写到 `driver/loops/`，而不是仅自然语言流程

### 11.3 添加工具

1. 放在 `tools/` 或 `tools/<domain>/`
2. 尽可能让工具从 `task.toml` 读取配置，而不是硬编码特定任务
3. 工具应可从 repo 根运行
4. 将可复用逻辑提取到公共模块，如 `tools/tuners/_common.py`

## 12. 框架配置覆盖

`framework_cfg.json` 为单次运行提供图搜索、语义选点和调优器参数的覆盖配置。放置在运行目录中：

```text
runs/<task>/<tag>/framework_cfg.json
```

**用途**：在不修改代码默认值的情况下，为特定实验微调框架行为。适用于：
- 消融研究（例如，改变 bootstrap 大小、调优预算）
- 任务特定的预算约束（例如，最大评估次数、单次评估时间限制）
- 调整探索与利用的权衡

**使用方法**：正常情况下直接用 `driver run` 的 CLI 标志，不需要手改该文件。新运行也可用 `python tools/init_run.py <task> <tag> --dimension-strategy llm_induced --llm-intelligence-score 61 --semantic-policy coverage_attempt --scheduler-policy anchor_challenger_v1 --inner-tuner-policy hebo24-hebo20 --max-evaluations 200 --timeout 60` 单独初始化；恢复已有运行时可更新预算和超时，但策略、维度来源和 intelligence score 在相关产物生成后被冻结。配置文件仍可用于没有 CLI 暴露的研究参数。

主要配置包括：
- **`got.*`**：外层 S-GoT 图搜索参数（bootstrap 大小、PUCB 批次大小、停滞阈值、渐进加宽等）
- **`space_initialization.dimension_strategy`**：维度来源；默认 `catalog_subset` 使用内置目录，`llm_induced` 让 background researcher 在检索前生成并完整采用通过验证的 `dimension_catalog.json`
- **`semantic_search.*`**：语义点策略及 gain / uncertainty / cost / coverage 权重；新 run 默认使用 `coverage_attempt`。`coverage`、`coverage_experience` 等对照臂可通过 `--semantic-policy` 显式选择
- **`tuner.scheduler_policy`**：新 experiment run 默认 `anchor_challenger_v1`；`v3_2` 与旧 percentile/alternation 调度器均通过 `--scheduler-policy` 显式选择
- **`tuner.inner_policy`**：新 experiment run 默认 `hebo24-hebo20`（24+10+10 三段全程使用 LLM pool + official HEBO MACE）；其他对照臂可通过 `--inner-tuner-policy` 显式选择，包括 `hebo24-turbo20-v1`、`mixup24-turbo20-v1`、`deferred-random8-hebo10-spsa10-v1`、`localtr8-hebo10-spsa10-v1`、`localtr8-hebo10-hebo10-v1`、`selfrank8-hebo10-hebo10` 和 `legacy`
- **`tuner.*`**：内层 HPO 调优器参数（热启动配置数量、深度调优门控阈值、BO 试验预算、patience 等）。其中 `tuner.K`（每个候选提出的热启动配置数，默认 5）和 `tuner.K_eval`（step 0+1 实际评估的条数，默认 3）分别通过 `--k-warm` / `--k-eval` 暴露；deferred 配置数 = K − K_eval，两者都在 run 产物生成后冻结
- **`max_evaluations`**：全局停止预算（所有候选方案的试验总和）
- **`per_runtime_limit`**：单次评估超时（秒）（超时配置被强制终止）

**优先级**：显式 CLI 标志 > `framework_cfg.json` > 代码默认值

该文件被 git 忽略（`runs/` 仅本地）。完整的带注释参考（所有可用键及其语义）见 `tasks/framework_cfg.example.json`。
