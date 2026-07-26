# autoresearch-automl

由 Claude Code 或 OpenCode 驱动的自主 AutoML 实验框架。LLM runtime 负责角色隔离与判断，确定性的状态、图搜索、评估和调优由共享的 `tools/` 实现。

## 1. 框架功能

该项目从单任务脚本演化为多任务实验框架，采用两层搜索架构：

- **外层搜索（S-GoT）**：候选方案形成开发 DAG。确定性图搜索（PUCB + 结构互补性 `c̃_dag` + bootstrap/stall fresh 规则）决定每轮采取的行动（`fresh` / `improve` / `crossover`）。LLM 智能体将每个行动转化为具体想法。
- **内层搜索（解耦调优）**：超参数调优器在每个候选方案的结构内搜索，但**与外层搜索解耦**。框架不会内联调优每个候选方案，而是每轮选择一个有前景的候选方案进行深度调优。
- 每个候选方案位于独立目录；过往结果永不覆盖。
- 每个任务是独立的 uv 项目，拥有各自的依赖。

## 2. 项目结构概览

框架包含：

- **`CLAUDE.md`**：Claude Code 打开项目时首先读取的入口文档。
- **`.claude/agents/`**：用于复杂多步骤任务的专用子智能体（想法生成、代码编写、合约提取、调优编排）。
- **`.claude/skills/`**：可复用的内联方法论（当前：崩溃诊断）。
- **`AGENTS.md`**：OpenCode 打开项目时读取的精简 runtime 入口。
- **`.opencode/agents/` / `.opencode/skills/`**：OpenCode 的主代理、子代理和内联方法论。
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

推荐用 OpenCode 启动完整实验：

```bash
opencode --agent autoresearch-experiment \
  --model moonshotai/kimi-k3 --auto
```

也可以使用 Claude Code：

```bash
claude --agent autoresearch-experiment
```

然后提供：

```text
task_name: tabular-model-search
tag: <你的运行标签>
```

对于并行实验，使用不同的 `tag` 值启动多个终端会话。

## 4. 框架工作原理

### 4.1 核心文档

**`CLAUDE.md` / `AGENTS.md`**：分别是 Claude Code 与 OpenCode 的项目入口。完整实验协议编码在对应 runtime 的 `autoresearch-experiment` 主代理中；任务语义来自 `TASK.md` 和 `task.toml`。

**`tasks/<task-name>/TASK.md`**：供人和 LLM 阅读的任务描述。包含 `## Evaluation Contract` 部分，描述：
- 训练表面：候选方案在训练期间做什么
- 官方评分表面：候选方案如何产生分数
- 报告表面：候选方案记录什么
- 调优器评估表面：单一 `config → score` 函数（如果支持调优）
- 行为规则

每个子智能体在工作前读取此合约。主循环在每轮开始时重新读取以避免长时间运行中的漂移。

**`tasks/<task-name>/task.toml`**：机器可读配置。包含：
- uv 环境和超时
- `[evaluation].score_fn`：单一 `config → score` 函数名（官方 = 调优表面；warm-start 和 Phase C 都调用它；没有单独的官方运行）
- 指标字段名
- 可编辑/只读文件约束
- 依赖规则
- 可选的 `[candidate]` 覆盖（仅在需要非默认目录结构时）

**拆分原则**：函数*名称*放在 toml（配置）；函数*语义*放在 TASK.md（散文描述）。

**`tasks/<task-name>/prepare.py`**：固定评估表面。正常实验期间不修改。

**`tasks/<task-name>/train.py`**：可选的用户提供基线。多数任务省略它——循环通过 `fresh` 自举，`candidate-writer` 在 `runs/` 下按已验证的完整语义点从头生成 `train.py`。

### 4.2 实验流程

手动通过主会话运行实验时，**一轮 = ① 一代（≤B 个想法，每个经过 step 0+1）+ ② 一次解耦深度调优步骤**：

```text
读取主代理 prompt / task.toml / TASK.md
        ↓
background-researcher: 解析维度策略 → 多后端知识侦察 → background.md + background_retrieval.json
    (llm_induced 额外先生成 dimension_catalog.json；随后冻结显式基线、hyp-* 与关系)
        ↓
(每 N 轮) experience-extractor: 提炼全局经验 → ledger.json experience 块
        ↓
idea-generator:
    SELECT-1: got_select.py decide → 获取图行动与数字父代
            (bootstrap/stall → fresh; 否则在前沿上 PUCB → ≤B improve/crossover)
    SELECT-2: semantic_search.py → 合法点集 → coverage/gain/gain_uncertainty/gain_uncertainty_nocost 选点
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
       → 对每次崩溃内联 crash-diagnosis skill(修复配置或修复代码)
       → 构建 BASE_PARAMS, 记录 best_warm_score
       → extractor 调用 record-run (final_best_score=best_warm_score + keep/discard status)
         + set-tuning (元数据, 无 --mark-tuned) 到 ledger.json
        ↓
(每轮一次) tuner-orchestrator:
    tune_tools.py select-candidate → 从整个种群中选择一个符合条件的候选方案
    → Phase C 搜索 + Apply → 原地写回 BASE_PARAMS
    → tuner 调用 record-run + set-tuning --mark-tuned 原地更新分数 (无重新运行)
```

使用 `opencode --agent autoresearch-experiment` 或 `claude --agent autoresearch-experiment` 时，主代理内部编码此协议。OpenCode 通过原生 `permission.task` 将调用闭包限制为六个角色；崩溃诊断使用 `crash-diagnosis` skill 内联。

### 4.3 两层搜索架构

**外层搜索（图行动 + 语义选点）**：候选方案形成开发 DAG。`got_select.py decide` 使用 PUCB + 结构互补性 `c̃_dag` + bootstrap/stall 规则，只决定行动与父代：
- 空图或停滞 → `fresh`
- 否则 → 在前沿叶子上 PUCB → ≤B `improve`（单亲）/ `crossover`（多亲）

随后 `semantic_search.py` 在冻结的层级空间中生成有界合法点：`coverage` 是无需 LLM 打分的确定性探索基线；`gain` 与 `gain_uncertainty` 将预期收益、不确定性、成本、覆盖分别保存并组合；`gain_uncertainty_nocost` 与 `gain_uncertainty` 相同但不预测成本（实现前的成本估计通常是噪声）。LLM 再把选定点落成完整方案。图策略与语义采集策略互不混写。

**内层搜索（解耦调优）**：每个候选方案结构内的超参数搜索，分为两个阶段，**与外层搜索解耦**：

- **Step 0+1**（tunable-contract-extractor；一个子智能体完成；对每个候选方案运行）：
  ① 行为保持地重构构造逻辑为 `make_model(<task-input>, params)`（首个参数与返回对象的接口由任务的 Evaluation Contract 定义）+ 声明 `PARAM_SCHEMA`
  ② 结合**血统证据**与数据提出 K=5 个热启动配置 + 数据驱动的 `SEARCH_SPACE`；一致性预检 + `check-search-space` + `apply_search_space`
  ③ **评估 K 个配置**（warmstart_eval；顺序/可恢复）；**对每次崩溃内联 crash-diagnosis skill**（config-invalid → 修复配置；code-incompatible → 最小化修复代码 ≤10 次）；全部通过 → 写 `BASE_PARAMS`=最佳-K′ + `phase_a`，记录 `best_warm_score`；无法修复 → 记录 `status:crash`
  深度调优（step 2）被解耦；所有候选方案在此停在 step 0+1。

- **Step 2（解耦深度调优）**（tuner-orchestrator；**每轮在整个运行上运行一次**，而非每个候选方案）：
  - 选择候选方案：运行 `tools/tuners/tune_tools.py select-candidate`——门控：种群 ≥ N_min=10 且按 `best_warm_score` 的最佳未调优候选方案在前 20% —— 选择**一个**候选方案；不足 → 返回 `none`（有效的无操作）
  - Phase C：基于维度的方法选择（grid n_dims≤2 / bo=多元 TPE ≥3；cmaes 仅作后备；见 HPO 基准 `dev_plan/hpo-benchmark-report.md`）；注入 step 1 热启动试验作为先验
  - Apply：从热启动 + Phase C 试验的全局最佳（select-best）→ 用 `apply_base_params` 原地写回 BASE_PARAMS；tuner **自己**调用 `record-run` + `set-tuning --mark-tuned` 原地更新 `final_best_score`（**无重新运行**——相同的 `config→score` 函数，select-best 保证无回归）

**关键洞察**：没有单独的官方运行。有**一个全局 `config → score` 函数**（task.toml `[evaluation].score_fn`）。热启动评估和 Phase C 调优都调用它。调优后的最佳值就是候选方案的新分数。

## 5. 智能体

智能体在两个 runtime 中保持相同的职责边界：

```text
.claude/agents/
.opencode/agents/
```

当前智能体：`autoresearch-experiment`、`background-researcher`、`idea-generator`、`experience-extractor`、`candidate-writer`、`tunable-contract-extractor`、`tuner-orchestrator`。

**重要约束**：`autoresearch-experiment` 必须作为 primary 启动，不能作为子代理嵌套。OpenCode 子代理均为 `task: deny`，`candidate-writer` 另外为 `bash: deny`；主代理只允许调用六个项目角色。

### 5.1 autoresearch-experiment

管理一个完整的实验运行（一个 `task_name + tag + run_dir`）。

职责：
- 完全自包含地执行完整实验协议，不依赖 `program.md`
- 初始化新的 `runs/<task>/<tag>/`（设置 = 仅 `background-researcher`；无种子阶段——循环通过 `fresh` 候选方案自举）
- 按**轮次**推进循环：一代 ≤B 个想法经过 step 0+1，然后一次解耦深度调优步骤
- 调用 `idea-generator` 用于 SELECT + IDEATE
- 调用 `candidate-writer` 编写候选代码
- 调用 `tunable-contract-extractor` 提取调优器合约并执行 step 0+1
- 调用 `tuner-orchestrator` 用于每轮一次的解耦深度调优
- 内联使用 `crash-diagnosis` skill 进行崩溃分析
- 无单独的候选运行；extractor/tuner 各自使用 `record-run` + `set-tuning` 记录分数（无 `parse_result`）
- 通过 `tools/ledger.py` 维护 `ledger.json` 和派生的 `loop_state.md`
- 持续直到硬停止条件

不做：
- 同时管理多个实验
- 作为无法继续调度角色子代理的嵌套子智能体运行
- 内联执行 `idea-generator` / `candidate-writer` / `tunable-contract-extractor` / `tuner-orchestrator` 工作（`crash-diagnosis` skill 是例外——它是内联的）
- 修改其他运行目录

### 5.2 background-researcher

证据感知的文献侦察员，**在设置期间必需一次**（在循环之前；唯一的设置步骤）。读取 `TASK.md` / `task.toml` / `prepare.py` 的候选可见接口，并先解析 `space_initialization.dimension_strategy`。默认 `catalog_subset` 从内置目录选取任务相关维度；`llm_induced` 则在检索文献前按需读取 `docs/dimension-induction.md`，直接生成最终的任务维度集 `<run_dir>/dimension_catalog.json`。随后才分解研究问题并检索证据。可复现主条件使用 pinned JSON corpus 的本地 `frozen` backend；DeepXiv 是显式选择的 open-world 学术条件，Jina 仅作为显式 live-web fallback/ablation。外部 backend 全部模块化且可选，失效或缺失只改变覆盖，不影响本地合约、去重、验证或 frozen replay。结果按 canonical URL / arXiv work 去重，以不同 query 的支持数排序，再轮询补齐各 query 的覆盖。随后按 grounding lane（6000 tokens）渐进阅读，并同时寻找反证、复现和官方工件。它总是产生 `<run_dir>/background.md` 与访问轨迹 `<run_dir>/background_retrieval.json`，在 `llm_induced` 下另加维度目录。

`background.md` 现在是 schema-3 的语义搜索空间：每个维度复制已解析目录的定义/边界/来源，登记显式任务基线与稳定 `hyp-*` 值；`catalog_subset` 可使用内置目录的子集，`llm_induced` 必须完整、按序使用 run-local 目录。`activates` / `requires` / `excludes` 关系表示条件激活与不兼容组合。标量设置仍属于内层 HPO。人类可读的 Dimension coverage / Dimensions / Relations 与 JSON 层级必须一致。

来源、结构化负面指导 `g-*` 与每个假设继续使用同一组五轴范围：模型家族、数据情境、指标、干预机制、评估协议。只有直接覆盖假设的指导可影响资格；`unverified` / `contested` 负面只能提示。每个非基线假设保存 claim、比较项、重开条件、来源关系与独立文献可信度标签。绑定负面仍必须保留范围外 `scope_probe`，不会删除邻近机制。

Search space registry 中的每个来源必须在 retrieval manifest 中存在成功且包含正文的 grounding visit（`section`、`preview`、`full_text` 或原样抓取的 `page`）；DeepXiv 的 `head` / `brief` 只用于筛选，不能作为证据。DeepXiv 的 `auto` 会先读取 `head`，再按检索问题选择并读取最多三个正文 section，必要时回退到 preview。只出现在搜索摘要或 novelty lane（2048 tokens）中不算访问。Claude 或 OpenCode 的原生 web 工具仍可作为本地 backend 全部失败时的 fallback，但成功访问必须通过 `record-visit` 写入同一 manifest。

冻结语料的约定路径是 `tasks/<task>/background_corpus.json`；普通 open-world 开发可不提供，但 frozen / network-disabled 评测必须提供该文件或显式等价路径。

层级语义空间、证据与作用域语义、fallback 和验证契约详见 `docs/background-research.md`。

- 假设是可复用的空间取值，不会在首次使用后“消耗”；每个候选保存完整、带版本的 `semantic_point`
- `source_run_ids` 只保存数字父代，假设归因、策略预测与分数观测分别存放
- 证据为基础（每个声明可追溯；无虚构论文）；对任务/账本只读；不运行实验
- schema-3 空间在 setup 后冻结；运行时剪枝只发生在账本 `search_space_state` 覆盖层，不修改冻结 registry；空间扩展留到 P4
- `model: inherit`（继承调用会话的模型配置）

### 5.3 idea-generator

外层搜索的 LLM 着陆点，通过三步产生下一代：

- **SELECT-1（图）**：`got_select.py decide` 确定 `fresh` / `improve` / `crossover` 与数字父代。**不通过目测适应度改选父代。**
- **SELECT-2（语义点）**：`semantic_search.py` 为该行动生成有界合法点集（按账本当前 `search_space_state` revision 过滤/排序）；根据配置应用 `coverage` / `gain` / `gain_uncertainty` / `gain_uncertainty_nocost`。后三者用 `[0,1]` rubric 分别评估 predicted gain、uncertainty、cost（`gain_uncertainty_nocost` 不含 cost），并与确定性 coverage 分栏写入 `policy_receipt`，不冒充校准后验。
- **IDEATE**：把选定点转成自包含的完整具体方案；用 `ledger.py add-record` 同时保存数字祖先、完整 `semantic_point` 与独立策略收据。映射是归因，不是完整代码规格；同一点可有不同实现。

替换旧的 `idea-proposer` skill 和固定的"一个 crossover + 一个 mutation"代数——行动计数和 op 混合由 `decide` 决定（PUCB 代产生 B 个行动；fresh 代每轮自举 1 个，stall 注入 B 个——fresh 计数折叠到 B 中，无单独的 m_fresh）。

### 5.4 experience-extractor

每 N 轮运行一次，从 DAG 增量、Top/Bottom 锚点、机械语义点差异中提炼有界全局经验（schema 3）。除通用 promising regions / lessons / bottlenecks 外，还生成双层 `dimension_evidence` / `hypothesis_evidence` 信念；其引用边 id、逐边观测、评估状态与比较计数只取自 `background_contract.py target-evidence`，绝不从 Top/Bottom 窗口重建。信念只"建议"运行时状态：`set-experience` 成功后调用一次 `ledger.py apply-space-state`，由确定性 helper 拥有所有 append-only `search_space_state` 转移（两阶段剪枝、基线保护、重开即追加）。不把点成员关系当因果，永不改写 background、映射、策略收据或原始观测。

### 5.5 candidate-writer

将账本记录的想法实现到候选目录的 `train.py` 中。**输入仅是候选目录**——它通过 `ledger.py show` 读取自己的账本记录以获取 `idea` + `source_run_ids`，派生父 `train.py` 引用，编写行为取决于文件系统状态，适应三种情况：

- 目标目录已有 `train.py`（提供的基线）：不写；按原样保留；返回 `wrote: false`
- `source_run_ids` 为空（`fresh`）：按记录的完整语义点从头编写，不得偷偷换成另一个更简单的点
- `source_run_ids` 是数字父 run ids（`improve`/`crossover`）：使用父代代码实现想法，同时保持与 `semantic_point` 一致

职责：
- 通过 `ledger.py show` 读取 idea + 数字父代 + semantic_point + policy_receipt
- 读取从 source_run_ids 派生的父 `train.py`
- 读取只读 `prepare.py` 以理解评估 API
- 仅写目标候选的 `train.py`

不做：
- 提出想法
- 提取调优器合约（之后由 `tunable-contract-extractor` 完成）
- 运行实验
- 解析结果
- 修改 `prepare.py`

### 5.6 tunable-contract-extractor

Step 0+1：在候选 `train.py` 准备好后运行，在一个子智能体中完成所有事情：

① 行为保持地将构造逻辑重构为 `make_model(<task-input>, params)`（首个参数与返回对象的接口由任务的 Evaluation Contract 定义）并声明 `PARAM_SCHEMA`（仅列出可调参数 + 类型，无范围/默认值）
② 结合**血统证据**（`lineage-evidence`）与数据一次性提出 K=5 个热启动配置 + 一个数据驱动的 `SEARCH_SPACE`，自运行 `check-search-space`（扩展边界以包含配置）+ `apply_search_space` 写回
③ 评估这 K 个配置（`warmstart_eval`；顺序/可恢复/崩溃时停止）；**对每次崩溃内联调用 `crash-diagnosis` skill**（config-invalid → 修复配置 / code-incompatible → 最小化修复代码 ≤10 次）直到全部通过 → 写 `BASE_PARAMS`=最佳-K′ + `phase_a`，记录 `best_warm_score`；无法修复 → 记录 `status:crash`

主循环在 `candidate-writer` 返回后对每个新候选方案运行一次此操作。深度调优（step 2）被解耦；所有候选方案在此停在 step 0+1。

### 5.7 tuner-orchestrator

Step 2（解耦深度调优，设计 §15）：**每轮在整个运行上运行一次**，而非每个候选方案。**无热启动**——`phase_a`（热配置 + `best_warm_score`）是 step 0+1（extractor）输出用作输入。

流程：

1. 选择候选方案：运行 `tools/tuners/tune_tools.py select-candidate`——门控：种群 ≥ `N_min=10` 且按 `best_warm_score` 的最佳未调优候选方案在前 20%（贪婪选择最佳未调优）→ 选择**一个**候选方案
2. Phase C：所选候选方案的基于维度的方法选择（`grid`/`bo`/`cmaes`），使用 step 1 热试验作为先验
3. Apply：从热试验 + Phase C 试验的全局最佳（select-best）→ 用 `apply_base_params` 原地写回 `BASE_PARAMS`，生成 `tune_report.json`；tuner 自己调用 `record-run` + `set-tuning --mark-tuned` 原地更新 `final_best_score`（**无重新运行**；原地回填）

资格不足（种群太小或顶层已调优）返回 `none`——有效的无操作。

## 6. Skills

Skills 是可复用的方法论描述，位于：

```text
.claude/skills/
```

当前：1 个 skill（`idea-proposer`、`hyperparam-tuner-llm`、`task-initializer` 全部退役——前两个合并到智能体；种子阶段溶解到 `fresh` 自举）：

### 6.1 crash-diagnosis

诊断一个候选崩溃并决定恢复方法的方法论，**由运行候选的上下文内联遵循**（非单独智能体）——主要是 `tunable-contract-extractor` 在 eval-K 崩溃期间（无单独的官方运行，因此主线程不再有官方运行崩溃路径），因为子智能体无法生成诊断子智能体。三向判决：

- `config_invalid`：配置值本身无法挽救 → 修复配置
- `code_incompatible`：配置合理，代码不兼容 → **最小化修复代码**以适应（首选）
- `abandon`：需要编辑只读 / 添加禁止的依赖 / 根本不兼容 → 放弃

> 注：外层想法生成从 `idea-proposer` skill 改为两个**智能体**（见上面的智能体部分）：`idea-generator`（S-GoT：每代首先调用 `got_select.py decide` 进行 SELECT，然后 IDEATE 每个行动）和 `experience-extractor`（每 N 轮将全局经验提炼到 ledger.json experience）。智能体转换通过用持久账本/经验替换对话上下文依赖实现。原始 `task-initializer` skill 也退役——无单独的种子阶段；循环通过 `fresh` 候选方案自举。

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
python tools/new_candidate.py <task-name> <tag> <run_id> --skip-entrypoint
python tools/new_candidate.py <task-name> <tag> <run_id> --from-candidate <best_run_id>
```

- 默认：复制 `prepare.py` 和任务根的 `train.py`（提供的基线）
- `--skip-entrypoint`：要求对应账本记录已存在，复制 `prepare.py` 并从该记录生成精简 `_candidate_brief.json`；`train.py` 稍后由 candidate-writer 编写（`fresh` 和所有 `improve`/`crossover` 候选方案的标准）
- `--from-candidate`：从历史候选复制 `train.py`（兼容性保留；正常流程不再使用——`improve`/`crossover` 候选方案引用代码由 candidate-writer 自己从 `source_run_ids` 派生，而非预复制）

### 7.2 ledger.py

`ledger.json` 的唯一写入者，以及实验状态的结构化真相源（每个候选一个 JSON 记录，合并原始 `idea_log.md` 想法字段与 `results.tsv` 分数/状态）。**永不手动编辑**；仅通过子命令变更，确保长时间运行中的模式稳定性（见 `.claude/rules/ledger.md`）：

```bash
python tools/ledger.py add-record      ...   # idea-generator 创建记录 (带 --op)
python tools/ledger.py set-tuning      ...   # 填充调优元数据 (extractor 无标记 / tuner --mark-tuned 设置 tune:true)
python tools/ledger.py set-experience --background <background.md> ...   # 校验后写全局 experience 块
python tools/ledger.py record-run      ...   # extractor/tuner 用 config-eval 最佳调用: 写 final_best_score + 计算 keep/discard/crash
python tools/ledger.py percentile      ...   # 按字段的跨记录百分位 (tuner 门控 / select-candidate 使用; 只读)
python tools/ledger.py evaluations     ...   # 预算检查: 跨记录的 Σ trials_completed
python tools/ledger.py loop-state      ...   # 从 ledger.json 重新生成 loop_state.md
python tools/ledger.py show            ...   # 读取单个记录或整个账本
```

每个记录携带 `op`；`kind` 始终为 `optimization`。`source_run_ids` 只含数字父代（fresh 为空）；`semantic_point` 保存对精确 background revision 的完整归因；`policy_receipt` 单独保存采集配置、证据与 gain/uncertainty/cost/coverage。首条记录还冻结顶层 `search_space` 收据。缺映射、旧平面记录或 revision 漂移都会被拒绝。MCTS 统计仍由 `got_graph.from_ledger` 重算。

`loop_state.md` 是 `ledger.json` 的派生视图，由 `ledger.py` 重新生成；不单独手动编辑。

### 7.3 parse_result.py（遗留/未使用）

S-GoT 单函数模型没有运行日志——分数通过 extractor/tuner `record-run` 直接写入。**此脚本不再被循环调用**（仅保留以满足 `validate_tasks` 对 `result.parser` 文件存在的检查）。原始行为：解析运行日志，然后通过 `ledger.py` 写入：

```text
runs/<task-name>/<tag>/ledger.json
runs/<task-name>/<tag>/loop_state.md
```

根据 task.toml 指标判断改进（分数始终越小越好）。仅改进结果标记为 `keep`。

### 7.4 got_graph.py / got_cdag.py / got_select.py

外层 S-GoT 图搜索确定性计算层（纯函数；单元可测试）：

- `got_graph.py`：开发 DAG + 反向传播（`V_max`/`V_med`/`N`/`ec`，`cap=⌈C·N^α⌉` 渐进加宽，`select_leaf`，前沿 `F`/`alive`，血统）；`from_ledger(ledger)` 从记录重建图
- `got_cdag.py`：结构互补性 `c̃_dag`——开发 DAG 上的祖先影响扩散向量 + 余弦；计算余弦前排除两个目标节点各自的单位 self 分量，避免人为制造结构正交性
- `got_select.py`：SELECT 层。`idea-generator` 调用 `python tools/got_select.py decide --ledger <path>` 获取本轮的行动：bootstrap/stall fresh 规则，或在前沿叶子上按 op 解耦定额选 ≤B 个动作：op 级收购 `U_op=ḡ_op+c_pucb·√σN/(1+Nop_op)` → `W=softmax(U/τ)` → 最大余数法分配 B 个槽位（确定性，受可用动作数封顶），各 op 内按 `Q` 取顶（`improve: V_max`；`crossover: geomean(V)·(1+c̃_dag)`），两 op 不混排竞争同一排序。所有全局派生量从记录重新计算；无持久状态。

### 7.5 background_contract.py

`background.md` 与候选语义归因之间的确定性合约层：

- `catalog`：校验并输出内置目录或显式 `--path` 目录及内容摘要
- `validate`：检查 schema-3 层级、目录解析、显式基线、关系、五轴范围、来源/指导证据、人类视图以及所有 ledger 点/祖先/策略收据/语义边收据与 `search_space_state` 覆盖层
- `render`：输出有界的维度、假设、关系与覆盖视图
- `preflight`：合法 P2 状态只返回 `none`；旧平面或 mixed-mode 直接拒绝，不迁移
- `validate-point`：检查完整点、条件激活、requires 与 excludes
- `lineage`：并列呈现数字祖先与机械 point diff，不生成因果边
- `validate-experience`：校验 schema-3 快照——通用 belief 的 run-id 证据，以及双层 dimension/hypothesis 信念的引用边、评估状态与比较计数（按持久化收据机械重算，必须一致）
- `target-evidence`：从持久化语义边收据为每个 target 输出有界比较证据（引用边 id、逐边观测、评估状态、引用计数），是 experience-extractor 的权威有界来源

`python tools/validate_background.py` 使用合成 DAG 回归这些合约。

### 7.6 semantic_search.py

图行动之后的语义选点层：`propose` 为 fresh/improve/crossover 生成有界合法点（按账本当前 `search_space_state` revision 过滤/排序：运行时剪枝的假设出局，被剪维度钉在显式基线）；`select` 可替换 `coverage`、`gain`、`gain_uncertainty`、`gain_uncertainty_nocost` 策略并输出 point + policy receipt，提案集与收据都带 revision 戳，过期即拒。策略可替换而不改变 registry、祖先或观测历史。

### 7.7 search_backends.py

受 Arbor 检索层启发的轻量适配器，但不导入 Arbor runtime：

- 本地 frozen-corpus backend 是可复现条件；DeepXiv / Jina 必须显式启用且单个失败不阻塞其他结果
- canonical work/URL 去重，统计不同 query 与 backend 的支持数，并平衡每个 query 的候选覆盖
- `visit` 渐进阅读并自动写 visit receipt；runtime 原生 web fetch 可用 `record-visit` 接入
- 独立的 `grounding=6000` 与 `novelty=2048` token lane
- 保留 raw response、retrieval timestamp、backend/client version、corpus cutoff/hash 和访问内容 hash；token 永不进入运行产物
- `python tools/validate_search_backends.py` 提供完全离线的回归检查

### 7.7 validate_skills.py

检查 `.claude/skills/` 结构和元数据。

```bash
python tools/validate_skills.py
```

### 7.8 validate_tasks.py

检查 `tasks/` 任务包结构和 `task.toml`。

```bash
python tools/validate_tasks.py
```

### 7.9 tools/tuners/

调优脚本：

- `_common.py`：共享加载、搜索空间、试验读取逻辑
- `tune_tools.py`：调优编排的确定性 CLI——`select-candidate`（解耦深度调优的候选选择门控：种群 ≥ N_min 且前 20% 中的最佳未调优候选）、`select-method`（基于维度的 grid/bo/cmaes 选择）、`select-best`（全局最佳试验）、`lineage-evidence`、`check-search-space` 等。
- `warmstart_eval.py`：顺序评估热配置（恢复 / 崩溃时停止），构建 `BASE_PARAMS` + 写 `phase_a`
- `grid_search.py`：低维搜索空间
- `bo_search.py`：使用贝叶斯优化（通过 Optuna 的多元 TPE）的中维搜索空间
- `cmaes_search.py`：使用 CMA-ES 的高维搜索空间

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
2. **`CLAUDE.md` / `AGENTS.md`**：理解所用 runtime 如何进入项目
3. **`.claude/agents/*.md` / `.opencode/agents/*.md`**：理解主代理与子代理的责任边界
4. **`.claude/skills/*/SKILL.md` / `.opencode/skills/*/SKILL.md`**：理解可复用的方法论
5. **`tools/*.py` 和 `tools/tuners/*.py`**：理解确定性执行层
6. **`tasks/tabular-model-search/`**：理解当前主要验证任务

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

### 11.2 添加 Skill

1. 创建 `.claude/skills/<skill-name>/SKILL.md`
2. 使用小写连字符命名
3. 前置必须有 `name` 和 `description`
4. 将长参考材料放在 skill 自己的 `references/` 中
5. 将确定性脚本放在 skill 自己的 `scripts/` 中
6. 运行：

```bash
python tools/validate_skills.py
```

### 11.3 添加智能体

1. 创建 `.claude/agents/<agent-name>.md`
2. 明确其输入、输出、边界和禁止的行动
3. 保持智能体范围集中。一个智能体应完成一个明确的子任务
4. 如果智能体需要确定性逻辑，写到 `tools/` 而不是仅自然语言流程

### 11.4 添加工具

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

**使用方法**：从 `tasks/framework_cfg.example.json` 复制，**仅保留**你想覆盖的键。删除其余部分——任何省略的键使用代码默认值。新运行也可用 `python tools/init_run.py <task> <tag> --dimension-strategy llm_induced --max-evaluations 200 --timeout 60` 直接持久化维度策略、总评估预算和单次评估超时；恢复已有运行时也可更新后两项。

主要配置包括：
- **`got.*`**：外层 S-GoT 图搜索参数（bootstrap 大小、PUCB 批次大小、停滞阈值、渐进加宽等）
- **`space_initialization.dimension_strategy`**：维度来源；默认 `catalog_subset` 使用内置目录，`llm_induced` 让 background researcher 在检索前生成并完整采用通过验证的 `dimension_catalog.json`
- **`semantic_search.*`**：语义点策略及 gain / uncertainty / cost / coverage 权重；默认使用 `gain_uncertainty_nocost`（不预测成本），`coverage` 保留为确定性消融或失败回退策略
- **`tuner.*`**：内层 HPO 调优器参数（热启动配置数量、深度调优门控阈值、BO 试验预算、patience 等）
- **`max_evaluations`**：全局停止预算（所有候选方案的试验总和）
- **`per_runtime_limit`**：单次评估超时（秒）（超时配置被强制终止）

**优先级**：显式 CLI 标志 > `framework_cfg.json` > 代码默认值

该文件被 git 忽略（`runs/` 仅本地）。完整的带注释参考（所有可用键及其语义）见 `tasks/framework_cfg.example.json`。
