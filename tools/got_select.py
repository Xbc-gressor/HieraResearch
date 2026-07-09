"""got_select.py — 选择层 / SELECT(设计稿 §5 / §6)。

确定性图搜索:给定当前 DAG + 全局量,选出本代的"做什么(op)+ 对谁(父代/方向)"。
idea-generator(LLM agent)调它拿指派,再做 IDEATE(写具体 idea)。本模块不含 LLM。

- decide_gen:每代 fresh 代 / PUCB 代 二选一(自举 / 停滞 / 否则 PUCB)。
- PUCB 代:动作池 = improve(L) + crossover(L 中所有不同对);
  Q(improve)=V_max;Q(crossover)=geomean(V)·(1+c̃_dag);P=softmax(ḡ_op/τ);PUCB 选 B。

对节点 id 类型无关(只用 graph.V_max / total_N / nodes[nid].ec + got_cdag)。
"""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path

from got_cdag import c_dag
from got_graph import Graph


def geomean(vals) -> float:
    p = 1.0
    for v in vals:
        p *= max(v, 0.0)
    return p ** (1.0 / len(vals))


def softmax(d: dict, tau: float) -> dict:
    if not d:
        return {}
    mx = max(d.values())
    ex = {k: math.exp((v - mx) / tau) for k, v in d.items()}
    z = sum(ex.values())
    return {k: v / z for k, v in ex.items()}


def ucb_node(graph, nid, c_leaf) -> float:
    """select_leaf 下降用的节点级 UCB(§4)。"""
    sigmaN = max(2, graph.total_N())
    return graph.V_max(nid) + c_leaf * math.sqrt(math.log(sigmaN) / (1 + graph.nodes[nid].ec))


# ---------- 每代 fresh / PUCB 决策(§6.1)----------
def decide_gen(n_roots, stall, cfg):
    if n_roots < cfg["n_seed"]:
        return ("fresh", 1)                       # 自举:每代 1 个种子
    if stall >= cfg["S"]:
        return ("fresh", cfg["B"])                # 反早熟:产 B 个 fresh(原 m_fresh 并入 B)
    return ("pucb", cfg["B"])


# ---------- Q(§6.2)----------
def Q(graph, action, gamma) -> float:
    op, args = action
    if op == "improve":
        return graph.V_max(args[0])
    return geomean([graph.V_max(args[0]), graph.V_max(args[1])]) * (1.0 + c_dag(graph, args[0], args[1], gamma))


# ---------- 动作池 + PUCB 选 B(PUCB 代,§6.1/§6.4)----------
def build_pool(L):
    acts = [("improve", (x,)) for x in L]
    Ll = list(L)
    for i in range(len(Ll)):
        for j in range(i + 1, len(Ll)):
            acts.append(("crossover", (Ll[i], Ll[j])))
    return acts


def pick_pucb(graph, L, gbar, Nop, cfg):
    pool = build_pool(L)
    if not pool:
        return []
    P = softmax(gbar, cfg["tau"])                 # over {'improve','crossover'}
    sigmaN = max(1, graph.total_N())

    def pucb(a):
        op = a[0]
        return (Q(graph, a, cfg["gamma"])
                + cfg["c_pucb"] * P.get(op, 0.0) * math.sqrt(sigmaN) / (1 + Nop.get(op, 0)))

    pool.sort(key=pucb, reverse=True)
    return pool[: cfg["B"]]


# ============================ 派生量重算 + decide CLI(SELECT,§14.1/§14.3)============================
# idea-generator 调 `got_select.py decide --ledger <path>` 拿本代指派(确定性);
# 再对每个 action 做 IDEATE(LLM)→ ledger.py add-record。全局量全部从 records 重算,无持久状态。
# Outer-search defaults, tuned under an EQUAL total-validation budget (max_evals=200,
# 8 seeds, 3 toys) on the two-level toy — the budget-fair re-tune that replaced the
# invalid fixed-generation one. See dev_plan/got-benchmark-report.md +
# got-equalbudget-{data.md,viz.png}; tools/got_benchmark/{bench_inner,retune,viz_inner}.py.
# The ONLY robust, base-independent change from the equal-budget re-tune is n_seed 3→5
# (clear minimum on all 3 toys). Everything else is flat / noise-level / base-dependent
# at a realistic ~200 budget and is kept at the original default:
#   - c_pucb: flat in [0.1,0.8]; its "winner" flipped with the base (0.8 under B=4, 0.1
#     under the landed B=2) → kept 0.4.  - S: flat → 10.  - B: toy liked 4 but that carries
#     real-loop cost the toy can't model → kept 2.  - alpha/gamma/c_leaf/tau/C: default.
DEFAULT_CFG = dict(n_seed=5, S=10, B=2, C=1.5, alpha=0.5,
                   gamma=0.6, c_pucb=0.4, c_leaf=0.4, tau=0.3)


def _replay_stall(graph) -> int:
    """连续多少个候选(run_id)没刷新 best(§6.1)。按 run_id 序重放(from_ledger 已按序插入):
    遇成功的 fresh 或刷新 best → 归零,否则 +1;crash 也算一个没刷新的 run_id(+1)。
    不需要代号——这是"多少个 run_id 没生成新的",不是"多少代"。"""
    best, stall = float("inf"), 0
    for nid in graph.nodes:                          # insertion order = run_id 升序
        n = graph.nodes[nid]
        if n.status == "crash":
            stall += 1
        elif n.op == "fresh" or n.score < best:      # 注入新血 或 刷新 best → 归零
            best = min(best, n.score)
            stall = 0
        else:
            stall += 1
    return stall


def derive_state(graph) -> dict:
    """从 records(经 from_ledger)重算全局派生量,无持久状态(§14.1 决策 B)。
    ḡ_op 用当前 r 重算(stateless);best/consumed/Nop/n_roots 同理。"""
    r = graph.r_map()
    scored = [n.score for n in graph.nodes.values() if n.status != "crash"]
    best = min(scored) if scored else float("inf")
    consumed = sorted({s for nid in graph.nodes for s in graph.nodes[nid].source_run_ids
                       if not graph._is_ref(s)})
    Nop = {"improve": 0, "crossover": 0}
    for n in graph.nodes.values():
        if n.op in Nop:
            Nop[n.op] += 1                       # 含 crash(与主循环 Nop 计数一致)
    gbar = {"improve": 0.0, "crossover": 0.0}
    cnt = {"improve": 0, "crossover": 0}
    for nid in graph.nodes:                      # insertion order(run_id 升序)
        n = graph.nodes[nid]
        if n.op in gbar and n.status != "crash" and nid in r:
            ps = [p for p in graph.parents(nid) if p in r]
            if ps:
                gval = max(0.0, r[nid] - max(r[p] for p in ps))
                cnt[n.op] += 1
                gbar[n.op] += (gval - gbar[n.op]) / cnt[n.op]   # running mean ≡ 均值
    n_roots = sum(1 for rt in graph.roots() if graph.nodes[rt].status != "crash")
    return dict(best=best, consumed=consumed, Nop=Nop, gbar=gbar,
                stall=_replay_stall(graph), n_alive=len(graph.alive_roots()), n_roots=n_roots)


def decide(graph, cfg) -> dict:
    """一代的 SELECT。返回 {gen, kind, actions, diag}:
    PUCB 代 actions=[{op,parents}…];fresh 代 actions=[{op:'fresh'}…](方向留给 IDEATE)。"""
    st = derive_state(graph)
    kind, k = decide_gen(st["n_roots"], st["stall"], cfg)
    actions: list = []
    if kind == "pucb":
        key = lambda nid: ucb_node(graph, nid, cfg["c_leaf"])
        L = []
        for s in graph.alive_roots():
            leaf = graph.select_leaf(s, key)
            if leaf is not None:
                L.append(leaf)
        L = list(dict.fromkeys(L))
        if not L:                                # 没料 → 退化 fresh
            kind, k = "fresh", 1
        else:
            actions = [{"op": op, "parents": list(args)}
                       for op, args in pick_pucb(graph, L, st["gbar"], st["Nop"], cfg)]
    if kind == "fresh":
        actions = [{"op": "fresh"} for _ in range(k)]   # 方向由 IDEATE 从 background 优先级 − consumed 选
    diag = dict(st)
    if diag["best"] == float("inf"):
        diag["best"] = None                      # 合法 JSON(无 non-crash 节点时)
    return {"kind": kind, "actions": actions, "diag": diag}


def load_run_cfg(ledger_path: Path, section: str) -> dict:
    """Per-run framework-hyperparameter overrides from `<run_dir>/framework_cfg.json`.

    Lets a single run (e.g. a Phase-3 OFAT trial) override framework meta-params
    without code edits or CLI flags, so a headless `autoresearch-experiment` run
    honors them too. Shape: `{"got": {...s-got keys...}, "tuner": {...}}`.
    Returns the requested section ({} if the file/section is absent).
    """
    p = Path(ledger_path).parent / "framework_cfg.json"
    if p.is_file():
        try:
            return dict(json.loads(p.read_text()).get(section, {}))
        except (ValueError, OSError):
            return {}
    return {}


def cmd_decide(args) -> int:
    cfg = dict(DEFAULT_CFG)
    cfg.update(load_run_cfg(Path(args.ledger), "got"))   # per-run framework_cfg.json
    if args.cfg:
        cfg.update(json.loads(args.cfg))                 # explicit --cfg wins
    # Round-1 bootstrap: ledger.json is created on the first add-record, but
    # SELECT runs before it. A missing ledger == empty run -> bootstrap fresh.
    path = Path(args.ledger)
    data = json.loads(path.read_text()) if path.exists() else {"records": []}
    graph = Graph.from_ledger(data, C=cfg["C"], alpha=cfg["alpha"])
    print(json.dumps(decide(graph, cfg), indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="S-GoT SELECT 层:一代的 op+父代指派(确定性图搜索)。")
    sub = parser.add_subparsers(dest="command", required=True)
    dec = sub.add_parser("decide", help="读 ledger → 输出本代 {gen,kind,actions,diag}")
    dec.add_argument("--ledger", required=True, help="Path to ledger.json")
    dec.add_argument("--cfg", help="JSON,覆盖 DEFAULT_CFG 的部分键(如 '{\"B\":3}')")
    dec.set_defaults(func=cmd_decide)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
