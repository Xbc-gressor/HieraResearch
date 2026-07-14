"""got_graph.py — S-GoT 计算层:发展 DAG + MCTS 统计(设计稿 §3 / §4 / §14.2)。

节点身份 = `run_id`(str)。`r / V_max / V_med / N / cap` 每代从 DAG 重算(规模小,
重算最简单,§14.1);`ec` 在 add 时增量维护(创建子代时每个真父代 +1,含产出 crash 的那次)。

`source_run_ids` 是通用"来源"列表(§9):improve/crossover 放父代 run_id;
fresh 放方向 tag `tf-*`。`parents()` 只取能解析成现有节点的项 → fresh 自动是根、tf-* 不连边。

两种构造:
- `add(op, source_run_ids, ...)`         自增 str id —— 离线 toy / 单测用。
- `Graph.from_ledger(ledger)`            用 ledger 的 run_id 作 id 建图 —— 生产用。
"""
from __future__ import annotations
import bisect
import math
import statistics
from dataclasses import dataclass

CRASH = float("inf")  # crash 的原始 score 哨兵(越低越好,+inf = 最差)


@dataclass
class Node:
    id: str                  # = run_id
    op: str                  # 'fresh' | 'improve' | 'crossover'
    source_run_ids: list     # list[str]:父代 run_id 或 'tf-*'(fresh 的方向)
    score: float             # 原始 score,越低越好;crash = +inf
    status: str              # 'kept' | 'discard' | 'crash'(ledger 的 keep/discard 等价 non-crash)
    genome: object = None    # 可选 payload(toy 是 list[float];生产为 None)
    ec: int = 0              # expand_count:当过父代的次数(含产出 crash 的那次)


class Graph:
    def __init__(self, C: float = 1.5, alpha: float = 0.5):
        self.nodes: dict = {}
        self.C, self.alpha = C, alpha
        self._next = 0
        self._cache: dict = {}
        self._N: dict = {}      # 子树 non-crash 计数:增量维护(r-无关,不随 r 漂移重算)

    # ---------- construction ----------
    def add(self, op, source_run_ids, genome=None, score=CRASH, status="kept") -> str:
        """自增 id 的便捷构造(离线/单测)。返回新节点 run_id(str)。"""
        rid = str(self._next)
        self._next += 1
        return self._add(rid, op, source_run_ids, genome, score, status)

    def _add(self, run_id, op, source_run_ids, genome, score, status) -> str:
        rid = str(run_id)
        self.nodes[rid] = Node(rid, op, [str(s) for s in source_run_ids], score, status, genome)
        crashed = status == "crash"
        self._N[rid] = 0 if crashed else 1
        for p in self.parents(rid):          # ec:产出该子代的 expansion 计入每个真父代(含 crash 那次)
            self.nodes[p].ec += 1
        if not crashed:                      # N:non-crash 子代 → 所有祖先 +1(backprop)
            for a in self.ancestors(rid):
                self._N[a] += 1
        self._cache.clear()
        return rid

    # ---------- ledger 适配器 ----------
    @classmethod
    def from_ledger(cls, ledger: dict, *, C: float = 1.5, alpha: float = 0.5) -> "Graph":
        """从 ledger.json dict 建图。按 run_id 升序(= 创建序 = 拓扑序:父先于子)插入,
        故 `_infer_op` 时父代已在图中。只纳入已跑过的 record(有数值 score 或 status=crash);
        pending / 未跑(无 score)跳过。record 带 `op` 时为准,否则按可解析父代数推断。"""
        g = cls(C=C, alpha=alpha)
        for r in sorted(ledger.get("records", []), key=lambda r: _rid_key(r.get("run_id"))):
            status = r.get("status")
            score = r.get("final_best_score")
            if status == "crash":
                score = CRASH
            elif not isinstance(score, (int, float)):
                continue                                  # pending / 未跑 → 跳过
            src = [str(s) for s in (r.get("source_run_ids") or [])]
            op = r.get("op") or g._infer_op(src)
            g._add(r["run_id"], op, src, None, float(score), status or "kept")
        return g

    def _infer_op(self, source_run_ids) -> str:
        """缺 `op` 字段时的回退:按"能解析成现有节点的父代数"判 0/1/≥2。"""
        np = sum(1 for s in source_run_ids if s in self.nodes)
        return "fresh" if np == 0 else ("improve" if np == 1 else "crossover")

    # ---------- structure ----------
    def _is_ref(self, s: str) -> bool:
        return s in self.nodes

    def parents(self, nid) -> list:
        return [s for s in self.nodes[nid].source_run_ids if self._is_ref(s)]

    def children_map(self) -> dict:
        if "kids" not in self._cache:
            kids = {nid: [] for nid in self.nodes}
            for nid in self.nodes:
                for p in self.parents(nid):
                    kids[p].append(nid)
            self._cache["kids"] = kids
        return self._cache["kids"]

    def children(self, nid) -> list:
        return self.children_map()[nid]

    def roots(self) -> list:
        return [nid for nid in self.nodes if not self.parents(nid)]

    def ancestors(self, nid) -> set:
        seen, stack = set(), list(self.parents(nid))
        while stack:
            a = stack.pop()
            if a in seen:
                continue
            seen.add(a)
            stack.extend(self.parents(a))
        return seen

    def descendants(self, nid) -> set:
        seen, stack = set(), list(self.children(nid))
        while stack:
            c = stack.pop()
            if c in seen:
                continue
            seen.add(c)
            stack.extend(self.children(c))
        return seen

    def lin(self, nid) -> set:
        """x 可达的根集合(= 祖先里的根,或 x 自己若是根)。"""
        roots = set(self.roots())
        return (self.ancestors(nid) | {nid}) & roots

    # ---------- reward r = 1 − percentile(score)(只对 non-crash;score 越低 → r 越高)----------
    def r_map(self) -> dict:
        if "r" not in self._cache:
            ok = [(nid, n.score) for nid, n in self.nodes.items() if n.status != "crash"]
            m = len(ok)
            srt = sorted(s for _, s in ok)                     # O(m log m)
            denom = max(1, m - 1)
            # better = 严格更优(更低)的个数 = bisect_left(srt, s);并列同分得同 r
            self._cache["r"] = {nid: 1.0 - bisect.bisect_left(srt, s) / denom for nid, s in ok}
        return self._cache["r"]

    # ---------- 拓扑序(父在前;随 add 失效)----------
    def topo(self) -> list:
        if "topo" not in self._cache:
            indeg = {n: len(self.parents(n)) for n in self.nodes}
            q = [n for n in self.nodes if indeg[n] == 0]
            order, kids = [], self.children_map()
            while q:
                u = q.pop()
                order.append(u)
                for c in kids[u]:
                    indeg[c] -= 1
                    if indeg[c] == 0:
                        q.append(c)
            self._cache["topo"] = order                        # parents before children
        return self._cache["topo"]

    # ---------- subtree 统计 ----------
    # V_max:r 每代漂移 → 重算,但一趟反向拓扑 O(m+E)(max 幂等,DAG 共享无妨)。
    # N    :r-无关 → add() 增量维护(见上),O(1) 命中,不重算。
    # V_med:当前无决策消费 → 按需算(O(subtree)),不进热路径。
    def _vmax_map(self) -> dict:
        if "vmax" not in self._cache:
            r = self.r_map()
            vmax = {}
            for x in reversed(self.topo()):                    # 子在前 → 处理 x 时子已就绪
                best = r.get(x, float("-inf"))                 # crash → -inf(排除自身)
                for c in self.children(x):
                    if vmax[c] > best:
                        best = vmax[c]
                vmax[x] = best if best != float("-inf") else 0.0
            self._cache["vmax"] = vmax
        return self._cache["vmax"]

    def V_max(self, nid): return self._vmax_map()[nid]
    def N(self, nid):     return self._N.get(nid, 0)
    def V_med(self, nid):                                      # 按需:无缓存、不在每代热路径
        r = self.r_map()
        rs = [r[a] for a in ({nid} | self.descendants(nid)) if a in r]
        return statistics.median(rs) if rs else 0.0

    def total_N(self) -> int:
        return sum(1 for n in self.nodes.values() if n.status != "crash")

    def cap(self, nid) -> int:
        n = self.N(nid)
        return math.ceil(self.C * (n ** self.alpha)) if n > 0 else 1

    # ---------- 前沿 / alive(§4)----------
    def is_frontier(self, nid) -> bool:
        n = self.nodes[nid]
        return n.status != "crash" and n.ec < self.cap(nid)

    def F(self) -> list:
        return [nid for nid in self.nodes if self.is_frontier(nid)]

    def alive_roots(self) -> list:
        al = set()
        for f in self.F():
            al |= self.lin(f)
        # 确定性顺序(run_id 数值序):str id 的 set 迭代受 PYTHONHASHSEED 随机化,必须排序,
        # 否则 select_leaf 遍历顺序 → L → PUCB 平手在不同进程间漂移(§4)。
        return sorted((r for r in al if self.nodes[r].status != "crash"), key=_rid_key)

    # ---------- select_leaf(§4):从 root 按 key 沿 UCB 下降到前沿叶 ----------
    def select_leaf(self, root, key):
        x = root
        while not self.is_frontier(x):                         # 饱和 → 继续下降
            ch = [c for c in self.children(x) if self.nodes[c].status != "crash"]
            if not ch:
                return None                                    # 饱和死路(子代全 crash)
            x = max(ch, key=key)
        return x


def _rid_key(rid):
    """run_id 排序键:数值 id 按数值序(免 '100' < '99' 的字典序坑),非数值置后。"""
    s = str(rid)
    return (0, int(s)) if s.isdigit() else (1, s)


# ---------- render:带注释的祖先轨迹视图(只读投影,设计 dev_plan/plan_change_effect)----------
# 给定本轮要用的父代,摘出它们的祖先发展轨迹:节点=结果(ledger 的 idea,自包含),
# 边=过程(ledger 的 change)+ 效果 Δ(现算)。纯读 ledger,不写、不调模型、仅 stdlib。

def _clean(s):
    """折叠空白/换行成单行,但**不截断**(完整保留 idea/change 内容)。"""
    return " ".join(str(s or "").split())


def _change_for_parent(change, parent):
    """若 change 写成 'vs <父>: … ; vs <父>: …'(crossover 分父代),抽出该父代那段;否则原样返回。"""
    if not change:
        return change
    for seg in str(change).split(";"):
        head, sep, body = seg.partition(":")
        if sep and parent in head and head.strip().lower().startswith("vs"):
            return body.strip() or seg.strip()
    return str(change).strip()


def _within(step, nid, depth):
    """从 nid 出发,沿 step(=g.parents 或 g.children)走最多 depth 跳的节点集(不含 nid)。
    depth=None 表示走到底。"""
    seen, cur, d = set(), {nid}, 0
    while cur and (depth is None or d < depth):
        nxt = set()
        for x in cur:
            for y in step(x):
                if y not in seen:
                    seen.add(y)
                    nxt.add(y)
        cur, d = nxt, d + 1
    return seen


def render_trajectory(ledger: dict, query_ids: list, depth=3) -> dict:
    """围绕 query 父代抠一张局部 DAG,从新到旧展示:
    - 往新方向 1 代:query 父代的**直接子代**(clone/improve 检查——这俩父代已被组合/改进过没有)。
    - 往旧方向 depth 层:query 父代的**祖先**(最多 depth 跳的发展轨迹)。
    节点=结果(idea)+分数,边=过程(change)+Δ;run_id 降序(新→旧)。纯读。"""
    g = Graph.from_ledger(ledger)
    recs = {str(r.get("run_id")): r for r in ledger.get("records", [])}
    query = [str(q) for q in query_ids]
    qset = {q for q in query if q in g.nodes}

    children, ancestors = set(), set()
    for q in qset:
        children |= _within(g.children, q, 1)         # 往新:只取 1 代直接子代
        ancestors |= _within(g.parents, q, depth)     # 往旧:depth 层祖先
    focus = qset | children | ancestors

    best_id, best_score = None, float("inf")          # 全局最优(最低 non-crash 分)
    for nid, n in g.nodes.items():
        if n.status != "crash" and n.score < best_score:
            best_id, best_score = nid, n.score

    nodes = []
    for nid in sorted(focus, key=_rid_key, reverse=True):   # 新 → 旧
        n = g.nodes[nid]
        root_tag = n.source_run_ids[0] if (n.op == "fresh" and n.source_run_ids) else n.op
        nodes.append({"id": nid, "op": n.op, "status": n.status,
                      "score": None if n.score == CRASH else round(n.score, 4),
                      "result": recs.get(nid, {}).get("idea"), "root": root_tag,
                      "is_query": nid in qset, "is_best": nid == best_id,
                      "is_child": nid in children and nid not in qset})

    edges = []
    for child in sorted(focus, key=_rid_key, reverse=True):
        for parent in sorted(g.parents(child), key=_rid_key):
            if parent not in focus:
                continue
            cs, ps = g.nodes[child].score, g.nodes[parent].score
            delta = None if (cs == CRASH or ps == CRASH) else round(cs - ps, 4)
            edges.append({"child": child, "parent": parent, "delta": delta,
                          "change": _change_for_parent(recs.get(child, {}).get("change"), parent)})
    return {"query": query, "best": best_id, "nodes": nodes, "edges": edges}


def render_global(ledger: dict) -> dict:
    """全局副模式(无 --nodes):投影**整个 run** 的全部节点 + 全部边(change+Δ),新→旧。
    不设 cap——给 experience-extractor 当全样本找「change→Δ」规律。纯读。"""
    g = Graph.from_ledger(ledger)
    recs = {str(r.get("run_id")): r for r in ledger.get("records", [])}

    best_id, best_score = None, float("inf")
    for nid, n in g.nodes.items():
        if n.status != "crash" and n.score < best_score:
            best_id, best_score = nid, n.score

    order = sorted(g.nodes, key=_rid_key, reverse=True)   # 新 → 旧
    nodes = []
    for nid in order:
        n = g.nodes[nid]
        root_tag = n.source_run_ids[0] if (n.op == "fresh" and n.source_run_ids) else n.op
        nodes.append({"id": nid, "op": n.op, "status": n.status,
                      "score": None if n.score == CRASH else round(n.score, 4),
                      "result": recs.get(nid, {}).get("idea"), "root": root_tag,
                      "is_query": False, "is_child": False, "is_best": nid == best_id})
    edges = []
    for child in order:
        for parent in sorted(g.parents(child), key=_rid_key):
            cs, ps = g.nodes[child].score, g.nodes[parent].score
            delta = None if (cs == CRASH or ps == CRASH) else round(cs - ps, 4)
            edges.append({"child": child, "parent": parent, "delta": delta,
                          "change": _change_for_parent(recs.get(child, {}).get("change"), parent)})
    return {"query": None, "best": best_id, "nodes": nodes, "edges": edges}


def format_text(view: dict) -> str:
    title = (f"TRAJECTORY for ◀ {', '.join(view['query'])}" if view.get("query")
             else "FULL RUN DAG — all candidates (new → old)")
    lines = [title, "",
             "NODES (new → old; ◀parent = this round's parents, ↳child = their direct children",
             "       (clone/improve check), ★best = global best; result has no parent refs)"]
    for n in view["nodes"]:
        marks = " ".join(([" ◀parent"] if n["is_query"] else [])
                         + (["↳child"] if n.get("is_child") else [])
                         + (["★best"] if n["is_best"] else []))
        score = "crash" if n["score"] is None else f"{n['score']:.4f}"
        st = "" if n["status"] in (None, "kept", "keep") else f" {n['status']}"
        # 定宽字段在前对齐(id/分数/root/标记),完整 result 拖尾不截断
        lines.append(f"  {n['id']:>4}  {score:>9}  [{n['root']}]{st}{marks}  {_clean(n['result'])}".rstrip())
    lines += ["", "EDGES (process → effect; Δ = child − parent score, so negative Δ = improvement / lower is better)"]
    for e in view["edges"]:
        d = "  n/a " if e["delta"] is None else f"Δ {e['delta']:+.3f}"
        # parent→child + Δ 在前对齐,完整 change 拖尾不截断
        lines.append(f"  {e['parent']:>4} → {e['child']:<4}  {d:>9}  {_clean(e['change'])}".rstrip())
    return "\n".join(lines)


def _main(argv=None):
    import argparse
    import json
    from pathlib import Path
    p = argparse.ArgumentParser(description="S-GoT graph tools")
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("render", help="annotated ancestor trajectory of the given parent node(s)")
    r.add_argument("--ledger", required=True)
    r.add_argument("--nodes", help="comma-separated parent run_ids to trace (this round's decide "
                                    "parents). OMIT for global mode = the whole run's DAG.")
    r.add_argument("--depth", type=int, default=3,
                   help="how many layers of ANCESTORS to walk back (default 3); descendants are "
                        "always just 1 generation (direct children, for the clone/improve check).")
    r.add_argument("--format", choices=["text", "json"], default="text")
    args = p.parse_args(argv)
    if args.cmd == "render":
        ledger = json.loads(Path(args.ledger).read_text())
        if args.nodes:                                   # 主模式:围绕指定父代
            view = render_trajectory(ledger, [x.strip() for x in args.nodes.split(",") if x.strip()],
                                     depth=args.depth)
        else:                                            # 全局副模式:整个 run
            view = render_global(ledger)
        print(format_text(view) if args.format == "text"
              else json.dumps(view, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
