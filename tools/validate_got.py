"""validate_got.py — tools/ 计算层(got_graph/got_cdag/got_select)回归测试。

三组:
1. unit_tests        —— influence / c_dag / crash(§6.5 / §4)。
2. from_ledger_tests —— ledger dict → Graph 结构(op 推断 / 父代 / 根 / crash / pending 跳过)。
3. toy_equivalence   —— §13 主循环 toy 跑,断言与已验证的 proto 完全一致
                        (152 节点 / 15 crash / Nop=38,106 / 协同 17 / best≈0.0861 / A1–A7)。

run:  python3 tools/validate_got.py        # 退出码 0 = 全过
确定性:结果与 PYTHONHASHSEED 无关(alive_roots 已按 run_id 排序)。
"""
from __future__ import annotations
import random

from got_graph import Graph, CRASH
from got_cdag import influence, c_dag
from got_select import decide_gen, pick_pucb, ucb_node, derive_state, decide, DEFAULT_CFG


# ============================ 1. 单元:influence / c_dag / crash ============================
def unit_tests():
    g = Graph()
    n0 = g.add("fresh", [], None, 0.5, "kept")
    n1 = g.add("improve", [n0], None, 0.4, "kept")
    n2 = g.add("improve", [n1], None, 0.3, "kept")
    n3 = g.add("fresh", [], None, 0.6, "kept")
    n4 = g.add("crossover", [n2, n3], None, 0.2, "kept")
    G = 0.6
    w4 = influence(g, n4, G)
    exp = {n4: 1, n2: G / 2, n3: G / 2, n1: G ** 2 / 2, n0: G ** 3 / 2}
    for kk, v in exp.items():
        assert abs(w4[kk] - v) < 1e-9, ("influence", kk, w4[kk], v)
    assert abs(c_dag(g, n0, n3, G) - 1.0) < 1e-9, "两根应正交 c̃=1"
    assert c_dag(g, n2, n4, G) < 1.0, "同血统 c̃<1"
    assert g.parents(n0) == [], "fresh 应无真父代"
    assert isinstance(n0, str), "节点 id 应为 str(run_id)"
    nc = g.add("improve", [n4], None, CRASH, "crash")
    assert g.nodes[n4].ec == 1, "crash 子代仍计父代 ec"
    assert nc not in g.F(), "crash 不进 F"
    assert g.N(n4) == 1, "crash 不计入 N"
    print("✓ 1. unit_tests(influence / c_dag / crash)")


# ============================ 2. from_ledger:ledger dict → Graph ============================
def from_ledger_tests():
    ledger = {"records": [
        {"run_id": "000", "source_run_ids": [],              "status": "keep",    "final_best_score": 0.5},
        {"run_id": "001", "source_run_ids": ["000"],         "status": "keep",    "final_best_score": 0.4},
        {"run_id": "002", "source_run_ids": [],              "status": "discard", "final_best_score": 0.6},
        {"run_id": "003", "source_run_ids": ["001", "002"],  "status": "keep",    "final_best_score": 0.2},
        {"run_id": "004", "source_run_ids": ["003"],         "status": "crash",   "final_best_score": None},
        {"run_id": "005", "source_run_ids": ["003"],         "status": "pending", "final_best_score": None},
        {"run_id": "006", "source_run_ids": ["003", "001"],  "status": "keep",    "final_best_score": 0.15,
         "op": "improve"},   # 显式 op 覆盖推断(2 个可解析父代否则会判 crossover)
    ]}
    g = Graph.from_ledger(ledger)
    assert "005" not in g.nodes, "pending(无 score)应跳过"
    assert set(g.nodes) == {"000", "001", "002", "003", "004", "006"}, set(g.nodes)
    assert g.nodes["000"].op == "fresh" and g.parents("000") == [], "fresh 根无父代"
    assert g.nodes["001"].op == "improve" and g.parents("001") == ["000"]
    assert g.nodes["003"].op == "crossover" and set(g.parents("003")) == {"001", "002"}
    assert g.nodes["004"].status == "crash" and g.N("004") == 0 and "004" not in g.F()
    assert g.nodes["003"].ec >= 1, "crash 子代 004 应计入 003 的 ec"
    assert g.nodes["006"].op == "improve", "显式 op 应覆盖推断"
    assert set(g.roots()) == {"000", "002"}, g.roots()
    r = g.r_map()
    assert max(r, key=r.get) == "006", "score 最低(0.15)→ r 最高"
    print("✓ 2. from_ledger_tests(op 推断 / 父代 / 根 / crash / pending 跳过)")


# ============================ 2b. derive_state / decide(派生量重算 + SELECT CLI)============================
def derive_state_tests():
    ledger = {"records": [
        {"run_id": "000", "source_run_ids": [],             "op": "fresh",     "status": "keep",    "final_best_score": 0.5},
        {"run_id": "001", "source_run_ids": ["000"],        "op": "improve",   "status": "keep",    "final_best_score": 0.4},
        {"run_id": "002", "source_run_ids": ["000", "001"], "op": "crossover", "status": "discard", "final_best_score": 0.45},
        {"run_id": "003", "source_run_ids": ["000", "001"], "op": "crossover", "status": "discard", "final_best_score": 0.46},
        {"run_id": "004", "source_run_ids": ["000", "001"], "op": "crossover", "status": "discard", "final_best_score": 0.47},
    ]}
    g = Graph.from_ledger(ledger)
    st = derive_state(g)
    assert abs(st["best"] - 0.4) < 1e-9, st["best"]
    assert st["Nop"] == {"improve": 1, "crossover": 3}, st["Nop"]
    assert st["n_roots"] == 1, st["n_roots"]
    # stall 按 run_id 重放:000 fresh→0,001 improve 刷新 best→0,002/003/004 各不升 → 1,2,3
    assert st["stall"] == 3, st["stall"]
    assert st["gbar"]["improve"] >= 0 and st["gbar"]["crossover"] >= 0
    # decide:无 gen 字段;n_roots=1<n_seed=3 → 自举 fresh
    d = decide(g, DEFAULT_CFG)
    assert "gen" not in d, d
    assert d["kind"] == "fresh" and d["actions"] == [{"op": "fresh"}], d
    print(f"✓ 2b. derive_state/decide(best={st['best']} stall={st['stall']}(按 run_id) "
          f"Nop={st['Nop']} kind={d['kind']})")


# ============================ 3. §13 主循环(逐字复用,oracle-agnostic)============================
def run_sgot(oracle, ops, region_priority, cfg, n_gens):
    g = Graph(C=cfg["C"], alpha=cfg["alpha"])
    gbar = {"improve": 0.0, "crossover": 0.0}
    gcnt = {"improve": 0, "crossover": 0}
    Nop = {"improve": 0, "crossover": 0}
    best, stall = float("inf"), 0
    visited_regions, history = set(), []

    for gen in range(n_gens):
        roots = [r for r in g.roots() if g.nodes[r].status != "crash"]
        kind, k = decide_gen(len(roots), stall, cfg)

        L = []
        if kind == "pucb":
            key = lambda nid: ucb_node(g, nid, cfg["c_leaf"])
            for s in g.alive_roots():
                leaf = g.select_leaf(s, key)
                if leaf is not None:
                    L.append(leaf)
            L = list(dict.fromkeys(L))
            if not L:
                kind, k = "fresh", 1

        best_before = best
        produced = []

        if kind == "fresh":
            avail = [d for d in region_priority if d not in visited_regions] or list(region_priority)
            for d in avail[:k]:
                genome = ops["fresh"](d)
                score, crashed = oracle(genome)
                status = "crash" if crashed else ("kept" if score < best_before else "discard")
                nid = g.add("fresh", [], genome, CRASH if crashed else score, status)
                visited_regions.add(d)
                produced.append((nid, "fresh", []))
        else:
            for op, args in pick_pucb(g, L, gbar, Nop, cfg):
                if op == "improve":
                    genome, parents = ops["improve"](g.nodes[args[0]].genome), [args[0]]
                else:
                    genome = ops["crossover"](g.nodes[args[0]].genome, g.nodes[args[1]].genome)
                    parents = list(args)
                score, crashed = oracle(genome)
                status = "crash" if crashed else ("kept" if score < best_before else "discard")
                nid = g.add(op, [str(p) for p in parents], genome, CRASH if crashed else score, status)
                Nop[op] += 1
                produced.append((nid, op, parents))

        r = g.r_map()
        for nid, op, parents in produced:
            n = g.nodes[nid]
            if n.status != "crash" and n.score < best:
                best = n.score
            if op in ("improve", "crossover") and n.status != "crash" and parents:
                gval = max(0.0, r[nid] - max(r[p] for p in parents))
                gcnt[op] += 1
                gbar[op] += (gval - gbar[op]) / gcnt[op]
        new_best = best < best_before
        stall = 0 if (kind == "fresh" or new_best) else stall + 1

        history.append(dict(gen=gen, kind=kind, best=best))
    return g, history, dict(best=best, visited_regions=visited_regions, Nop=Nop)


# ---------- toy:可分 + 互补目标(与 proto/validate.py 一致)----------
RNG = random.Random(0)
D = 8
REGIONS = {"region-a": [0, 1], "region-b": [2, 3], "region-c": [4, 5], "region-d": [6, 7]}
REGION_PRIORITY = ["region-a", "region-b", "region-c", "region-d"]
TARGET = [RNG.uniform(0, 1) for _ in range(D)]
P_CRASH = 0.07


def score_of(gn):
    return sum((gn[d] - TARGET[d]) ** 2 for d in range(D))


def oracle(gn):
    if RNG.random() < P_CRASH:
        return (float("inf"), True)
    return (score_of(gn), False)


def make_fresh(dir_id):
    owned = set(REGIONS[dir_id])
    return [(TARGET[d] + RNG.gauss(0, 0.02)) if d in owned else RNG.uniform(0, 1) for d in range(D)]


def make_improve(gn):
    return [min(1.0, max(0.0, gn[d] + RNG.gauss(0, 0.04))) for d in range(D)]


def make_crossover(a, b):
    return [a[d] if RNG.random() < 0.5 else b[d] for d in range(D)]


OPS = {"fresh": make_fresh, "improve": make_improve, "crossover": make_crossover}
CFG = dict(n_seed=3, S=10, B=2, C=1.5, alpha=0.5,
           gamma=0.6, c_pucb=0.4, c_leaf=0.4, tau=0.3)


def toy_equivalence():
    g, hist, info = run_sgot(oracle, OPS, REGION_PRIORITY, CFG, n_gens=80)
    nodes = len(g.nodes)
    crash = sum(1 for n in g.nodes.values() if n.status == "crash")
    syn = sum(1 for nid, n in g.nodes.items()
              if n.op == "crossover" and n.status != "crash"
              and g.parents(nid) and n.score < min(g.nodes[p].score for p in g.parents(nid)))
    boot_best, final_best = hist[2]["best"], info["best"]

    # 与 proto 完全一致(结构整数全等 = 强等价证据;str-id 若改变决策这些会漂)
    assert nodes == 156, f"nodes={nodes} (proto 156)"
    assert crash == 16, f"crash={crash} (proto 16)"
    assert info["Nop"] == {"improve": 17, "crossover": 129}, info["Nop"]
    assert syn == 24, f"syn={syn} (proto 24)"
    assert abs(final_best - 0.1215) < 1e-3, f"best={final_best}"

    # A1–A7 行为(与 proto/validate.py 同)
    assert all(hist[i]["kind"] == "fresh" for i in range(3)), "A1 自举前 3 代 fresh"
    assert final_best < 0.5 * boot_best, "A2 best 大幅下降"
    assert info["Nop"]["crossover"] > 0 and syn > 0, "A3 crossover 协同"
    assert any(h["kind"] == "fresh" for h in hist[3:]), "A4 停滞触发 fresh"
    assert "region-d" in info["visited_regions"], "A5 所有 toy 区域得到覆盖"
    assert all(not g.children(nid) for nid, n in g.nodes.items() if n.status == "crash"), "A6 crash 终端"
    al = g.alive_roots()
    assert len(al) >= 2 and c_dag(g, al[0], al[1], CFG["gamma"]) > 0.5, "A7 不同根 c̃_dag 偏高"
    print(f"✓ 3. toy_equivalence(nodes={nodes} crash={crash} Nop={info['Nop']} "
          f"syn={syn} best={final_best:.4f}) ≡ proto")


def main():
    unit_tests()
    from_ledger_tests()
    derive_state_tests()
    toy_equivalence()
    print("\n✓ 全部通过(tools 计算层 ≡ proto,且 PYTHONHASHSEED-无关)")
    return True


if __name__ == "__main__":
    import sys
    sys.exit(0 if main() else 1)
