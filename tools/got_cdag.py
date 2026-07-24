"""got_cdag.py — 结构互补 c̃_dag(设计稿 §6.5)。

从 x 反向扩散影响力:infl[x]=1;沿父代边每层 ×γ,crossover 处 /K(等分守恒)。
闭式 w_x(a) = Σ_{a→…→x 路径} γ^(路径长) · Π_(沿途每个 crossover) (1/K)。
c̃_dag(x,y) = 1 − cos(w_x\\{x}, w_y\\{y})(只比较祖先影响;目标节点自身不进范数)。

对节点 id 类型无关(只用 graph.parents / children_map / ancestors)。
"""
from __future__ import annotations
import math


def _topo(graph, nodes):
    """Kahn 拓扑序(父在前)over 诱导子图。"""
    nodes = set(nodes)
    indeg = {n: 0 for n in nodes}
    for n in nodes:
        for p in graph.parents(n):
            if p in nodes:
                indeg[n] += 1
    q = [n for n in nodes if indeg[n] == 0]
    order, kids = [], graph.children_map()
    while q:
        u = q.pop()
        order.append(u)
        for c in kids[u]:
            if c in nodes:
                indeg[c] -= 1
                if indeg[c] == 0:
                    q.append(c)
    return order                                   # parents before children


def influence(graph, x, gamma) -> dict:
    anc = graph.ancestors(x) | {x}
    order = _topo(graph, anc)                       # 父在前
    infl = {a: 0.0 for a in anc}
    infl[x] = 1.0
    for n in reversed(order):                       # 子在前(x 侧先)→ 处理 n 时其贡献已齐
        ps = [p for p in graph.parents(n) if p in anc]
        if ps:
            share = infl[n] * gamma / len(ps)       # /K 等分给父代
            for p in ps:
                infl[p] += share
    return infl


def cosine(wx: dict, wy: dict) -> float:
    keys = set(wx) & set(wy)
    dot = sum(wx[k] * wy[k] for k in keys)
    nx = math.sqrt(sum(v * v for v in wx.values()))
    ny = math.sqrt(sum(v * v for v in wy.values()))
    if nx == 0 or ny == 0:
        return 0.0
    return dot / (nx * ny)


def c_dag(graph, x, y, gamma) -> float:
    wx = influence(graph, x, gamma)
    wy = influence(graph, y, gamma)
    # For distinct candidate nodes, each unit self-coordinate contributes to
    # only one norm and can never contribute to the dot product.  Keeping it
    # makes every pair look artificially orthogonal (c_dag has a large floor).
    # Complementarity should compare ancestry, so remove each target itself.
    wx.pop(x, None)
    wy.pop(y, None)
    return 1.0 - cosine(wx, wy)
