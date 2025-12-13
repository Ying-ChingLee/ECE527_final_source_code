#!/usr/bin/env python3

""" Usage: python list_scheduling.py graph.txt timing.txt constraints.txt --list """

from __future__ import annotations
import re
import time
from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple
from collections import defaultdict


# ==================
# Data structures
# ==================
@dataclass
class TaskNode:
    nid: int
    op: str
    children: List[int] = field(default_factory=list)
    parents: List[int] = field(default_factory=list)


@dataclass
class TaskGraph:
    nodes: Dict[int, TaskNode]
    latency: Dict[str, int]
    units: Dict[str, int]

    @property
    def num_nodes(self) -> int:
        return len(self.nodes)


# ============
# Parsers
# ============

_DEF_LINE_RE = re.compile(r"^\s*(\w+)\s+(\d+)\s*$")
_GRAPH_LINE_RE = re.compile(r"^\s*(\d+)\s*,\s*\[\s*([0-9\s]*)\s*\]\s*,\s*(\w+)\s*$")


def read_lines(path: str) -> List[str]:
    with open(path) as f:
        return [l.strip() for l in f if l.strip() and not l.startswith("#")]


def parse_def(path: str) -> Dict[str, int]:
    out = {}
    for l in read_lines(path):
        m = _DEF_LINE_RE.match(l)
        if m:
            out[m.group(1)] = int(m.group(2))
    return out


def parse_graph(path: str) -> Dict[int, TaskNode]:
    lines = read_lines(path)
    n = int(lines[0])
    nodes = {i: TaskNode(i, "") for i in range(n)}

    for l in lines[1:]:
        m = _GRAPH_LINE_RE.match(l)
        if not m:
            continue
        nid, ch, op = int(m.group(1)), m.group(2), m.group(3)
        nodes[nid].op = op
        nodes[nid].children = [int(x) for x in ch.split()] if ch else []

    for u in nodes.values():
        for v in u.children:
            nodes[v].parents.append(u.nid)

    return nodes


def load_graph(g, t, c) -> TaskGraph:
    latency = parse_def(t)
    units = parse_def(c)
    nodes = parse_graph(g)

    for n in nodes.values():
        if n.op not in units:
            units[n.op] = 10**9

    return TaskGraph(nodes, latency, units)


# ========================
# ASAP / ALAP / Slack
# ========================

def topo_order(tg: TaskGraph) -> List[int]:
    indeg = {i: len(n.parents) for i, n in tg.nodes.items()}
    q = [i for i, d in indeg.items() if d == 0]
    order = []

    while q:
        u = q.pop(0)
        order.append(u)
        for v in tg.nodes[u].children:
            indeg[v] -= 1
            if indeg[v] == 0:
                q.append(v)

    return order


def compute_asap(tg: TaskGraph) -> Tuple[Dict[int, int], int]:
    asap, finish = {}, {}
    for u in topo_order(tg):
        s = max((finish[p] for p in tg.nodes[u].parents), default=0)
        asap[u] = s
        finish[u] = s + tg.latency[tg.nodes[u].op]
    return asap, max(finish.values())


def compute_alap(tg: TaskGraph, T: int) -> Dict[int, int]:
    alap = {}
    for u in reversed(topo_order(tg)):
        d = tg.latency[tg.nodes[u].op]
        alap[u] = min((alap[c] - d for c in tg.nodes[u].children), default=T - d)
    return alap


def compute_slack(asap, alap):
    return {i: alap[i] - asap[i] for i in asap}


# ========================================
# Resource-constrained list scheduling
# ========================================

def list_schedule(tg: TaskGraph, asap, slack):
    run, fin = {}, {}
    usage = defaultdict(list)
    unscheduled: Set[int] = set(tg.nodes.keys())

    def ensure(op, t):
        if len(usage[op]) < t:
            usage[op].extend([0] * (t - len(usage[op])))

    def can_start(i, t):
        node = tg.nodes[i]
        d = tg.latency[node.op]

        if any(p not in fin for p in node.parents):
            return False
        if node.parents and t < max(fin[p] for p in node.parents):
            return False

        ensure(node.op, t + d)
        return all(usage[node.op][tt] < tg.units[node.op] for tt in range(t, t + d))

    def place(i, t):
        d = tg.latency[tg.nodes[i].op]
        ensure(tg.nodes[i].op, t + d)
        for tt in range(t, t + d):
            usage[tg.nodes[i].op][tt] += 1
        run[i], fin[i] = t, t + d

    t = 0
    while unscheduled:
        ready = [i for i in unscheduled if all(p in fin for p in tg.nodes[i].parents)]
        ready.sort(key=lambda i: (asap[i], slack[i]))

        for i in ready:
            if can_start(i, t):
                place(i, t)
                unscheduled.remove(i)

        t += 1

    return run, fin, max(fin.values())


# ==========================
# Metrics + timeline
# ==========================

def cycle_usage(tg, run, fin):
    T = max(fin.values())
    out = []
    for t in range(T):
        c = defaultdict(int)
        for i, s in run.items():
            if s <= t < fin[i]:
                c[tg.nodes[i].op] += 1
        out.append((t, dict(c)))
    return out


def resource_variance(usage):
    total = [sum(c.values()) for _, c in usage]
    m = sum(total) / len(total)
    return sum((x - m) ** 2 for x in total) / len(total)


def power_switching(usage):
    sw = 0
    for i in range(1, len(usage)):
        sw += len(set(usage[i-1][1]) ^ set(usage[i][1]))
    return sw


# ========
# Main
# ========

def main():
    import sys
    g, t, c = sys.argv[1:4]

    tg = load_graph(g, t, c)
    asap, T = compute_asap(tg)
    alap = compute_alap(tg, T)
    slack = compute_slack(asap, alap)

    t0 = time.perf_counter()
    run, fin, latency = list_schedule(tg, asap, slack)
    t1 = time.perf_counter()

    usage = cycle_usage(tg, run, fin)
    var = resource_variance(usage)
    sw = power_switching(usage)

    peak = defaultdict(int)
    for _, c in usage:
        for op, v in c.items():
            peak[op] = max(peak[op], v)

    total_resources = sum(peak.values())

    print(f"Latency           : {latency}")
    print(f"Total resources   : {total_resources}")
    print(f"Resource variance : {var:.4f}")
    print(f"Power switches    : {sw}")
    print(f"Runtime (ms)      : {(t1 - t0) * 1000:.3f}\n")

    print("Peak units per op:")
    for op in sorted(peak.keys()):
        print(f"  {op:<6}: {peak[op]} unit(s)")
    print()

    ops = sorted(peak)
    print("Cycle | " + " ".join(f"{op:>4}" for op in ops) + " | Total")
    for t, c in usage:
        row = [c.get(op, 0) for op in ops]
        print(f"{t:5d} | " + " ".join(f"{v:4d}" for v in row) + f" | {sum(row):5d}")



if __name__ == "__main__":
    main()

