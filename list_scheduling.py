#!/usr/bin/env python3
"""
Usage:
    python list_scheduling.py graph.txt timing.txt constraints.txt --list
"""

from __future__ import annotations

import time
import argparse
import os
import re
from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple
from collections import defaultdict


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



def _read_lines(path: str) -> List[str]:
    with open(path, 'r', encoding='utf-8') as f:
        raw = f.readlines()
    out: List[str] = []
    for line in raw:
        line = line.strip()
        if not line:
            continue
        if line.startswith('#'):
            continue
        out.append(line)
    return out


_DEF_LINE_RE = re.compile(r"^\s*(?P<op>[A-Za-z_][A-Za-z0-9_]*)\s+(?P<val>\d+)\s*$")


def parse_def_file(path: str) -> Dict[str, int]:
    
    lines = _read_lines(path)
    out: Dict[str, int] = {}
    for raw in lines:
        m = _DEF_LINE_RE.match(raw)
        if not m:
            continue
        op = m.group('op')
        val = int(m.group('val'))
        out[op] = val
    return out


_GRAPH_LINE_RE = re.compile(
    r"^\s*(?P<nid>\d+)\s*,\s*\[\s*(?P<ch>[0-9\s]*)\s*\]\s*,\s*(?P<op>[A-Za-z_][A-Za-z0-9_]*)\s*$"
)


def parse_graph_file(path: str) -> Dict[int, TaskNode]:
    
    lines = _read_lines(path)
    if not lines:
        raise ValueError("Empty graph file")

    num_nodes = int(lines[0])
    nodes: Dict[int, TaskNode] = {}

    for i in range(num_nodes):
        nodes[i] = TaskNode(nid=i, op="", children=[], parents=[])

    for raw in lines[1:]:
        m = _GRAPH_LINE_RE.match(raw)
        if not m:
            continue
        nid = int(m.group('nid'))
        ch_raw = m.group('ch').strip()
        op = m.group('op').strip()
        if ch_raw:
            children = [int(x) for x in ch_raw.split()]
        else:
            children = []
        nodes[nid].op = op
        nodes[nid].children = children

    for nid, node in nodes.items():
        for c in node.children:
            nodes[c].parents.append(nid)

    return nodes


def load_inputs(graph_path: str, timing_path: str, constraints_path: str) -> TaskGraph:
    latency = parse_def_file(timing_path)
    units = parse_def_file(constraints_path)
    nodes = parse_graph_file(graph_path)

    for n in nodes.values():
        if n.op not in units:
            units[n.op] = 999999

    return TaskGraph(nodes=nodes, latency=latency, units=units)


def topo_order(tg: TaskGraph) -> List[int]:
    indeg: Dict[int, int] = {i: len(n.parents) for i, n in tg.nodes.items()}
    q: List[int] = [i for i, d in indeg.items() if d == 0]
    order: List[int] = []
    while q:
        u = q.pop(0)
        order.append(u)
        for v in tg.nodes[u].children:
            indeg[v] -= 1
            if indeg[v] == 0:
                q.append(v)
    if len(order) != tg.num_nodes:
        raise ValueError("Graph is not a DAG or node count mismatch")
    return order


def topo_order_rev(tg: TaskGraph) -> List[int]:
    order = topo_order(tg)
    order.reverse()
    return order


def compute_asap(tg: TaskGraph) -> Tuple[Dict[int, int], int]:
    
    order = topo_order(tg)
    t: Dict[int, int] = {}
    finish: Dict[int, int] = {}

    for u in order:
        if not tg.nodes[u].parents:
            start = 0
        else:
            start = max(finish[p] for p in tg.nodes[u].parents)
        t[u] = start
        finish[u] = start + tg.latency[tg.nodes[u].op]

    finish_global = max(finish.values()) if finish else 0
    return t, finish_global


def compute_alap(tg: TaskGraph, finish_global: int) -> Dict[int, int]:
   
    order = topo_order_rev(tg)
    LS: Dict[int, int] = {}

    for u in order:
        dur = tg.latency[tg.nodes[u].op]
        if not tg.nodes[u].children:
            LS[u] = finish_global - dur
        else:
            LS[u] = min(LS[c] - dur for c in tg.nodes[u].children)
    return LS


def compute_slack(asap: Dict[int, int], alap: Dict[int, int]) -> Dict[int, int]:
    slk: Dict[int, int] = {}
    for i in asap:
        slk[i] = alap[i] - asap[i]
    return slk



def list_schedule(tg: TaskGraph,
                  asap: Dict[int, int],
                  slack: Dict[int, int]) -> Tuple[Dict[int, List[int]],
                                                  Dict[int, int],
                                                  Dict[int, int],
                                                  int]:
   
    nodes = tg.nodes
    dur = lambda i: tg.latency[nodes[i].op]

    unscheduled: Set[int] = set(nodes.keys())

    usage: Dict[str, List[int]] = defaultdict(list)

    run: Dict[int, int] = {}
    fin: Dict[int, int] = {}
    ready_map: Dict[int, List[int]] = {i: n.parents for i, n in nodes.items()}

    t = 0
    scheduled_count = 0

    def ensure_usage_len(op: str, t_max: int) -> None:
        arr = usage[op]
        if len(arr) < t_max:
            arr.extend([0] * (t_max - len(arr)))

    def can_start_at(i: int, t0: int) -> bool:
        op = nodes[i].op
        d = dur(i)
        if nodes[i].parents:
            if any(p not in fin for p in nodes[i].parents):
                return False
            earliest = max(fin[p] for p in nodes[i].parents)
            if t0 < earliest:
                return False
        t1 = t0 + d
        ensure_usage_len(op, t1)
        for tt in range(t0, t1):
            if usage[op][tt] >= tg.units[op]:
                return False
        return True

    def schedule_at(i: int, t0: int) -> None:
        op = nodes[i].op
        d = dur(i)
        t1 = t0 + d
        ensure_usage_len(op, t1)
        for tt in range(t0, t1):
            usage[op][tt] += 1
        run[i] = t0
        fin[i] = t1

    while scheduled_count < tg.num_nodes:
        candidates: List[int] = []
        for i in unscheduled:
            if nodes[i].parents:
                if any(p not in fin for p in nodes[i].parents):
                    continue
            candidates.append(i)

        if not candidates:
            t += 1
            continue

        candidates.sort(key=lambda i: (asap.get(i, 0), slack.get(i, 0)))

        scheduled_this_t = 0
        for i in candidates:
            if i not in unscheduled:
                continue
            if can_start_at(i, t):
                schedule_at(i, t)
                unscheduled.remove(i)
                scheduled_count += 1
                scheduled_this_t += 1

        t += 1  

    finish_global = max(fin.values()) if fin else 0
    return ready_map, run, fin, finish_global

def compute_peak_units(tg: TaskGraph,
                       run: Dict[int, int],
                       fin: Dict[int, int]) -> Dict[str, int]:
    if not run or not fin:
        return {}

    t_max = max(fin.values())
    peak: Dict[str, int] = defaultdict(int)

    for t in range(t_max):
        active: Dict[str, int] = defaultdict(int)
        for nid, start in run.items():
            end = fin[nid]
            if start <= t < end:
                op = tg.nodes[nid].op
                active[op] += 1
        for op, cnt in active.items():
            if cnt > peak[op]:
                peak[op] = cnt

    return dict(peak)


def compute_cycle_usage(tg: TaskGraph,
                        run: Dict[int, int],
                        fin: Dict[int, int]) -> List[Tuple[int, Dict[str, int]]]:
    
    usage: List[Tuple[int, Dict[str, int]]] = []
    if not run or not fin:
        return usage

    t_max = max(fin.values())
    for t in range(t_max):
        active: Dict[str, int] = defaultdict(int)
        for nid, start in run.items():
            end = fin[nid]
            if start <= t < end:
                op = tg.nodes[nid].op
                active[op] += 1
        usage.append((t, dict(active)))

    return usage

def compute_resource_variance(usage: List[Tuple[int, Dict[str, int]]]) -> float:
    if not usage:
        return 0.0
    totals = [sum(counts.values()) for _, counts in usage]
    if len(totals) <= 1:
        return 0.0
    mean = sum(totals) / len(totals)
    var = sum((x - mean) ** 2 for x in totals) / len(totals)
    return var


def compute_power_switches(usage: List[Tuple[int, Dict[str, int]]]) -> int:
    switches = 0
    for i in range(1, len(usage)):
        prev_ops = set(usage[i - 1][1].keys())
        curr_ops = set(usage[i][1].keys())
        switches += len(prev_ops.symmetric_difference(curr_ops))
    return switches

_OUTDIR = 'list_output/'


def write_list(tg: TaskGraph,
               run: Dict[int, int],
               fin: Dict[int, int],
               finish_global: int) -> None:
    
    p = os.path.join(_OUTDIR, 'list_schedule.txt')
    with open(p, 'w', encoding='utf-8') as f:
        f.write("Node   Op       Start  End\n")
        f.write("---------------------------------\n")
        for nid in sorted(tg.nodes.keys()):
            op = tg.nodes[nid].op
            s = run.get(nid, -1)
            e = fin.get(nid, -1)
            f.write(f"{nid:<6} {op:<8} {s:<6} {e:<6}\n")
        f.write(f"\nLatency: {finish_global} cycles\n")


def main() -> None:
    ap = argparse.ArgumentParser(add_help=True)
    ap.add_argument('graph')
    ap.add_argument('timing')
    ap.add_argument('constraints')
    ap.add_argument('--asap', action='store_true')
    ap.add_argument('--alap', action='store_true')
    ap.add_argument('--slack', action='store_true')
    ap.add_argument('--list', action='store_true')
    ap.add_argument('--all', action='store_true')
    args = ap.parse_args()

    print("=" * 70)
    print("Resource-Constrained List Scheduler")
    print("=" * 70)
    print()
    print("[Step 1] Loading input files")
    print("-" * 70)
    print(f"Graph file      : {args.graph}")
    print(f"Timing file     : {args.timing}")
    print(f"Constraint file : {args.constraints}")

    tg = load_inputs(args.graph, args.timing, args.constraints)

    print("\nTask graph information:")
    print(f"  Number of nodes   : {tg.num_nodes}")
    print(f"  Operation latency : {tg.latency}")
    print(f"  FU constraints    : {tg.units}")
    print()

    run_asap = args.asap or args.all
    run_alap = args.alap or args.all
    run_slack = args.slack or args.all
    run_list = args.list or args.all

    asap: Dict[int, int] = {}
    alap: Dict[int, int] = {}
    slk: Dict[int, int] = {}
    finish_global = 0

    if run_asap:
        print("[Step 2] Computing ASAP schedule")
        print("-" * 70)
        asap, finish_global = compute_asap(tg)
        print(f"ASAP minimum latency: {finish_global} cycles\n")

    if run_alap:
        print("[Step 3] Computing ALAP schedule")
        print("-" * 70)
        if not asap:
            asap, finish_global = compute_asap(tg)
        alap = compute_alap(tg, finish_global)
        print(f"ALAP schedule computed with target latency {finish_global} cycles\n")

    if run_slack:
        print("[Step 4] Computing slack")
        print("-" * 70)
        if not alap:
            if not asap:
                asap, finish_global = compute_asap(tg)
            alap = compute_alap(tg, finish_global)
        slk = compute_slack(asap, alap)
        print("Slack values computed.\n")

        if run_list:
            print("[Step 5] Running list scheduling")
            print("-" * 70)

        if not slk:
            if not alap:
                if not asap:
                    asap, finish_global = compute_asap(tg)
                alap = compute_alap(tg, finish_global)
            slk = compute_slack(asap, alap)

        t0 = time.perf_counter()
        ready, run, fin, finish_global = list_schedule(tg, asap, slk)
        t1 = time.perf_counter()
        runtime_ms = (t1 - t0) * 1000.0
        print(f"[Timing] List runtime: {runtime_ms:.3f} ms")


        latency = finish_global
        print(f"List scheduling latency: {latency} cycles\n")

        write_list(tg, run, fin, finish_global)

        peak_units = compute_peak_units(tg, run, fin)
        usage = compute_cycle_usage(tg, run, fin)
        resource_variance = compute_resource_variance(usage)
        power_switches = compute_power_switches(usage)
        
        total_peak = 0
        for _, counts in usage:
            total_peak = max(total_peak, sum(counts.values()))

        print("Peak functional-unit usage (per op type):")
        for op in sorted(peak_units.keys()):
            print(f"  {op:<6}: {peak_units[op]} unit(s)")
        print(f"  (sum of per-op peaks: {sum(peak_units.values())})\n")
        print(f"Resource usage variance: {resource_variance:.4f}")
        print(f"Operation-type switches: {power_switches}\n")

        print("Resource usage per cycle:")
        if peak_units:
            op_order = sorted(peak_units.keys())
        else:
            op_order = sorted({n.op for n in tg.nodes.values()})
        header = "Cycle    " + "  ".join(f"{op:>4}" for op in op_order) + "    Total"
        print(header)
        print("-" * len(header))
        for t, counts in usage:
            row = [counts.get(op, 0) for op in op_order]
            tot = sum(row)
            mark = "<--PEAK" if tot == total_peak and total_peak > 0 else ""
            print(f"{t:>5}    " + "  ".join(f"{v:>4}" for v in row) + f"    {tot:>5} {mark}")
        print()

        print("=" * 70)
        print("List scheduling summary")
        print("=" * 70)
        print(f"Latency           : {latency} cycles")
        print(f"Peak total usage  : {total_peak}")
        print(f"Resource variance    : {resource_variance:.4f}")
        print(f"Power switch count   : {power_switches}")
        print("Peak units per op :")
        for op in sorted(peak_units.keys()):
            print(f"  - {op:<6}: {peak_units[op]} unit(s)")
        print("=" * 70)
        print("Schedule written to list_schedule.txt")
        print("=" * 70)



if __name__ == '__main__':
    main()
