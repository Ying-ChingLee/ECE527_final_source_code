#!/usr/bin/env python3
import math
import random
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple, Set
from collections import defaultdict, deque

# ---------------------------
# DFG Data Structures
# ---------------------------

class Node:
    def __init__(self, node_id: int, op: str):
        self.node_id = node_id
        self.op = op
        self.parents: List[int] = []
        self.children: List[int] = []

class TaskGraph:
    def __init__(self):
        self.nodes: Dict[int, Node] = {}
        self.num_nodes = 0
        self.latency: Dict[str, int] = {}

    def add_node(self, node_id: int, op: str):
        self.nodes[node_id] = Node(node_id, op)
        self.num_nodes += 1

    def add_edge(self, from_id: int, to_id: int):
        self.nodes[from_id].children.append(to_id)
        self.nodes[to_id].parents.append(from_id)

    def set_latency(self, op: str, latency: int):
        self.latency[op] = latency


def load_inputs(graph_file: str, timing_file: str) -> TaskGraph:
    tg = TaskGraph()

    # timing.txt: "<OP> <LAT>"
    with open(timing_file, "r") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                op, lat = line.split()[:2]
                tg.set_latency(op, int(lat))

    # graph.txt:
    # line0 = num_nodes
    # each line: "id , [ child0 child1 ... ] , OP"
    temp_edges = []
    with open(graph_file, "r") as f:
        lines = [ln.strip() for ln in f.readlines()]

    _ = int(lines[0])
    for line in lines[1:]:
        if not line or line.startswith("#"):
            continue
        parts = [p.strip() for p in line.split(",")]
        node_id = int(parts[0])
        children_str = parts[1].strip().strip("[]")
        children = [int(x) for x in children_str.split()] if children_str else []
        op = parts[2]
        tg.add_node(node_id, op)
        for c in children:
            temp_edges.append((node_id, c))

    for u, v in temp_edges:
        tg.add_edge(u, v)

    return tg

# ---------------------------
# ASAP / ALAP / Slack
# ---------------------------

def compute_asap(tg: TaskGraph) -> Tuple[Dict[int, int], int]:
    asap: Dict[int, int] = {}
    indeg = {i: len(tg.nodes[i].parents) for i in tg.nodes}
    q = deque(sorted([i for i, d in indeg.items() if d == 0]))

    while q:
        u = q.popleft()
        if not tg.nodes[u].parents:
            asap[u] = 0
        else:
            asap[u] = max(asap[p] + tg.latency[tg.nodes[p].op] for p in tg.nodes[u].parents)

        for v in tg.nodes[u].children:
            indeg[v] -= 1
            if indeg[v] == 0:
                q.append(v)
        if q:
            q = deque(sorted(q))

    finish = max(asap[i] + tg.latency[tg.nodes[i].op] for i in tg.nodes)
    return asap, finish


def compute_alap(tg: TaskGraph, target_latency: int) -> Dict[int, int]:
    alap: Dict[int, int] = {}
    outdeg = {i: len(tg.nodes[i].children) for i in tg.nodes}
    q = deque(sorted([i for i, d in outdeg.items() if d == 0]))

    while q:
        u = q.popleft()
        if not tg.nodes[u].children:
            alap[u] = target_latency - tg.latency[tg.nodes[u].op]
        else:
            alap[u] = min(alap[c] - tg.latency[tg.nodes[u].op] for c in tg.nodes[u].children)

        for p in tg.nodes[u].parents:
            outdeg[p] -= 1
            if outdeg[p] == 0:
                q.append(p)
        if q:
            q = deque(sorted(q))

    return alap


def compute_slack(asap: Dict[int, int], alap: Dict[int, int]) -> Dict[int, int]:
    return {i: alap[i] - asap[i] for i in asap}

# ---------------------------
# Metrics
# ---------------------------

@dataclass
class ScheduleMetrics:
    latency: int
    resource_requirements: Dict[str, int]
    resource_peak: float
    resource_variance: float
    power_switches: int

    def total_cost(self, weights: Dict[str, float]) -> float:
        total_resources = sum(self.resource_requirements.values())
        return (weights["resource"] * total_resources +
                weights["variance"] * self.resource_variance +
                weights["power"] * self.power_switches)

# ---------------------------
# HFSA Scheduler
# ---------------------------

class LatencyConstrainedScheduler:
    def __init__(self, tg: TaskGraph,
                 asap: Dict[int, int],
                 alap: Dict[int, int],
                 slack: Dict[int, int],
                 target_latency: int):

        self.tg = tg
        self.asap = asap
        self.alap = alap
        self.slack = slack
        self.target_latency = target_latency
        self.cpl = self._compute_cpl()

        # SA params
        self.initial_temp = 100.0
        self.final_temp = 0.1
        self.cooling_rate = 0.95
        self.iterations_per_temp = 50

        # cost weights
        self.weights = {"resource": 1.0, "variance": 0.2, "power": 0.2}

    def _compute_cpl(self) -> Dict[int, int]:
        indeg = {i: len(self.tg.nodes[i].parents) for i in self.tg.nodes}
        q = deque(sorted([i for i, d in indeg.items() if d == 0]))
        topo: List[int] = []

        while q:
            u = q.popleft()
            topo.append(u)
            for v in self.tg.nodes[u].children:
                indeg[v] -= 1
                if indeg[v] == 0:
                    q.append(v)
            if q:
                q = deque(sorted(q))

        cpl: Dict[int, int] = {}
        for u in reversed(topo):
            lat_u = self.tg.latency[self.tg.nodes[u].op]
            if not self.tg.nodes[u].children:
                cpl[u] = lat_u
            else:
                cpl[u] = lat_u + max(cpl[v] for v in self.tg.nodes[u].children)
        return cpl

    # -------- Force --------
    def compute_force(self, schedule: Dict[int, int]) -> Dict[int, Dict[int, float]]:
        forces: Dict[int, Dict[int, float]] = {}

        resource_distribution = defaultdict(lambda: defaultdict(int))
        for nid, t in schedule.items():
            if t >= 0:
                op = self.tg.nodes[nid].op
                lat = self.tg.latency[op]
                for c in range(t, t + lat):
                    resource_distribution[c][op] += 1

        for nid in self.tg.nodes:
            if schedule.get(nid, -1) >= 0:
                continue

            forces[nid] = {}
            op = self.tg.nodes[nid].op
            lat = self.tg.latency[op]

            for t in range(self.asap[nid], min(self.alap[nid] + 1, self.target_latency)):
                if t + lat > self.target_latency:
                    forces[nid][t] = float("inf")
                    continue

                # self-force
                self_force = 0.0
                for c in range(t, t + lat):
                    u = resource_distribution[c][op]
                    self_force += u * u
                self_force /= lat

                # predecessor force
                pred_force = 0.0
                for p in self.tg.nodes[nid].parents:
                    if schedule.get(p, -1) >= 0:
                        p_end = schedule[p] + self.tg.latency[self.tg.nodes[p].op]
                        if t > p_end:
                            pred_force += (t - p_end) * 0.5
                    else:
                        pred_force += 1.0 / (self.slack[p] + 1)

                # successor force
                succ_force = 0.0
                for s in self.tg.nodes[nid].children:
                    if schedule.get(s, -1) >= 0:
                        if t + lat > schedule[s]:
                            succ_force += 10.0
                    else:
                        succ_force += 1.0 / (self.cpl[s] + 1)

                # distribution force
                total_usage = 0
                for c in range(t, min(t + lat, self.target_latency)):
                    total_usage += sum(resource_distribution[c].values())
                dist_force = total_usage * 0.3

                forces[nid][t] = (1.5 * self_force +
                                  0.5 * pred_force +
                                  0.5 * succ_force +
                                  0.8 * dist_force)
        return forces

    def fds_initial_schedule(self) -> Dict[int, int]:
        schedule = {i: -1 for i in self.tg.nodes}
        scheduled: Set[int] = set()

        while len(scheduled) < self.tg.num_nodes:
            forces = self.compute_force(schedule)

            best_node, best_time, best_force = None, None, float("inf")

            for nid, time_forces in forces.items():
                # only schedule when all parents already scheduled
                if not all(p in scheduled for p in self.tg.nodes[nid].parents):
                    continue

                for t, f in time_forces.items():
                    # enforce precedence with already-scheduled parents
                    earliest = 0
                    for p in self.tg.nodes[nid].parents:
                        p_end = schedule[p] + self.tg.latency[self.tg.nodes[p].op]
                        earliest = max(earliest, p_end)
                    if t < earliest:
                        continue

                    if f < best_force:
                        best_force = f
                        best_node = nid
                        best_time = t

            # if no candidate found
            if best_node is None:
                break

            schedule[best_node] = best_time
            scheduled.add(best_node)

        return schedule

    # -------- Evaluation --------
    def evaluate_schedule(self, schedule: Dict[int, int]) -> ScheduleMetrics:
        latency = max(schedule[i] + self.tg.latency[self.tg.nodes[i].op] for i in self.tg.nodes)

        resource_usage = defaultdict(lambda: defaultdict(int))
        for nid, t in schedule.items():
            op = self.tg.nodes[nid].op
            lat = self.tg.latency[op]
            for c in range(t, t + lat):
                resource_usage[c][op] += 1

        resource_requirements = {}
        for op in self.tg.latency.keys():
            resource_requirements[op] = max(resource_usage[c][op] for c in range(latency))

        cycle_usage = []
        resource_peak = 0.0
        for c in range(latency):
            total = sum(resource_usage[c].values())
            cycle_usage.append(total)
            resource_peak = max(resource_peak, total)

        mean = sum(cycle_usage) / len(cycle_usage) if cycle_usage else 0.0
        resource_variance = (sum((u - mean) ** 2 for u in cycle_usage) / len(cycle_usage)) if cycle_usage else 0.0

        power_switches = 0
        for c in range(latency - 1):
            ops_c = {op for op, cnt in resource_usage[c].items() if cnt > 0}
            ops_n = {op for op, cnt in resource_usage[c + 1].items() if cnt > 0}
            power_switches += len(ops_c.symmetric_difference(ops_n))

        return ScheduleMetrics(latency, resource_requirements, resource_peak, resource_variance, power_switches)

    # -------- SA neighbor moves --------
    def _check_dependencies(self, schedule: Dict[int, int]) -> bool:
        for nid in self.tg.nodes:
            t = schedule[nid]
            for p in self.tg.nodes[nid].parents:
                p_end = schedule[p] + self.tg.latency[self.tg.nodes[p].op]
                if t < p_end:
                    return False
        return True

    def _check_latency(self, schedule: Dict[int, int]) -> bool:
        for nid in self.tg.nodes:
            t = schedule[nid]
            lat = self.tg.latency[self.tg.nodes[nid].op]
            if t + lat > self.target_latency:
                return False
        return True

    def _get_neighborhood(self, nid: int, radius: int = 2) -> Set[int]:
        visited = {nid}
        q = [(nid, 0)]
        while q:
            u, d = q.pop(0)
            if d >= radius:
                continue
            for v in self.tg.nodes[u].children + self.tg.nodes[u].parents:
                if v not in visited:
                    visited.add(v)
                    q.append((v, d + 1))
        return visited

    def perturb_schedule(self, schedule: Dict[int, int], temp: float) -> Dict[int, int]:
        new_s = schedule.copy()
        strategy = random.choice(["shift", "swap", "regional"]) if temp > 50 else random.choice(["shift", "shift", "swap"])

        if strategy == "shift":
            nid = random.choice(list(self.tg.nodes.keys()))
            lat = self.tg.latency[self.tg.nodes[nid].op]

            earliest = self.asap[nid]
            for p in self.tg.nodes[nid].parents:
                p_end = new_s[p] + self.tg.latency[self.tg.nodes[p].op]
                earliest = max(earliest, p_end)

            latest = min(self.alap[nid], self.target_latency - lat)
            for s in self.tg.nodes[nid].children:
                latest = min(latest, new_s[s] - lat)

            if earliest <= latest:
                new_s[nid] = random.randint(earliest, latest)

        elif strategy == "swap":
            nodes = list(self.tg.nodes.keys())
            random.shuffle(nodes)
            for i in range(len(nodes) - 1):
                n1, n2 = nodes[i], nodes[i + 1]
                if (n2 not in self.tg.nodes[n1].children and n1 not in self.tg.nodes[n2].children):
                    t1, t2 = new_s[n1], new_s[n2]
                    new_s[n1], new_s[n2] = t2, t1
                    if self._check_dependencies(new_s) and self._check_latency(new_s):
                        break
                    new_s[n1], new_s[n2] = t1, t2

        else:  # regional
            start = random.choice(list(self.tg.nodes.keys()))
            region = self._get_neighborhood(start, radius=2)
            forces = self.compute_force(new_s)
            for nid in region:
                if nid in forces and forces[nid]:
                    candidates = [(t, f) for t, f in forces[nid].items() if f < float("inf")]
                    if candidates:
                        new_s[nid] = min(candidates, key=lambda x: x[1])[0]

        return new_s

    def simulated_annealing(self, initial_schedule: Dict[int, int]) -> Tuple[Dict[int, int], List[float]]:
        current = initial_schedule.copy()
        current_cost = self.evaluate_schedule(current).total_cost(self.weights)

        best = current.copy()
        best_cost = current_cost

        history = [current_cost]
        temp = self.initial_temp

        while temp > self.final_temp:
            for _ in range(self.iterations_per_temp):
                neighbor = self.perturb_schedule(current, temp)
                neighbor_cost = self.evaluate_schedule(neighbor).total_cost(self.weights)

                delta = neighbor_cost - current_cost
                accept = (delta < 0) or (random.random() < math.exp(-delta / temp))

                if accept:
                    current = neighbor
                    current_cost = neighbor_cost
                    if current_cost < best_cost:
                        best = current.copy()
                        best_cost = current_cost

                history.append(current_cost)

            temp *= self.cooling_rate

        return best, history
    
    def print_resource_usage(self, schedule: Dict[int, int], tag: str) -> None:
        resource_usage = defaultdict(lambda: defaultdict(int))
        for node_id, t in schedule.items():
            op = self.tg.nodes[node_id].op
            lat = self.tg.latency[op]
            for cycle in range(t, t + lat):
                resource_usage[cycle][op] += 1

        latency = max(t + self.tg.latency[self.tg.nodes[nid].op] for nid, t in schedule.items())

        print(f"\n===== {tag} Resource Usage Per Cycle =====")
        ops = sorted(self.tg.latency.keys())
        header = "Cycle | " + " ".join([f"{op:>5s}" for op in ops]) + " | Total"
        print(header)
        print("-" * len(header))

        for cycle in range(latency):
            usage = resource_usage[cycle]
            counts = [usage.get(op, 0) for op in ops]
            total = sum(counts)
            row = f"{cycle:5d} | " + " ".join([f"{c:5d}" for c in counts]) + f" | {total:5d}"
            print(row)

    def run(self) -> None:
        t0 = time.perf_counter()
        init = self.fds_initial_schedule()
        init_m = self.evaluate_schedule(init)
        t1 = time.perf_counter()

        final, hist = self.simulated_annealing(init)
        final_m = self.evaluate_schedule(final)
        t2 = time.perf_counter()

        print("\n=== FDS (init) ===")
        print(f"Latency: {init_m.latency} / target {self.target_latency}")
        print(f"Resources: {init_m.resource_requirements}  (total={sum(init_m.resource_requirements.values())})")
        print(f"Variance: {init_m.resource_variance:.4f}  Switches: {init_m.power_switches}")
        print(f"Cost: {init_m.total_cost(self.weights):.4f}  Runtime: {(t1 - t0) * 1000:.2f} ms")
        self.print_resource_usage(init, tag="FDS")
        
        print("\n=== HFSA (after SA) ===")
        print(f"Latency: {final_m.latency} / target {self.target_latency}")
        print(f"Resources: {final_m.resource_requirements}  (total={sum(final_m.resource_requirements.values())})")
        print(f"Variance: {final_m.resource_variance:.4f}  Switches: {final_m.power_switches}")
        print(f"Cost: {final_m.total_cost(self.weights):.4f}  Runtime: {(t2 - t1) * 1000:.2f} ms")
        self.print_resource_usage(final, tag="HFSA")

        if sum(init_m.resource_requirements.values()) > 0:
            r0 = sum(init_m.resource_requirements.values())
            r1 = sum(final_m.resource_requirements.values())
            print(f"Resource reduction: {(r0 - r1) / r0 * 100:.2f}%")

# ---------------------------
# Main
# ---------------------------

def main():
    tg = load_inputs("graph.txt", "timing.txt")
    asap, asap_latency = compute_asap(tg)
    target_latency = asap_latency
    alap = compute_alap(tg, target_latency)
    slack = compute_slack(asap, alap)

    scheduler = LatencyConstrainedScheduler(tg, asap, alap, slack, target_latency)
    scheduler.run()

if __name__ == "__main__":
    main()
