#!/usr/bin/env python3

import math
import random
import time
import os
from typing import Dict, List, Tuple, Set
from dataclasses import dataclass
from collections import defaultdict, deque

import sys
import datetime

timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = open(f"run_log_{timestamp}.txt", "w")

class Logger:
    """Redirect print to both console and log file."""
    def write(self, message):
        sys.__stdout__.write(message)  
        log_file.write(message)         
    def flush(self):
        sys.__stdout__.flush()
        log_file.flush()

sys.stdout = Logger()

# Data Structure Definitions

class Node:
    """Node in the task graph"""
    def __init__(self, node_id: int, op: str):
        self.node_id = node_id
        self.op = op
        self.parents: List[int] = []
        self.children: List[int] = []
    
    def __repr__(self):
        return f"Node({self.node_id}, {self.op})"

class TaskGraph:
    """Task Graph (DFG)"""
    def __init__(self):
        self.nodes: Dict[int, Node] = {}
        self.num_nodes = 0
        self.latency: Dict[str, int] = {}
    
    def add_node(self, node_id: int, op: str):
        """Add a node"""
        self.nodes[node_id] = Node(node_id, op)
        self.num_nodes += 1
    
    def add_edge(self, from_id: int, to_id: int):
        """Add an edge (dependency relation)"""
        if from_id in self.nodes and to_id in self.nodes:
            self.nodes[from_id].children.append(to_id)
            self.nodes[to_id].parents.append(from_id)
    
    def set_latency(self, op: str, latency: int):
        """Set operation latency"""
        self.latency[op] = latency
    
    def __repr__(self):
        return f"TaskGraph(nodes={self.num_nodes})"

# Input File Loading

def load_inputs(graph_file: str, timing_file: str) -> TaskGraph:
    
    tg = TaskGraph()
    
    # Read timing.txt
    print(f"Loading {timing_file}...")
    with open(timing_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                parts = line.split()
                if len(parts) >= 2:
                    op, lat = parts[0], int(parts[1])
                    tg.set_latency(op, lat)
    print(f"  Latency settings: {tg.latency}")
    
    # Read graph.txt
    print(f"Loading {graph_file}...")
    with open(graph_file, 'r') as f:
        lines = f.readlines()
        
        num_nodes = int(lines[0].strip())
        print(f"  Number of nodes: {num_nodes}")
        
        for i in range(1, len(lines)):
            line = lines[i].strip()
            if not line or line.startswith('#'):
                continue
            
            parts = line.split(',')
            if len(parts) >= 3:
                node_id = int(parts[0].strip())
                
                children_str = parts[1].strip()
                children_str = children_str.strip('[]')
                if children_str:
                    children = [int(x.strip()) for x in children_str.split()]
                else:
                    children = []
                op = parts[2].strip()
                
                tg.add_node(node_id, op)
                
                for child_id in children:
                    if not hasattr(tg, '_temp_edges'):
                        tg._temp_edges = []
                    tg._temp_edges.append((node_id, child_id))
        
        if hasattr(tg, '_temp_edges'):
            for from_id, to_id in tg._temp_edges:
                tg.add_edge(from_id, to_id)
            delattr(tg, '_temp_edges')
    
    print(f"  Successfully loaded {tg.num_nodes} nodes")
    
    return tg

# ASAP/ALAP Computation

def compute_asap(tg: TaskGraph) -> Tuple[Dict[int, int], int]:
    """Compute ASAP schedule"""
    asap: Dict[int, int] = {}
    
    indeg = {i: len(tg.nodes[i].parents) for i in tg.nodes}
    q = deque(sorted([i for i, d in indeg.items() if d == 0]))
    
    while q:
        u = q.popleft()
        
        if not tg.nodes[u].parents:
            asap[u] = 0
        else:
            asap[u] = max(asap[p] + tg.latency[tg.nodes[p].op] 
                         for p in tg.nodes[u].parents)
        
        for v in tg.nodes[u].children:
            indeg[v] -= 1
            if indeg[v] == 0:
                q.append(v)
        
        if q:
            q = deque(sorted(q))
    
    finish_global = max(asap[i] + tg.latency[tg.nodes[i].op] 
                       for i in tg.nodes)
    
    return asap, finish_global

def compute_alap(tg: TaskGraph, target_latency: int) -> Dict[int, int]:
    
    alap: Dict[int, int] = {}
    outdeg = {i: len(tg.nodes[i].children) for i in tg.nodes}
    q = deque(sorted([i for i, d in outdeg.items() if d == 0]))
    
    while q:
        u = q.popleft()
        
        if not tg.nodes[u].children:
            alap[u] = target_latency - tg.latency[tg.nodes[u].op]
        else:
            alap[u] = min(alap[c] - tg.latency[tg.nodes[u].op]
                         for c in tg.nodes[u].children)
        
        for v in tg.nodes[u].parents:
            outdeg[v] -= 1
            if outdeg[v] == 0:
                q.append(v)
        
        if q:
            q = deque(sorted(q))
    
    return alap

def compute_slack(asap: Dict[int, int], alap: Dict[int, int]) -> Dict[int, int]:
    
    return {i: alap[i] - asap[i] for i in asap}

# ScheduleMetrics 
@dataclass
class ScheduleMetrics:
    latency: int
    resource_requirements: Dict[str, int]
    resource_peak: float
    resource_variance: float
    power_switches: int
    
    def total_cost(self, weights: Dict[str, float]) -> float:
        
        total_resources = sum(self.resource_requirements.values())
        return (weights['resource'] * total_resources +
                weights['variance'] * self.resource_variance +
                weights['power'] * self.power_switches)

# LatencyConstrainedScheduler

class LatencyConstrainedScheduler:
    """
    Latency-Constrained Scheduler: Force-Directed + Simulated Annealing
    
    Given a target latency, find minimum resources needed to meet the constraint.
    """
    
    def __init__(self, tg, asap: Dict[int,int], alap: Dict[int,int], 
                 slack: Dict[int,int], target_latency: int):
        self.tg = tg
        self.asap = asap
        self.alap = alap
        self.slack = slack
        self.target_latency = target_latency
        self.cpl = self._compute_critical_path_length()
        
        # SA parameters
        self.initial_temp = 100.0
        self.final_temp = 0.1
        self.cooling_rate = 0.95
        self.iterations_per_temp = 50
        
        # Multi-objective weights
        self.weights = {
            'resource': 1.0,   
            'variance': 0.2,   
            'power': 0.2     
        }
    
    def _compute_critical_path_length(self) -> Dict[int, int]:
        """Compute longest path from each node to sink"""
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
    
    def compute_force(self, schedule: Dict[int,int]) -> Dict[int, Dict[int,float]]:
        """
        Compute Force (modified for latency-constrained version)
        Goal: Distribute operations to minimize peak resource usage
        """
        forces: Dict[int, Dict[int,float]] = {}
        
        resource_distribution = defaultdict(lambda: defaultdict(int))
        for node_id, t in schedule.items():
            if t >= 0:
                op = self.tg.nodes[node_id].op
                lat = self.tg.latency[op]
                for cycle in range(t, t + lat):
                    resource_distribution[cycle][op] += 1
        
        for node_id in self.tg.nodes:
            if node_id in schedule and schedule[node_id] >= 0:
                continue
            
            forces[node_id] = {}
            op = self.tg.nodes[node_id].op
            lat = self.tg.latency[op]
            
            for t in range(self.asap[node_id], min(self.alap[node_id] + 1, self.target_latency)):
              
                finish_time = t + lat
                if finish_time > self.target_latency:
                    forces[node_id][t] = float('inf') 
                    continue
                
                self_force = 0.0
                for cycle in range(t, t + lat):
                    current_usage = resource_distribution[cycle][op]
                    self_force += current_usage ** 2  
                self_force /= lat
                
                pred_force = 0.0
                for pred_id in self.tg.nodes[node_id].parents:
                    if pred_id in schedule and schedule[pred_id] >= 0:
                        pred_end = schedule[pred_id] + self.tg.latency[self.tg.nodes[pred_id].op]
                        if t > pred_end:
                            pred_force += (t - pred_end) * 0.5
                    else:
                        pred_force += 1.0 / (self.slack[pred_id] + 1)
                
                succ_force = 0.0
                for succ_id in self.tg.nodes[node_id].children:
                    if succ_id in schedule and schedule[succ_id] >= 0:
                        if t + lat > schedule[succ_id]:
                            succ_force += 10.0
                    else:
                        succ_force += 1.0 / (self.cpl[succ_id] + 1)
                
                total_usage = 0
                for cycle in range(t, min(t + lat, self.target_latency)):
                    total_usage += sum(resource_distribution[cycle].values())
                dist_force = total_usage * 0.3
                
                forces[node_id][t] = (
                    1.5 * self_force +      
                    0.5 * pred_force +
                    0.5 * succ_force +
                    0.8 * dist_force       
                )
        
        return forces
    
    def fds_initial_schedule(self) -> Dict[int,int]:
        """Generate initial schedule using Force-Directed Scheduling"""
        schedule: Dict[int,int] = {i: -1 for i in self.tg.nodes}
        scheduled: Set[int] = set()
        
        while len(scheduled) < self.tg.num_nodes:
            forces = self.compute_force(schedule)
            
            best_node, best_time, best_force = None, None, float('inf')
            
            for node_id, time_forces in forces.items():
                if not all(p in scheduled for p in self.tg.nodes[node_id].parents):
                    continue
                
                for t, force in time_forces.items():
                    earliest = 0
                    for pred_id in self.tg.nodes[node_id].parents:
                        pred_end = schedule[pred_id] + self.tg.latency[self.tg.nodes[pred_id].op]
                        earliest = max(earliest, pred_end)
                    
                    if t < earliest:
                        continue
                    
                    if t + self.tg.latency[self.tg.nodes[node_id].op] > self.target_latency:
                        continue
                    
                    if force < best_force:
                        best_force = force
                        best_node = node_id
                        best_time = t
            
            if best_node is None:
                print("  WARNING: Cannot meet target latency constraint!")
                print(f"  Target latency {self.target_latency} is too tight.")
                candidates = [i for i in self.tg.nodes 
                            if i not in scheduled 
                            and all(p in scheduled for p in self.tg.nodes[i].parents)]
                if candidates:
                    best_node = min(candidates)
                    best_time = self.asap[best_node]
                else:
                    break
            
            schedule[best_node] = best_time
            scheduled.add(best_node)
        
        return schedule
    
    def evaluate_schedule(self, schedule: Dict[int,int]) -> ScheduleMetrics:
        """Evaluate schedule and calculate minimum resource requirements"""
        if any(t < 0 for t in schedule.values()):
            return ScheduleMetrics(
                latency=999999,
                resource_requirements={},
                resource_peak=999999,
                resource_variance=999999,
                power_switches=999999
            )
        
        # Calculate actual latency
        latency = max(schedule[i] + self.tg.latency[self.tg.nodes[i].op] 
                     for i in self.tg.nodes)
        
        # Calculate resource usage per cycle
        resource_usage = defaultdict(lambda: defaultdict(int))
        for node_id, t in schedule.items():
            op = self.tg.nodes[node_id].op
            lat = self.tg.latency[op]
            for cycle in range(t, t + lat):
                resource_usage[cycle][op] += 1
        
        # Calculate resources needed
        resource_requirements = {}
        for op in self.tg.latency.keys():
            max_usage = 0
            for cycle in range(latency):
                usage = resource_usage[cycle][op]
                max_usage = max(max_usage, usage)
            resource_requirements[op] = max_usage
        
        # Calculate overall resource metrics
        resource_peak = 0.0
        cycle_usage = []
        for cycle in range(latency):
            total = sum(resource_usage[cycle].values())
            cycle_usage.append(total)
            resource_peak = max(resource_peak, total)
        
        if cycle_usage:
            mean_usage = sum(cycle_usage) / len(cycle_usage)
            resource_variance = sum((u - mean_usage)**2 for u in cycle_usage) / len(cycle_usage)
        else:
            resource_variance = 0.0
        
        # Calculate power switches
        power_switches = 0
        for cycle in range(latency - 1):
            ops_current = {op for op, cnt in resource_usage[cycle].items() if cnt > 0}
            ops_next = {op for op, cnt in resource_usage[cycle + 1].items() if cnt > 0}
            power_switches += len(ops_current.symmetric_difference(ops_next))
        
        return ScheduleMetrics(
            latency=latency,
            resource_requirements=resource_requirements,
            resource_peak=resource_peak,
            resource_variance=resource_variance,
            power_switches=power_switches
        )
    
    def print_resource_usage(self, schedule: Dict[int, int], tag: str = "FDS"):
       
        from collections import defaultdict  
        resource_usage = defaultdict(lambda: defaultdict(int))
        for node_id, t in schedule.items():
            op = self.tg.nodes[node_id].op
            lat = self.tg.latency[op]
            for cycle in range(t, t + lat):
                resource_usage[cycle][op] += 1

        if not resource_usage:
            print(f"\n[{tag}] schedule is empty, nothing to print.")
            return

        print(f"\n===== {tag} Resource Usage Per Cycle =====")
        print("Cycle | ADD  SUB  MULT  DIV  | Total")
        print("---------------------------------------")

        for cycle in sorted(resource_usage.keys()):
            usage = resource_usage[cycle]
            add_cnt  = usage.get("ADD", 0)
            sub_cnt  = usage.get("SUB", 0)
            mult_cnt = usage.get("MULT", 0)
            div_cnt  = usage.get("DIV", 0)
            total = add_cnt + sub_cnt + mult_cnt + div_cnt

            print(f"{cycle:5d} |"
                  f"  {add_cnt:2d}   {sub_cnt:2d}   {mult_cnt:2d}    {div_cnt:2d}  |"
                  f"  {total:2d}")


    def perturb_schedule(self, schedule: Dict[int,int], temp: float) -> Dict[int,int]:
        
        new_schedule = schedule.copy()
        
        if temp > 50:
            strategy = random.choice(['shift', 'swap', 'regional'])
        else:
            strategy = random.choice(['shift', 'shift', 'swap'])
        
        if strategy == 'shift':
            node_id = random.choice(list(self.tg.nodes.keys()))
            
            earliest = self.asap[node_id]
            for pred_id in self.tg.nodes[node_id].parents:
                pred_end = new_schedule[pred_id] + self.tg.latency[self.tg.nodes[pred_id].op]
                earliest = max(earliest, pred_end)
            
            latest = min(self.alap[node_id], 
                        self.target_latency - self.tg.latency[self.tg.nodes[node_id].op])
            
            for succ_id in self.tg.nodes[node_id].children:
                succ_start = new_schedule[succ_id]
                latest = min(latest, succ_start - self.tg.latency[self.tg.nodes[node_id].op])
            
            if earliest <= latest:
                new_time = random.randint(earliest, latest)
                new_schedule[node_id] = new_time
        
        elif strategy == 'swap':
            nodes = list(self.tg.nodes.keys())
            random.shuffle(nodes)
            
            for i in range(len(nodes) - 1):
                n1, n2 = nodes[i], nodes[i+1]
                
                if (n2 not in self.tg.nodes[n1].children and 
                    n1 not in self.tg.nodes[n2].children):
                    
                    t1, t2 = new_schedule[n1], new_schedule[n2]
                    new_schedule[n1], new_schedule[n2] = t2, t1
                    
                    if self._check_dependencies(new_schedule) and self._check_latency_constraint(new_schedule):
                        break
                    else:
                        new_schedule[n1], new_schedule[n2] = t1, t2
        
        elif strategy == 'regional':
            start_node = random.choice(list(self.tg.nodes.keys()))
            region = self._get_neighborhood(start_node, radius=2)
            
            for node_id in region:
                if node_id in self.tg.nodes:
                    forces = self.compute_force(new_schedule)
                    if node_id in forces and forces[node_id]:
                        valid_times = [(t, f) for t, f in forces[node_id].items() if f < float('inf')]
                        if valid_times:
                            best_time = min(valid_times, key=lambda x: x[1])[0]
                            new_schedule[node_id] = best_time
        
        return new_schedule
    
    def _check_dependencies(self, schedule: Dict[int,int]) -> bool:
        
        for node_id in self.tg.nodes:
            t = schedule[node_id]
            for pred_id in self.tg.nodes[node_id].parents:
                pred_end = schedule[pred_id] + self.tg.latency[self.tg.nodes[pred_id].op]
                if t < pred_end:
                    return False
        return True
    
    def _check_latency_constraint(self, schedule: Dict[int,int]) -> bool:
   
        for node_id in self.tg.nodes:
            t = schedule[node_id]
            lat = self.tg.latency[self.tg.nodes[node_id].op]
            if t + lat > self.target_latency:
                return False
        return True
    
    def _get_neighborhood(self, node_id: int, radius: int) -> Set[int]:
       
        visited = {node_id}
        queue = [(node_id, 0)]
        
        while queue:
            curr, dist = queue.pop(0)
            if dist < radius:
                for child in self.tg.nodes[curr].children:
                    if child not in visited:
                        visited.add(child)
                        queue.append((child, dist + 1))
                for parent in self.tg.nodes[curr].parents:
                    if parent not in visited:
                        visited.add(parent)
                        queue.append((parent, dist + 1))
        
        return visited
    
    def simulated_annealing(self, initial_schedule: Dict[int,int]) -> Tuple[Dict[int,int], List[float]]:
       
        current = initial_schedule.copy()
        current_cost = self.evaluate_schedule(current).total_cost(self.weights)
        
        best = current.copy()
        best_cost = current_cost
        
        cost_history = [current_cost]
        temp = self.initial_temp
        
        iteration = 0
        while temp > self.final_temp:
            for _ in range(self.iterations_per_temp):
                neighbor = self.perturb_schedule(current, temp)
                neighbor_cost = self.evaluate_schedule(neighbor).total_cost(self.weights)
                
                delta = neighbor_cost - current_cost
                
                if delta < 0:
                    accept = True
                else:
                    accept_prob = math.exp(-delta / temp)
                    accept = random.random() < accept_prob
                
                if accept:
                    current = neighbor
                    current_cost = neighbor_cost
                    
                    if current_cost < best_cost:
                        best = current.copy()
                        best_cost = current_cost
                
                cost_history.append(current_cost)
                iteration += 1
            
            temp *= self.cooling_rate
            
            if iteration % 100 == 0:
                print(f"  Iteration {iteration}: T={temp:.2f}, Current={current_cost:.2f}, Best={best_cost:.2f}")
        
        return best, cost_history
    
    def schedule(self) -> Tuple[Dict[int,int], ScheduleMetrics, List[float]]:
        """Main scheduling method"""
        print("\n=== Phase 1: Force-Directed Scheduling (Initialization) ===")
        print(f"Target latency constraint: {self.target_latency} cycles")
        
        fds_t0 = time.perf_counter()
        initial_schedule = self.fds_initial_schedule()
        initial_metrics = self.evaluate_schedule(initial_schedule)
        fds_t1 = time.perf_counter()
        fds_runtime_ms = (fds_t1 - fds_t0) * 1000.0
        print(f"Initial solution (Pure FDS):")
        print(f"  Actual latency:          {initial_metrics.latency} cycles")
        print(f"  Resource requirements:   {initial_metrics.resource_requirements}")
        print(f"  Total resources:         {sum(initial_metrics.resource_requirements.values())} units")
        print(f"  Resource usage peak:     {initial_metrics.resource_peak:.2f}")
        print(f"  Resource usage variance: {initial_metrics.resource_variance:.4f}")
        print(f"  Power switch count:      {initial_metrics.power_switches}")
        print(f"  Total cost (FDS):        {initial_metrics.total_cost(self.weights):.4f}")
        print(f"  [Timing] Pure FDS runtime: {fds_runtime_ms:.3f} ms")

        self.print_resource_usage(initial_schedule, tag="FDS")
        
        print("\n=== Phase 2: Simulated Annealing (Resource Minimization) ===")
        final_schedule, cost_history = self.simulated_annealing(initial_schedule)
        final_metrics = self.evaluate_schedule(final_schedule)
        
        print(f"\nFinal solution (HFSA after SA):")
        print(f"  Actual latency:                 {final_metrics.latency} cycles")
        print(f"  Minimum resource requirements:  {final_metrics.resource_requirements}")
        print(f"  Total resources:                {sum(final_metrics.resource_requirements.values())} units")
        print(f"  Resource usage peak:            {final_metrics.resource_peak:.2f}")
        print(f"  Resource usage variance:        {final_metrics.resource_variance:.4f}")
        print(f"  Power switch count:             {final_metrics.power_switches}")
        print(f"  Total cost (HFSA):              {final_metrics.total_cost(self.weights):.4f}")

        self.print_resource_usage(final_schedule, tag="HFSA")
        
        initial_resources = sum(initial_metrics.resource_requirements.values())
        final_resources = sum(final_metrics.resource_requirements.values())
        if initial_resources > 0:
            improvement = (initial_resources - final_resources) / initial_resources * 100
            print(f"  Resource reduction:             {improvement:.2f}%")
        
        return final_schedule, final_metrics, cost_history



def main():
    """Main program"""
    print("=" * 70)
    print("Latency-Constrained Hybrid Scheduler")
    print("Force-Directed Scheduling + Simulated Annealing")
    print("=" * 70)
    print("\nTRADITIONAL FDS APPROACH:")
    print("  Input:  Task graph + Target latency")
    print("  Output: Minimum resources needed")
    print("  Goal:   Minimize resource requirements while meeting deadline")
    
    # Load input files 
    print("\n[Step 1] Loading input files")
    print("-" * 70)
    
    try:
        tg = load_inputs('graph.txt', 'timing.txt')
    except FileNotFoundError as e:
        print(f"Error: File not found: {e.filename}")
        print("Please ensure the following files exist in current directory:")
        print("  - graph.txt")
        print("  - timing.txt")
        print("\nNOTE: constraints.txt is NOT needed for latency-constrained version!")
        return
    
    print(f"\nTask graph information:")
    print(f"  Number of nodes: {tg.num_nodes}")
    print(f"  Operation latency: {dict(sorted(tg.latency.items()))}")
    
    # Calculate ASAP 
    print("\n[Step 2] Computing ASAP (minimum possible latency)")
    print("-" * 70)
    
    asap, asap_latency = compute_asap(tg)
    print(f"ASAP schedule: {dict(sorted(asap.items()))}")
    print(f"ASAP minimum latency: {asap_latency} cycles")
    
    # Get target latency 
    print("\n[Step 3] Setting target latency constraint")
    print("-" * 70)
    
    # Use ASAP latency 
    target_latency = asap_latency
    print(f"Using ASAP latency as target: {target_latency} cycles")
    print("(This finds minimum resources for fastest possible execution)")
    
    
    # 4. Compute ALAP and Slack
    alap = compute_alap(tg, target_latency)
    slack = compute_slack(asap, alap)
    
    print(f"ALAP schedule (for target={target_latency}): {dict(sorted(alap.items()))}")
    print(f"Slack values: {dict(sorted(slack.items()))}")
    
    critical_nodes = [i for i in slack if slack[i] == 0]
    print(f"Critical path nodes: {critical_nodes}")
    
    # Execute latency-constrained scheduling
    print("\n[Step 4] Executing latency-constrained scheduling")
    print("-" * 70)
    
    scheduler = LatencyConstrainedScheduler(tg, asap, alap, slack, target_latency)
        
    t0 = time.perf_counter()
    final_schedule, final_metrics, cost_history = scheduler.schedule()
    t1 = time.perf_counter()
    runtime_ms = (t1 - t0) * 1000.0
    print(f"[Timing] HFSA runtime: {runtime_ms:.3f} ms")
    # Output results
    print("\n" + "=" * 70)
    print("FINAL RESULTS: Minimum Resource Requirements")
    print("=" * 70)
    
    schedule_by_time = []
    for node_id in sorted(final_schedule.keys()):
        t = final_schedule[node_id]
        op = tg.nodes[node_id].op
        lat = tg.latency[op]
        schedule_by_time.append((t, node_id, op, lat))
    
    schedule_by_time.sort()
    
    print(f"\nSchedule (meeting {target_latency}-cycle constraint):")
    print(f"{'Node':<6} {'Op':<8} {'Start':<6} {'End':<6} {'Latency':<8}")
    print("-" * 70)
    for t, node_id, op, lat in schedule_by_time:
        print(f"{node_id:<6} {op:<8} {t:<6} {t+lat-1:<6} {lat:<8}")
    
    print(f"\n{'='*70}")
    print("MINIMUM RESOURCE REQUIREMENTS")
    print(f"{'='*70}")
    print(f"To meet {target_latency}-cycle latency constraint, you need:")
    print()
    
    total_resources = 0
    for op in sorted(final_metrics.resource_requirements.keys()):
        count = final_metrics.resource_requirements[op]
        total_resources += count
        print(f"  {op:8s}: {count} unit(s)")
    
    print(f"\n  TOTAL:    {total_resources} resource units")
    
    print(f"\nPerformance Metrics:")
    print("-" * 70)
    print(f"Actual completion time:  {final_metrics.latency} cycles")
    print(f"Target latency:          {target_latency} cycles")
    if final_metrics.latency <= target_latency:
        print(f"Status:                  MEETS CONSTRAINT")
    else:
        print(f"Status:                  VIOLATES CONSTRAINT")
    print(f"Resource usage peak:     {final_metrics.resource_peak:.2f}")
    print(f"Resource usage variance: {final_metrics.resource_variance:.2f}")
    print(f"Power switch count:      {final_metrics.power_switches}")
    
    # Resource usage 
    print(f"\n{'='*70}")
    print("RESOURCE USAGE ANALYSIS")
    print(f"{'='*70}")
    
    resource_usage_by_cycle = defaultdict(lambda: defaultdict(int))
    for node_id, t in final_schedule.items():
        op = tg.nodes[node_id].op
        lat = tg.latency[op]
        for cycle in range(t, t + lat):
            resource_usage_by_cycle[cycle][op] += 1
    
    print("\nResource usage per cycle:")
    print(f"{'Cycle':<8} {'ADD':<6} {'SUB':<6} {'MULT':<6} {'DIV':<6} {'Total':<6}")
    print("-" * 70)
    
    for cycle in sorted(resource_usage_by_cycle.keys()):
        usage = resource_usage_by_cycle[cycle]
        add_count = usage.get('ADD', 0)
        sub_count = usage.get('SUB', 0)
        mult_count = usage.get('MULT', 0)
        div_count = usage.get('DIV', 0)
        total = add_count + sub_count + mult_count + div_count
        
        marker = " <--PEAK" if total == final_metrics.resource_peak else ""
        print(f"{cycle:<8} {add_count:<6} {sub_count:<6} {mult_count:<6} {div_count:<6} {total:<6}{marker}")
    
    # Save results
    print("\n[Step 5] Saving results to files")
    print("-" * 70)
    
    output_file = 'hybrid_output/latency_constrained_results.txt'
    with open(output_file, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("Latency-Constrained Scheduling Results\n")
        f.write("=" * 70 + "\n\n")
        
        f.write(f"Target latency constraint: {target_latency} cycles\n")
        f.write(f"Actual latency achieved:   {final_metrics.latency} cycles\n\n")
        
        f.write("MINIMUM RESOURCE REQUIREMENTS:\n")
        f.write("-" * 70 + "\n")
        for op in sorted(final_metrics.resource_requirements.keys()):
            count = final_metrics.resource_requirements[op]
            f.write(f"  {op:8s}: {count} unit(s)\n")
        f.write(f"\n  TOTAL:    {total_resources} resource units\n\n")
        
        f.write("Schedule:\n")
        f.write("-" * 70 + "\n")
        f.write(f"{'Node':<6} {'Op':<8} {'Start':<6} {'End':<6} {'Latency':<8}\n")
        f.write("-" * 70 + "\n")
        for t, node_id, op, lat in schedule_by_time:
            f.write(f"{node_id:<6} {op:<8} {t:<6} {t+lat-1:<6} {lat:<8}\n")
        
        f.write("\nResource Usage Per Cycle:\n")
        f.write("-" * 70 + "\n")
        for cycle in sorted(resource_usage_by_cycle.keys()):
            usage = resource_usage_by_cycle[cycle]
            f.write(f"Cycle {cycle:2d}: ")
            for op in sorted(tg.latency.keys()):
                count = usage.get(op, 0)
                if count > 0:
                    f.write(f"{op}={count} ")
            f.write("\n")
    
    print(f"  Results saved to: {output_file}")
    
    output_graph_file = 'hybrid_output/schedule_graph.txt'
    with open(output_graph_file, 'w') as f:
        f.write(f"{tg.num_nodes}\n")
        for node_id in sorted(final_schedule.keys()):
            children = tg.nodes[node_id].children
            op = tg.nodes[node_id].op
            t = final_schedule[node_id]
            children_str = ' '.join(map(str, sorted(children))) if children else ''
            f.write(f"{node_id} , [ {children_str} ] , {op} , scheduled_at={t}\n")
    
    print(f"  Schedule graph saved to: {output_graph_file}")
    
    output_table_file = 'hybrid_output/schedule_table.csv'
    with open(output_table_file, 'w') as f:
        f.write("Node,Operation,Start_Cycle,End_Cycle,Latency,ASAP,ALAP,Slack,On_Critical_Path\n")
        for node_id in sorted(final_schedule.keys()):
            t = final_schedule[node_id]
            op = tg.nodes[node_id].op
            lat = tg.latency[op]
            is_critical = "Yes" if slack[node_id] == 0 else "No"
            f.write(f"{node_id},{op},{t},{t+lat-1},{lat},{asap[node_id]},{alap[node_id]},{slack[node_id]},{is_critical}\n")
    
    print(f"  Schedule table saved to: {output_table_file}")
    
    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)
    print(f"\nTo achieve {target_latency}-cycle latency, the minimum hardware")
    print(f"configuration required is:")
    print()
    for op in sorted(final_metrics.resource_requirements.keys()):
        count = final_metrics.resource_requirements[op]
        print(f"  - {count} {op} unit(s)")
    print(f"\nTotal: {total_resources} functional units needed.")
    print("\n" + "=" * 70)

if __name__ == "__main__":
    main()
