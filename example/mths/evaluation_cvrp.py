# Module Name: TSPEvaluation
# Last Revision: 2025/2/16
# Description: Evaluates the constructive heuristic for Traveling Salseman Problem (TSP).
#              Given a set of locations,
#              the goal is to find optimal route to travel all locations and back to start point
#              while minimizing the total travel distance.
#              This module is part of the LLM4AD project (https://github.com/Optima-CityU/llm4ad).
#
# Parameters:
#    - timeout_seconds: Maximum allowed time (in seconds) for the evaluation process: int (default: 30).
#    - n_instance: Number of problem instances to generate: int (default: 16).
#    - problem_size: Number of customers to serve: int (default: 50).
#
#
# References:
#   - Fei Liu, Xialiang Tong, Mingxuan Yuan, and Qingfu Zhang.
#     "Algorithm Evolution using Large Language Model." arXiv preprint arXiv:2311.15249 (2023).
#
# ------------------------------- Copyright --------------------------------
# Copyright (c) 2025 Optima Group.
#
# Permission is granted to use the LLM4AD platform for research purposes.
# All publications, software, or other works that utilize this platform
# or any part of its codebase must acknowledge the use of "LLM4AD" and
# cite the following reference:
#
# Fei Liu, Rui Zhang, Zhuoliang Xie, Rui Sun, Kai Li, Xi Lin, Zhenkun Wang,
# Zhichao Lu, and Qingfu Zhang, "LLM4AD: A Platform for Algorithm Design
# with Large Language Model," arXiv preprint arXiv:2412.17287 (2024).
#
# For inquiries regarding commercial use or licensing, please contact
# http://www.llm4ad.com/contact.html
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
from __future__ import annotations

import copy
from typing import Any
import matplotlib.pyplot as plt
import numpy as np
import traceback
from llm4ad.base import Evaluation
from llm4ad.task.optimization.cvrp_construct.get_instance import GetData
from template_cvrp import template_program, task_description
import pickle as pkl
import time

class CVRPEvaluation(Evaluation):
    def __init__(self,
                 timeout_seconds=20,
                 return_list = True,
                 **kwargs):

        super().__init__(
            template_program=template_program,
            task_description=task_description,
            use_numba_accelerate=False,
            timeout_seconds=timeout_seconds
        )

        #getData = GetData(self.n_instance, self.problem_size, self.capacity)
        self.n_instance = 64
        self._datasets = pkl.load(open("./cvrp_instances_train.pkl",'rb'))
        self._datasets = self._datasets[:self.n_instance]
        # Set the seed for numpy's random number generator
        #np.random.seed(2025)  # You can use any number here as the seed
        # Assuming self._datasets is a list or an array of datasets
        # First, generate 10 random indices from the range of self._datasets' length
        #indices = np.random.choice(len(self._datasets), size=self.n_instance, replace=False)

        # Now use these indices to select the datasets
        #self._datasets = [self._datasets[i] for i in indices]
        print("test")


    def tour_cost(self, instance, solution):
        cost = 0
        for j in range(len(solution) - 1):
            cost += np.linalg.norm(instance[int(solution[j])] - instance[int(solution[j + 1])])
        cost += np.linalg.norm(instance[int(solution[-1])] - instance[int(solution[0])])
        return cost

    def check_feasibility(self, solution, demands, vehicle_capacity):
        """
        Check if the given CVRP solution is feasible given the one-dimensional list format.

        Args:
        solution (list of int): The solution where all routes are concatenated into a single list,
                                with '0' representing the depot and marking the start and end of routes.
        demands (list of int): The demand of each customer node.
        vehicle_capacity (int): The maximum capacity of each vehicle.

        Returns:
        bool: True if the solution is feasible, False otherwise.
        """
        if solution[0] != 0 or solution[-1] != 0:
            # Solution must start and end at the depot
            return False

        current_demand = 0
        visited_nodes = set()
        route_start = True

        for i in range(1, len(solution)):
            node = solution[i]

            if node == 0:
                if current_demand > vehicle_capacity + 1E-8:
                    # Check if the previous route exceeded vehicle capacity
                    print("Check if the previous route exceeded vehicle capacity")
                    return False
                current_demand = 0  # Reset demand for the next route
                route_start = True  # Mark the start of a new route
            else:
                if node in visited_nodes or node >= len(demands):
                    # Ensure each customer node is visited only once and node index is valid
                    print("Ensure each customer node is visited only once and node index is valid")
                    return False
                visited_nodes.add(node)
                current_demand += demands[node]
                route_start = False

                if current_demand > vehicle_capacity + 1E-8:
                    # Check if the current demand exceeds the vehicle capacity
                    print("Check if the current demand exceeds the vehicle capacity")
                    # print(visited_nodes)
                    # print(current_demand)
                    # print(vehicle_capacity)
                    return False

        # print(len(visited_nodes))
        # print(len(demands))
        # input()

        # Ensure all customer nodes are visited exactly once
        if len(visited_nodes) != len(demands) - 1:  # demands include the depot with typically zero demand
            return False

        return True

    def evaluate(self, alg):

        start_time = time.time()
        try:

            dis_list = []

            for coordinates, distance_matrix, demands, vehicle_capacity, baseline in self._datasets:

                cvrp_solver = alg(coordinates, distance_matrix, demands, vehicle_capacity)


                cvrp_solution = cvrp_solver.solve()

                # print(cvrp_solution)

                # Check if solution is multi-dimensional (nested lists)
                if isinstance(cvrp_solution, list) and len(cvrp_solution) > 0 and isinstance(cvrp_solution[0], list):
                    # Solution contains multiple routes as nested lists
                    cvrp_solution_cat = []
                    for route in cvrp_solution:
                        cvrp_solution_cat.extend(route)
                    cvrp_solution = cvrp_solution_cat
                    print("Flattened solution:", cvrp_solution_cat)
                # else:
                    # # Solution is already one-dimensional
                    # print("Solution is already one-dimensional")

                if not self.check_feasibility(cvrp_solution,demands, vehicle_capacity):
                    print("invalid solution!")
                    return None

                LLM_dis = self.tour_cost(coordinates, cvrp_solution)

                dis_list.append((LLM_dis-baseline)/baseline)


                #print(f"gap = {(LLM_dis-baseline)/baseline}, time cost = {time.time()-start_time}")


            ave_dis = np.average(dis_list)

        except Exception as e:
            print(f"Error occurred: {e}")
            print("Traceback:")
            traceback.print_exc()
            return None

        return -ave_dis

    def evaluate_program(self, program_str: str, callable_func: callable) -> Any | None:
        g = {}
        exec(program_str, g)
        class_callable = g['CVRPSolver']
        return self.evaluate(class_callable)


if __name__ == '__main__':
    template_program = '''
# our updated program here
import numpy as np
import random
import math
from collections import defaultdict, Counter
import time

class CVRPSolver:
    def __init__(self, coordinates: np.ndarray, distance_matrix: np.ndarray, demands: list, vehicle_capacity: int):
        """
        Initialize the CVRP solver.

        Args:
            coordinates: Numpy array of shape (n, 2) containing the (x, y) coordinates of each node, including the depot.
            distance_matrix: Numpy array of shape (n, n) containing pairwise distances between nodes.
            demands: List of integers representing the demand of each node (first node is typically the depot with zero demand).
            vehicle_capacity: Integer representing the maximum capacity of each vehicle.
        """
        self.coordinates = coordinates
        self.distance_matrix = distance_matrix
        self.demands = demands
        self.vehicle_capacity = vehicle_capacity

        self.n = len(coordinates)
        if len(demands) != self.n or distance_matrix.shape != (self.n, self.n):
            raise ValueError("Dimensions of inputs do not match.")

        # Quick impossibility check
        max_demand = max(demands[1:]) if self.n > 1 else 0
        if max_demand > self.vehicle_capacity:
            raise ValueError("At least one customer's demand exceeds vehicle capacity; problem infeasible.")

        # Parameters (tunable)
        self.max_iter = max(200, 30 * int(math.sqrt(self.n)))  # main loop iterations
        self.population_size = max(10, min(40, self.n))  # number of candidate solutions kept
        self.elite_size = max(5, min(15, self.population_size // 3))
        self.random_seed = 42
        random.seed(self.random_seed)
        np.random.seed(self.random_seed)

    # ---------------- Utility functions ----------------

    def solution_to_routes(self, sol):
        """
        Convert flat solution list with 0 separators into list of routes (list of lists, excluding depot).
        """
        routes = []
        curr = []
        for v in sol:
            if v == 0:
                if curr:
                    routes.append(curr)
                    curr = []
            else:
                curr.append(v)
        # In case list doesn't end with 0
        if curr:
            routes.append(curr)
        return routes

    def routes_to_solution(self, routes):
        """
        Convert list of routes (lists of nodes excluding depot) to flat solution with 0 separators.
        """
        sol = []
        for r in routes:
            sol.append(0)
            sol.extend(r)
        sol.append(0)
        return sol

    def route_cost(self, route):
        if not route:
            return 0.0
        cost = 0.0
        prev = 0
        for v in route:
            cost += self.distance_matrix[prev, v]
            prev = v
        cost += self.distance_matrix[prev, 0]
        return cost

    def solution_cost(self, routes):
        return sum(self.route_cost(r) for r in routes)

    def route_load(self, route):
        return sum(self.demands[v] for v in route)

    def is_feasible(self, routes):
        # Check all customers visited once and capacity constraints
        visited = set()
        for r in routes:
            load = 0
            for v in r:
                if v in visited:
                    return False
                visited.add(v)
                load += self.demands[v]
            if load > self.vehicle_capacity:
                return False
        # check all customers visited
        all_customers = set(range(1, self.n))
        return visited == all_customers

    # ---------------- Initial solution generators ----------------

    def nearest_neighbor_solution(self, randomized=False):
        # Greedy build: start new route, pick nearest unvisited within capacity
        unvisited = set(range(1, self.n))
        routes = []
        while unvisited:
            curr = 0
            load = 0
            route = []
            while True:
                # find candidates within capacity
                candidates = [v for v in unvisited if load + self.demands[v] <= self.vehicle_capacity]
                if not candidates:
                    break
                # choose nearest (optionally randomized among best k)
                dists = [(self.distance_matrix[curr, v], v) for v in candidates]
                dists.sort()
                if randomized and len(dists) > 3 and random.random() < 0.5:
                    # choose among top-3 randomly
                    pick = random.choice(dists[:3])[1]
                else:
                    pick = dists[0][1]
                route.append(pick)
                unvisited.remove(pick)
                load += self.demands[pick]
                curr = pick
            routes.append(route)
        return routes

    def savings_solution(self):
        # Clarke-Wright Savings heuristic (basic)
        n = self.n
        savings = []
        for i in range(1, n):
            for j in range(i + 1, n):
                s = self.distance_matrix[0, i] + self.distance_matrix[0, j] - self.distance_matrix[i, j]
                savings.append((s, i, j))
        savings.sort(reverse=True)
        # initial routes: each customer alone
        routes = {i: [i] for i in range(1, n)}
        loads = {i: self.demands[i] for i in range(1, n)}
        def find_route_containing(node):
            for k, r in routes.items():
                if node in r:
                    return k, r
            return None, None
        for s, i, j in savings:
            ri_key, ri = find_route_containing(i)
            rj_key, rj = find_route_containing(j)
            if ri_key is None or rj_key is None or ri_key == rj_key:
                continue
            # check if i is at one end of ri and j at one end of rj (to preserve sequence)
            if (ri[0] == i or ri[-1] == i) and (rj[0] == j or rj[-1] == j):
                # capacity check
                if loads[ri_key] + loads[rj_key] <= self.vehicle_capacity:
                    # merge: preserve order
                    if ri[-1] == i and rj[0] == j:
                        new_route = ri + rj
                    elif ri[0] == i and rj[-1] == j:
                        new_route = rj + ri
                    elif ri[-1] == i and rj[-1] == j:
                        new_route = ri + rj[::-1]
                    elif ri[0] == i and rj[0] == j:
                        new_route = rj[::-1] + ri
                    else:
                        continue
                    new_key = min(ri_key, rj_key)
                    routes[new_key] = new_route
                    loads[new_key] = loads[ri_key] + loads[rj_key]
                    # remove the other
                    del routes[max(ri_key, rj_key)]
                    del loads[max(ri_key, rj_key)]
        # gather final routes
        final = list(routes.values())
        return final

    def random_greedy_solution(self):
        # Random perturbation of nearest neighbor
        return self.nearest_neighbor_solution(randomized=True)

    def generate_initial_population(self):
        pop = []
        # deterministic seeds
        pop.append(self.savings_solution())
        pop.append(self.nearest_neighbor_solution(randomized=False))
        # random variants
        for _ in range(max(0, self.population_size - len(pop))):
            if random.random() < 0.4:
                pop.append(self.random_greedy_solution())
            else:
                # shuffle nodes then greedy
                nodes = list(range(1, self.n))
                random.shuffle(nodes)
                # build routes greedily from shuffled order
                routes = []
                curr_route = []
                curr_load = 0
                for v in nodes:
                    if curr_load + self.demands[v] > self.vehicle_capacity:
                        routes.append(curr_route)
                        curr_route = [v]
                        curr_load = self.demands[v]
                    else:
                        curr_route.append(v)
                        curr_load += self.demands[v]
                if curr_route:
                    routes.append(curr_route)
                pop.append(routes)
        # ensure feasible and each customer once
        cleaned = []
        for routes in pop:
            cleaned_routes = self.repair_routes_unique_and_capacity(routes)
            cleaned.append(cleaned_routes)
        # remove duplicates by canonical string
        seen = set()
        unique = []
        for r in cleaned:
            key = tuple(tuple(route) for route in r)
            if key not in seen:
                seen.add(key)
                unique.append(r)
        return unique[:self.population_size]

    # ---------------- Repair and builders/modifiers ----------------

    def flatten_routes_nodes(self, routes):
        nodes = []
        for r in routes:
            nodes.extend(r)
        return nodes

    def repair_routes_unique_and_capacity(self, routes):
        # Ensure every customer appears exactly once and no over-capacity.
        # Start by collecting nodes and missing nodes
        present = []
        for r in routes:
            present.extend(r)
        present_counts = Counter(present)
        # remove duplicates: keep first occurrence
        seen = set()
        cleaned_routes = []
        for r in routes:
            newr = []
            for v in r:
                if v not in seen:
                    newr.append(v)
                    seen.add(v)
            if newr:
                cleaned_routes.append(newr)
        missing = [v for v in range(1, self.n) if v not in seen]
        # try to insert missing greedily by cheapest insertion (respecting capacity)
        for v in missing:
            best = None
            best_inc = None
            best_pos = None
            best_route_idx = None
            for idx, r in enumerate(cleaned_routes):
                load = self.route_load(r)
                if load + self.demands[v] > self.vehicle_capacity:
                    continue
                # try all insertion positions
                if not r:
                    inc = self.distance_matrix[0, v] * 2
                    pos = 0
                else:
                    # evaluate insert between nodes
                    inc = None
                    pos = 0
                    prev = 0
                    for k in range(len(r) + 1):
                        if k < len(r):
                            nxt = r[k]
                        else:
                            nxt = 0
                        cost_before = self.distance_matrix[prev, nxt]
                        cost_after = self.distance_matrix[prev, v] + self.distance_matrix[v, nxt]
                        delta = cost_after - cost_before
                        if inc is None or delta < inc:
                            inc = delta
                            pos = k
                        if k < len(r):
                            prev = r[k]
                if best_inc is None or inc < best_inc:
                    best_inc = inc
                    best = r
                    best_route_idx = idx
                    best_pos = pos
            if best is None:
                # create new route
                cleaned_routes.append([v])
            else:
                cleaned_routes[best_route_idx].insert(best_pos, v)
        # now fix any overloads by splitting overloaded routes
        final_routes = []
        for r in cleaned_routes:
            if self.route_load(r) <= self.vehicle_capacity:
                final_routes.append(r)
            else:
                # split greedily: build new routes from r's nodes
                nodes = r[:]
                nodes_sorted = sorted(nodes, key=lambda x: -self.demands[x])
                # simple pack into new routes
                packs = []
                for v in nodes_sorted:
                    placed = False
                    for p in packs:
                        if sum(self.demands[u] for u in p) + self.demands[v] <= self.vehicle_capacity:
                            p.append(v)
                            placed = True
                            break
                    if not placed:
                        packs.append([v])
                # try to order each pack by nearest neighbor
                ordered_packs = []
                for p in packs:
                    ordered = self._order_route_by_nn(p)
                    ordered_packs.append(ordered)
                final_routes.extend(ordered_packs)
        return final_routes

    def _order_route_by_nn(self, nodes):
        if not nodes:
            return []
        nodes = set(nodes)
        curr = random.choice(list(nodes))
        order = [curr]
        nodes.remove(curr)
        while nodes:
            nextv = min(nodes, key=lambda x: self.distance_matrix[curr, x])
            order.append(nextv)
            nodes.remove(nextv)
            curr = nextv
        return order

    # ---------------- Cooperative learner and equilibrium ----------------

    class Learner:
        def __init__(self):
            self.edge_counts = Counter()  # ordered edges (a,b)
            self.seq_counts = Counter()   # sequences as tuples
            self.total_updates = 0

        def register_solution(self, routes, weight=1.0):
            for r in routes:
                if not r:
                    continue
                # edges including depot edges
                prev = 0
                for v in r:
                    self.edge_counts[(prev, v)] += weight
                    prev = v
                self.edge_counts[(prev, 0)] += weight
                # sequences up to length 3 (customer-only)
                L = len(r)
                for l in range(2, min(4, L + 1)):  # length 2..3
                    for i in range(0, L - l + 1):
                        seq = tuple(r[i:i + l])
                        self.seq_counts[seq] += weight
            self.total_updates += 1

        def top_edges(self, k=50):
            return [e for e, _ in self.edge_counts.most_common(k)]

        def top_seqs(self, k=50, max_len=3):
            return [s for s, _ in self.seq_counts.most_common(k) if len(s) <= max_len]

        def edge_score(self, edge):
            return self.edge_counts.get(edge, 0)

    # ---------------- Operators ----------------

    def two_opt_route(self, route):
        """
        Optimized 2-opt first-improvement tailored for a single route (list of customer node ids).
        It uses cached nearest-neighbor lists across the full graph but restricts candidates to nodes
        present in the current route. Behavior preserves the original 2-opt semantics for open routes
        (with depot at ends) and avoids the trivial reversal that would flip the entire route.
        """
        if not route:
            return []
        n_r = len(route)
        if n_r <= 2:
            return route[:]

        import numpy as _np

        dm = self.distance_matrix  # local reference

        # best: numpy array of node ids (customers, excluding depot)
        best = _np.array(route, dtype=int)

        # position map for quick lookup: index by node id (size self.n), -1 for absent nodes
        pos = _np.full(self.n, -1, dtype=int)
        pos[best] = _np.arange(n_r, dtype=int)

        # choose k_nn proportional to log(self.n), bounded and deterministic-ish
        n_total = int(self.n)
        k_nn = int(max(10, min(n_total - 1, 5 + 5 * _np.log(max(2, n_total)))))
        k_nn = min(n_total - 1, k_nn)

        # build / reuse nearest-neighbor candidate lists for the full graph (exclude self)
        if not hasattr(self, "_nn_list") or getattr(self, "_nn_list_n", None) != n_total or getattr(self, "_nn_list_k", None) != k_nn:
            nn_list = [None] * n_total
            for u in range(n_total):
                row = dm[u]
                if k_nn + 1 < n_total:
                    part = _np.argpartition(row, k_nn + 1)[: (k_nn + 1)]
                    part = part[_np.argsort(row[part])]
                    part = part[part != u]
                    nn_list[u] = part[:k_nn].astype(int)
                else:
                    inds = _np.argsort(row)
                    inds = inds[inds != u]
                    nn_list[u] = inds[:k_nn].astype(int)
            self._nn_list = nn_list
            self._nn_list_n = n_total
            self._nn_list_k = k_nn
        nn_list = self._nn_list

        it = 0
        improved = True
        tol = 1e-12
        max_iters = 50  # limit iterations to keep runtime reasonable

        # main 2-opt loop (first improvement)
        while improved and it < max_iters:
            improved = False
            it += 1
            # scan edges (i, i+1) for i from 0 .. n_r-2
            for i in range(n_r - 1):
                a = int(best[i])
                b = int(best[i + 1])
                # iterate candidate c from nearest neighbors of a, but restrict to nodes present in route
                for c in nn_list[a]:
                    j = int(pos[int(c)])
                    # skip nodes not in route
                    if j == -1:
                        continue
                    # require a non-trivial segment i+1 .. j (j must be at least i+2)
                    if j <= i + 1:
                        continue
                    # skip trivial full reversal that recreates same route (closing edge)
                    if j == n_r - 1 and i == 0:
                        continue
                    c_node = int(best[j])
                    # determine d_node: node after c_node in route or depot(0) if c is last in route
                    if j == n_r - 1:
                        d_node = 0
                    else:
                        d_node = int(best[j + 1])
                    # delta = (a-c + b-d) - (a-b + c-d)
                    delta = (dm[a, c_node] + dm[b, d_node]) - (dm[a, b] + dm[c_node, d_node])
                    if delta < -tol:
                        # perform 2-opt reversal between indices seg_start .. seg_end-1
                        seg_start = i + 1
                        seg_end = j + 1  # exclusive
                        # reverse segment
                        seg = best[seg_start:seg_end]
                        rev = seg[::-1]
                        best[seg_start:seg_end] = rev
                        # vectorized update of pos for affected nodes
                        pos[rev.astype(int)] = _np.arange(seg_start, seg_end, dtype=int)
                        improved = True
                        break  # restart scanning from scratch after any improvement
                if improved:
                    break
            # continue until no improvement or max_iters exhausted
        return best.tolist()

    def intra_route_opt(self, routes):
        return [self.two_opt_route(r) for r in routes]

    def relocate_node(self, routes, from_idx, pos_in_from, to_idx, pos_in_to):
        # move node from one route to another at specified positions
        if from_idx == to_idx:
            r = routes[from_idx]
            v = r.pop(pos_in_from)
            # adjust pos if needed
            if pos_in_from < pos_in_to:
                pos_in_to -= 1
            r.insert(pos_in_to, v)
            routes[from_idx] = r
        else:
            v = routes[from_idx].pop(pos_in_from)
            routes[to_idx].insert(pos_in_to, v)
        return routes

    def random_relocation(self, routes, remove_k=3):
        # Remove remove_k nodes randomly and reinsert via cheapest insertion (randomized)
        all_nodes = [v for r in routes for v in r]
        if not all_nodes:
            return routes
        remove_k = min(remove_k, len(all_nodes))
        removed = random.sample(all_nodes, remove_k)
        # remove them
        new_routes = []
        for r in routes:
            newr = [v for v in r if v not in removed]
            if newr:
                new_routes.append(newr)
        # reinsert
        for v in removed:
            # find feasible insertions
            candidates = []
            for idx, r in enumerate(new_routes):
                load = self.route_load(r)
                if load + self.demands[v] > self.vehicle_capacity:
                    continue
                if not r:
                    inc = self.distance_matrix[0, v] * 2
                    pos = 0
                    candidates.append((inc, idx, pos))
                else:
                    prev = 0
                    for k in range(len(r) + 1):
                        nxt = r[k] if k < len(r) else 0
                        delta = self.distance_matrix[prev, v] + self.distance_matrix[v, nxt] - self.distance_matrix[prev, nxt]
                        candidates.append((delta, idx, k))
                        if k < len(r):
                            prev = r[k]
            if not candidates:
                # create new route
                new_routes.append([v])
            else:
                candidates.sort(key=lambda x: x[0])
                # choose among top few to keep diversity
                topk = min(3, len(candidates))
                pick = random.choice(candidates[:topk])
                _, ridx, pos = pick
                new_routes[ridx].insert(pos, v)
        return new_routes

    def swap_segments(self, routes, max_len=3):
        # choose two routes and swap small segments
        if len(routes) < 2:
            return routes
        a, b = random.sample(range(len(routes)), 2)
        ra = routes[a]
        rb = routes[b]
        if not ra or not rb:
            return routes
        la = random.randint(1, min(max_len, len(ra)))
        lb = random.randint(1, min(max_len, len(rb)))
        ia = random.randint(0, len(ra) - la)
        ib = random.randint(0, len(rb) - lb)
        sa = ra[ia:ia + la]
        sb = rb[ib:ib + lb]
        # perform swap and check capacity
        new_ra = ra[:ia] + sb + ra[ia + la:]
        new_rb = rb[:ib] + sa + rb[ib + lb:]
        if self.route_load(new_ra) <= self.vehicle_capacity and self.route_load(new_rb) <= self.vehicle_capacity:
            routes[a] = new_ra
            routes[b] = new_rb
        return routes

    # ---------------- Builder & Modifier ----------------

    def builder_propose(self, seed_routes, learner, intensity=1.0):
        # Build new solution by assembling high-value subsequences from learner,
        # then filling remaining nodes via cheapest insertion.
        top_seqs = learner.top_seqs(k=40, max_len=3)
        used = set()
        chains = []
        # pick some sequences probabilistically
        if top_seqs:
            picks = random.sample(top_seqs, min(len(top_seqs), max(1, int(3 * intensity))))
            for s in picks:
                seq = list(s)
                # ensure nodes not already used
                if any(v in used for v in seq):
                    continue
                # only take if feasible as route or part (sum demand <= capacity)
                if sum(self.demands[v] for v in seq) <= self.vehicle_capacity:
                    chains.append(seq[:])
                    for v in seq:
                        used.add(v)
        # add some routes from seed that are good and don't exceed capacity
        for r in seed_routes:
            if sum(self.demands[v] for v in r) <= self.vehicle_capacity:
                # keep route if it doesn't conflict with used
                if not any(v in used for v in r):
                    chains.append(r[:])
                    for v in r:
                        used.add(v)
        # remaining nodes
        remaining = [v for v in range(1, self.n) if v not in used]
        # shuffle remaining to diversify
        random.shuffle(remaining)
        # try to insert remaining into existing chains by cheapest insertion
        for v in remaining[:]:
            best = None
            best_inc = None
            best_chain_idx = None
            best_pos = None
            for idx, c in enumerate(chains):
                load = sum(self.demands[u] for u in c)
                if load + self.demands[v] > self.vehicle_capacity:
                    continue
                if not c:
                    inc = self.distance_matrix[0, v] * 2
                    pos = 0
                else:
                    prev = 0
                    inc = None
                    pos = 0
                    for k in range(len(c) + 1):
                        nxt = c[k] if k < len(c) else 0
                        delta = self.distance_matrix[prev, v] + self.distance_matrix[v, nxt] - self.distance_matrix[prev, nxt]
                        if inc is None or delta < inc:
                            inc = delta
                            pos = k
                        if k < len(c):
                            prev = c[k]
                if best_inc is None or inc < best_inc:
                    best_inc = inc
                    best_chain_idx = idx
                    best_pos = pos
            if best_chain_idx is not None:
                chains[best_chain_idx].insert(best_pos, v)
            else:
                # start new chain
                chains.append([v])
            remaining.remove(v)
        # final repair to ensure capacities and uniqueness
        new_routes = self.repair_routes_unique_and_capacity(chains)
        # small local improvement
        new_routes = self.intra_route_opt(new_routes)
        return new_routes

    def modifier_propose(self, seed_routes, intensity=1.0, diversify=False):
        # Modifier applies disruptive moves
        routes = [r[:] for r in seed_routes]
        # choose number of operations based on intensity
        ops = max(1, int(2 * intensity))
        for _ in range(ops):
            if random.random() < 0.5 or diversify:
                remove_k = 1 + int(2 * intensity)
                routes = self.random_relocation(routes, remove_k=remove_k)
            else:
                routes = self.swap_segments(routes, max_len=1 + int(2 * intensity))
        routes = self.repair_routes_unique_and_capacity(routes)
        return routes

    def reconcile_proposals(self, builder_routes, modifier_routes):
        # Reconcile by picking best non-overlapping route pieces from both
        # Strategy: collect candidate routes, greedily pick routes by best cost-per-demand ratio
        candidates = []
        for r in builder_routes + modifier_routes:
            load = self.route_load(r)
            if load == 0:
                continue
            cost = self.route_cost(r)
            candidates.append((cost / (load + 1e-6), cost, load, r))
        # sort by lowest cost per load
        candidates.sort(key=lambda x: x[0])
        picked_nodes = set()
        final = []
        for _, cost, load, r in candidates:
            if any(v in picked_nodes for v in r):
                continue
            # check capacity
            if load <= self.vehicle_capacity:
                final.append(r[:])
                for v in r:
                    picked_nodes.add(v)
        # remaining nodes fill
        remaining = [v for v in range(1, self.n) if v not in picked_nodes]
        # attempt to insert remaining nodes into final routes cheapest-first
        for v in remaining[:]:
            best = None
            best_inc = None
            best_ridx = None
            best_pos = None
            for idx, r in enumerate(final):
                if self.route_load(r) + self.demands[v] > self.vehicle_capacity:
                    continue
                if not r:
                    inc = self.distance_matrix[0, v] * 2
                    pos = 0
                else:
                    prev = 0
                    inc = None
                    pos = 0
                    for k in range(len(r) + 1):
                        nxt = r[k] if k < len(r) else 0
                        delta = self.distance_matrix[prev, v] + self.distance_matrix[v, nxt] - self.distance_matrix[prev, nxt]
                        if inc is None or delta < inc:
                            inc = delta
                            pos = k
                        if k < len(r):
                            prev = r[k]
                if best_inc is None or inc < best_inc:
                    best_inc = inc
                    best_ridx = idx
                    best_pos = pos
            if best_ridx is not None:
                final[best_ridx].insert(best_pos, v)
            else:
                final.append([v])
            remaining.remove(v)
        final = self.repair_routes_unique_and_capacity(final)
        final = self.intra_route_opt(final)
        return final

    # ---------------- Main solve method ----------------

    def solve(self) -> list:
        n = self.n
        start_time = time.time()
        # Initialize learner and equilibrium model
        learner = CVRPSolver.Learner()
        # Initialize population
        population = self.generate_initial_population()
        # Evaluate and keep elites
        scored = []
        for routes in population:
            cost = self.solution_cost(routes)
            scored.append((cost, routes))
            # update learner mildly
            learner.register_solution(routes, weight=0.2)
        scored.sort(key=lambda x: x[0])
        elites = [r for _, r in scored[:self.elite_size]]
        best_routes = elites[0]
        best_cost = self.solution_cost(best_routes)
        # Archive elites as list of tuples
        archive = [(best_cost, best_routes)]
        no_improve = 0
        iter_count = 0

        while iter_count < self.max_iter and (time.time() - start_time) < 9.0:
            iter_count += 1
            proposals = []
            # select seeds: mix of elites and random population
            seeds = []
            # keep top 2 elites
            seeds.extend(elites[:2])
            # random elite or population
            for _ in range(3):
                seeds.append(random.choice(elites))
            # add some random greedy
            for _ in range(2):
                seeds.append(self.random_greedy_solution())
            # generate proposals from builders and modifiers
            for seed in seeds:
                intensity = 1.0
                # adjust intensity from stagnation
                if no_improve > 20:
                    intensity = 1.5
                if random.random() < 0.6:
                    bprop = self.builder_propose(seed, learner, intensity=intensity)
                    proposals.append(bprop)
                else:
                    mprop = self.modifier_propose(seed, intensity=intensity, diversify=(no_improve > 30))
                    proposals.append(mprop)
                # produce some hybrid reconstructions between builder and modifier
                if random.random() < 0.7:
                    b = self.builder_propose(seed, learner, intensity=intensity)
                    m = self.modifier_propose(seed, intensity=intensity)
                    recon = self.reconcile_proposals(b, m)
                    proposals.append(recon)
            # evaluate and repair proposals
            evaluated = []
            for p in proposals:
                p_rep = self.repair_routes_unique_and_capacity(p)
                cost = self.solution_cost(p_rep)
                evaluated.append((cost, p_rep))
                # update learner stronger for good ones
                if cost < best_cost * 1.05:
                    learner.register_solution(p_rep, weight=1.0)
                else:
                    learner.register_solution(p_rep, weight=0.1)
            # combine with current elites and select new elites
            combined = evaluated + [(self.solution_cost(r), r) for r in elites]
            combined.sort(key=lambda x: x[0])
            # keep unique by node order signature
            new_elites = []
            seen = set()
            for cost, r in combined:
                key = tuple(tuple(route) for route in r)
                if key in seen:
                    continue
                seen.add(key)
                new_elites.append(r)
                if len(new_elites) >= self.elite_size:
                    break
            # update elites and best
            elites = new_elites
            current_best = elites[0]
            current_best_cost = self.solution_cost(current_best)
            if current_best_cost + 1e-9 < best_cost:
                best_cost = current_best_cost
                best_routes = current_best
                archive.append((best_cost, best_routes))
                no_improve = 0
            else:
                no_improve += 1
            # adapt exploration: occasionally add more diversification if stagnating
            if no_improve > 40:
                # generate randomized perturbations and inject into elites
                for _ in range(3):
                    m = self.modifier_propose(random.choice(elites), intensity=2.0, diversify=True)
                    m = self.repair_routes_unique_and_capacity(m)
                    elites.append(m)
                # trim elites
                elites = elites[:self.elite_size]
                no_improve = 0
            # small intensification: local polish of top elites
            for i in range(min(2, len(elites))):
                polished = self.intra_route_opt(elites[i])
                polished = self.repair_routes_unique_and_capacity(polished)
                elites[i] = polished
            # limit loop time to keep solver fast
        # Post-processing: intensify and polish best solution
        polished = best_routes
        improved = True
        polish_iter = 0
        while improved and polish_iter < 50:
            polish_iter += 1
            improved = False
            # intra-route 2-opt
            new_polished = self.intra_route_opt(polished)
            # try relocations between routes greedily
            nodes = [v for r in new_polished for v in r]
            random.shuffle(nodes)
            for v in nodes:
                # find its route
                ridx = None
                pos = None
                for idx, r in enumerate(new_polished):
                    if v in r:
                        ridx = idx
                        pos = r.index(v)
                        break
                if ridx is None:
                    continue
                # try moving to every other route and positions
                best_conf = None
                best_routes = None
                for j in range(len(new_polished)):
                    if j == ridx:
                        continue
                    rfrom = new_polished[ridx][:]
                    rto = new_polished[j][:]
                    vload = self.demands[v]
                    if sum(self.demands[u] for u in rto) + vload > self.vehicle_capacity:
                        continue
                    rfrom.pop(pos)
                    # try insert positions
                    for k in range(len(rto) + 1):
                        candidate_routes = [r[:] for r in new_polished]
                        candidate_routes[ridx] = rfrom[:]
                        candidate_routes[j] = rto[:]
                        candidate_routes[j].insert(k, v)
                        candidate_routes = self.repair_routes_unique_and_capacity(candidate_routes)
                        if not self.is_feasible(candidate_routes):
                            continue
                        if best_conf is None or self.solution_cost(candidate_routes) + 1e-9 < best_conf:
                            best_conf = self.solution_cost(candidate_routes)
                            best_routes = candidate_routes
                if best_routes is not None and best_conf + 1e-9 < self.solution_cost(new_polished):
                    new_polished = best_routes
                    improved = True
                    break
            if new_polished and self.solution_cost(new_polished) + 1e-9 < self.solution_cost(polished):
                polished = new_polished
                improved = True
            else:
                improved = False
        best_routes = polished
        final_solution = self.routes_to_solution(best_routes)
        # final sanity repair
        final_routes = self.repair_routes_unique_and_capacity(self.solution_to_routes(final_solution))
        final_solution = self.routes_to_solution(final_routes)
        return final_solution
            
    '''
    cvrp = CVRPEvaluation()
    results = cvrp.evaluate_program(template_program, None)
    print(results)
