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
from __future__ import annotations

from typing import Any
import numpy as np
from llm4ad.base import Evaluation
from llm4ad.task.optimization.tsp_construct.get_instance import GetData
from template_tsp import template_program, task_description
import pickle as pkl
import random
__all__ = ['TSPEvaluation']


class TSPEvaluation(Evaluation):
    """Evaluator for traveling salesman problem."""

    def __init__(self,
                 timeout_seconds=30,
                 n_instance=64,
                 problem_size=50,
                 **kwargs):

        """
            Args:
                None
            Raises:
                AttributeError: If the data key does not exist.
                FileNotFoundError: If the specified data file is not found.
        """

        super().__init__(
            template_program=template_program,
            task_description=task_description,
            use_numba_accelerate=False,
            timeout_seconds=timeout_seconds
        )

        self.n_instance = 64
        self._datasets = pkl.load(open("./tsp_instances_train.pkl",'rb'))
        self._datasets = self._datasets[:self.n_instance]
        # Set the seed for numpy's random number generator
        np.random.seed(2025)  # You can use any number here as the seed
        # # Assuming self._datasets is a list or an array of datasets
        # # First, generate 10 random indices from the range of self._datasets' length
        # indices = np.random.choice(len(self._datasets), size=self.n_instance, replace=False)

        # Now use these indices to select the datasets
        #self._datasets = [self._datasets[i] for i in indices]
        print("test")

    def evaluate_program(self, program_str: str, callable_func: callable) -> Any | None:
        g = {}
        exec(program_str, g)
        class_callable = g['TSPSolver']
        return self.evaluate(class_callable)

    def tour_cost(self, instance, solution, problem_size):
        cost = 0
        for j in range(problem_size - 1):
            cost += np.linalg.norm(instance[int(solution[j])] - instance[int(solution[j + 1])])
        cost += np.linalg.norm(instance[int(solution[-1])] - instance[int(solution[0])])
        return cost


    def check_feasibility(self, solution, problem_size):
        """
        Check if the TSP solution is feasible.

        Args:
            solution (list or array): Tour order as a sequence of city indices.
            problem_size (int): Expected number of cities.

        Returns:
            bool: True if feasible, False otherwise.
        """
        # 1. Check type and length
        if not isinstance(solution, (list, np.ndarray)):
            return False
        if len(solution) != problem_size:
            return False

        # 2. Ensure all elements are unique and in correct range
        unique_cities = set(solution)
        if unique_cities != set(range(problem_size)):
            return False

        return True


    def evaluate(self, eva: callable) -> float:

        dis_list = []

        for coordinates, opt_tour, distance_matrix, baseline in self._datasets:

            tsp_solver = eva(coordinates, distance_matrix)

            tsp_solution = tsp_solver.solve()

            if not self.check_feasibility(tsp_solution,len(coordinates)):
                return None

            LLM_dis = self.tour_cost(coordinates, tsp_solution, len(coordinates))


            dis_list.append((LLM_dis-baseline)/baseline)


        ave_dis = np.average(dis_list)

        return -ave_dis



if __name__ == '__main__':
    template_program = '''
import numpy as np
import random

class TSPSolver:
    def __init__(self, coordinates: np.ndarray, distance_matrix: np.ndarray):
        """
        Initialize the TSP solver.

        Args:
            coordinates: Numpy array of shape (n, 2) containing the (x, y) coordinates of each city.
            distance_matrix: Numpy array of shape (n, n) containing pairwise distances between cities.
        """
        self.coordinates = coordinates
        self.distance_matrix = distance_matrix

    def _apply_2opt(self, tour):
        improved = True
        while improved:
            improved = False
            for i in range(1, len(tour) - 3):
                for j in range(i + 1, len(tour) - 2):
                    a, b = tour[i], tour[i + 1]
                    c, d = tour[j], tour[j + 1]
                    current_dist = self.distance_matrix[a, b] + self.distance_matrix[c, d]
                    new_dist = self.distance_matrix[a, c] + self.distance_matrix[b, d]
                    if new_dist < current_dist:
                        tour[i + 1 : j + 1] = reversed(tour[i + 1 : j + 1])
                        improved = True
        return tour

    def solve(self) -> np.ndarray:
        n = len(self.coordinates)
        unvisited = list(range(n))
        start_city = random.choice(unvisited)
        unvisited.remove(start_city)

        # Find nearest city to start
        nearest_city = min(unvisited, key=lambda c: self.distance_matrix[start_city, c])
        unvisited.remove(nearest_city)

        # Initialize with two cities and loop back
        tour = [start_city, nearest_city, start_city]

        # Nearest Insertion Heuristic
        while unvisited:
            best_increase = float('inf')
            best_city = None
            insert_position = None
            for city in unvisited:
                for idx in range(len(tour) - 1):
                    a, b = tour[idx], tour[idx + 1]
                    increase = (
                        self.distance_matrix[a, city]
                        + self.distance_matrix[city, b]
                        - self.distance_matrix[a, b]
                    )
                    if increase < best_increase:
                        best_increase = increase
                        best_city = city
                        insert_position = idx + 1
            tour.insert(insert_position, best_city)
            unvisited.remove(best_city)

        # 2-opt optimization
        tour = self._apply_2opt(tour)

        # Return tour without duplicate final city
        return np.array(tour[:-1])
    '''
    tsp = TSPEvaluation()
    results = tsp.evaluate_program(template_program, None)
    print(results)
