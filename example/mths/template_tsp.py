template_program = '''

import numpy as np

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
        
    # --- your code here ---

    def solve(self) -> np.ndarray:
        """
        Solve the Traveling Salesman Problem (TSP).

        Returns:
            A numpy array of shape (n,) containing a permutation of integers
            [0, 1, ..., n-1] representing the order in which the cities are visited.

            The tour must:
            - Start and end at the same city (implicitly, since it's a loop)
            - Visit each city exactly once
        """
        n = len(self.coordinates)

        # --- your code here ---

        # Example (naive ordered tour — replace with your algorithm):
        tour = np.arange(n)

        return tour
'''

task_description = "Develop an algorithm to address the Traveling Salesman Problem. The objective is to determine the shortest route that visits each city in a given list exactly once and then returns to the starting city, thereby minimizing the total distance traveled."