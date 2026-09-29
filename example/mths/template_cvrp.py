template_program = '''
import numpy as np

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

    # --- your code here ---
    
    def solve(self) -> list:
        """
        Solve the Capacitated Vehicle Routing Problem (CVRP).
    
        Returns:
            A one-dimensional list of integers representing the sequence of nodes visited by all vehicles.
            The depot (node 0) is used to separate different vehicle routes and appears at the start and end
            of each route. For example: [0, 1, 4, 0, 2, 3, 0] represents:
              - Route 1: 0 → 1 → 4 → 0
              - Route 2: 0 → 2 → 3 → 0
            
            Requirements:
            - Each route must start and end at the depot (node 0)
            - The total demand of nodes in each route must not exceed vehicle capacity
            - Each customer node (non-zero) must be visited exactly once across all routes
            - The output must be a flat list (not nested lists)
            - Depot nodes (0) separate routes and mark route boundaries
        """
        n = len(self.coordinates)

        # --- your code here ---

        # Example (naive solution — replace with your algorithm):
        # This example simply assigns nodes to routes in order until the vehicle capacity is reached.
        solution = [0]  # Start at the depot
        current_capacity = 0

        for i in range(1, n):
            if current_capacity + self.demands[i] > self.vehicle_capacity:
                solution.append(0)  # return to depot and start a new route
                current_capacity = 0

            solution.append(i)
            current_capacity += self.demands[i]

        if solution[-1] != 0:
            solution.append(0)  # end the last route at the depot

        return solution
'''
task_description = "Develop an algorithm to solve the Capacitated Vehicle Routing Problem (CVRP). The objective is to determine the optimal set of routes for a fleet of vehicles that all start and end at a central depot. Each vehicle has a maximum capacity, and the routes must collectively serve all customer nodes exactly once without exceeding the vehicle's capacity. The goal is to minimize the total distance traveled across all routes."

