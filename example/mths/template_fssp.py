template_program = '''
import numpy as np

class FSSPSolver:
    def __init__(self, num_jobs: int, num_machines: int, processing_times: list):
        """
        Initialize the FSSP solver.

        Args:
            num_jobs: Number of jobs in the problem
            num_machines: Number of machines in the problem
            processing_times: List of lists where processing_times[j][m] is the processing time of job j on machine m
        """
        self.num_jobs = num_jobs
        self.num_machines = num_machines
        self.processing_times = processing_times

    def solve(self) -> list:
        """
        Solve the Flow Shop Scheduling Problem (FSSP).

        Returns:
            A list representing the sequence of jobs to be processed.
            For example, [0, 2, 1] means job 0 is processed first, then job 2, then job 1.
            All jobs must be processed on all machines in the same order.

            The sequence must include all jobs exactly once.
        """
        # --- Implement your scheduling algorithm here ---

        # Simple solution: process jobs in their original order (0, 1, 2, ...)
        job_sequence = list(range(self.num_jobs))

        return job_sequence
        '''

task_description = "Develop an algorithm to solve the Flow Shop Scheduling Problem (FSSP) by determining the optimal sequence of jobs to minimize makespan. In FSSP, all jobs must be processed on all machines in the same order (machine 0, then machine 1, then machine 2, etc.). The goal is to find the job sequence that minimizes the makespan (total completion time) while ensuring that: (1) all jobs follow the same machine processing order, (2) each machine processes only one job at a time, and (3) each job can only be processed on one machine at a time. The algorithm should return a permutation of job indices representing the order in which jobs should be processed."
