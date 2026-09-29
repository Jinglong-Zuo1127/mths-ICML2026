# Module Name: FSSPEvaluation
# Last Revision: 2025/9/8
# Description: Evaluates the constructive heuristic for Flow Shop Scheduling Problem (FSSP).
#              Given a set of jobs that must be processed on all machines in the same order,
#              the goal is to find optimal sequence to minimize the makespan.
#              This module is part of the LLM4AD project (https://github.com/Optima-CityU/llm4ad).
#
# Parameters:
#    - timeout_seconds: Maximum allowed time (in seconds) for the evaluation process: int (default: 20).
#    - n_instance: Number of problem instances to generate: int (default: 10).
#    - problem_size: Number of jobs to schedule: int (default: varies by instance).
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

from llm4ad.base import Evaluation
from template_fssp import template_program, task_description
import pickle as pkl
import traceback


class FSSPEvaluation(Evaluation):
    def __init__(self,
                 timeout_seconds=20,
                 return_list=True,
                 **kwargs):

        super().__init__(
            template_program=template_program,
            task_description=task_description,
            use_numba_accelerate=False,
            timeout_seconds=timeout_seconds
        )

        self.n_instance = 64
        # Load the FSSP instances
        with open("./fssp_instances_train.pkl", 'rb') as f:
            all_instances = pkl.load(f)
        self._datasets = all_instances
        self._datasets = self._datasets[:self.n_instance]
        # Set the seed for numpy's random number generator
        np.random.seed(2025)

        # Convert dictionary to list of (name, instance) tuples


        print(f"Loaded {len(self._datasets)} FSSP instances for evaluation")

    def plot_solution(self, instance, job_sequence, instance_name="Unknown"):
        """
        Plot the solution as a Gantt chart for Flow Shop Scheduling.

        Args:
            instance (dict): FSSP instance with processing_times
            job_sequence (list): Sequence of job IDs
            instance_name (str): Name of the instance for the plot title
        """
        if not job_sequence:
            print("Cannot plot: Empty job sequence")
            return

        # Calculate the schedule from job sequence
        schedule = self.calculate_schedule_from_sequence(instance, job_sequence)

        # Calculate makespan for the title
        makespan = self.calculate_makespan_from_sequence(instance, job_sequence)

        # Create a figure and axis
        fig, ax = plt.subplots(figsize=(12, 8))

        # Define colors for jobs (using a colormap)
        colors = plt.cm.get_cmap('tab20', instance['num_jobs'])

        # For each machine, we'll plot a row of operations
        machine_y_positions = {}
        for machine_id in range(instance['num_machines']):
            machine_y_positions[machine_id] = instance['num_machines'] - machine_id - 1

        # Plot operations
        for machine_id in range(instance['num_machines']):
            for job_idx, job_id in enumerate(job_sequence):
                start_time = schedule[machine_id][job_idx][0]
                end_time = schedule[machine_id][job_idx][1]
                proc_time = end_time - start_time

                y_pos = machine_y_positions[machine_id]

                # Plot the operation as a colored rectangle
                rect = plt.Rectangle((start_time, y_pos - 0.4), proc_time, 0.8,
                                     facecolor=colors(job_id), edgecolor='black', alpha=0.7)
                ax.add_patch(rect)

                # Add text label for the job
                ax.text(start_time + proc_time / 2, y_pos, f"J{job_id}",
                        ha='center', va='center', color='black', fontsize=8)

        # Set the limits and labels
        ax.set_xlim(0, makespan * 1.05)
        ax.set_ylim(-0.5, instance['num_machines'] - 0.5)

        # Set y-ticks for machines
        ax.set_yticks(list(range(instance['num_machines'])))
        ax.set_yticklabels([f"Machine {m}" for m in range(instance['num_machines'])])

        # Set title and labels
        ax.set_title(f"Flow Shop Schedule - Instance: {instance_name}\nMakespan: {makespan}")
        ax.set_xlabel("Time")

        # Add a grid
        ax.grid(True, axis='x', linestyle='--', alpha=0.7)

        # Add a legend for jobs
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor=colors(j), edgecolor='black', alpha=0.7, label=f'Job {j}')
                           for j in range(instance['num_jobs'])]
        ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.1, 1))

        plt.tight_layout()
        plt.show()
        return

    def calculate_schedule_from_sequence(self, instance, job_sequence):
        """
        Calculate the detailed schedule from a job sequence for FSSP.

        Args:
            instance (dict): FSSP instance with processing_times
            job_sequence (list): Sequence of job IDs

        Returns:
            list: Schedule where schedule[m][j] = (start_time, end_time) for job j on machine m
        """
        num_machines = instance['num_machines']
        num_jobs = len(job_sequence)

        # Initialize completion times for each machine
        machine_completion_time = [0] * num_machines
        # Track completion time of each job (for precedence constraints)
        job_completion_time = [0] * num_jobs

        # Schedule to store (start_time, end_time) for each job on each machine
        schedule = [[None for _ in range(num_jobs)] for _ in range(num_machines)]

        # Process jobs in the given sequence
        for job_idx, job_id in enumerate(job_sequence):
            for machine_id in range(num_machines):
                # Start time is max of machine availability and job's previous operation completion
                if machine_id == 0:
                    # First machine: only consider machine availability
                    start_time = machine_completion_time[machine_id]
                else:
                    # Other machines: consider both machine availability and job precedence
                    start_time = max(machine_completion_time[machine_id], job_completion_time[job_idx])

                # Processing time for this job on this machine
                proc_time = instance['processing_times'][job_id][machine_id]
                end_time = start_time + proc_time

                # Update schedule
                schedule[machine_id][job_idx] = (start_time, end_time)

                # Update completion times
                machine_completion_time[machine_id] = end_time
                job_completion_time[job_idx] = end_time

        return schedule

    def calculate_makespan_from_sequence(self, instance, job_sequence):
        """
        Calculate the makespan from a job sequence for FSSP.

        Args:
            instance (dict): FSSP instance with processing_times
            job_sequence (list): Sequence of job IDs

        Returns:
            int: The makespan (completion time of the last job on the last machine)
        """
        if not job_sequence or len(job_sequence) != instance['num_jobs']:
            return float('inf')

        num_machines = instance['num_machines']
        num_jobs = len(job_sequence)

        # Initialize completion times for each machine
        machine_completion_time = [0] * num_machines
        # Track completion time of each job (for precedence constraints)
        job_completion_time = [0] * num_jobs

        # Process jobs in the given sequence
        for job_idx, job_id in enumerate(job_sequence):
            for machine_id in range(num_machines):
                # Start time is max of machine availability and job's previous operation completion
                if machine_id == 0:
                    # First machine: only consider machine availability
                    start_time = machine_completion_time[machine_id]
                else:
                    # Other machines: consider both machine availability and job precedence
                    start_time = max(machine_completion_time[machine_id], job_completion_time[job_idx])

                # Processing time for this job on this machine
                proc_time = instance['processing_times'][job_id][machine_id]
                end_time = start_time + proc_time

                # Update completion times
                machine_completion_time[machine_id] = end_time
                job_completion_time[job_idx] = end_time

        # Makespan is the completion time of the last machine
        return machine_completion_time[-1]

    def check_feasibility(self, instance, job_sequence):
        """
        Check if the job sequence is feasible for the given FSSP instance.

        Args:
            instance (dict): FSSP instance with processing_times
            job_sequence (list): Sequence of job IDs

        Returns:
            bool: True if the sequence is feasible, False otherwise
        """
        # Check if all jobs are included exactly once
        if len(job_sequence) != instance['num_jobs']:
            return False

        if set(job_sequence) != set(range(instance['num_jobs'])):
            return False

        # Check if all job IDs are valid
        for job_id in job_sequence:
            if job_id < 0 or job_id >= instance['num_jobs']:
                return False

        return True

    def evaluate(self, alg):
        """
        Evaluate the FSSP algorithm on the test instances.

        Args:
            alg: The FSSP solver algorithm class

        Returns:
            float: The negative of the average relative gap to the best known solution
        """

        try:
            gap_list = []

            for name, instance in self._datasets:
                # Create a solver instance
                fssp_solver = alg(
                    num_jobs=instance['num_jobs'],
                    num_machines=instance['num_machines'],
                    processing_times=instance['processing_times']
                )

                # Solve the problem
                job_sequence = fssp_solver.solve()

                # Check feasibility
                if not self.check_feasibility(instance, job_sequence):
                    print("infeasible solution")
                    return None

                # Calculate makespan
                makespan = self.calculate_makespan_from_sequence(instance, job_sequence)

                # self.plot_solution(instance, job_sequence)
                # input()

                # Calculate gap to best known solution
                best_known = instance['best_known']

                gap = (makespan - best_known) / best_known
                gap_list.append(gap)

            # Return negative average gap (lower gap is better)
            avg_gap = np.mean(gap_list)

        except Exception as e:
            print(f"Error occurred: {e}")
            print("Traceback:")
            traceback.print_exc()
            return None

        return -avg_gap

    def evaluate_program(self, program_str: str, callable_func: callable) -> Any | None:
        g = {}
        exec(program_str, g)
        class_callable = g['FSSPSolver']
        return self.evaluate(class_callable)


if __name__ == '__main__':
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
    fssp = FSSPEvaluation()
    results = fssp.evaluate_program(template_program, None)
    print(results)
