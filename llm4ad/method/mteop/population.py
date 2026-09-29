from __future__ import annotations

import math
from threading import Lock
from typing import List, Optional
import numpy as np


# Assuming Program is defined elsewhere, e.g.:
class Program:
    def __init__(self, code: str):
        self.code = code
        self.task_scores: Optional[List[Optional[float]]] = None
        self.dom_score: float = float('inf')  # Default dom_score to infinity

    def __repr__(self):
        return f"Program(scores={self.task_scores})"


from ...base import *
import traceback


class Population:
    def __init__(self, pop_size, use_pareto=False, generation=0, pop: List[Program] | Population | None = None):
        if pop is None:
            self._population = []
        elif isinstance(pop, list):
            self._population = pop
        else:
            self._population = pop._population

        self._pop_size = pop_size
        self._lock = Lock()
        self._next_gen_pop = []
        self._generation = generation
        self._use_pareto = use_pareto

    def __len__(self):
        return len(self._population)

    def __getitem__(self, item) -> Program:
        return self._population[item]

    def __setitem__(self, key, value):
        self._population[key] = value

    @property
    def population(self):
        return self._population

    @property
    def generation(self):
        return self._generation

    @property
    def next_population(self):
        return self._next_gen_pop

    # <<< ADDED >>>
    # Helper function to safely calculate the mean score of a program.
    def _calculate_safe_mean_score(self, program: Program) -> float:
        """
        Calculates the mean of a program's task scores, handling None values.
        Returns -inf if no valid scores are found.
        """
        if program.task_scores is None:
            return float('-inf')

        # Filter out None values before calculating the mean
        valid_scores = [s for s in program.task_scores if s is not None]

        if not valid_scores:
            return float('-inf')

        return np.mean(valid_scores)

        # ... (Keep the rest of the class the same as the previous corrected version) ...

        # <<< ADDED >>>
        # Helper to safely get a score for a specific task, used for finding champions.
    def _get_safe_score_for_task(self, program: Program, task_id: int) -> float:
        """Safely retrieves a score for a given task_id, returning -inf if unavailable."""
        if program.task_scores and len(program.task_scores) > task_id and program.task_scores[task_id] is not None:
            return program.task_scores[task_id]
        return float('-inf')

    def survival(self):
        # 种群管理
        """
        Performs survival selection to create the next generation.
        - If use_pareto is False, it uses simple fitness-based truncation.
        - If use_pareto is True, it uses a two-step process:
            1. Elitism: Guarantees survival for the best program in each task.
            2. Pareto Dominance: Fills remaining slots based on Pareto dominance ranking.
        """
        try:
            pop = self._population + self._next_gen_pop

            if not pop:
                return
            
            pop = [p for p in pop if p.task_scores is not None and p.task_scores != [float('-inf')] and  p.task_scores != float('-inf') ]
            
            
            if self._use_pareto:
                # --- Step 1: Elitism - Preserve the champion of each task ---
                # <<< FIXED: Changed from set() to list() to avoid unhashable type error >>>
                elite_champions = []
                n_tasks = 0
                # Determine the number of tasks from the first valid program
                for p in pop:
                    if p.task_scores:
                        n_tasks = len(p.task_scores)
                        break

                if n_tasks > 0:
                    for task_id in range(n_tasks):
                        # Find the best program for the current task_id
                        best_for_task = max(pop, key=lambda p: self._get_safe_score_for_task(p, task_id))
                        # <<< FIXED: Check for existence before appending to the list >>>
                        if best_for_task not in elite_champions:
                            elite_champions.append(best_for_task)

                # --- Handle the case where champions already fill the population ---
                if len(elite_champions) >= self._pop_size:
                    # If champions overflow the population size, sort them by mean score and take the best.
                    sorted_champions = sorted(elite_champions, key=self._calculate_safe_mean_score, reverse=True)
                    self._population = sorted_champions[:self._pop_size]
                    self._next_gen_pop = []
                    self._generation += 1
                    return  # Survival selection is complete

                # --- Step 2: Pareto Dominance for the rest ---
                # Calculate dominance scores for all programs in the full context of the combined population.
                for prog in pop:
                    if prog.task_scores is None:
                        prog.dom_score = len(pop)  # Penalize programs with no scores
                    else:
                        prog.dom_score = 0
                        for other in pop:
                            if self._dominates(other, prog):
                                prog.dom_score += 1

                # Identify the non-champion candidates for the remaining slots
                non_champions = [p for p in pop if p not in elite_champions]

                # Sort the non-champions using Pareto dominance score (lower is better)
                # and mean score as a tie-breaker (higher is better).
                sorted_non_champions = sorted(non_champions,
                                            key=lambda f: (f.dom_score, -self._calculate_safe_mean_score(f)))

                # --- Step 3: Assemble the new population ---
                new_population = list(elite_champions)  # Create a copy
                num_to_add = self._pop_size - len(new_population)
                new_population.extend(sorted_non_champions[:num_to_add])

                self._population = new_population

            else:  # if self._use_pareto is False
                # Sort by the mean score (higher is better).
                pop = sorted(pop, key=self._calculate_safe_mean_score, reverse=True)
                # Select the top programs
                self._population = pop[:self._pop_size]

        except Exception as e:
            traceback.print_exc()
            print("Error in population survival!")

        # --- Finalization ---
        # Clear the next generation pool and increment the generation counter
        self._next_gen_pop = []
        self._generation += 1

    def _dominates(self, prog1, prog2):
        """
        Check if prog1 dominates prog2 according to Pareto dominance.
        A program dominates another if it's at least as good in all objectives
        and strictly better in at least one objective.
        """
        if prog1.task_scores is None or prog2.task_scores is None:
            return False

        # Ensure we are comparing lists of the same length, filtering out Nones might be needed
        # depending on strict definition, but for now we assume valid lists.
        # This simple comparison assumes scores are numeric. A more robust version might
        # handle internal None values if that's a valid state.
        prog1_scores = [s if s is not None else float('-inf') for s in prog1.task_scores]
        prog2_scores = [s if s is not None else float('-inf') for s in prog2.task_scores]

        # Check if prog1 is at least as good as prog2 in all objectives
        at_least_as_good = all(s1 >= s2 for s1, s2 in zip(prog1_scores, prog2_scores))
        # Check if prog1 is strictly better than prog2 in at least one objective
        strictly_better = any(s1 > s2 for s1, s2 in zip(prog1_scores, prog2_scores))

        return at_least_as_good and strictly_better

    def register_program(self, func: Program, n_tasks: int):
        # in population initialization, we only accept valid functions
        if self._generation == 0 and func.task_scores is None:
            return

        try:
            self._lock.acquire()
            # If a program is a duplicate, penalize it to prevent it from being selected.
            if self.has_duplicate_program(func):
                func.task_scores = [float('-inf')] * n_tasks

            # register to next_gen
            self._next_gen_pop.append(func)

            # update: perform survival if the next generation is full
            if len(self._next_gen_pop) >= self._pop_size:
                self.survival()
        except Exception as e:
            # Print error message
            print(f"[ERROR] Failed to register program: {e}")
            # Print the full traceback for debugging
            traceback.print_exc()
        finally:
            self._lock.release()

    def register_program_low_level(self, func: Program, task_id: int):
        # This method seems designed for a different evolutionary model (e.g., steady-state)
        # and might conflict with the generational model. For now, ensuring it's safe.
        if func.task_scores is None or func.task_scores[task_id] is None:
            return  # Ignore if the relevant score is missing

        try:
            self._lock.acquire()
            if self.has_duplicate_program(func):
                func.task_scores[task_id] = float('-inf')

            # This directly appends to the main population, bypassing the generational model.
            # Be cautious using this method with the generational `survival` logic.
            self.population.append(func)
        except Exception as e:
            print(f"[ERROR] Failed to register program_low_level: {e}")
            traceback.print_exc()
        finally:
            self._lock.release()

    def has_duplicate_program(self, program: Program) -> bool:
        # A more robust check might be on the program's code/structure,
        # but checking scores is a valid way to check for functional duplicates.
        if program.task_scores is None:
            return False  # Cannot be a duplicate if it has no scores

        all_programs = self._population + self._next_gen_pop
        for p in all_programs:
            if p.task_scores == program.task_scores:
                return True
        return False

    def selection(self) -> Program:
        # <<< CHANGED (Major fix) >>>
        """
        Selects a program from the population using rank-based selection.
        - If use_pareto is True, selection is based on dominance score.
        - If use_pareto is False, selection is based on mean task score.
        """
        if not self._population:
            raise ValueError("Cannot select from an empty population.")

        #eligible_programs = [p for p in self._population if math.isfinite(p.dom_score)]

        # if self._use_pareto:
        #     # Filter out programs with non-finite dominance scores and sort by dominance
        #     eligible_programs = [p for p in self._population if math.isfinite(p.dom_score)]
        #     # Sort by dom_score (lower is better)
        #     sorted_programs = sorted(eligible_programs, key=lambda p: p.dom_score)
        # else:
        #     # When not using Pareto, sort by mean score (higher is better)
        #     sorted_programs = sorted(self._population, key=self._calculate_safe_mean_score, reverse=True)
        #
        # if not sorted_programs:
        #     # Fallback: if no programs are eligible (e.g., all have inf scores), return a random one
        #     return np.random.choice(self._population)
        #
        # # Create probability distribution using inverse rank weighting
        # # This gives higher probability to programs at the front of the sorted list
        # ranks = range(len(sorted_programs))
        # p = [1 / (r + 1) for r in ranks]
        # p_sum = sum(p)
        # p_normalized = [val / p_sum for val in p]

        # Select a program based on the probability distribution
        return np.random.choice(self._population)

    def selection_from_topk(self, k: int = 10, task_id: int = 0) -> Program:
        """
        Selects an individual from the top-k ranked programs for a specific task.
        Probabilities are recalculated using only the top-k subset.
        """
        if not self._population:
            raise ValueError("Cannot select from an empty population.")

        # <<< CHANGED (made more robust) >>>
        # Safely get the score for the given task_id
        def get_safe_score(p: Program):
            if p.task_scores and len(p.task_scores) > task_id and p.task_scores[task_id] is not None:
                return p.task_scores[task_id]
            return float('-inf')

        # Sort by score for the specific task (descending)
        func_sorted = sorted(self._population, key=get_safe_score, reverse=True)

        # Handle case where population < k
        actual_k = min(k, len(func_sorted))
        if actual_k == 0:
            # Fallback if no programs have valid scores for this task
            return np.random.choice(self._population)

        topk_funcs = func_sorted[:actual_k]

        # Assign probabilities based on rank within the top-k
        ranks = range(len(topk_funcs))
        p = [1 / (r + 1) for r in ranks]
        p = np.array(p, dtype=float)
        p /= p.sum()  # normalize

        # Randomly choose from top-k
        return np.random.choice(topk_funcs, p=p)

    def get_best(self, task_id: int) -> Program:
        """
        Returns the program with the highest score for a specific task.
        """
        if not self._population:
            raise ValueError("Population is empty")

        # <<< CHANGED (minor simplification) >>>
        def get_safe_score(p: Program):
            if p.task_scores and len(p.task_scores) > task_id and p.task_scores[task_id] is not None:
                return p.task_scores[task_id]
            return float('-inf')

        return max(self._population, key=get_safe_score)

