from __future__ import annotations

import math
from threading import Lock
from typing import List
import numpy as np

from ...base import *


class Population:
    def __init__(self, pop_size, generation=0, pop: List[Program] | Population | None = None):
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

    def survival(self):
        pop = self._population + self._next_gen_pop
        pop = sorted(pop, key=lambda f: f.score, reverse=True)
        self._population = pop[:self._pop_size]
        self._next_gen_pop = []
        self._generation += 1

    def register_program(self, func: Program):
        # in population initialization, we only accept valid functions
        if self._generation == 0 and func.score is None:
            return
        # if the score is None, we still put it into the population,
        # we set the score to '-inf'
        if func.score is None:
            func.score = float('-inf')
        try:
            self._lock.acquire()
            if self.has_duplicate_program(func):
                func.score = float('-inf')
            # register to next_gen
            self._next_gen_pop.append(func)
            # update: perform survival if reach the pop size
            if len(self._next_gen_pop) >= self._pop_size:
                self.survival()
        except Exception as e:
            return
        finally:
            self._lock.release()

    def register_program_low_level(self, func: Program):
        # in population initialization, we only accept valid functions
        if self._generation == 0 and func.score is None:
            return
        # if the score is None, we still put it into the population,
        # we set the score to '-inf'
        if func.score is None:
            func.score = float('-inf')
        try:
            self._lock.acquire()
            if self.has_duplicate_program(func):
                func.score = float('-inf')
            # register to next_gen
            self.population.append(func)
        except Exception as e:
            return
        finally:
            self._lock.release()

    def has_duplicate_program(self, program: str | Program) -> bool:
        for f in self._population:
            if str(f) == str(program) or program.score == f.score:
                return True
        for f in self._next_gen_pop:
            if str(f) == str(program) or program.score == f.score:
                return True
        return False

    def selection(self) -> Program:
        funcs = [f for f in self._population if not math.isinf(f.score)]
        func = sorted(funcs, key=lambda f: f.score, reverse=True)
        p = [1 / (r + len(func)) for r in range(len(func))]
        p = np.array(p)
        p = p / np.sum(p)
        return np.random.choice(func, p=p)

    def selection_from_topk(self, k: int = 10) -> Program:
        """
        Selects an individual from the top-k ranked programs in the population.
        Probabilities are recalculated using only the top-k subset.
        """
        # Filter & sort by score
        funcs = [f for f in self._population if not math.isinf(f.score)]
        # Sort by score (descending)
        func_sorted = sorted(funcs, key=lambda f: f.score, reverse=True)

        # Handle case where population < k
        actual_k = min(k, len(func_sorted))
        topk_funcs = func_sorted[:actual_k]

        # Assign probabilities based only on this top-k (rank-based)
        p = [1 / (r + actual_k) for r in range(len(topk_funcs))]
        p = np.array(p, dtype=float)
        p /= p.sum()  # normalize

        # Randomly choose from top-k
        return np.random.choice(topk_funcs, p=p)

    def get_best(self) -> Program:
        """
        Returns the program with the highest score in the population.

        Returns:
            Program: The program with the highest score.

        Raises:
            ValueError: If the population is empty.
        """
        if not self._population:
            raise ValueError("Population is empty")

        return max(self._population, key=lambda p: p.score if p.score is not None else float('-inf'))
