# Module Name: EoH
# Last Revision: 2025/2/16
# This file is part of the LLM4AD project (https://github.com/Optima-CityU/llm4ad).
#
# Reference:
#   - Fei Liu, Tong Xialiang, Mingxuan Yuan, Xi Lin, Fu Luo, Zhenkun Wang, Zhichao Lu, and Qingfu Zhang.
#       "Evolution of Heuristics: Towards Efficient Automatic Algorithm Design Using Large Language Model."
#       In Forty-first International Conference on Machine Learning (ICML). 2024.
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

import concurrent.futures
import time
import traceback
from threading import Thread
from typing import Optional, Literal, Dict, List

from .population import Population
from .profiler import EoHProfiler
from .prompt import EoHPrompt
from .sampler import EoHSampler
from ...base import (
    Evaluation, LLM, Program, SecureEvaluator
)
from ...tools.profiler import ProfilerBase
from threading import Lock
import copy
import math

class TwoLevelEoP:
    def __init__(self,
                 llm: LLM,
                 evaluation: Evaluation,
                 profiler: ProfilerBase = None,
                 max_generations: Optional[int] = 10,
                 max_sample_nums: Optional[int] = 100,
                 high_level_pop_size: Optional[int] = 10,
                 low_level_max_samples_each_program: Optional[int] = 20,
                 low_level_pop_size: Optional[int] = 5,
                 selection_num=2,
                 use_e2_operator: bool = True,
                 use_m1_operator: bool = True,
                 use_m2_operator: bool = True,
                 num_samplers: int = 1,
                 num_evaluators: int = 1,
                 *,
                 resume_mode: bool = False,
                 debug_mode: bool = False,
                 multi_thread_or_process_eval: Literal['thread', 'process'] = 'thread',
                 **kwargs):
        """Two-Level Evolutionary of Programs.

        This implements a hierarchical search where:
        1) High-level: Program design population
        2) Low-level: Key function design population for each program

        Args:
            llm             : an instance of 'llm4ad.base.LLM', which provides the way to query LLM.
            evaluation      : an instance of 'llm4ad.base.Evaluator', which defines the way to calculate the score of a generated function.
            profiler        : an instance of 'llm4ad.method.eoh.EoHProfiler'. If you do not want to use it, you can pass a 'None'.
            max_generations : terminate after evolving 'max_generations' generations or reach 'max_sample_nums',
                              pass 'None' to disable this termination condition.
            max_sample_nums : terminate after evaluating max_sample_nums functions (no matter the function is valid or not) or reach 'max_generations',
                              pass 'None' to disable this termination condition.
            high_level_pop_size : population size for program designs, if set to 'None', will automatically adjust this parameter.
            low_level_pop_size  : population size for key function variants of each program.
            selection_num   : number of selected individuals while crossover.
            use_e2_operator : if use e2 operator.
            use_m1_operator : if use m1 operator.
            use_m2_operator : if use m2 operator.
            resume_mode     : in resume_mode, will not evaluate the template_program, and will skip the init process.
            debug_mode      : if set to True, we will print detailed information.
            multi_thread_or_process_eval: use 'concurrent.futures.ThreadPoolExecutor' or 'concurrent.futures.ProcessPoolExecutor' for the usage of
                multi-core CPU while evaluation.
            **kwargs        : some args pass to 'llm4ad.base.SecureEvaluator'. Such as 'fork_proc'.
        """
        self._template_program_str = evaluation.template_program
        self._task_description_str = evaluation.task_description
        self._max_generations = max_generations
        self._max_sample_nums = max_sample_nums
        self._high_level_pop_size = high_level_pop_size
        self._low_level_max_samples_each_program = low_level_max_samples_each_program
        self._low_level_pop_size = low_level_pop_size
        self._selection_num = selection_num
        self._use_e2_operator = use_e2_operator
        self._use_m1_operator = use_m1_operator
        self._use_m2_operator = use_m2_operator

        # samplers and evaluators
        self._num_samplers = num_samplers
        self._num_evaluators = num_evaluators
        self._resume_mode = resume_mode
        self._debug_mode = debug_mode
        llm.debug_mode = debug_mode
        self._multi_thread_or_process_eval = multi_thread_or_process_eval

        # adjust population size
        self._adjust_pop_size()

        # high-level population for program designs
        self._high_level_population = Population(pop_size=self._high_level_pop_size)

        # dictionary to store low-level populations for each program
        self._low_level_populations: Dict[str, Population] = {}

        # dictionary to store key function templates for each program
        self._key_function_templates: Dict[str, str] = {}

        self._sampler = EoHSampler(llm, self._template_program_str)
        self._evaluator = SecureEvaluator(evaluation, debug_mode=debug_mode, **kwargs)
        self._profiler = profiler

        # statistics
        self._tot_sample_nums = 0

        # reset _initial_sample_nums_max
        self._initial_sample_nums_max = min(
            self._max_sample_nums,
            self._high_level_pop_size*self._low_level_max_samples_each_program
        )

        # multi-thread executor for evaluation
        assert multi_thread_or_process_eval in ['thread', 'process']
        if multi_thread_or_process_eval == 'thread':
            self._evaluation_executor = concurrent.futures.ThreadPoolExecutor(
                max_workers=num_evaluators
            )
        else:
            self._evaluation_executor = concurrent.futures.ProcessPoolExecutor(
                max_workers=num_evaluators
            )

        # pass parameters to profiler
        if profiler is not None:
            self._profiler.record_parameters(llm, evaluation, self)
        self._lock = Lock()

    def _adjust_pop_size(self):
        # adjust high-level population size
        if self._max_sample_nums >= 10000:
            if self._high_level_pop_size is None:
                self._high_level_pop_size = 40
            elif abs(self._high_level_pop_size - 40) > 20:
                print(f'Warning: high-level population size {self._high_level_pop_size} '
                      f'is not suitable, please reset it to 40.')
        elif self._max_sample_nums >= 1000:
            if self._high_level_pop_size is None:
                self._high_level_pop_size = 20
            elif abs(self._high_level_pop_size - 20) > 10:
                print(f'Warning: high-level population size {self._high_level_pop_size} '
                      f'is not suitable, please reset it to 20.')
        elif self._max_sample_nums >= 200:
            if self._high_level_pop_size is None:
                self._high_level_pop_size = 10
            elif abs(self._high_level_pop_size - 10) > 5:
                print(f'Warning: high-level population size {self._high_level_pop_size} '
                      f'is not suitable, please reset it to 10.')
        else:
            if self._high_level_pop_size is None:
                self._high_level_pop_size = 5
            elif abs(self._high_level_pop_size - 5) > 5:
                print(f'Warning: high-level population size {self._high_level_pop_size} '
                      f'is not suitable, please reset it to 5.')

    def _identify_key_function(self, program_str: str) -> str:
        """Use LLM to identify the key function in a program and create a template for it."""
        prompt = f"""
You are given a program. Please identify the most important function in this program 
that would benefit most from optimization.  

Program:
{program_str}

Return only the key function. It should be enclosed between ```python and ``` markers exactly as shown below:

```python
# Your key function here
```
        """

        if self._debug_mode:
            print(f"check identify key function prompt: {prompt}")

        key_function_template = self._sampler.get_python_response(prompt)

        if self._debug_mode:
            print(f"check identify key function response: {key_function_template}")

        return key_function_template

    def _generate_full_program(self, program_str: str, key_function_body: str) -> str:
        """Generate a complete program by replacing the key function template with the optimized function body."""

        # Use LLM to integrate the optimized function back into the program
        prompt = f"""
You are given a program and an optimized function. Please replace the original function in the program
with the optimized version and return the complete updated program.

Original program:
{program_str}

Optimized function:
{key_function_body}

Return the complete updated program.  It should be enclosed between ```python and ``` markers exactly as shown below:

```python
# Your updated program
```
        """

        if self._debug_mode:
            print(f"check _generate_full_program prompt: {prompt}")

        updated_program = self._sampler.get_python_response(prompt)

        if self._debug_mode:
            print(f"check _generate_full_program response: {updated_program}")

        return updated_program

    def _sample_evaluate_register_high_level(self, prompt):
        """Sample, evaluate, and register a high-level program design."""
        sample_start = time.time()
        thought, program_str = self._sampler.get_thought_and_function(prompt)
        sample_time = time.time() - sample_start
        if thought is None or program_str is None:
            return

        # evaluate
        score, eval_time = self._evaluation_executor.submit(
            self._evaluator.evaluate_program_record_time,
            program_str
        ).result()

        if score is not None and not math.isinf(score) :

            # register to profiler
            program = Program()
            program.code = program_str
            program.score = score
            program.evaluate_time = eval_time
            program.algorithm = thought
            program.sample_time = sample_time


            # Identify key function and create template
            key_function_template = None
            max_attempts = 5
            attempts = 0

            while key_function_template is None and attempts < max_attempts:
                key_function_template = self._identify_key_function(program_str)
                attempts += 1
                if key_function_template is None and attempts < max_attempts:
                    if self._debug_mode:
                        print(f"Failed to identify key function (attempt {attempts}/{max_attempts}), retrying...")
                    #time.sleep(1)  # Short delay before retry

            if key_function_template is None:
                if self._debug_mode:
                    print(f"Failed to identify key function after {max_attempts} attempts, skipping this program")
                return

            program.function_body = key_function_template

            self._lock.acquire()
            try:
                # Create a unique identifier for this program
                program_id = str(len(self._low_level_populations)+1)

                self._key_function_templates[program_id] = key_function_template

                # Create a low-level population for this program
                self._low_level_populations[program_id] = Population(pop_size=self._low_level_max_samples_each_program)

            finally:
                self._lock.release()

            program.id = program_id


            # Initialize the low-level population with the original key function
            self._initialize_low_level_population(program_id, program_str, copy.deepcopy(program))

            # Evolve low-level populations for existing programs
            self._evolve_low_level_populations(program_id, program_str)

            best_low_program = self._low_level_populations[program_id].get_best()

            if best_low_program.score > program.score:
                program.score = best_low_program.score
                program.code = best_low_program.code

            if self._profiler is not None:
                self._profiler.register_function(program, program=str(program))
                if isinstance(self._profiler, EoHProfiler):
                    self._profiler.register_population(self._high_level_population)
                self._tot_sample_nums += 1

        # register to the high-level population
        self._high_level_population.register_program(program)

    def _initialize_low_level_population(self, program_id: str, program_str: str, program_high: Program):
        """Initialize the low-level population for a specific program."""
        # Extract the key function body from the original program
        original_function = self._key_function_templates[program_id]


        # Create initial variants for the low-level population
        for i in range(self._low_level_pop_size):
            if i == 0:
                # First variant is the original function body
                # Register to the low-level population
                self._low_level_populations[program_id].register_program_low_level(program_high)

                # if self._profiler is not None:
                #     self._profiler.register_function(program_high, program=str(program))
            else:
                # Generate variations
                prompt = f"""
You are given a function. Please create a variation of this function that 
with the same inputs and outputs but might be more effective or use a different approach.
The function is part of a larger program solving the following task:

Task description:
{self._task_description_str}

Original function body:
{original_function}

Return only the modified function body.  It should be enclosed between ```python and ``` markers exactly as shown below:

```python
# Your new function
```
                """

                if self._debug_mode:
                    print(f"create a variation of this function prompt: {prompt}")

                function_body = self._sampler.get_python_response(prompt)
                if function_body is None:
                    continue

                if self._debug_mode:
                    print(f"create a variation of this function response: {function_body}")

                # Generate full program with this function variant
                full_program = self._generate_full_program(program_str, function_body)

                # Evaluate
                score, eval_time = self._evaluation_executor.submit(
                    self._evaluator.evaluate_program_record_time,
                    full_program
                ).result()

                # Create program object
                program = Program()
                program.code = full_program  # Store the full program
                program.function_body = function_body  # Store just the function body
                program.score = score
                program.evaluate_time = eval_time
                program.algorithm = f"Function variant {i} for program {program_id}"


                # Register to the low-level population
                self._low_level_populations[program_id].register_program_low_level(program)

                if self._profiler is not None:
                    self._profiler.register_function(program, program=str(program))
                    self._tot_sample_nums += 1

    def _sample_evaluate_register_low_level(self, program_id: str,program_str: str, prompt):
        """Sample, evaluate, and register a low-level function variant."""
        sample_start = time.time()
        function_body = self._sampler.get_python_response(prompt)
        sample_time = time.time() - sample_start
        if function_body is None:
            return


        # Generate full program with this function variant
        full_program = self._generate_full_program(program_str, function_body)

        # evaluate
        score, eval_time = self._evaluation_executor.submit(
            self._evaluator.evaluate_program_record_time,
            full_program
        ).result()

        # register to profiler
        program = Program()
        program.code = full_program  # Store the full program
        program.function_body = function_body  # Store just the function body
        program.score = score
        program.evaluate_time = eval_time
        program.algorithm = ''
        program.sample_time = sample_time

        if self._profiler is not None:
            self._profiler.register_function(program, program=str(program))
            if isinstance(self._profiler, EoHProfiler):
                self._profiler.register_low_level_population(self._low_level_populations[program_id],program_id)
            self._tot_sample_nums += 1

        # register to the low-level population
        self._low_level_populations[program_id].register_program_low_level(program)

    def _get_best_program_for_id(self, program_id: str) -> Program:
        """Get the best program variant from a low-level population."""
        return self._low_level_populations[program_id].get_best()

    def _continue_loop(self) -> bool:
        if self._max_generations is None and self._max_sample_nums is None:
            return True
        elif self._max_generations is not None and self._max_sample_nums is None:
            return self._high_level_population.generation < self._max_generations
        elif self._max_generations is None and self._max_sample_nums is not None:
            return self._tot_sample_nums < self._max_sample_nums
        else:
            return (self._high_level_population.generation < self._max_generations
                    and self._tot_sample_nums < self._max_sample_nums)

    def _iteratively_use_high_level_operators(self):
        """Evolve the high-level population (program designs)."""
        while self._continue_loop():
            try:
                # get a new program using e1
                indivs = [self._high_level_population.selection() for _ in range(self._selection_num)]
                prompt = EoHPrompt.get_prompt_e1(self._task_description_str, indivs, self._template_program_str)
                if self._debug_mode:
                    print(f'High-Level E1 Prompt: {prompt}')
                self._sample_evaluate_register_high_level(prompt)
                if not self._continue_loop():
                    break

                # get a new program using e2
                if self._use_e2_operator:
                    indivs = [self._high_level_population.selection() for _ in range(self._selection_num)]
                    prompt = EoHPrompt.get_prompt_e2(self._task_description_str, indivs, self._template_program_str)
                    if self._debug_mode:
                        print(f'High-Level E2 Prompt: {prompt}')
                    self._sample_evaluate_register_high_level(prompt)
                    if not self._continue_loop():
                        break

                # get a new program using m1
                if self._use_m1_operator:
                    indiv = self._high_level_population.selection()
                    prompt = EoHPrompt.get_prompt_m1(self._task_description_str, indiv, self._template_program_str)
                    if self._debug_mode:
                        print(f'High-Level M1 Prompt: {prompt}')
                    self._sample_evaluate_register_high_level(prompt)
                    if not self._continue_loop():
                        break

                # get a new program using m2
                if self._use_m2_operator:
                    indiv = self._high_level_population.selection()
                    prompt = EoHPrompt.get_prompt_m2(self._task_description_str, indiv, self._template_program_str)
                    if self._debug_mode:
                        print(f'High-Level M2 Prompt: {prompt}')
                    self._sample_evaluate_register_high_level(prompt)
                    if not self._continue_loop():
                        break


            except KeyboardInterrupt:
                break
            except Exception as e:
                if self._debug_mode:
                    traceback.print_exc()
                    exit()
                continue

    def _evolve_low_level_populations(self,program_id,program_str):
        """Evolve the low-level populations (key function variants) for each program."""
        # Select a random program to evolve its key function
        if not self._low_level_populations:
            return

        # Get the key function template for this program
        key_function_template = self._key_function_templates[program_id]

        while len(self._low_level_populations[program_id].population) < self._low_level_max_samples_each_program:

            try:
                # Get a new function variant using mutation
                indiv = self._low_level_populations[program_id].selection_from_topk(self._low_level_pop_size)
                function_body = indiv.code  # This contains just the function body

                prompt = f"""
You are given a function. Please improve this function. The function is part of a larger program solving the following task:

Task description:
{self._task_description_str}

Current function:
{function_body}

Return only the improved function. It should be enclosed between ```python and ``` markers exactly as shown below:

```python
# Your improved function here
```
                """

                if self._debug_mode:
                    print(f'Low-Level Mutation Prompt for program {program_id}')

                self._sample_evaluate_register_low_level(program_id,program_str, prompt)

                # # Perform crossover if there are enough individuals
                # if len(self._low_level_populations[program_id]) >= 2:
                #     indiv1 = self._low_level_populations[program_id].selection()
                #     indiv2 = self._low_level_populations[program_id].selection()
                #
                #     function_body1 = indiv1.function_body
                #     function_body2 = indiv2.function_body
                #
                #     prompt = f"""
                #     You are given two function bodies implementing the same functionality.
                #     Please create a new improved function body.
                #     The function is part of a larger program solving the following task:
                #
                #     Task description:
                #     {self._task_description_str}
                #
                #     Function body 1:
                #     {function_body1}
                #
                #     Function body 2:
                #     {function_body2}
                #
                #     Key function template:
                #     {key_function_template}
                #
                #     Return only the new improved function body in
                #     """
                #
                #     if self._debug_mode:
                #         print(f'Low-Level Crossover Prompt for program {program_id}')
                #
                #     self._sample_evaluate_register_low_level(program_id, prompt)

            except Exception as e:
                if self._debug_mode:
                    traceback.print_exc()

    def _iteratively_init_high_level_population(self):
        """Initialize the high-level population (program designs)."""
        while self._high_level_population.generation == 0:
            try:
                # get a new program using i1
                prompt = EoHPrompt.get_prompt_i1(self._task_description_str, self._template_program_str)
                self._sample_evaluate_register_high_level(prompt)
                if self._tot_sample_nums >= self._initial_sample_nums_max:
                    print(
                        f'Note: During initialization, TwoLevelEoP gets {len(self._high_level_population) + len(self._high_level_population._next_gen_pop)} algorithms '
                        f'after {self._initial_sample_nums_max} trails.')
                    break
            except Exception:
                if self._debug_mode:
                    traceback.print_exc()
                    exit()
                continue

    def _multi_threaded_sampling(self, fn: callable, *args, **kwargs):
        """Execute `fn` using multithreading."""
        # threads for sampling
        sampler_threads = [
            Thread(target=fn, args=args, kwargs=kwargs)
            for _ in range(self._num_samplers)
        ]
        for t in sampler_threads:
            t.start()
        for t in sampler_threads:
            t.join()

    def get_best_solution(self) -> Program:
        """Get the best solution from all populations."""
        best_program = None
        best_score = float('-inf')

        # Check high-level population
        high_level_best = self._high_level_population.get_best()
        if high_level_best and high_level_best.score > best_score:
            best_program = high_level_best
            best_score = high_level_best.score

        # Check all low-level populations
        for program_id, population in self._low_level_populations.items():
            low_level_best = population.get_best()
            if low_level_best and low_level_best.score > best_score:
                best_program = low_level_best
                best_score = low_level_best.score

        return best_program

    def run(self):
        if not self._resume_mode:
            # do initialization of high-level population
            self._multi_threaded_sampling(self._iteratively_init_high_level_population)
            self._high_level_population.survival()

            # terminate searching if
            if len(self._high_level_population) < self._selection_num:
                print(
                    f'The search is terminated since unable to obtain {self._selection_num} feasible algorithms during initialization. '
                    f'Please increase the `initial_sample_nums_max` argument (currently {self._initial_sample_nums_max}). '
                    f'Please also check your evaluation implementation and LLM implementation.')
                return

        # two-level evolutionary search
        self._multi_threaded_sampling(self._iteratively_use_high_level_operators)

        # finish
        if self._profiler is not None:
            self._profiler.finish()

        self._sampler.llm.close()

