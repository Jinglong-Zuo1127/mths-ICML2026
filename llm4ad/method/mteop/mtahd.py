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
from threading import Thread, Lock
from typing import Optional, Literal, Dict, List, Any

from .population import Population
from .profiler import EoHProfiler
from .sampler import EoHSampler
from ...base import (
    Evaluation, LLM, Program, SecureEvaluator
)
from ...tools.profiler import ProfilerBase
import copy
import math
from .prompt import EoHPrompt
from .prompt_abstract import EoHPrompt_abstract
from .prompt_free import EoHPrompt_free
from .prompt_thought import EoHPrompt_thought

class MTAHD:
    def __init__(self,
                 llm: LLM,
                 evaluations: List[Evaluation],  # Changed to list of evaluations for multiple tasks
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
                 enable_knowledge_transfer: bool = False,
                 enable_low_level_search: bool = False,
                 enable_pareto_population_management: bool = False,
                 metaheuristic_type: int = 1,
                 *,
                 resume_mode: bool = False,
                 debug_mode: bool = False,
                 multi_thread_or_process_eval: Literal['thread', 'process'] = 'thread',
                 **kwargs):
        """Multi-Task Two-Level Evolutionary of Programs.

        This implements a hierarchical search where:
        1) High-level: Program design population shared across tasks
        2) Low-level: Task-specific key function design population for each program

        Args:
            llm             : an instance of 'llm4ad.base.LLM', which provides the way to query LLM.
            evaluations     : a list of 'llm4ad.base.Evaluator' instances, one for each task.
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
        # 对元启发式形式进行消融实验

        if metaheuristic_type == 1:
            self.EoHPrompt = EoHPrompt
        elif metaheuristic_type == 2:
            self.EoHPrompt = EoHPrompt_abstract
        elif metaheuristic_type == 3:
            self.EoHPrompt = EoHPrompt_free
        elif metaheuristic_type == 4:
            self.EoHPrompt = EoHPrompt_thought
        else:
            print('metaheuristic_type must be 1, 2 or 3.')
        self._evaluations = evaluations
        self._num_tasks = len(evaluations)
        self._template_program_strs = [eval.template_program for eval in evaluations]
        self._task_description_strs = [eval.task_description for eval in evaluations]
        self._max_generations = max_generations
        self._max_sample_nums = max_sample_nums
        self._high_level_pop_size = high_level_pop_size
        self._low_level_max_samples_each_program = low_level_max_samples_each_program
        self._low_level_pop_size = low_level_pop_size
        self._selection_num = selection_num
        self._use_e2_operator = use_e2_operator
        self._use_m1_operator = use_m1_operator
        self._use_m2_operator = use_m2_operator
        self._knowledge_transfer = enable_knowledge_transfer
        self._low_level_search = enable_low_level_search
        self._pareto_population_management = enable_pareto_population_management

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
        self._high_level_population = Population(pop_size=self._high_level_pop_size,use_pareto=self._pareto_population_management)

        # dictionary to store low-level populations for each program and each task
        # {program_id: [population_for_task1, population_for_task2, ...]}
        self._low_level_populations: Dict[str, List[Population]] = {}

        # dictionary to store key function templates for each program and each task
        # {program_id: [template_for_task1, template_for_task2, ...]}
        self._key_function_templates: Dict[str, List[str]] = {}

        # Create samplers and evaluators for each task
        self._samplers = EoHSampler(llm)
        self._evaluators = [SecureEvaluator(evaluation, debug_mode=debug_mode, **kwargs)
                            for evaluation in self._evaluations]
        self._profiler = profiler

        # statistics
        self._tot_sample_nums = 0

        # reset _initial_sample_nums_max
        self._initial_sample_nums_max = min(
            self._max_sample_nums,
            self._high_level_pop_size * self._low_level_max_samples_each_program * self._num_tasks
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
            self._profiler.record_parameters(llm, evaluations[0],
                                             self)  # Using first evaluation for backward compatibility
        self._lock = Lock()

    def _adjust_pop_size(self):
        # 为什么这样调整高层种群size呢？

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

    def _identify_key_function(self, program_str: str, task_idx: int) -> str:
        # 识别任务特定算法中的关键函数

        """Use LLM to identify the key function in a program and create a template for it."""
        prompt = self.EoHPrompt.get_prompt_keyfunc(self._task_description_strs[task_idx], program_str)

        if self._debug_mode:
            print(f"check identify key function prompt for task {task_idx}: {prompt}")

        key_function_template = self._samplers.get_python_response(prompt)

        if self._debug_mode:
            print(f"check identify key function response for task {task_idx}: {key_function_template}")

        return key_function_template


    def _generate_full_program(self, program_str: str, key_function_body: str, task_idx: int) -> str:
        """Generate a complete program by replacing the key function template with the optimized function body."""

        # Use LLM to integrate the optimized function back into the program
        prompt = self.EoHPrompt.get_prompt_fullprogram(self._task_description_strs[task_idx], program_str, key_function_body)

        if self._debug_mode:
            print(f"check _generate_full_program prompt for task {task_idx}: {prompt}")

        updated_program = self._samplers.get_python_response(prompt)

        if self._debug_mode:
            print(f"check _generate_full_program response for task {task_idx}: {updated_program} ")

        return updated_program

    def _sample_evaluate_register_high_level(self, prompt, max_attempts=3):
        """Sample, evaluate, and register a high-level program design for a specific task."""
        print(f"DEBUG: Starting high-level sampling with max_attempts={max_attempts}")

        pseudocode, response = self._samplers.get_thought(prompt)
        if self._debug_mode:
            print(f"\n Check pseudocode: {pseudocode} \n")

        # register to profiler
        program = Program()
        program.algorithm = pseudocode
        program.task_programs = [''] * self._num_tasks
        program.function_bodies = [''] * self._num_tasks
        program.task_scores = [float('-inf')] * self._num_tasks

        self._lock.acquire()
        try:
            # Create a unique identifier for this program
            high_level_program_id = str(len(self._low_level_populations) + 1)
            print(f"DEBUG: Created program ID: {high_level_program_id}")

            program.id = high_level_program_id

            # Initialize task-specific data structures
            self._key_function_templates[high_level_program_id] = [''] * self._num_tasks

            # Create low-level populations for each task
            self._low_level_populations[high_level_program_id] = [
                Population(pop_size=self._low_level_max_samples_each_program)
                for _ in range(self._num_tasks)]

        finally:
            self._lock.release()

        try:

            print(f"DEBUG: Processing {self._num_tasks} tasks")
            for task_idx in range(self._num_tasks):

                if self._high_level_population.generation == 0 and self._tot_sample_nums >= self._initial_sample_nums_max:
                    print(f"Reach max init evaluation!")
                    return


                print(f"DEBUG: Starting task {task_idx}")

                # Generate variations
                prompt = self.EoHPrompt.get_prompt_program(pseudocode, self._template_program_strs[task_idx])

                attempts = 0
                score = None
                while score is None and attempts < max_attempts:
                    print(f"DEBUG: Task {task_idx}, attempt {attempts + 1}/{max_attempts}")

                    if self._debug_mode:
                        print(f"\n Check prompt: {prompt} \n")

                    program_str, response = self._samplers.get_program(prompt)

                    if self._debug_mode:
                        print(f"\n Check response: {response} \n")
                        print(f"\n Check program: {program_str} \n")

                    # evaluate for the specific task
                    score, eval_time = self._evaluation_executor.submit(
                        self._evaluators[task_idx].evaluate_program_record_time,
                        program_str
                    ).result()

                    attempts += 1

                if self._debug_mode:
                    print(f"\n score for task {task_idx}: {score} \n")

                if score is None:
                    print(f"\n no valid program found for task {task_idx}\n")
                    return

                if score is not None and not math.isinf(score):
                    print(f"DEBUG: Valid score {score} for task {task_idx}")

                    # For multi-task, we need to create a list of program_str for each task
                    # Initially, only the current task has a valid program_str

                    program.task_programs[task_idx] = program_str

                    program.task_scores[task_idx] = score


                    if self._low_level_search:
                        ######    start of key function identification and evolution

                        # Identify key function and create template
                        key_function_template = None
                        attempts = 0

                        while key_function_template is None and attempts < max_attempts:
                            key_function_template = self._identify_key_function(program_str, task_idx)
                            attempts += 1
                            if key_function_template is None and attempts < max_attempts:
                                if self._debug_mode:
                                    print(
                                        f"Failed to identify key function for task {task_idx} (attempt {attempts}/{max_attempts}), retrying...")

                        if key_function_template is None:
                            if self._debug_mode:
                                print(
                                    f"Failed to identify key function for task {task_idx} after {max_attempts} attempts, skipping this program")
                            return

                        print(f"DEBUG: Key function identified for program  {high_level_program_id} task {task_idx}")
                        program.function_bodies[task_idx] = key_function_template

                        self._key_function_templates[high_level_program_id][task_idx] = key_function_template

                        # Initialize the low-level population for the current task with the original key function
                        print(f"DEBUG: Initializing low-level population for program  {high_level_program_id} task {task_idx}")
                        self._initialize_low_level_population(high_level_program_id, program_str, copy.deepcopy(program),
                                                              task_idx, pseudocode)

                        # Evolve low-level population for the current task
                        print(f"DEBUG: Evolving low-level population for program  {high_level_program_id} task {task_idx}")
                        self._evolve_low_level_populations(high_level_program_id, program_str, task_idx, pseudocode)

                        best_low_program = self._low_level_populations[high_level_program_id][task_idx].get_best(task_idx)

                        if best_low_program and best_low_program.task_scores[task_idx] > program.task_scores[task_idx]:
                            print(
                                f"DEBUG: Best low-level program improved score for task {task_idx}: {best_low_program.task_scores[task_idx]}")
                            program.task_scores[task_idx] = best_low_program.task_scores[task_idx]
                            program.task_programs[task_idx] = best_low_program.task_programs[task_idx]


                        ######    end of key function identification and evolution
                    else:
                        if self._profiler is not None:
                            self._profiler.register_function(program, ' ')
                            self._tot_sample_nums += 1


        except Exception as e:
            if self._debug_mode:
                print("error in register high level program")
                traceback.print_exc()



        # register to the high-level population
        print(f"DEBUG: Registering program {high_level_program_id} to high-level population")
        self._high_level_population.register_program(program,self._num_tasks)

        if self._profiler is not None:
            if isinstance(self._profiler, EoHProfiler):
                print(f"DEBUG: Registering to profiler")
                self._profiler.register_population(self._high_level_population)


        try:
            # 知识迁移
            if self._high_level_population.generation > 0 and self._knowledge_transfer:

                print(f"DEBUG: Start knowledge transfer for population {self._high_level_population.generation} program {high_level_program_id}")

                for task_idx in range(self._num_tasks):

                    print(f"DEBUG: Start knowledge transfer best_program from task {task_idx}")
                    # --- REFACTORED: Find the best program for the current task across both populations ---
                    best_program_for_task = None
                    best_score_for_task = float('-inf')

                    # Combine both populations for a single, clean search
                    # itertools.chain is efficient as it doesn't create a new list
                    all_programs = self._high_level_population.population + self._high_level_population.next_population

                    print("get all programs")

                    for program_candidate in all_programs:
                        if program_candidate.task_scores[task_idx] > best_score_for_task:
                            best_score_for_task = program_candidate.task_scores[task_idx]
                            best_program_for_task = program_candidate

                    # If no program was found (e.g., populations are empty), skip to the next task
                    if best_program_for_task is None:
                        print("no program was found (e.g., populations are empty), skip to the next task")
                        continue

                    # Create a deep copy to modify, leaving the original untouched
                    program_to_transfer = copy.deepcopy(best_program_for_task)

                    # --- End of refactoring ---

                    # The program from the best-performing task is used as a basis for the prompt
                    base_program_str = program_to_transfer.task_programs[task_idx]

                    needs_registration = False
                    for task_idx_transfer in range(self._num_tasks):

                        if task_idx_transfer != task_idx:
                            print(f"DEBUG: Start knowledge transfer to task {task_idx_transfer}")
                            score = None
                            attempts = 0  # FIXED: Initialize attempts
                            while score is None and attempts < max_attempts:


                                prompt = self.EoHPrompt.get_prompt_transfer(
                                    self._task_description_strs[task_idx_transfer],
                                    base_program_str,
                                    self._template_program_strs[task_idx_transfer]
                                )
                                program_str, response = self._samplers.get_program(prompt)

                                if self._debug_mode:
                                    print(f"\n Check response: {response} \n")
                                    print(f"\n Check program: {program_str} \n")

                                # evaluate for the specific task
                                score, eval_time = self._evaluation_executor.submit(
                                    self._evaluators[task_idx_transfer].evaluate_program_record_time,
                                    program_str
                                ).result()

                                attempts += 1  # FIXED: Increment attempts to prevent infinite loop

                                print(f"DEBUG: attempts {attempts}/{max_attempts}, score {score}, original score {program_to_transfer.task_scores[task_idx_transfer]}")

                                # If the new program is better, update the copied program object
                                if score is not None:
                                    if score > program_to_transfer.task_scores[task_idx_transfer]:
                                        program_to_transfer.task_scores[task_idx_transfer] = score
                                        program_to_transfer.task_programs[task_idx_transfer] = f"#Improved in knowledge transfer\n{program_str}"
                                        # program_to_transfer.function_bodies[
                                        #     task_idx_transfer] = f'Improved in knowledge transfer'
                                        needs_registration = True
                                    # else:
                                    #     # program_to_transfer.task_scores[task_idx_transfer] = score
                                    #     # program_to_transfer.task_programs[task_idx_transfer] = program_str
                                    #     # program_to_transfer.function_bodies[
                                    #     #     task_idx_transfer] = f'Not improved in knowledge transfer'
                                    if self._profiler is not None:
                                        self._profiler.register_function(program_to_transfer, '')
                                        self._tot_sample_nums += 1

                    if needs_registration:
                        # FIXED: Use an attribute from the program object for the ID
                        program_id = getattr(program_to_transfer, 'id', 'N/A')

                        print(f"DEBUG: Removing original program to replace with improved version.")
                        with self._high_level_population._lock:
                            print(f"DEBUG: Acquired lock. Removing original program.")
                            try:
                                # Attempt to remove from the current generation's population
                                self._high_level_population.population.remove(best_program_for_task)
                                print(f"DEBUG: Removed from 'population' list.")
                            except ValueError:
                                # If not found, it must be in the next generation's staging list
                                try:
                                    self._high_level_population.next_population.remove(best_program_for_task)
                                    print(f"DEBUG: Removed from 'next_population' list.")
                                except ValueError:
                                    # This case is unlikely if the logic is sound, but it's safe to handle.
                                    print(f"WARNING: Could not find original program to remove.")

                        print(f"DEBUG: Registering transfer-improved program {program_id} to high-level population")

                        self._high_level_population.register_program(program_to_transfer, self._num_tasks)

                        # SIMPLIFIED: Combined profiler check
                        if isinstance(self._profiler, EoHProfiler):
                            print(f"DEBUG: Registering transfer to profiler")
                            self._profiler.register_population(self._high_level_population)

        except Exception as e:
            print("error in knowledge transfer")
            traceback.print_exc()




    # def _sample_evaluate_register_high_level(self, prompt,max_attempts = 5):
    #     """Sample, evaluate, and register a high-level program design for a specific task."""
    #     pseudocode, response = self._samplers.get_thought(prompt)
    #     if self._debug_mode:
    #         print(f"\n Check pseudocode: {pseudocode} \n")
    #
    #     # register to profiler
    #     program = Program()
    #     program.algorithm = pseudocode
    #     program.task_programs = [''] * self._num_tasks
    #     program.function_bodies = [''] * self._num_tasks
    #     program.task_scores = [float('-inf')] * self._num_tasks
    #
    #     self._lock.acquire()
    #     try:
    #         # Create a unique identifier for this program
    #         high_level_program_id = str(len(self._low_level_populations) + 1)
    #
    #         # Initialize task-specific data structures
    #         self._key_function_templates[high_level_program_id] = [''] * self._num_tasks
    #
    #         # Create low-level populations for each task
    #         self._low_level_populations[high_level_program_id] = [
    #             Population(pop_size=self._low_level_max_samples_each_program)
    #             for _ in range(self._num_tasks)]
    #
    #     finally:
    #         self._lock.release()
    #
    #     for task_idx in range(self._num_tasks):
    #
    #         # Generate variations
    #         prompt = EoHPrompt.get_prompt_program(pseudocode, self._template_program_strs[task_idx])
    #
    #         attempts = 0
    #         score = None
    #         while score is None and attempts < max_attempts:
    #
    #             program_str, response = self._samplers.get_program(prompt)
    #
    #             if self._debug_mode:
    #                 print(f"\n Check program: {program_str} \n")
    #
    #             if program_str is None:
    #                 return
    #
    #             # evaluate for the specific task
    #             score, eval_time = self._evaluation_executor.submit(
    #                 self._evaluators[task_idx].evaluate_program_record_time,
    #                 program_str
    #             ).result()
    #
    #             attempts += 1
    #
    #         if self._debug_mode:
    #             print(f"\n score for task {task_idx}: {score} \n")
    #
    #         if score is None:
    #             print(f"\n no valid program found for task {task_idx}\n")
    #             return
    #
    #         if score is not None and not math.isinf(score):
    #
    #
    #             # For multi-task, we need to create a list of program_str for each task
    #             # Initially, only the current task has a valid program_str
    #
    #             program.task_programs[task_idx] = program_str
    #
    #             program.task_scores[task_idx] = score
    #
    #             # Identify key function and create template
    #             key_function_template = None
    #             attempts = 0
    #
    #             while key_function_template is None and attempts < max_attempts:
    #                 key_function_template = self._identify_key_function(program_str, task_idx)
    #                 attempts += 1
    #                 if key_function_template is None and attempts < max_attempts:
    #                     if self._debug_mode:
    #                         print(
    #                             f"Failed to identify key function for task {task_idx} (attempt {attempts}/{max_attempts}), retrying...")
    #
    #             if key_function_template is None:
    #                 if self._debug_mode:
    #                     print(
    #                         f"Failed to identify key function for task {task_idx} after {max_attempts} attempts, skipping this program")
    #                 return
    #
    #             program.function_bodies[task_idx] = key_function_template
    #
    #             self._key_function_templates[high_level_program_id][task_idx] = key_function_template
    #
    #             program.id = high_level_program_id
    #
    #             # Initialize the low-level population for the current task with the original key function
    #             self._initialize_low_level_population(high_level_program_id, program_str, copy.deepcopy(program), task_idx, pseudocode)
    #
    #             # Evolve low-level population for the current task
    #             self._evolve_low_level_populations(high_level_program_id, program_str, task_idx, pseudocode)
    #
    #             best_low_program = self._low_level_populations[high_level_program_id][task_idx].get_best(task_idx)
    #
    #             if best_low_program and best_low_program.task_scores[task_idx] > program.task_scores[task_idx]:
    #                 program.task_scores[task_idx] = best_low_program.task_scores[task_idx]
    #                 program.task_programs[task_idx] = best_low_program.task_programs[task_idx]
    #                 program.task_scores[task_idx] = best_low_program.task_scores[task_idx]  # Update overall score to the task-specific score
    #
    #             # if self._profiler is not None:
    #             #     self._profiler.register_function(program, program=str(program))
    #             #     # if isinstance(self._profiler, EoHProfiler):
    #             #     #     self._profiler.register_population(self._high_level_population)
    #             #     self._tot_sample_nums += 1
    #
    #             # register to the high-level population
    #             #self._high_level_population.register_program(program)
    #
    #     # register to the high-level population
    #     self._high_level_population.register_program(program)
    #
    #     if self._profiler is not None:
    #         if isinstance(self._profiler, EoHProfiler):
    #             self._profiler.register_population(self._high_level_population)

    def _initialize_low_level_population(self, program_id: str, program_str: str, program_high: Program, task_idx: int, pseudocode: str):
        """Initialize the low-level population for a specific program and task."""
        # Extract the key function body from the original program
        original_function = self._key_function_templates[program_id][task_idx]
        if not original_function:
            return

        # Create initial variants for the low-level population
        for i in range(self._low_level_pop_size):
            if i == 0:
                # First variant is the original function body
                # Create a task-specific copy of the program
                program = copy.deepcopy(program_high)
                program.task_idx = task_idx
                full_program = program.task_programs[task_idx]

                # Register to the low-level population for this task
                self._low_level_populations[program_id][task_idx].register_program_low_level(program,task_idx)

            else:
                # Generate variations
                prompt = self.EoHPrompt.get_prompt_newfunc(self._task_description_strs[task_idx],original_function)

                if self._debug_mode:
                    print(f"Create a variation of this function prompt for program {program_id}, task {task_idx}")

                function_body = self._samplers.get_python_response(prompt)
                if function_body is None:
                    continue

                if self._debug_mode:
                    print(
                        f"Create a variation of this function response for program {program_id}, task {task_idx}: {function_body}")

                # Generate full program with this function variant
                full_program = self._generate_full_program(program_str, function_body, task_idx)

                # Evaluate for this specific task
                score, eval_time = self._evaluation_executor.submit(
                    self._evaluators[task_idx].evaluate_program_record_time,
                    full_program
                ).result()

                # Create program object
                program = Program()

                program.evaluate_time = eval_time
                program.algorithm = pseudocode
                program.task_specific = True
                program.task_idx = task_idx

                # For task-specific programs, initialize task_programs and task_scores
                program.task_programs = [''] * self._num_tasks
                program.task_programs[task_idx] = full_program
                program.task_scores = [float('-inf')] * self._num_tasks
                program.task_scores[task_idx] = score
                program.function_bodies = [''] * self._num_tasks
                program.function_bodies[task_idx] = function_body  # Store just the function body

                # Register to the low-level population for this task
                self._low_level_populations[program_id][task_idx].register_program_low_level(program,task_idx)

            if self._profiler is not None:
                self._profiler.register_function(program, full_program)
                self._tot_sample_nums += 1
                if isinstance(self._profiler, EoHProfiler):
                    self._profiler.register_low_level_population(self._low_level_populations[program_id][task_idx],
                                                                 f"{program_id}_task_{str(task_idx)}")


    def _sample_evaluate_register_low_level(self, program_id: str, program_str: str, prompt, task_idx: int, pseudocode: str):
        """Sample, evaluate, and register a low-level function variant for a specific task."""
        sample_start = time.time()
        function_body = self._samplers.get_python_response(prompt)
        sample_time = time.time() - sample_start
        if function_body is None:
            return

        # Generate full program with this function variant
        full_program = self._generate_full_program(program_str, function_body, task_idx)

        # evaluate for this specific task
        score, eval_time = self._evaluation_executor.submit(
            self._evaluators[task_idx].evaluate_program_record_time,
            full_program
        ).result()

        # register to profiler
        program = Program()

        program.evaluate_time = eval_time
        program.algorithm = pseudocode
        program.sample_time = sample_time
        program.task_idx = task_idx

        # For task-specific programs, initialize task_programs and task_scores
        program.task_programs = [''] * self._num_tasks
        program.task_programs[task_idx] = full_program
        program.task_scores = [float('-inf')] * self._num_tasks
        program.task_scores[task_idx] = score
        program.function_bodies = [''] * self._num_tasks
        program.function_bodies[task_idx] = function_body  # Store just the function body


        # register to the low-level population for this task
        self._low_level_populations[program_id][task_idx].register_program_low_level(program,task_idx)

        if self._profiler is not None:
            self._profiler.register_function(program, program=str(program))
            if isinstance(self._profiler, EoHProfiler):
                self._profiler.register_low_level_population(self._low_level_populations[program_id][task_idx],
                                                             f"{program_id}_task_{str(task_idx)}")
            self._tot_sample_nums += 1



    def _get_best_program_for_id_and_task(self, program_id: str, task_idx: int) -> Program:
        """Get the best program variant from a low-level population for a specific task."""
        return self._low_level_populations[program_id][task_idx].get_best(task_idx)


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
        """Evolve the high-level population (program designs) for all tasks."""
        while self._continue_loop():
            # For each iteration, randomly select a task to optimize
                try:
                    # get a new program using e1 for the current task
                    indivs = [self._high_level_population.selection() for _ in range(self._selection_num)]
                    prompt = self.EoHPrompt.get_prompt_e1(self._task_description_strs, indivs)

                    if self._debug_mode:
                        print(f'High-Level E1 Prompt: {prompt}')
                    self._sample_evaluate_register_high_level(prompt)
                    if not self._continue_loop():
                        break

                    # get a new program using e2 for the current task
                    if self._use_e2_operator:
                        indivs = [self._high_level_population.selection() for _ in range(self._selection_num)]
                        prompt = self.EoHPrompt.get_prompt_e2(self._task_description_strs, indivs)
                        if self._debug_mode:
                            print(f'High-Level E2 Prompt for task: {prompt}')
                        self._sample_evaluate_register_high_level(prompt)
                        if not self._continue_loop():
                            break

                except KeyboardInterrupt:
                    return
                except Exception as e:
                    if self._debug_mode:
                        traceback.print_exc()
                        exit()
                    continue


    def _evolve_low_level_populations(self, program_id: str, program_str: str, task_idx: int, pseudocode: str):
        """Evolve the low-level populations (key function variants) for a specific program and task."""
        # Get the key function template for this program and task
        key_function_template = self._key_function_templates[program_id][task_idx]
        if not key_function_template:
            return

        population = self._low_level_populations[program_id][task_idx]
        n_low_sampled = len(population.population)
        while n_low_sampled < self._low_level_max_samples_each_program:
            try:
                n_low_sampled += 1
                # Get a new function variant using mutation
                indiv = population.selection_from_topk(self._low_level_pop_size,task_idx)
                if not indiv:
                    break

                function_body = indiv.function_bodies  # This contains just the function body

                prompt = self.EoHPrompt.get_prompt_mutfunc(self._task_description_strs[task_idx], function_body)

                if self._debug_mode:
                    print(f'Low-Level Mutation Prompt for program {program_id}, task {task_idx}')

                self._sample_evaluate_register_low_level(program_id, program_str, prompt, task_idx, pseudocode)

            except Exception as e:
                if self._debug_mode:
                    traceback.print_exc()


    def _iteratively_init_high_level_population(self):
        """Initialize the high-level population (program designs) for all tasks."""
        while self._high_level_population.generation == 0:

            try:
                # get a new program using i1 for the current task
                prompt = self.EoHPrompt.get_prompt_i1(self._task_description_strs)
                if self._debug_mode:
                    print(f"Check initial prompt for task: {prompt}")
                self._sample_evaluate_register_high_level(prompt)
                if self._tot_sample_nums >= self._initial_sample_nums_max:
                    print(
                        f'Note: During initialization, MTAHD gets {len(self._high_level_population) + len(self._high_level_population._next_gen_pop)} algorithms '
                        f'after {self._initial_sample_nums_max} trails.')
                    return
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


    def get_best_solution(self, task_idx: Optional[int] = None) -> Program:
        """
        Get the best solution.

        Args:
            task_idx: If provided, get the best solution for a specific task.
                     If None, get the overall best solution across all tasks.

        Returns:
            The best Program object
        """
        if task_idx is not None:
            # Get best solution for a specific task
            best_program = None
            best_score = float('-inf')

            # Check high-level population
            for program in self._high_level_population.population:
                if program.task_scores[task_idx] > best_score:
                    best_program = program
                    best_score = program.task_scores[task_idx]

            # Check all low-level populations for this task
            for program_id, populations in self._low_level_populations.items():
                low_level_best = populations[task_idx].get_best(task_idx)
                if low_level_best and low_level_best.task_scores[task_idx] > best_score:
                    best_program = low_level_best
                    best_score = low_level_best.task_scores[task_idx]

            return best_program
        else:
            # Get overall best solution (highest average score across tasks)
            best_program = None
            best_avg_score = float('-inf')

            # Check high-level population
            for program in self._high_level_population.population:
                valid_scores = [score for score in program.task_scores if score != float('-inf')]
                if valid_scores:
                    avg_score = sum(valid_scores) / len(valid_scores)
                    if avg_score > best_avg_score:
                        best_program = program
                        best_avg_score = avg_score

            return best_program


    def run(self):
        if not self._resume_mode:
            # do initialization of high-level population
            self._multi_threaded_sampling(self._iteratively_init_high_level_population)

            print("start survival")
            print(len(self._high_level_population))
            self._high_level_population.survival()
            print("end survival")

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

        # Close all samplers
        self._samplers.llm.close()

