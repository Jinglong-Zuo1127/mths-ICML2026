from __future__ import annotations

from typing import List, Dict

from ...base import *


class EoHPrompt_abstract:

    @classmethod
    def create_instruct_prompt(cls, prompt: str) -> List[Dict]:
        content = [
            {'role': 'system', 'message': cls.get_system_prompt()},
            {'role': 'user', 'message': prompt}
        ]
        return content

    @classmethod
    def get_system_prompt(cls) -> str:
        return ''

    @classmethod
    def get_prompt_i1(cls, task_prompts):

        # Format task prompts
        tasks_formatted = '\n'.join(f'- {task}' for task in task_prompts)

        # Create prompt
        prompt_content = f"""You are an expert algorithm designer. Your task is to create one novel algorithm for the following tasks:
{tasks_formatted}

Design and present the high-level task-agnostic code abstract for your new algorithm in 10–20 lines.

Enclose the entire pseudocode block within a single code block marked by ```pseudocode and ```.
"""
        return prompt_content

    @classmethod
    def get_prompt_e1(cls, task_prompts, indivs: List[Program]):
        for indi in indivs:
            assert hasattr(indi, 'algorithm')

        # create prompt content for all individuals
        indivs_prompt = ''
        for i, indi in enumerate(indivs):
            indi.docstring = ''
            indivs_prompt += f'No. {i + 1} algorithm is:\n{indi.algorithm}'

        # Format task prompts
        tasks_formatted = '\n'.join(f'- {task}' for task in task_prompts)

        # Create final prompt
        prompt_content = f"""You are an expert algorithm designer. Your task is to create one novel algorithm for the following tasks:
{tasks_formatted}

I have {len(indivs)} existing algorithms as follows:
{indivs_prompt}
Please help me create a new algorithm that has a totally different form from the given ones. 

Design and present the high-level task-agnostic code abstract for your new algorithm in 10–20 lines.

Enclose the entire pseudocode block within a single code block marked by ```pseudocode and ```.
"""
        return prompt_content

    @classmethod
    def get_prompt_e2(cls, task_prompts: List[str], indivs: List[Program]):
        for indi in indivs:
            assert hasattr(indi, 'algorithm')

        # create prompt content for all individuals
        indivs_prompt = ''
        for i, indi in enumerate(indivs):
            indi.docstring = ''
            indivs_prompt += f'No. {i + 1} algorithm is:\n{indi.algorithm}'

        # Format task prompts
        tasks_formatted = '\n'.join(f'- {task}' for task in task_prompts)

        # Create final prompt
        prompt_content = f"""You are an expert algorithm designer. Your task is to create one novel algorithm for the following tasks:
{tasks_formatted}

I have {len(indivs)} existing algorithms as follows:
{indivs_prompt}
Please help me create a new algorithm that is a revised version of given ones.

Design and present the high-level task-agnostic code abstract for your new algorithm in 10–20 lines.

Enclose the entire pseudocode block within a single code block marked by ```pseudocode and ```.
"""
        return prompt_content

    @classmethod
    def get_prompt_keyfunc(cls, task_description: str, program_str: str):
        prompt_content = f"""
You are given a program. Please identify the most important function in this program 
that would benefit most from optimization.  

Program:
{program_str}

Task Description:
{task_description}

Return only the key function. It should be enclosed between ```python and ``` markers exactly as shown below:

```python
# Your key function here
```
"""
        return prompt_content

    @classmethod
    def get_prompt_fullprogram(cls, task_description, program_str, key_function_body):
        prompt_content = f"""
You are given a program and an optimized function. Please replace the original function in the program
with the optimized version and return the complete updated program.

Original program:
{program_str}

Optimized function:
{key_function_body}

Task Description:
{task_description}

Return the complete updated program. It should be enclosed between ```python and ``` markers exactly as shown below:

```python
# our updated program here
```
            """
        return prompt_content

    @classmethod
    def get_prompt_newfunc(cls, task_description, original_function):
        prompt_content = f"""
You are given a function. Please create a variation of this function that
with the same inputs and outputs but might be more effective or use a different approach.
The function is part of a larger program solving the following task:

Task description:
{task_description}

Original function body:
{original_function}

Return only the modified function. It should be enclosed between ```python and ``` markers exactly as shown below:

```python
# Your new function
```
                    """
        return prompt_content

    @classmethod
    def get_prompt_mutfunc(cls, task_description, function_body):

        prompt_content = f"""
You are given a function. Please improve this function. The function is part of a larger program solving the following task:

Task description:
{task_description}

Current function:
{function_body}

Return only the improved function. It should be enclosed between ```python and ``` markers exactly as shown below:

```python
# Your improved function here
```
                    """
        return prompt_content

    @classmethod
    def get_prompt_program(cls, pseudocode, template_program_str):
        prompt_content = f"""
You are an expert algorithm implementer. Given a pseudocode algorithm, convert it to an efficient Python implementation.

PSEUDOCODE:
{pseudocode}

IMPLEMENTATION REQUIREMENTS:
1. Use the template structure provided below
2. Ensure the implementation runs in acceptable time complexity
3. Maintain the core logic of the pseudocode
4. Use appropriate Python data structures and libraries

TEMPLATE:
{template_program_str}

RESPONSE FORMAT:
Return ONLY the Python code without explanations or examples, enclosed between ```python and ``` markers as shown:

```python
# Your program here
```
            """
        return prompt_content

    @classmethod
    def get_prompt_transfer(cls, task_description, refer_program, template_program_str):

        prompt_content = f"""
You are an expert algorithm designer specialized in translating knowledge between related problems. Your task is to implement an algorithm for the specified target problem by drawing inspiration from a reference algorithm that solves a related task.

Target Problem:
{task_description}

Reference Algorithm:
The following is a high-quality implementation for a related problem that can inform your approach:

{refer_program}

Implementation Template:
Your implementation should follow this template structure:

{template_program_str}

RESPONSE FORMAT:
Return ONLY the Python code without explanations or examples, enclosed between ```python and ``` markers as shown:

```python
# Your implementation here
```
            """
        return prompt_content