from __future__ import annotations

import copy
from typing import List, Dict

from ...base import *


class EoHPrompt:
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
    def get_prompt_i1(cls, task_prompt: str, template_function: Function):
        # template
        temp_func = copy.deepcopy(template_function)
        
        # create prompt content
        prompt_content = f'''You are an expert algorithm designer. Your task is to create a novel algorithm for the following problem:

{task_prompt}

You must respond in two distinct parts. Do not combine them.

**PART 1: PSEUDOCODE**
First, design and present the high-level task-agnostic pseudocode for your new algorithm.
- The pseudocode must describe the core strategy and logical flow of the algorithm at a conceptual level.
- Crucially, avoid low-level task-specific implementation details. Do not include specific variable names, data structures, or numerical constants.
- Enclose the entire pseudocode block within a single code block marked by ```pseudocode and ```.

**PART 2: PYTHON IMPLEMENTATION**
Next, provide a complete, runnable Python implementation based on the following template:
{str(temp_func)}

- Your code must be a direct implementation of the pseudocode from Part 1.
- You must enclose your entire Python code within a single code block marked by ```python and ```.
- Do not include any text, explanations, or comments outside of these code block markers. The code must be ready to be copied and executed.'''
        return prompt_content

    @classmethod
    def get_prompt_e1(cls, task_prompt: str, indivs: List[Program], template_function: Function):
        for indi in indivs:
            assert hasattr(indi, 'algorithm')
        # template
        temp_func = copy.deepcopy(template_function)
        
        # create prompt content for all individuals
        indivs_prompt = ''
        for i, indi in enumerate(indivs):
            indi.docstring = ''
            indivs_prompt += f'No. {i + 1} algorithm is:\n{indi.algorithm}'
        # create prmpt content
        prompt_content = f'''You are an expert algorithm designer. Your task is to create a novel algorithm for the following problem:
{task_prompt}
I have {len(indivs)} existing algorithms as follows:
{indivs_prompt}
Please help me create a new algorithm that has a totally different form from the given ones. 

You must respond in two distinct parts. Do not combine them.

**PART 1: PSEUDOCODE**
First, design and present the high-level task-agnostic pseudocode for your new algorithm.
- The pseudocode must describe the core strategy and logical flow of the algorithm at a conceptual level.
- Crucially, avoid low-level task-specific implementation details. Do not include specific variable names, data structures, or numerical constants.
- Enclose the entire pseudocode block within a single code block marked by ```pseudocode and ```.

**PART 2: PYTHON IMPLEMENTATION**
Next, provide a complete, runnable Python implementation based on the following template:
{str(temp_func)}

- Your code must be a direct implementation of the pseudocode from Part 1.
- You must enclose your entire Python code within a single code block marked by ```python and ```.
- Do not include any text, explanations, or comments outside of these code block markers. The code must be ready to be copied and executed.'''
        return prompt_content

    @classmethod
    def get_prompt_e2(cls, task_prompt: str, indivs: List[Program], template_function: Function):
        for indi in indivs:
            assert hasattr(indi, 'algorithm')

        # template
        temp_func = copy.deepcopy(template_function)
        
        # create prompt content for all individuals
        indivs_prompt = ''
        for i, indi in enumerate(indivs):
            indi.docstring = ''
            indivs_prompt += f'No. {i + 1} algorithm is:\n{indi.algorithm}'
        # create prmpt content
        prompt_content = f'''You are an expert algorithm designer. Your task is to create a novel algorithm for the following problem:
{task_prompt}
I have {len(indivs)} existing algorithms as follows:
{indivs_prompt}
Please help me create a new algorithm that is different from the given ones but can be motivated from them.

You must respond in two distinct parts. Do not combine them.

**PART 1: PSEUDOCODE**
First, design and present the high-level task-agnostic pseudocode for your new algorithm.
- The pseudocode must describe the core strategy and logical flow of the algorithm at a conceptual level.
- Crucially, avoid low-level task-specific implementation details. Do not include specific variable names, data structures, or numerical constants.
- Enclose the entire pseudocode block within a single code block marked by ```pseudocode and ```.

**PART 2: PYTHON IMPLEMENTATION**
Next, provide a complete, runnable Python implementation based on the following template:
{str(temp_func)}

- Your code must be a direct implementation of the pseudocode from Part 1.
- You must enclose your entire Python code within a single code block marked by ```python and ```.
- Do not include any text, explanations, or comments outside of these code block markers. The code must be ready to be copied and executed.'''
        return prompt_content

    @classmethod
    def get_prompt_m1(cls, task_prompt: str, indi: Function, template_function: Function):
        assert hasattr(indi, 'algorithm')
        # template
        temp_func = copy.deepcopy(template_function)
        

        # create prmpt content
        prompt_content = f'''{task_prompt}
I have one algorithm with its code as follows. Algorithm description:
{indi.algorithm}
Code:
{str(indi)}
Please assist me in creating a new algorithm that has a different form but can be a modified version of the algorithm provided.
1. First, describe your new algorithm and main steps in one sentence. The description must be inside within boxed {{}}.
2. Next, implement the following program template:
{str(temp_func)}

Your implementation should be enclosed between ```python and ``` markers exactly as shown below:

```python
# Your program here
```

Do not give additional explanations.'''
        return prompt_content

    @classmethod
    def get_prompt_m2(cls, task_prompt: str, indi: Function, template_function: Function):
        assert hasattr(indi, 'algorithm')
        # template
        temp_func = copy.deepcopy(template_function)
        
        # create prmpt content
        prompt_content = f'''{task_prompt}
I have one algorithm with its code as follows. Algorithm description:
{indi.algorithm}
Code:
{str(indi)}
Please identify the main algorithm parameters and assist me in creating a new algorithm that has a different parameter settings of the score function provided.
1. First, describe your new algorithm and main steps in one sentence. The description must be inside within boxed {{}}.
2. Next, implement the following program template:
{str(temp_func)}

Your implementation should be enclosed between ```python and ``` markers exactly as shown below:

```python
# Your program here
```

Do not give additional explanations.'''
        return prompt_content
