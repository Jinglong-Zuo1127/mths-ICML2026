from __future__ import annotations

import re
from typing import Tuple, List, Dict

from .prompt import EoHPrompt
from ...base import LLM, SampleTrimmer, Function, Program
from ...base.modify_code import ModifyCode


class EoHSampler:
    def __init__(self, llm: LLM, template_program: str | Program):
        self.llm = llm
        self._template_program = template_program

    def get_thought_and_function(self, prompt: str) -> Tuple[str, Function]:
        response = self.llm.draw_sample(prompt)

        thought = self.__class__.trim_thought_from_response(response)
        program = self.__class__.trim_program_from_response(response)

        return thought, program, response

    @classmethod
    def trim_thought_from_response(cls, response: str) -> str | None:
        try:
            pattern = r'```pseudocode\s*(.*?)\s*```'  # Compared with r'\{(.*)\}'
            matches = re.findall(pattern, response, re.DOTALL)
            if matches:
                return matches[0].strip()
            return None
        except:
            return None

    @classmethod
    def trim_program_from_response(cls, response: str) -> str | None:
        try:
            # Match content between ```python and ``` markers
            pattern = r'```python\s*(.*?)\s*```'
            matches = re.findall(pattern, response, re.DOTALL)
            if matches:
                return matches[0].strip()
            return None
        except:
            return None

    def get_python_response(self, prompt: str) -> str | None:

        response = self.llm.draw_sample(prompt)
        program = self.__class__.trim_program_from_response(response)
        return program
