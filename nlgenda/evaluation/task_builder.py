import logging
from abc import ABC, abstractmethod
from typing import Any

from nlgenda.evaluation.example import EvaluationExample, MultichoiceExample

logger = logging.getLogger(__name__)


class TaskBuilder(ABC):
    @abstractmethod
    def build_example(self, row: dict[str, Any]) -> EvaluationExample:
        ...


class MultichoiceBuilder(TaskBuilder):
    prompt_feature: str = "question"

    def process_prompt(self, prompt: str) -> str:
        return f"{prompt} "

    def process_option(self, option: str) -> str:
        return option

    def get_options(self, row: dict[str, Any]) -> list[str]:
        options = []
        for name, val in row.items():
            if "option" in name and isinstance(val, str):
                options.append(self.process_option(val))
        return options

    def get_correct_idx(self, row: dict[str, Any]) -> int:
        assert isinstance(row["correct"], int)
        return row["correct"]

    def build_example(self, row: dict[str, Any]) -> MultichoiceExample:
        return MultichoiceExample(
            self.process_prompt(row[self.prompt_feature]),
            self.get_options(row),
            self.get_correct_idx(row),
        )


class HyggeSwagMultichoiceBuilder(MultichoiceBuilder):
    prompt_feature = "ctx"


def get_task_builder(key: str) -> TaskBuilder:
    match key:
        case "hyggeswag":
            return HyggeSwagMultichoiceBuilder()
        case _:
            logger.error("Unknown key %s.", key)
            raise ValueError("Key unknown.")
