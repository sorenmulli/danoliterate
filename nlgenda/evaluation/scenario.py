from typing import Generator

from datasets import Dataset, load_dataset
from omegaconf import DictConfig

from nlgenda.evaluation.example import EvaluationExample
from nlgenda.evaluation.task_builder import TaskBuilder, get_task_builder


class EvaluatorScenario:
    def __init__(self, name: str, path: str, task: DictConfig):
        self.name = name
        self.path = path
        self.task_builder: TaskBuilder = get_task_builder(task.builder)

        self.dataset: Dataset = load_dataset(self.path, split="train")

    def generate_examples(self) -> Generator[EvaluationExample, None, None]:
        for data_example in self.dataset:
            yield self.task_builder.build_example(data_example)

    def __len__(self):
        return len(self.dataset)
