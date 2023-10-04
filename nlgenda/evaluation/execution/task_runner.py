import logging
from abc import ABC, abstractmethod
from typing import Any

from nlgenda.evaluation.execution.model_inference import InferenceMethod, ModelInference
from nlgenda.evaluation.results import ExecutionExample

logger = logging.getLogger(__name__)


class TaskRunner(ABC):
    id_features: tuple[str, ...] = ("id",)

    @abstractmethod
    def build_example(self, row: dict[str, Any]) -> ExecutionExample:
        ...

    @abstractmethod
    def get_prediction(
        self, example: ExecutionExample, inference: ModelInference
    ) -> ExecutionExample:
        ...

    def get_example_id(self, row: dict[str, Any]) -> str:
        return "-".join(str(row[id_feature]) for id_feature in self.id_features)


class MultichoiceRunner(TaskRunner):
    def __init__(self, prompt_feature: str = "text", id_features: tuple[str, ...] = ("id",)):
        self.prompt_feature = prompt_feature
        # Allow ID to be concatenation of features
        self.id_features = id_features

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

    def build_example(self, row: dict[str, Any]) -> ExecutionExample:
        return ExecutionExample(
            prompt=self.process_prompt(row[self.prompt_feature]),
            id_=self.get_example_id(row),
            options=self.get_options(row),
            index_label=self.get_correct_idx(row),
        )

    def get_prediction(
        self, example: ExecutionExample, inference: ModelInference
    ) -> ExecutionExample:
        assert example.options is not None

        match inference.inference_method:
            case InferenceMethod.LM:
                example.options_model_likelihoods = [
                    inference.likelihood(example.prompt, option) for option in example.options
                ]

            case InferenceMethod.NLG:
                example.generated_text = inference.generate_text(example.prompt)

        return example


class MultichoiceRunnerLetterOptions(MultichoiceRunner):
    """
    A multiple choice task that has the options saved as letters A, B, C, D instead of indices
    """

    def get_correct_idx(self, row: dict[str, Any]) -> int:
        num_options = sum("option" in key for key in row)
        # Generates the string ABCDE...
        letter_options = "".join(chr(i) for i in range(65, 65 + num_options))
        return letter_options.index(row["correct"])


class AnswerSimilarityRunner(TaskRunner):
    def __init__(
        self,
        prompt_feature="prompt",
        answer_feature="answer",
        id_features: tuple[str, ...] = ("id",),
    ):
        self.prompt_feature = prompt_feature
        self.answer_feature = answer_feature
        self.id_features = id_features

    def build_example(self, row: dict[str, Any]) -> ExecutionExample:
        return ExecutionExample(
            prompt=row[self.prompt_feature],
            id_=self.get_example_id(row),
            target_answer=row[self.answer_feature],
        )

    def get_prediction(
        self, example: ExecutionExample, inference: ModelInference
    ) -> ExecutionExample:
        assert example.target_answer is not None

        match inference.inference_method:
            case InferenceMethod.LM:
                logger.error("AnswerSimilarityRunner does not support language modeling")
                raise ValueError("Unsupported inference method")

            case InferenceMethod.NLG:
                example.generated_text = inference.generate_text(example.prompt)

        return example
