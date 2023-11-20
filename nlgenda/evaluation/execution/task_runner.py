import logging
from abc import ABC, abstractmethod
from typing import Any, Optional

from nlgenda.evaluation.execution.model_inference import ModelInference
from nlgenda.evaluation.results import ExecutionExample

logger = logging.getLogger(__name__)


class TaskRunner(ABC):
    id_features: tuple[str, ...] = ("id",)

    @abstractmethod
    def build_example(
        self, row: dict[str, Any], pre_prompt="", post_prompt="", idx: Optional[int] = None
    ) -> ExecutionExample:
        ...

    @abstractmethod
    def get_prediction(
        self, example: ExecutionExample, inference: ModelInference
    ) -> ExecutionExample:
        ...

    def get_example_id(self, row: dict[str, Any], idx: Optional[int] = None) -> str:
        if self.id_features:
            return "-".join(str(row[id_feature]) for id_feature in self.id_features)
        if idx is not None:
            return str(idx)
        raise ValueError("Neither ID features nor index where given; cannot identify col")

    def prepare_prompt(self, text: str, pre_prompt: str, post_prompt: str) -> str:
        return pre_prompt + text + post_prompt


class MultichoiceRunner(TaskRunner):
    def __init__(self, prompt_feature: str = "text", id_features: tuple[str, ...] = ("id",)):
        self.prompt_feature = prompt_feature
        # Allow ID to be concatenation of features
        self.id_features = id_features

    def get_options(self, row: dict[str, Any]) -> list[str]:
        options = []
        for name, val in row.items():
            if "option" in name and isinstance(val, str):
                options.append(val)
        return options

    def get_correct_idx(self, row: dict[str, Any]) -> int:
        assert isinstance(row["correct"], int)
        return row["correct"]

    def build_example(
        self, row: dict[str, Any], pre_prompt="", post_prompt="", idx: Optional[int] = None
    ) -> ExecutionExample:
        return ExecutionExample(
            prompt=self.prepare_prompt(row[self.prompt_feature], pre_prompt, post_prompt),
            id_=self.get_example_id(row, idx),
            options=self.get_options(row),
            index_label=self.get_correct_idx(row),
        )

    def get_prediction(
        self, example: ExecutionExample, inference: ModelInference
    ) -> ExecutionExample:
        assert example.options is not None

        if inference.can_do_lm:
            example.options_model_likelihoods = [
                inference.query_likelihood(example.prompt, option) for option in example.options
            ]
        if inference.can_do_nlg:
            example.generated_text = inference.query_generate_text(example.prompt)
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


class MultiChoiceRunnerSameOptions(MultichoiceRunner):
    """
    A multiple choice task that has the a constant set of possible label options globally,
    across exaples
    """

    def __init__(
        self,
        all_labels: list[str],
        label_feature: str = "label",
        prompt_feature: str = "text",
        id_features: tuple[str, ...] = ("id",),
    ):
        super().__init__(prompt_feature=prompt_feature, id_features=id_features)
        self.label_feature = label_feature
        self.all_labels = [str(x) for x in all_labels]

    def get_correct_idx(self, row: dict[str, Any]) -> int:
        return self.all_labels.index(row[self.label_feature])

    def get_options(self, _: dict[str, Any]) -> list[str]:
        return self.all_labels


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

    def build_example(
        self, row: dict[str, Any], pre_prompt="", post_prompt="", idx: Optional[int] = None
    ) -> ExecutionExample:
        return ExecutionExample(
            prompt=self.prepare_prompt(row[self.prompt_feature], pre_prompt, post_prompt),
            id_=self.get_example_id(row, idx),
            target_answer=row[self.answer_feature],
        )

    def get_prediction(
        self, example: ExecutionExample, inference: ModelInference
    ) -> ExecutionExample:
        assert example.target_answer is not None

        if inference.can_do_nlg:
            example.generated_text = inference.query_generate_text(example.prompt)

        return example
