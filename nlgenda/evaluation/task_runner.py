import logging
from abc import ABC, abstractmethod
from typing import Any

from nlgenda.evaluation.example import EvaluationExample
from nlgenda.evaluation.model_inference import InferenceMethod, ModelInference
from nlgenda.modeling.text_comparison import TextCompareFun

logger = logging.getLogger(__name__)


class TaskRunner(ABC):
    @abstractmethod
    def build_example(self, row: dict[str, Any]) -> EvaluationExample:
        ...

    @abstractmethod
    def get_prediction(
        self, example: EvaluationExample, inference: ModelInference, text_compare: TextCompareFun
    ) -> EvaluationExample:
        ...


class MultichoiceRunner(TaskRunner):
    def __init__(self, prompt_feature: str = "text", id_features: tuple[str, ...] = ("id",)):
        self.prompt_feature = prompt_feature
        # Allow ID to be concatenation of features
        self.id_features = id_features

    def process_prompt(self, prompt: str) -> str:
        return f"{prompt} "

    def process_option(self, option: str) -> str:
        return option

    def get_example_id(self, row: dict[str, Any]) -> str:
        return "-".join(str(row[id_feature]) for id_feature in self.id_features)

    def get_options(self, row: dict[str, Any]) -> list[str]:
        options = []
        for name, val in row.items():
            if "option" in name and isinstance(val, str):
                options.append(self.process_option(val))
        return options

    def get_correct_idx(self, row: dict[str, Any]) -> int:
        assert isinstance(row["correct"], int)
        return row["correct"]

    def build_example(self, row: dict[str, Any]) -> EvaluationExample:
        return EvaluationExample(
            prompt=self.process_prompt(row[self.prompt_feature]),
            id_=self.get_example_id(row),
            options=self.get_options(row),
            index_label=self.get_correct_idx(row),
        )

    def get_prediction(
        self, example: EvaluationExample, inference: ModelInference, text_compare: TextCompareFun
    ) -> EvaluationExample:
        assert example.options is not None

        match inference.inference_method:
            case InferenceMethod.LM:
                scores = [
                    inference.likelihood(example.prompt, option) for option in example.options
                ]

            case InferenceMethod.NLG:
                assert text_compare is not None
                example.generated_text = inference.generate_text(example.prompt)
                scores = [
                    text_compare(option, example.generated_text) for option in example.options
                ]

        # Prediction is argmax of scores
        example.index_prediction = scores.index(max(scores))
        example.options_model_scores = scores
        return example
