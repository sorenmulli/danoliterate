import logging
from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Optional

import numpy as np
from omegaconf import DictConfig, OmegaConf
from scipy import stats

from nlgenda.evaluation.execution.task_runner import AnswerSimilarityRunner, MultichoiceRunner
from nlgenda.evaluation.registries.get import get_task_runner
from nlgenda.evaluation.results import ExecutionExample, MetricResult
from nlgenda.evaluation.serialization import OutDictType
from nlgenda.modeling.text_comparison import COMPARE_FUNCTIONS, TextCompareFun, get_compare_fun

logger = logging.getLogger(__name__)


class Metric(ABC):
    confidence_level = 0.95

    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @property
    @abstractmethod
    def description(self) -> str:
        ...

    @property
    @abstractmethod
    def higher_is_better(self) -> bool:
        ...

    # Not an abstractmethod as you might want to not use this way to do it
    def compute_single(self, example: ExecutionExample) -> float:
        raise NotImplementedError("compute_single should be overwritten")

    def error(self, scores: np.ndarray) -> float:
        std_error = np.std(scores, ddof=1) / np.sqrt(len(scores))

        # Two sided Students t test with N - 1 degrees of freedom
        t_value = stats.t.ppf((1 + self.confidence_level) / 2, len(scores) - 1)

        return std_error * t_value

    def __call__(self, examples: list[ExecutionExample]) -> MetricResult:
        results = {example.id_: self.compute_single(example) for example in examples}
        scores = np.array(list(results.values()))
        mean = float(scores.mean())
        error = self.error(scores)
        return MetricResult(
            short_name=self.name,
            description=self.description,
            example_results=results,
            mean=mean,
            error=error,
            higher_is_better=self.higher_is_better,
        )


class Accuracy(Metric):
    def __init__(
        self,
        comparison_name: Optional[str] = None,
        comparison_function: Optional[TextCompareFun] = None,
    ):
        self.comparison_name = comparison_name
        self.comparison_function = comparison_function

    @property
    def name(self) -> str:
        return "Accuracy"

    @property
    def description(self) -> str:
        return "Simple true prediction frequency. " "LM: Max likelihood of options." + (
            ""
            if self.comparison_name is None
            else f" NLG: Max similarity ({self.comparison_name})."
        )

    @property
    def higher_is_better(self) -> bool:
        return True

    # TODO: Calculate proportion normality uncertainty

    def compute_single(self, example: ExecutionExample) -> float:
        if example.index_label is None:
            logger.error("Example with ID %s had no index label.", example.id_)
            raise ValueError("ExecutionExample had missing required fields.")
        if example.options_model_likelihoods is not None:
            scores = example.options_model_likelihoods
        elif (
            example.generated_text is not None
            and self.comparison_function is not None
            and example.options is not None
        ):
            scores = [
                self.comparison_function(example.generated_text, option)
                for option in example.options
            ]
        else:
            logger.error(
                "Example with ID %s had neither likelihood "
                "nor generated text for accuracy calculation.",
                example.id_,
            )
            raise ValueError("ExecutionExample had missing required fields.")
        model_prediction = scores.index(max(scores))
        return float(model_prediction == example.index_label)


class TextSimilarityMetric(Metric):
    def __init__(self, comparison_name: str, comparison_function: TextCompareFun):
        self.comparison_name = comparison_name
        self.comparison_function = comparison_function

    @property
    def name(self) -> str:
        return f"Similarity ({self.comparison_name})"

    @property
    def description(self) -> str:
        return (
            f"Comparing similarity of prediction and reference texts using {self.comparison_name}."
        )

    @property
    def higher_is_better(self) -> bool:
        return True

    def compute_single(self, example: ExecutionExample) -> float:
        if example.target_answer is None or example.generated_text is None:
            logger.error("Example with ID %s lacked target answer or generated text.", example.id_)
            raise ValueError("ExecutionExample had missing required fields.")
        return self.comparison_function(example.target_answer, example.generated_text)


def get_compatible_metrics(eval_cfg: DictConfig, scenario_cfg: OutDictType) -> Sequence[Metric]:
    task = get_task_runner(OmegaConf.create(scenario_cfg))
    compatible: list[Metric] = []
    if isinstance(task, MultichoiceRunner):
        compare_name = eval_cfg.compare_for_accuracy
        compare_fun = get_compare_fun(compare_name) if compare_name is not None else None
        # TODO: Also look to model config and make this govern whether to do accuracy with
        # likelihood or with similarity
        compatible.append(Accuracy(compare_name, compare_fun))

        # TODO: Also allow similarity between correct option and generated
    if isinstance(task, AnswerSimilarityRunner):
        compatible.extend(
            TextSimilarityMetric(name, function) for name, function in COMPARE_FUNCTIONS.items()
        )
    return compatible
