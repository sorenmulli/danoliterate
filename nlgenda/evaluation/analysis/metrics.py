import logging
from abc import ABC, abstractmethod
from collections.abc import Sequence

import numpy as np
from omegaconf import OmegaConf
from scipy import stats

from nlgenda.evaluation.execution.model_inference import InferenceMethod
from nlgenda.evaluation.execution.task_runner import AnswerSimilarityRunner, MultichoiceRunner
from nlgenda.evaluation.registries.get import get_inference, get_task_runner
from nlgenda.evaluation.results import ExecutionExample, MetricResult
from nlgenda.evaluation.serialization import OutDictType
from nlgenda.modeling.text_comparison import COMPARERS, Comparer

logger = logging.getLogger(__name__)

# TODO: Instead of passing name everywhere, take name from comparer object


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
    def higher_is_better(self) -> bool:
        return True

    # Not an abstractmethod as you might want to not use this way to do it
    def compute_single(self, example: ExecutionExample) -> float | tuple[float, ...]:
        raise NotImplementedError("compute_single should be overwritten")

    def std_error(self, _: float, scores: np.ndarray) -> float:
        return float(np.std(scores, ddof=1) / np.sqrt(len(scores)))

    def error(self, aggregate: float, scores: np.ndarray) -> float:
        # Two sided Students t test with N - 1 degrees of freedom
        t_value = float(stats.t.ppf((1 + self.confidence_level) / 2, len(scores) - 1))
        return self.std_error(aggregate, scores) * t_value

    def aggregate(self, scores: np.ndarray) -> float:
        return float(scores.mean())

    def __call__(self, examples: list[ExecutionExample]) -> MetricResult:
        results = {example.id_: self.compute_single(example) for example in examples}
        scores = np.array(list(results.values()))
        aggregate = self.aggregate(scores)
        error = self.error(aggregate, scores)
        return MetricResult(
            short_name=self.name,
            description=self.description,
            example_results=results,
            aggregate=aggregate,
            error=error,
            higher_is_better=self.higher_is_better,
        )


class BaseAccuracy(Metric, ABC):
    @abstractmethod
    def get_model_prediction(self, example: ExecutionExample) -> int:
        ...

    def compute_single(self, example: ExecutionExample) -> tuple[float, float]:
        if example.index_label is None:
            logger.error("Example with ID %s had no index label.", example.id_)
            raise ValueError("ExecutionExample had missing required fields.")

        return example.index_label, self.get_model_prediction(example)

    # pylint: disable=arguments-renamed
    def aggregate(self, examples: np.ndarray) -> float:
        return float((examples[:, 0] == examples[:, 1]).mean())

    def std_error(self, aggregate: float, scores: np.ndarray) -> float:
        return aggregate * (1 - aggregate) / len(scores) ** 0.5


class MaxLikelihoodAccuracy(BaseAccuracy):
    @property
    def name(self) -> str:
        return "Accuracy (LM)"

    @property
    def description(self) -> str:
        return "Frequency of true predictions (max model likelihood)"

    def get_model_prediction(self, example: ExecutionExample) -> int:
        if example.options_model_likelihoods is None:
            logger.error(
                "Example with ID %s lacked likelihoods",
                example.id_,
            )
            raise ValueError("ExecutionExample had missing required fields.")
        scores = example.options_model_likelihoods
        return scores.index(max(scores))


class MaxSimilarityAccuracy(BaseAccuracy):
    def __init__(
        self,
        comparison_name: str,
        comparison_function: Comparer,
    ):
        self.comparison_name = comparison_name
        self.comparison_function = comparison_function

    @property
    def name(self) -> str:
        return f"Accuracy ({self.comparison_name})"

    @property
    def description(self) -> str:
        return f"Frequency of true predictions (max {self.comparison_name})"

    def get_model_prediction(self, example: ExecutionExample) -> int:
        if example.generated_text is None or example.options is None:
            logger.error(
                "Example with ID %s lacked text fields for similarity",
                example.id_,
            )
            raise ValueError("ExecutionExample had missing required fields.")
        scores = [
            self.comparison_function(option, example.generated_text) for option in example.options
        ]
        return scores.index(max(scores))


# pylint: disable=invalid-name
def f1_from_examples(examples: np.ndarray) -> float:
    targets = examples[:, 0]
    predictions = examples[:, 1]

    tp = np.sum((targets == 1) & (predictions == 1))
    fp = np.sum((targets == 0) & (predictions == 1))
    fn = np.sum((targets == 1) & (predictions == 0))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0

    return 2 * ((precision * recall) / (precision + recall)) if (precision + recall) > 0 else 0


class MaxLikelihoodF1(MaxLikelihoodAccuracy):
    @property
    def name(self) -> str:
        return "F1 score (LM)"

    @property
    def description(self) -> str:
        return "Harmonic mean of precision and recall (max model likelihood)"

    def aggregate(self, examples: np.ndarray) -> float:
        return f1_from_examples(examples)


class MaxSimilarityF1(MaxSimilarityAccuracy):
    @property
    def name(self) -> str:
        return f"F1 score ({self.comparison_name})"

    @property
    def description(self) -> str:
        return f"Harmonic mean of precision and recall (max {self.comparison_name})"

    def aggregate(self, examples: np.ndarray) -> float:
        return f1_from_examples(examples)


class TextSimilarityMetric(Metric):
    def __init__(self, comparison_name: str, comparison_function: Comparer):
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

    def compute_single(self, example: ExecutionExample) -> float:
        if example.generated_text is not None:
            if example.target_answer is not None:
                return self.comparison_function(example.target_answer, example.generated_text)
            if example.options is not None and example.index_label is not None:
                return self.comparison_function(
                    example.options[example.index_label], example.generated_text
                )
        logger.error(
            "Example with ID %s lacked target answer, generated text or options.", example.id_
        )
        raise ValueError("ExecutionExample had missing required fields.")

    def std_error(self, aggregate: float, scores: np.ndarray) -> float:
        return aggregate * (1 - aggregate) / len(scores) ** 0.5


def get_compatible_metrics(scenario_cfg: OutDictType, model_cfg: OutDictType) -> Sequence[Metric]:
    task = get_task_runner(OmegaConf.create(scenario_cfg))
    compatible: list[Metric] = []

    compare_functions = {name: comparer() for name, comparer in COMPARERS.items()} # type: ignore

    if isinstance(task, MultichoiceRunner):
        method_str = model_cfg["inference"].get("method")  # type: ignore
        inference_method = (
            get_inference(OmegaConf.create(model_cfg)).inference_method
            if method_str is None
            else InferenceMethod(method_str)
        )
        match inference_method:
            case InferenceMethod.LM:
                compatible.extend([MaxLikelihoodAccuracy(), MaxLikelihoodF1()])
            case InferenceMethod.NLG:
                for name, function in compare_functions.items():
                    compatible.extend(
                        [
                            MaxSimilarityAccuracy(name, function),
                            MaxSimilarityF1(name, function),
                            TextSimilarityMetric(name, function),
                            TextSimilarityMetric(name, function),
                        ]
                    )
    if isinstance(task, AnswerSimilarityRunner):
        compatible.extend(
            TextSimilarityMetric(name, function) for name, function in compare_functions.items()
        )
    return compatible
