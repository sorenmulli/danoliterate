import logging
from abc import ABC, abstractmethod
from collections.abc import Sequence

import numpy as np
from scipy import stats

from nlgenda.evaluation.results import ExecutionExample, MetricResult
from nlgenda.evaluation.serialization import OutDictType
from nlgenda.modeling.text_comparison import COMPARERS, Comparer

logger = logging.getLogger(__name__)


class Metric(ABC):
    confidence_level = 0.95
    is_fully_initialized = False

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
    def compute(self, examples: list[ExecutionExample]) -> list[float] | list[tuple[float, float]]:
        raise NotImplementedError("compute should be overwritten")

    def std_error(self, _: float, scores: np.ndarray) -> float:
        return float(np.std(scores, ddof=1) / np.sqrt(len(scores)))

    def error(self, aggregate: float, scores: np.ndarray) -> float:
        # Two sided Students t test with N - 1 degrees of freedom
        t_value = float(stats.t.ppf((1 + self.confidence_level) / 2, len(scores) - 1))
        return self.std_error(aggregate, scores) * t_value

    def aggregate(self, scores: np.ndarray) -> float:
        return float(scores.mean())

    def __call__(self, examples: list[ExecutionExample]) -> MetricResult:
        scores = self.compute(examples)
        score_arr = np.array(scores)
        aggregate = self.aggregate(score_arr)
        error = self.error(aggregate, score_arr)
        return MetricResult(
            short_name=self.name,
            description=self.description,
            # TODO: Fix mypy type confusion here
            example_results={
                example.id_: score for example, score in zip(examples, scores)  # type: ignore
            },
            aggregate=aggregate,
            error=error,
            higher_is_better=self.higher_is_better,
        )


class BaseAccuracy(Metric, ABC):
    @abstractmethod
    def get_model_predictions(self, examples: list[ExecutionExample]) -> list[int]:
        ...

    def compute(self, examples: list[ExecutionExample]) -> list[tuple[float, float]]:
        res: list[tuple[float, float]] = []
        model_predictions = self.get_model_predictions(examples)
        for example, pred in zip(examples, model_predictions):
            if example.index_label is None:
                logger.error("Example with ID %s had no index label.", example.id_)
                raise ValueError("ExecutionExample had missing required fields.")
            res.append((example.index_label, pred))
        return res

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

    def get_model_predictions(self, examples: list[ExecutionExample]) -> list[int]:
        out = []
        for example in examples:
            if example.options_model_likelihoods is None:
                logger.error(
                    "Example with ID %s lacked likelihoods",
                    example.id_,
                )
                raise ValueError("ExecutionExample had missing required fields.")
            scores = example.options_model_likelihoods
            out.append(scores.index(max(scores)))
        return out


class MaxSimilarityAccuracy(BaseAccuracy):
    def __init__(
        self,
        comparison_name: str,
    ):
        self.comparison_name = comparison_name

    @property
    def name(self) -> str:
        return f"Accuracy (NLG {self.comparison_name})"

    @property
    def description(self) -> str:
        return f"Frequency of true predictions (max {self.comparison_name})"

    def get_model_predictions(self, examples: list[ExecutionExample]) -> list[int]:
        comparison_function = COMPARERS[self.comparison_name]()
        targets = []
        predictions = []
        for example in examples:
            if example.generated_text is None or example.options is None:
                logger.error(
                    "Example with ID %s lacked text fields for similarity",
                    example.id_,
                )
                raise ValueError("ExecutionExample had missing required fields.")
            for option in example.options:
                targets.append(option)
                predictions.append(example.generated_text)
        scores = comparison_function(targets, predictions)
        max_indeces = []
        i = 0
        for example in examples:
            option_scores = []
            for _ in example.options:  # type: ignore
                option_scores.append(scores[i])
                i += 1
            max_indeces.append(option_scores.index(max(option_scores)))
        return max_indeces


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
    def __init__(self, comparison_name: str):
        self.comparison_name = comparison_name

    @property
    def name(self) -> str:
        return f"Similarity ({self.comparison_name})"

    @property
    def description(self) -> str:
        return (
            f"Comparing similarity of prediction and reference texts using {self.comparison_name}."
        )

    def compute(self, examples: list[ExecutionExample]) -> list[float]:
        comparison_function = COMPARERS[self.comparison_name]()
        targets: list[str] = []
        predictions: list[str] = []
        for example in examples:
            if example.generated_text is not None:
                if example.target_answer is not None:
                    targets.append(example.target_answer)
                    predictions.append(example.generated_text)
                    continue
                if example.options is not None and example.index_label is not None:
                    targets.append(example.options[example.index_label])
                    predictions.append(example.generated_text)
                    continue
            logger.error(
                "Example with ID %s lacked target answer, generated text or options.", example.id_
            )
            raise ValueError("ExecutionExample had missing required fields.")
        return comparison_function(targets, predictions)

    def std_error(self, aggregate: float, scores: np.ndarray) -> float:
        return aggregate * (1 - aggregate) / len(scores) ** 0.5


def get_compatible_metrics(scenario_cfg: OutDictType, model_cfg: OutDictType) -> Sequence[Metric]:
    task_type_str = scenario_cfg["task"]["type"]  # type: ignore
    compatible: list[Metric] = []

    # TODO: Remove backwards compatibility keys
    if task_type_str in {
        "default-mc",
        "default-mc-letter-options",
        "hyggeswag",
        "citizenship-test",
    }:
        # Save some time skipping likelihood metrics for text generators (would just give none)
        if model_cfg["inference"]["type"] != "openai-api":  # type: ignore
            compatible.extend([MaxLikelihoodAccuracy(), MaxLikelihoodF1()])
        for name in COMPARERS:
            compatible.extend(
                [
                    MaxSimilarityAccuracy(name),
                    MaxSimilarityF1(name),
                    TextSimilarityMetric(name),
                ]
            )
    if task_type_str in {"default-answer-similarity", "prompt-similarity"}:
        compatible.extend(TextSimilarityMetric(name) for name in COMPARERS)
    return compatible
