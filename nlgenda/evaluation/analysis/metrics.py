import logging
from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import Sequence
from typing import DefaultDict, Optional

import numpy as np
from datasets import Dataset, load_dataset
from scipy import stats
from seqeval.metrics import f1_score

from nlgenda.evaluation.results import ExecutionExample, MetricResult
from nlgenda.evaluation.serialization import OutDictType
from nlgenda.modeling.gpt_ner_alignment import parse_model_pred
from nlgenda.modeling.text_comparison import COMPARERS

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


class MaxTextSimilarity(TextSimilarityMetric):
    @property
    def name(self) -> str:
        return f"Max. similarity to references ({self.comparison_name})"

    @property
    def description(self) -> str:
        return (
            f"Maximum similarity of prediction over reference texts using {self.comparison_name}."
        )

    @staticmethod
    def norm_option_similarities(similarities: list[float]) -> float:
        return max(similarities)

    def compute(self, examples: list[ExecutionExample]) -> list[float]:
        comparison_function = COMPARERS[self.comparison_name]()
        targets: list[str] = []
        predictions: list[str] = []
        for example in examples:
            if example.generated_text is not None:
                if example.options is not None:
                    targets.extend(example.options)
                    predictions.extend(example.generated_text for _ in example.options)
                    continue
            logger.error(
                "Example with ID %s lacked generated text or reference options.", example.id_
            )
            raise ValueError("ExecutionExample had missing required fields.")
        res = []
        similarities = comparison_function(targets, predictions)
        for example in examples:
            option_similarities = [similarities.pop(0) for _ in example.options]  # type: ignore
            res.append(self.norm_option_similarities(option_similarities))
        return res


class MinTextSimilarity(MaxTextSimilarity):
    @property
    def name(self) -> str:
        return f"Min. similarity to references ({self.comparison_name})"

    @property
    def description(self) -> str:
        return (
            f"Minimum similarity of prediction over reference texts using {self.comparison_name}."
        )

    @staticmethod
    def norm_option_similarities(similarities: list[float]) -> float:
        return min(similarities)


class AverageTextSimilarity(MaxTextSimilarity):
    @property
    def name(self) -> str:
        return f"Avg. similarity to references ({self.comparison_name})"

    @property
    def description(self) -> str:
        return f"Average similarity of prediction of reference texts using {self.comparison_name}."

    @staticmethod
    def norm_option_similarities(similarities: list[float]) -> float:
        return sum(similarities) / len(similarities)


class OddOneOutAccuracy(TextSimilarityMetric):
    @property
    def name(self) -> str:
        return f"Frequency of odd-one-out ({self.comparison_name})"

    @property
    def description(self) -> str:
        return (
            "Accuracy of identifying generated text "
            f"as odd-one-out using max total {self.comparison_name}."
        )

    @property
    def higher_is_better(self) -> bool:
        return False

    def compute(self, examples: list[ExecutionExample]) -> list[float]:
        comparison_function = COMPARERS[self.comparison_name]()
        targets: list[str] = []
        predictions: list[str] = []
        for example in examples:
            if example.generated_text is not None:
                if example.options is not None:
                    for option in example.options:
                        targets.append(option)
                        # Always have the similarity as the first
                        predictions.append(example.generated_text)
                        for option_ in example.options:
                            targets.append(option)
                            predictions.append(option_)
                    continue
            logger.error(
                "Example with ID %s lacked generated text or reference options.", example.id_
            )
            raise ValueError("ExecutionExample had missing required fields.")
        res = []
        similarities = comparison_function(targets, predictions)
        for example in examples:
            generated_total_dist = 0.0
            reference_total_dists = [0.0 for _ in example.options]  # type: ignore
            for option in example.options:  # type: ignore
                generated_total_dist -= similarities.pop(0)
                for i in range(len(example.options)):  # type: ignore
                    reference_total_dists[i] -= similarities.pop(0)
            # Generated is odd one out if it has highest total dist of all option total dists
            res.append(
                float(all(generated_total_dist > ref_dist for ref_dist in reference_total_dists))
            )
        return res


class GptNerParsingF1(Metric):
    def __init__(self, dataset_path: str, dataset_split: str):
        self.dataset: Dataset = load_dataset(
            dataset_path,
            split=dataset_split,
        )

    @property
    def name(self) -> str:
        return "NER F1"

    @property
    def description(self) -> str:
        return "GPT-NER parsed micro class avg. F1 score"

    def extract_ner(self, examples: list[ExecutionExample]):
        labels = []
        preds = []
        idx_to_class_preds: DefaultDict[int, dict[str, ExecutionExample]] = defaultdict(dict)
        for example in examples:
            *id_, entity_class = example.id_.split("-")
            try:
                idx = int(id_[0])
            except ValueError as error:
                raise NotImplementedError(
                    "GPT-NER currently assumes that first part of ID is an index "
                ) from error
            idx_to_class_preds[idx][entity_class] = example
        for idx, class_examples in idx_to_class_preds.items():
            try:
                tokens = self.dataset[idx]["tokens"]
                labels.append(self.dataset[idx]["labels"])
            except KeyError as error:
                raise NotImplementedError("GPT-NER currently has hardcoded column names") from error
            combined_prediction: Optional[list[tuple[str, float]]] = None
            for entity_class, example in class_examples.items():
                assert example.generated_text is not None
                model_prediction = parse_model_pred(tokens, example.generated_text, entity_class)
                score = example.generated_score or 0.0
                if combined_prediction is None:
                    combined_prediction = [(pred, score) for pred in model_prediction]
                else:
                    combined_prediction = [
                        (new, score)
                        if combined == "O" or score > old_score
                        else (combined, old_score)
                        for (combined, old_score), new in zip(
                            combined_prediction, model_prediction, strict=True
                        )
                    ]
            assert combined_prediction is not None
            preds.append([entity_pred for entity_pred, _ in combined_prediction])
        return labels, preds

    def aggregate_ner(self, labels: list[list[str]], preds: list[list[str]]) -> float:
        return float(f1_score(labels, preds))

    def example_scores_ner(self, labels: list[list[str]], preds: list[list[str]]) -> list[float]:
        return [
            float(f1_score([label], [pred], zero_division=0))
            for label, pred in zip(labels, preds, strict=True)
        ]

    def std_error(self, aggregate: float, scores: np.ndarray) -> float:
        return aggregate * (1 - aggregate) / len(scores) ** 0.5

    def __call__(self, examples: list[ExecutionExample]) -> MetricResult:
        labels, preds = self.extract_ner(examples)
        aggregate = self.aggregate_ner(labels, preds)
        try:
            ids = sorted(list({example.id_.split("-")[0] for example in examples}), key=int)
        except ValueError as exception:
            raise NotImplementedError(
                "GPT-NER currently assumes that first part of ID is an index "
            ) from exception
        scores = np.array(self.example_scores_ner(labels, preds))
        error = self.std_error(aggregate, scores)
        return MetricResult(
            short_name=self.name,
            description=self.description,
            example_results=dict(zip(ids, scores, strict=True)),
            aggregate=aggregate,
            error=error,
            higher_is_better=self.higher_is_better,
        )


def get_compatible_metrics(scenario_cfg: OutDictType, model_cfg: OutDictType) -> Sequence[Metric]:
    task_type_str = scenario_cfg["task"]["type"]  # type: ignore
    scenario_name = scenario_cfg["name"]  # type: ignore
    compatible: list[Metric] = []

    if task_type_str in {
        "default-mc",
        "default-mc-letter-options",
        "default-mc-letter-context",
        "cloze-showing-options",
        "default-mc-same-options",
        # TODO: Remove backwards compatibility keys
        "hyggeswag",
        "citizenship-test",
    }:
        # Save some time skipping likelihood metrics for text generators (would just give none)
        if model_cfg["inference"]["type"] != "openai-api":  # type: ignore
            compatible.extend([MaxLikelihoodAccuracy(), MaxLikelihoodF1()])
        for name in COMPARERS:
            if (scenario_name == "Angry Tweets") != (name == "Parsing of chosen class"):
                continue
            # For sentiment classification, only output parsing makes sense
            compatible.extend(
                [
                    MaxSimilarityAccuracy(name),
                    MaxSimilarityF1(name),
                    TextSimilarityMetric(name),
                ]
            )
    elif task_type_str == "multi-answer-similarity":
        for name in COMPARERS:
            if name == "Parsing of chosen class":
                continue
            compatible.extend(
                [
                    MaxTextSimilarity(name),
                    MinTextSimilarity(name),
                    AverageTextSimilarity(name),
                    OddOneOutAccuracy(name),
                ]
            )
    elif task_type_str == "gpt-ner":
        compatible.append(
            GptNerParsingF1(scenario_cfg["path"], scenario_cfg["dataset_split"])  # type: ignore
        )
    elif task_type_str in {"default-answer-similarity", "prompt-similarity"}:
        compatible.extend(
            TextSimilarityMetric(name) for name in COMPARERS if name != "Parsing of chosen class"
        )
    return compatible
