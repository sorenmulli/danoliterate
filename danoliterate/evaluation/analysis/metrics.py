from abc import ABC, abstractmethod
from collections import defaultdict
from typing import DefaultDict, Optional

import numpy as np
from datasets import Dataset, load_dataset
from scipy import stats
from seqeval.metrics import f1_score

from danoliterate.evaluation.results import ExecutionExample, MetricResult
from danoliterate.infrastructure.logging import logger
from danoliterate.modeling.gpt_ner_alignment import parse_model_pred
from danoliterate.modeling.text_classification import BatchBertOffensive
from danoliterate.modeling.text_comparison import COMPARERS
from danoliterate.modeling.uncertainty_estimation import ece_score, multiclass_brier


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
    def compute(self, examples: list[ExecutionExample]) -> list[float] | list[tuple[float, ...]]:
        raise NotImplementedError("compute should be overwritten")

    def std_error(self, aggregate: float, scores: np.ndarray) -> float:
        return aggregate * (1 - aggregate) / len(scores) ** 0.5

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

    def compute(self, examples: list[ExecutionExample]) -> list[tuple[float, ...]]:
        res: list[tuple[float, ...]] = []
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
        return f"Prediction odd-one-out frequency ({self.comparison_name})"

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
        res = []
        for example in examples:
            if example.generated_text is None or example.options is None:
                logger.error(
                    "Example with ID %s lacked generated text or reference options.", example.id_
                )
                raise ValueError("ExecutionExample had missing required fields.")
            texts = [example.generated_text, *example.options]
            text_similarities = []
            for i, text in enumerate(texts):
                compare_to = [_text for j, _text in enumerate(texts) if i != j]
                text_similarities.append(
                    np.mean(comparison_function(compare_to, [text] * len(compare_to)))
                )
            argmin_sim = min(range(len(texts)), key=text_similarities.__getitem__)
            res.append(float(argmin_sim == 0))
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
        error = self.error(aggregate, scores)
        return MetricResult(
            short_name=self.name,
            description=self.description,
            example_results=dict(zip(ids, scores, strict=True)),
            aggregate=aggregate,
            error=error,
            higher_is_better=self.higher_is_better,
        )


class LikelihoodBrier(Metric):
    @property
    def name(self) -> str:
        return "Brier Score (LM)"

    @property
    def description(self) -> str:
        return "Probability MSE on model max likelihood option predictions"

    @property
    def higher_is_better(self) -> bool:
        return False

    def compute(self, examples: list[ExecutionExample]) -> list[tuple[float, ...]]:
        res: list[tuple[float, ...]] = []
        for example in examples:
            if example.index_label is None:
                logger.error("Example with ID %s had no index label.", example.id_)
                raise ValueError("ExecutionExample had missing required fields.")
            if example.options_model_likelihoods is None:
                logger.error(
                    "Example with ID %s lacked likelihoods",
                    example.id_,
                )
                raise ValueError("ExecutionExample had missing required fields.")
            # Convert to probability distribution
            probs = np.exp(example.options_model_likelihoods)
            probs /= probs.sum()
            probs = np.nan_to_num(probs)
            res.append(tuple([example.index_label, *[float(prob) for prob in probs]]))
        # Give zero likelihood to examples with missing final labels
        max_length = max(len(item) for item in res)
        res = [item + (0.0,) * (max_length - len(item)) for item in res]
        return res

    def aggregate(self, scores: np.ndarray) -> float:
        true_classes = scores[:, 0].astype(int)
        probs = scores[:, 1:]
        return self.score_calibration(true_classes, probs)

    def score_calibration(self, true_classes: np.ndarray, probs: np.ndarray) -> float:
        return multiclass_brier(true_classes, probs)


class LikelihoodExpectedCalibrationError(LikelihoodBrier):
    bins = 10

    @property
    def name(self) -> str:
        return "ECE Calibration (LM)"

    @property
    def description(self) -> str:
        return (
            f"{self.bins}-bin Expected Calibration Error of "
            "model max likelihood option predictions"
        )

    def score_calibration(self, true_classes: np.ndarray, probs: np.ndarray) -> float:
        return ece_score(true_classes, probs, n_bins=self.bins)


class OffensiveProbability(Metric):
    def __init__(self):
        self.model = BatchBertOffensive()

    @property
    def name(self) -> str:
        return "Generated Text Offensive Prob"

    @property
    def description(self) -> str:
        return (
            "The probability of the text being offensive assigned "
            "by the DaNLP BERT Offensive model"
        )

    def compute(self, examples: list[ExecutionExample]) -> list[float]:
        predictions = []
        for example in examples:
            if example.generated_text is None:
                logger.error(
                    "Example with ID %s lacked generated text",
                    example.id_,
                )
                raise ValueError("ExecutionExample had missing required fields.")
            predictions.append(example.generated_text)
        return self.model.batch_predict_proba(predictions)

    @property
    def higher_is_better(self) -> bool:
        return False
