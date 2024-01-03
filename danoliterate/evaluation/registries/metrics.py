from typing import Callable

from danoliterate.evaluation.analysis.dimensions import Dimension
from danoliterate.evaluation.analysis.metrics import (
    AverageTextSimilarity,
    GptNerParsingF1,
    LikelihoodBrier,
    LikelihoodExpectedCalibrationError,
    MaxLikelihoodAccuracy,
    MaxLikelihoodF1,
    MaxSimilarityAccuracy,
    MaxSimilarityF1,
    MaxTextSimilarity,
    Metric,
    MinTextSimilarity,
    OddOneOutAccuracy,
    OffensiveProbability,
    TextSimilarityMetric,
)
from danoliterate.evaluation.registries.inferences import (
    UnknownInference,
    inference_unsupported_metrics_registry,
)
from danoliterate.evaluation.registries.tasks import UnknownTask, task_supported_metrics_registry
from danoliterate.evaluation.serialization import OutDictType
from danoliterate.modeling.text_comparison import COMPARERS

MetricFunctionType = Callable[[OutDictType], Metric]

metric_registry: dict[str, MetricFunctionType] = {}
metric_dimension_registry: dict[str, Dimension] = {}


class UnknownMetric(KeyError):
    """A metric key was given without a registered metric"""


def register_metric(
    metric_name: str,
    dimension: Dimension = Dimension.CAPABILITY,
) -> Callable[[MetricFunctionType], MetricFunctionType]:
    def decorator(func: MetricFunctionType) -> MetricFunctionType:
        if metric_name in metric_registry:
            raise ValueError(f"Evaluation metric {metric_name} registered more than once!")
        metric_registry[metric_name] = func
        metric_dimension_registry[metric_name] = dimension
        return func

    return decorator


@register_metric("max-likelihood-accuracy")
def get_max_likelihood_accuracy(_: OutDictType) -> Metric:
    return MaxLikelihoodAccuracy()


@register_metric("max-likelihood-f1")
def get_max_likelihood_f1(_: OutDictType) -> Metric:
    return MaxLikelihoodF1()


@register_metric("gpt-ner")
def get_gpt_ner(scenario_cfg: OutDictType) -> Metric:
    return GptNerParsingF1(scenario_cfg["path"], scenario_cfg["dataset_split"])  # type: ignore


@register_metric("offensive-prob", dimension=Dimension.TOXICITY)
def get_offensive_prob(_: OutDictType) -> Metric:
    return OffensiveProbability()


METRICS_WITH_COMPARISONS = {
    "max-similarity-accuracy": MaxSimilarityAccuracy,
    "max-similarity-f1": MaxSimilarityF1,
    "text-similarity": TextSimilarityMetric,
    "max-text-similarity": MaxTextSimilarity,
    "min-text-similarity": MinTextSimilarity,
    "avg-text-similarity": AverageTextSimilarity,
    "odd-one-out-accuracy": OddOneOutAccuracy,
}


def comparison_metric_factory(metric_type: str, comparison_name: str) -> Metric:
    return METRICS_WITH_COMPARISONS[metric_type](comparison_name=comparison_name)  # type: ignore


for _metric_name, metric_class in METRICS_WITH_COMPARISONS.items():
    for comparer_name, comparer in COMPARERS.items():
        register_metric(f"{_metric_name}-{comparer.key}")(
            # pylint: disable=line-too-long
            lambda _, mn=_metric_name, ck=comparer_name: comparison_metric_factory(mn, ck)  # type: ignore
        )


@register_metric("likelihood-brier", dimension=Dimension.CALIBRATION)
def get_likelihood_brier(_: OutDictType) -> Metric:
    return LikelihoodBrier()


@register_metric("likelihood-ece", dimension=Dimension.CALIBRATION)
def get_likelihood_ece(_: OutDictType) -> Metric:
    return LikelihoodExpectedCalibrationError()


def get_compatible_metrics(task: str, inference: str, scenario_cfg: OutDictType) -> list[Metric]:
    try:
        task_supported = task_supported_metrics_registry[task]
    except KeyError as error:
        raise UnknownTask(f"No task registered with task type {task}") from error
    try:
        inference_unsupported = inference_unsupported_metrics_registry[inference]
    except KeyError as error:
        raise UnknownInference(
            f"No inference registered with model inference type {inference}"
        ) from error
    metric_keys = [metric for metric in task_supported if metric not in inference_unsupported]
    metrics = []
    for metric_key in metric_keys:
        try:
            metrics.append(metric_registry[metric_key](scenario_cfg))
        except KeyError as error:
            raise UnknownMetric(f"No metric registered with metric key {metric_key}") from error
    return metrics
