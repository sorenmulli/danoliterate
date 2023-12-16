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
from danoliterate.evaluation.registries.registration import register_metric
from danoliterate.evaluation.serialization import OutDictType
from danoliterate.modeling.text_comparison import COMPARERS


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


for metric_name, metric_class in METRICS_WITH_COMPARISONS.items():
    for comparer_name, comparer in COMPARERS.items():
        register_metric(f"{metric_name}-{comparer.key}")(
            # pylint: disable=line-too-long
            lambda _, mn=metric_name, ck=comparer_name: comparison_metric_factory(mn, ck)  # type: ignore
        )


@register_metric("likelihood-brier", dimension=Dimension.CALIBRATION)
def get_likelihood_brier(_: OutDictType) -> Metric:
    return LikelihoodBrier()


@register_metric("likelihood-ece", dimension=Dimension.CALIBRATION)
def get_likelihood_ece(_: OutDictType) -> Metric:
    return LikelihoodExpectedCalibrationError()
