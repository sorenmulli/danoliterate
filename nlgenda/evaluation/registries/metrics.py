from nlgenda.evaluation.analysis.metrics import (
    AverageTextSimilarity,
    GptNerParsingF1,
    MaxLikelihoodAccuracy,
    MaxLikelihoodF1,
    MaxSimilarityAccuracy,
    MaxSimilarityF1,
    MaxTextSimilarity,
    Metric,
    MinTextSimilarity,
    OddOneOutAccuracy,
    TextSimilarityMetric,
)
from nlgenda.evaluation.registries.registration import register_metric
from nlgenda.evaluation.serialization import OutDictType
from nlgenda.modeling.text_comparison import COMPARERS


@register_metric("max-likelihood-accuracy")
def get_max_likelihood_accuracy(_: OutDictType) -> Metric:
    return MaxLikelihoodAccuracy()


@register_metric("max-likelihood-f1")
def get_max_likelihood_f1(_: OutDictType) -> Metric:
    return MaxLikelihoodF1()


@register_metric("gpt-ner")
def get_gpt_ner(scenario_cfg: OutDictType) -> Metric:
    return GptNerParsingF1(scenario_cfg["path"], scenario_cfg["dataset_split"])  # type: ignore


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
    for comparer_key, comparer in COMPARERS.items():
        register_metric(f"{metric_name}-{comparer_key}")(
            lambda _, mn=metric_name, ck=comparer.name: comparison_metric_factory(mn, ck)
        )
