from collections import defaultdict
from copy import deepcopy
from typing import Optional

from danoliterate.evaluation.analysis.dimensions import Dimension
from danoliterate.evaluation.analysis.meta_scorings import META_SCORERS
from danoliterate.evaluation.execution.eval_types import DEFAULT_EVALUATION_TYPE
from danoliterate.evaluation.results import MetricResult, Scores, Scoring

SPECIAL_TO_SHOW: dict[str, tuple[str, ...]] = {
    "standard": (),
    "free-generation": ("DaNE", "Nordjylland News", "#twitterhjerne", "Angry Tweets"),
}


def get_relevant_metrics(
    scoring: Scoring, chosen_dimension: Optional[Dimension]
) -> list[MetricResult]:
    out = []
    for metric in scoring.metric_results:
        if "Offensive" in metric.description:
            if chosen_dimension == Dimension.TOXICITY:
                out.append(metric)
        elif "ECE" in metric.short_name or "Brier" in metric.short_name:
            if chosen_dimension == Dimension.CALIBRATION:
                out.append(metric)
        elif chosen_dimension == Dimension.CAPABILITY:
            out.append(metric)
    return out


# TODO: Clean up this code
def extract_metrics(
    scores: Scores,
    chosen_dimension: Optional[Dimension],
    chosen_type=DEFAULT_EVALUATION_TYPE,
    chosen_models: Optional[list[str]] = None,
):
    scores = deepcopy(scores)
    scores.scorings = [
        scoring
        for scoring in scores.scorings
        if (chosen_models is None or scoring.execution_metadata.model_cfg["name"] in chosen_models)
        and scoring.execution_metadata.scenario_cfg.get("type", DEFAULT_EVALUATION_TYPE)
        == chosen_type
        or scoring.execution_metadata.scenario_cfg["name"] in SPECIAL_TO_SHOW.get(chosen_type, [])
    ]

    out: defaultdict[str, dict[str, list[MetricResult]]] = defaultdict(dict)
    for meta_scorer in META_SCORERS:
        for scenario_name, model_name, dimension, metric_results in meta_scorer.meta_score(scores):
            if dimension == chosen_dimension:
                out[scenario_name][model_name] = metric_results  # type: ignore
    for scoring in scores.scorings:
        scenario_name: str = scoring.execution_metadata.scenario_cfg["name"]  # type: ignore
        model_name: str = scoring.execution_metadata.model_cfg["name"]  # type: ignore
        relevant_metric_results = get_relevant_metrics(scoring, chosen_dimension)
        if relevant_metric_results:
            if out[scenario_name].get(model_name) is None:  # type: ignore
                out[scenario_name][model_name] = relevant_metric_results  # type: ignore
            else:
                # TODO: Warn if metric already exists
                out[scenario_name][model_name].extend(relevant_metric_results)  # type: ignore
    return out


def default_choices(metric_structure):
    out = defaultdict(dict)
    for scenario, models in metric_structure.items():
        for model, metrics in models.items():
            out[scenario][model] = metrics[0]
    return out
