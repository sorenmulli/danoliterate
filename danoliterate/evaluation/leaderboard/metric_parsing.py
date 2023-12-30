from collections import defaultdict
from typing import Optional

from danoliterate.evaluation.analysis.dimensions import Dimension
from danoliterate.evaluation.analysis.meta_scorings import META_SCORERS
from danoliterate.evaluation.results import MetricResult, Scores, Scoring


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
def extract_metrics(scores: Scores, chosen_dimension: Optional[Dimension], chosen_type="standard"):
    scores.scorings = [
        scoring
        for scoring in scores.scorings
        if scoring.execution_metadata.scenario_cfg.get("type", "standard") == chosen_type
    ]

    out: defaultdict[str, dict[str, list[MetricResult]]] = defaultdict(dict)
    for meta_scorer in META_SCORERS:
        for scenario_name, model_name, dimension, metric_results in meta_scorer.meta_score(scores):
            if dimension == chosen_dimension:
                out[scenario_name][model_name] = metric_results  # type: ignore
    for scoring in scores.scorings:
        scenario_name: str = scoring.execution_metadata.scenario_cfg["name"]  # type: ignore
        model_name: str = scoring.execution_metadata.model_cfg["name"]  # type: ignore
        # TODO: Change to saving the metric key and use the registry for this instead of hard code
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
