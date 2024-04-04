from omegaconf import DictConfig

from danoliterate.evaluation.results import Scores, Scoring
from danoliterate.infrastructure.logging import format_config, logger


def report_for_one_scoring(scoring: Scoring):
    formatted_res = ",\t".join(
        f"{metric.short_name}: {metric.aggregate:.4f}" for metric in scoring.metric_results
    )
    logger.info(
        """
===============================================================
(Run %s) Scenario=%s, Model=%s
%s
""",
        scoring.timestamp,
        scoring.execution_metadata.scenario_cfg["name"],
        scoring.execution_metadata.model_cfg["name"],
        formatted_res,
    )


def report_scores(cfg: DictConfig):
    logger.debug("Running scorer report with arguments: %s", format_config(cfg))
    logger.info("Loading scores from %s", cfg.evaluation.local_results)
    if (scores := Scores.from_local_result_db(cfg)) is None:
        raise FileNotFoundError("Directory had no scores JSON")
    logger.info("Found %i scores, logging them below.", len(scores.scorings))
    for scoring in sorted(scores.scorings, key=lambda scoring: scoring.id_):
        report_for_one_scoring(scoring)
