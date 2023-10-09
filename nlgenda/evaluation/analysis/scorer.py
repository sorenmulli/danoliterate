import logging

from omegaconf import DictConfig
from tqdm import tqdm

from nlgenda.evaluation.analysis.metrics import get_compatible_metrics
from nlgenda.evaluation.artifact_integration import (
    get_results_wandb,
    get_scores_wandb,
    send_scores_wandb,
    setup_short_run,
)
from nlgenda.evaluation.results import ExecutionResult, Scores, Scoring
from nlgenda.infrastructure.logging import format_config
from nlgenda.infrastructure.timing import from_timestamp

logger = logging.getLogger(__name__)


class Scorer:
    def __init__(self, cfg: DictConfig):
        self.wandb_cfg = cfg.wandb
        self.eval_cfg = cfg.evaluation

        # TODO: Make it possible to load from existing result
        self.result = Scores.from_config(cfg)

        self.wandb = setup_short_run(self.result.name, "score", cfg.wandb)
        self.previous_result = (
            None
            if cfg.evaluation.rescore
            else get_scores_wandb(cfg.wandb.project, cfg.wandb.entity)
        )

    def run(self):
        logger.info("Fetching executed results ...")
        results = get_results_wandb(self.wandb_cfg.project, self.wandb_cfg.entity)
        if (min_time := self.eval_cfg.do_not_score_before) is not None:
            len_before = len(results)
            results = [
                result
                for result in results
                if from_timestamp(result.metadata.timestamp) >= from_timestamp(min_time)
            ]
            if (skipped := len_before - len(results)) > 0:
                logger.info("Skipped %i examples that were before %s.", skipped, min_time)
        if not results:
            logger.info("No results acquired; nothing to do. Exiting ...")
            return

        logger.info("Acquired %i execution results. Scoring ...", len(results))
        for result in tqdm(results):
            self.result.scorings.append(self.score_result(result))

    def score_result(self, result: ExecutionResult) -> Scoring:
        scoring = Scoring.from_execution_metadata(result.metadata)
        metrics = get_compatible_metrics(result.metadata.scenario_cfg, result.metadata.model_cfg)
        if self.previous_result is not None:
            for old_scoring in self.previous_result.scorings:
                if old_scoring.execution_metadata == scoring.execution_metadata and {
                    metric.short_name for metric in old_scoring.metric_results
                } == {metric.name for metric in metrics}:
                    logger.info(
                        "Skipping scoring of %s as a scoring from %s "
                        "already covered same execution and had same metrics. "
                        "set `evaluation.rerun` to disable this behaviour",
                        result.name,
                        old_scoring.timestamp,
                    )
                    return old_scoring

        logger.info(
            "Scoring result %s on metrics %s.",
            result.name,
            ", ".join(metric.name for metric in metrics),
        )
        for metric in metrics:
            scoring.metric_results.append(metric(result.examples))
        return scoring

    def save_scores(self):
        if self.wandb is not None:
            send_scores_wandb(self.result, self.wandb)
            logger.info("Sucessfully sent scores to W&B.")

        out = self.result.save_locally()
        logger.info("Scores were saved locally to %s.", out)


def score(cfg: DictConfig):
    logger.debug("Running scoring with arguments: %s", format_config(cfg))

    scorer = Scorer(cfg)
    scorer.run()
    scorer.save_scores()
