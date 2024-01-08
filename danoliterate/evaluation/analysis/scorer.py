from typing import Sequence

from omegaconf import DictConfig
from tqdm import tqdm

from danoliterate.evaluation.analysis.metrics import Metric
from danoliterate.evaluation.artifact_integration import (
    get_results_wandb,
    get_scores_wandb,
    send_scores_wandb,
    setup_short_run,
)
from danoliterate.evaluation.registries.metrics import get_compatible_metrics
from danoliterate.evaluation.results import ExecutionResult, Scores, Scoring
from danoliterate.infrastructure.logging import format_config, logger
from danoliterate.infrastructure.timing import from_timestamp


class Scorer:
    def __init__(self, cfg: DictConfig):
        self.wandb_cfg = cfg.wandb
        self.eval_cfg = cfg.evaluation

        self.result = Scores.from_config(cfg)

        self.wandb = setup_short_run(self.result.name, "score", self.wandb_cfg)
        self.previous_result = (
            None
            if cfg.evaluation.rescore
            else get_scores_wandb(cfg.wandb.project, cfg.wandb.entity)
        )
        self.combinations_to_skip: dict[str, Scoring] = {}

    def run(self):
        logger.info("Fetching executed results ...")
        results = get_results_wandb(
            self.wandb_cfg.project,
            self.wandb_cfg.entity,
            cache_file=self.wandb_cfg.artifact_cache,
            cache_update=self.wandb_cfg.cache_update,
        )
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

        if self.previous_result is not None and not self.eval_cfg.rescore:
            self.combinations_to_skip = {
                self._get_scoring_comparison_key(old_scoring, []): old_scoring
                for old_scoring in self.previous_result.scorings
            }
        logger.info("Acquired %i execution results. Scoring ...", len(results))
        for result in tqdm(results):
            if _should_skip(result):
                continue
            self.result.scorings.append(self.score_result(result))

    def score_result(self, result: ExecutionResult) -> Scoring:
        scoring = Scoring.from_execution_metadata(result.metadata)
        metrics = get_compatible_metrics(
            result.metadata.scenario_cfg["task"]["type"],  # type: ignore
            result.metadata.model_cfg["inference"]["type"],  # type: ignore
            result.metadata.scenario_cfg,
        )
        if (
            self.combinations_to_skip
            and (
                old_scoring := self.combinations_to_skip.get(
                    self._get_scoring_comparison_key(scoring, metrics)
                )
            )
            is not None
        ):
            logger.info(
                "Skipping scoring of %s as a scoring from %s "
                "already covered same execution and had same metrics. "
                "set `evaluation.rescore` to disable this behaviour",
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

    def _get_scoring_comparison_key(self, scoring: Scoring, metrics: Sequence[Metric]):
        metric_names = (
            [metric.name for metric in metrics]
            if metrics
            else [metric_res.short_name for metric_res in scoring.metric_results]
        )
        return scoring.execution_metadata.id_ + "-" + "-".join(sorted(metric_names))


def _should_skip(result: ExecutionResult):
    # TODO: Remove these partial results
    if (mname := result.metadata.model_cfg["name"]) in {
        "mGPT 13B",
        "Hestenettet LM",
        "SOLAR 10.7B",
        "OpenAI Davinci 003",
    }:
        logger.warning("Manually skipped scoring of result for model %s", mname)
        return True
    return False


def score(cfg: DictConfig):
    logger.debug("Running scoring with arguments: %s", format_config(cfg))

    scorer = Scorer(cfg)
    scorer.run()
    scorer.save_scores()
