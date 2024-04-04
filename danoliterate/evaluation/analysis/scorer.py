from pathlib import Path
from typing import Sequence

from omegaconf import DictConfig
from tqdm import tqdm

from danoliterate.evaluation.analysis.meta_scorings import META_SCORERS
from danoliterate.evaluation.analysis.metrics import Metric
from danoliterate.evaluation.registries.metrics import get_compatible_metrics
from danoliterate.evaluation.results import ExecutionResult, Scores, Scoring
from danoliterate.infrastructure.constants import EXECUTION_RESULT_NAME
from danoliterate.infrastructure.logging import format_config, logger, maybe_setup_wandb_logging_run
from danoliterate.infrastructure.timing import from_timestamp


class Scorer:
    def __init__(self, cfg: DictConfig):
        self.wandb_cfg = cfg.wandb
        self.eval_cfg = cfg.evaluation

        self.result = Scores.from_config(cfg)

        maybe_setup_wandb_logging_run(self.result.name, "score", self.wandb_cfg)
        self.previous_result = None if cfg.evaluation.rescore else Scores.from_local_result_db(cfg)
        self.combinations_to_skip: dict[str, Scoring] = {}

    def run(self):
        logger.info("Fetching executed results locally from %s ...", self.eval_cfg.local_results)
        results = [
            ExecutionResult.from_path(path)
            for path in Path(self.eval_cfg.local_results).glob(f"{EXECUTION_RESULT_NAME}*.json")
        ]
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
            self.result.scorings.append(self.score_result(result))
        logger.info("Running meta scores")
        meta_scorings = []
        for meta_scorer in tqdm(META_SCORERS):
            for metadata, metric_results in meta_scorer.meta_score(self.result):
                meta_scoring = Scoring.from_execution_metadata(metadata)
                meta_scoring.metric_results = metric_results
                meta_scoring.is_meta = True
                meta_scorings.append(meta_scoring)
        self.result.scorings.extend(meta_scorings)

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
        out = self.result.save_locally()
        logger.info("Scores were saved locally to %s.", out)

    def _get_scoring_comparison_key(self, scoring: Scoring, metrics: Sequence[Metric]):
        metric_names = (
            [metric.name for metric in metrics]
            if metrics
            else [metric_res.short_name for metric_res in scoring.metric_results]
        )
        return scoring.execution_metadata.id_ + "-" + "-".join(sorted(metric_names))


def score(cfg: DictConfig):
    logger.debug("Running scoring with arguments: %s", format_config(cfg))

    scorer = Scorer(cfg)
    scorer.run()
    scorer.save_scores()
