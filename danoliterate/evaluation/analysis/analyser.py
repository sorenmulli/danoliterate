import logging
from typing import Optional

import pandas as pd
from omegaconf import DictConfig

from danoliterate.evaluation.artifact_integration import get_scores_wandb
from danoliterate.infrastructure.logging import format_config

logger = logging.getLogger(__name__)


class Analyser:
    concat_key = "Concatenated"

    def __init__(self, cfg: DictConfig):
        self.wandb_cfg = cfg.wandb
        self.eval_cfg = cfg.evaluation

        self.result = get_scores_wandb(cfg.wandb.project, cfg.wandb.entity)
        self.df = self._get_df()
        self.options_dict = self._compute_options()

    def _get_df(self):
        data = []
        for scoring in self.result.scorings:
            already_metrics = set()
            for metric_result in scoring.metric_results:
                if metric_result.short_name in already_metrics:
                    continue
                already_metrics.add(metric_result.short_name)
                if "F1" in metric_result.short_name:
                    continue
                for example_id, result in metric_result.example_results.items():
                    data.append(
                        {
                            "model": scoring.execution_metadata.model_cfg["name"],
                            "scenario": scoring.execution_metadata.scenario_cfg["name"],
                            "metric": metric_result.short_name,
                            "score": result
                            if isinstance(result, float)
                            else float(result[0] == result[1]),
                            "timestamp": scoring.timestamp,
                            "example_id": example_id,
                        }
                    )
        return pd.DataFrame(data)

    def _compute_options(self):
        return {
            option: sorted(self.df[option].unique()) for option in ("model", "scenario", "metric")
        }

    def get_subset(
        self,
        metric: Optional[str] = None,
        model: Optional[str] = None,
        scenario: Optional[str] = None,
    ) -> dict[str, pd.DataFrame]:
        if sum(kwarg is None for kwarg in (model, scenario, metric)) > 1:
            raise ValueError
        conditions = []
        multi_queries = None
        for given_val, col in zip((metric, model, scenario), ("metric", "model", "scenario")):
            if given_val is None:
                multi_queries = {val: f"{col} == '{val}'" for val in self.options_dict[col]}
            elif given_val != self.concat_key:
                conditions.append(f"{col} == '{given_val}'")
        if multi_queries is None:
            return {"Chosen combination": self.df.query(" & ".join(conditions))}
        all_dfs = {
            val: self.df.query(" & ".join([*conditions, extra_condition]))
            for val, extra_condition in multi_queries.items()
        }
        return {val: df for val, df in all_dfs.items() if not df.empty}


def analyse(cfg: DictConfig):
    logger.debug("Running scoring with arguments: %s", format_config(cfg))
