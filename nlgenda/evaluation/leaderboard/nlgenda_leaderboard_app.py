"""
Should be run as streamlit application
"""
import logging
from collections import defaultdict

import hydra
import streamlit as st
from omegaconf import DictConfig

from nlgenda.evaluation.artifact_integration import get_scores_wandb
from nlgenda.evaluation.leaderboard.table import build_leaderboard_table
from nlgenda.evaluation.results import MetricResult, Scores
from nlgenda.infrastructure.constants import CONFIG_DIR

logger = logging.getLogger(__name__)


# TODO: Clean up this code
def extract_metrics(scores: Scores):
    out: defaultdict[str, dict[str, list[MetricResult]]] = defaultdict(dict)
    for scoring in scores.scorings:
        out[scoring.execution_metadata.scenario_cfg["name"]][  # type: ignore
            scoring.execution_metadata.model_cfg["name"]  # type: ignore
        ] = scoring.metric_results
    return out


def default_choices(metric_structure):
    out = defaultdict(dict)
    for scenario, models in metric_structure.items():
        for model, metrics in models.items():
            out[scenario][model] = metrics[0]
    return out


# TODO: Move to config dir more elegantly
@hydra.main(config_path=f"../../{CONFIG_DIR}", config_name="master", version_base=None)
def setup_app(cfg: DictConfig):
    st.title("NLGenDa Leaderboard")

    logger.info("Fetching scores ...")
    # TODO: Minimize loading times
    scores = get_scores_wandb(cfg.wandb.project, cfg.wandb.entity)
    metric_structure = extract_metrics(scores)
    default_metrics = default_choices(metric_structure)
    chosen_metrics = default_metrics

    with st.sidebar:
        with st.form(key="metrics_form"):
            for scenario, models in metric_structure.items():
                st.subheader(f"Metrics for {scenario}")
                for model, metrics in models.items():
                    options = [metric.short_name for metric in metrics]
                    selected_metric = st.selectbox(
                        f"{model} metric",
                        options,
                        index=options.index(default_metrics[scenario][model].short_name),
                        key=f"{scenario}-{model}",
                    )
                    chosen_metrics[scenario][model] = next(
                        (metric for metric in metrics if metric.short_name == selected_metric), None
                    )
                    st.caption(
                        f"Now showing: {chosen_metrics[scenario][model].short_name}.",
                        help=chosen_metrics[scenario][model].description or "",
                    )
            st.form_submit_button(label="Submit")

    logger.info("Building leaderboard table ...")
    table = build_leaderboard_table(scores, chosen_metrics)

    st.dataframe(table, use_container_width=True)

    logger.info("App built!")


if __name__ == "__main__":
    # pylint: disable=no-value-for-parameter
    setup_app()
