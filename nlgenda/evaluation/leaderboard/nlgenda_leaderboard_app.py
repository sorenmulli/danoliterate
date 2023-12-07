"""
Should be run as streamlit application
"""
import logging
from collections import defaultdict

import hydra
import streamlit as st
from omegaconf import DictConfig

from nlgenda.evaluation.analysis.meta_scorings import META_SCORERS
from nlgenda.evaluation.artifact_integration import get_scores_wandb
from nlgenda.evaluation.leaderboard.table import build_leaderboard_table
from nlgenda.evaluation.results import MetricResult, Scores
from nlgenda.infrastructure.constants import CONFIG_DIR

logger = logging.getLogger(__name__)


def group_models_by_metrics(models):
    metrics_to_models = defaultdict(list)
    for model, metrics in models.items():
        metrics_key = tuple(metric.short_name for metric in metrics)
        metrics_to_models[metrics_key].append(model)
    return metrics_to_models


# TODO: Clean up this code
def extract_metrics(scores: Scores):
    out: defaultdict[str, dict[str, list[MetricResult]]] = defaultdict(dict)
    for meta_scorer in META_SCORERS:
        for scenario_name, model_name, metric_results in meta_scorer.meta_score(scores):
            out[scenario_name][model_name] = metric_results  # type: ignore
    for scoring in scores.scorings:
        scenario_name = scoring.execution_metadata.scenario_cfg["name"]
        model_name = scoring.execution_metadata.model_cfg["name"]
        if out[scenario_name].get(model_name) is None:  # type: ignore
            out[scenario_name][model_name] = scoring.metric_results  # type: ignore
        else:
            # TODO: Warn if metric already exists
            out[scenario_name][model_name].extend(scoring.metric_results)  # type: ignore
    return out


def default_choices(metric_structure):
    out = defaultdict(dict)
    for scenario, models in metric_structure.items():
        for model, metrics in models.items():
            out[scenario][model] = metrics[0]
    return out


@st.cache_data
def fetch_scores_cached(_cfg: DictConfig):
    return get_scores_wandb(_cfg.wandb.project, _cfg.wandb.entity)


# TODO: Move to config dir more elegantly
# pylint: disable=too-many-locals
@hydra.main(config_path=f"../../{CONFIG_DIR}", config_name="master", version_base=None)
def setup_app(cfg: DictConfig):
    st.set_page_config("NLGenDa Leaderboard", page_icon="🇩🇰")
    # https://discuss.streamlit.io/t/remove-made-with-streamlit-from-bottom-of-app/1370/17
    hide_streamlit_style = """
            <style>
            [data-testid="stToolbar"] {visibility: hidden !important;}
            footer {visibility: hidden !important;}
            </style>
            """
    st.markdown(hide_streamlit_style, unsafe_allow_html=True)

    st.title("NLGenDa Leaderboard")

    logger.info("Fetching scores ...")
    # TODO: Minimize loading times
    scores = fetch_scores_cached(cfg)
    metric_structure = extract_metrics(scores)
    default_metrics = default_choices(metric_structure)
    chosen_metrics = default_metrics

    with st.sidebar:
        with st.form(key="metrics_form"):
            for scenario, models in metric_structure.items():
                st.subheader(f"Metrics for {scenario}")
                models_grouped_by_metrics = group_models_by_metrics(models)
                for metrics_key, models_group in models_grouped_by_metrics.items():
                    options = list(metrics_key)
                    common_key = f"{scenario}-{'-'.join(models_group)}"
                    selected_metric = st.selectbox(
                        f"Metric for {', '.join(models_group)}",
                        options,
                        index=options.index(default_metrics[scenario][models_group[0]].short_name),
                        key=common_key,
                    )
                    model = None
                    for model in models_group:
                        chosen_metrics[scenario][model] = next(
                            (
                                metric
                                for metric in models[model]
                                if metric.short_name == selected_metric
                            ),
                            None,
                        )
                    if model is None:
                        st.error("No models were found in results")
                    else:
                        st.caption(
                            f"Currently showing: {selected_metric}.",
                            help=chosen_metrics[scenario][model].description or "",
                        )
            st.form_submit_button(label="Submit")

    logger.info("Building leaderboard table ...")
    table = build_leaderboard_table(metric_structure, chosen_metrics)

    st.dataframe(table, use_container_width=True)

    logger.info("App built!")


if __name__ == "__main__":
    # pylint: disable=no-value-for-parameter
    setup_app()
