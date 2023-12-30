"""
Should be run as streamlit application
"""
from collections import defaultdict

import hydra
import streamlit as st
from omegaconf import DictConfig

from danoliterate.evaluation.analysis.dimensions import Dimension
from danoliterate.evaluation.artifact_integration import get_scores_wandb
from danoliterate.evaluation.leaderboard.metric_parsing import default_choices, extract_metrics
from danoliterate.evaluation.leaderboard.table import (
    build_leaderboard_table,
    format_table_for_latex,
)
from danoliterate.infrastructure.constants import CONFIG_DIR
from danoliterate.infrastructure.logging import logger


def group_models_by_metrics(models):
    metrics_to_models = defaultdict(list)
    for model, metrics in models.items():
        metrics_key = tuple(metric.short_name for metric in metrics)
        metrics_to_models[metrics_key].append(model)
    return metrics_to_models


@st.cache_data
def fetch_scores_cached(_cfg: DictConfig):
    return get_scores_wandb(_cfg.wandb.project, _cfg.wandb.entity)


# TODO: Move to config dir more elegantly
# pylint: disable=too-many-locals
@hydra.main(config_path=f"../../{CONFIG_DIR}", config_name="master", version_base=None)
def setup_app(cfg: DictConfig):
    st.set_page_config("Danoliterate Leaderboard", page_icon="🇩🇰")
    # https://discuss.streamlit.io/t/remove-made-with-streamlit-from-bottom-of-app/1370/17
    hide_streamlit_style = """
            <style>
            [data-testid="stToolbar"] {visibility: hidden !important;}
            footer {visibility: hidden !important;}
            </style>
            """
    st.markdown(hide_streamlit_style, unsafe_allow_html=True)

    st.title("Danoliterate LLM Leaderboard")

    logger.info("Fetching scores ...")
    # TODO: Minimize loading times
    scores = fetch_scores_cached(cfg)
    chosen_dimension = st.selectbox(
        "Evaluation Dimension", Dimension, format_func=lambda dim: dim.value
    )
    chosen_type = st.selectbox(
        "Leaderboard type",
        ("standard", "free-generation"),
        format_func=lambda text: text.replace("-", " ").capitalize(),
    )
    index_micro = st.selectbox("Index Average", ["Micro Avg.", "Macro Avg."]) == "Micro Avg."
    metric_structure = extract_metrics(scores, chosen_dimension, chosen_type)
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
    table, lower_is_better = build_leaderboard_table(
        metric_structure,
        chosen_metrics,
        efficiency=chosen_dimension == Dimension.EFFICIENCY,
        micro=index_micro,
    )

    st.dataframe(table, use_container_width=True)

    if st.button("Log current table as LaTeX"):
        latex = format_table_for_latex(table, lower_is_better)
        logger.info("Table:\n%s", latex)

    logger.info("App built!")


if __name__ == "__main__":
    # pylint: disable=no-value-for-parameter
    setup_app()
