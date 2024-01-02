"""
Should be run as streamlit application
"""
from collections import defaultdict

import hydra
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
from omegaconf import DictConfig

from danoliterate.evaluation.analysis.analyser import Analyser
from danoliterate.evaluation.analysis.dimensions import Dimension
from danoliterate.evaluation.artifact_integration import get_scores_wandb
from danoliterate.evaluation.leaderboard.metric_parsing import default_choices, extract_metrics
from danoliterate.evaluation.leaderboard.table import (
    build_leaderboard_table,
    format_table_for_latex,
)
from danoliterate.evaluation.results import MetricResult
from danoliterate.infrastructure.constants import CONFIG_DIR
from danoliterate.infrastructure.logging import logger

ALL_KEY = "All"


def group_models_by_metrics(models):
    metrics_to_models = defaultdict(list)
    for model, metrics in models.items():
        metrics_key = tuple(metric.short_name for metric in metrics)
        metrics_to_models[metrics_key].append(model)
    return metrics_to_models


@st.cache_data
def fetch_scores_cached(_cfg: DictConfig):
    return get_scores_wandb(_cfg.wandb.project, _cfg.wandb.entity)


def setup_analysis(chosen_metrics: dict[str, dict[str, MetricResult]]):
    analyser = Analyser(chosen_metrics)
    model = st.selectbox(
        "Select a Model",
        options=[ALL_KEY, analyser.concat_key, *analyser.options_dict["model"]],
        index=0,
    )
    scenario = st.selectbox(
        "Select a Scenario",
        options=[ALL_KEY, analyser.concat_key, *analyser.options_dict["scenario"]],
        index=0,
    )

    # Button to perform action
    if st.button("Analyse"):
        # Call the get_subset function with user inputs
        subset = analyser.get_subset(
            model=None if model == ALL_KEY else model,
            scenario=None if scenario == ALL_KEY else scenario,
        )

        summary_stats = pd.DataFrame()
        for name, df in subset.items():
            df: pd.DataFrame  # type: ignore
            # Add these statistics to the summary_stats DataFrame with the name as the column header
            summary_stats[name] = df["score"].describe()
        st.dataframe(summary_stats)

        plt.figure(figsize=(10, 6))
        for name, df in subset.items():
            df: pd.DataFrame  # type: ignore
            sns.histplot(df["score"], kde=True, label=name)
        plt.legend(title="Dataset")
        plt.title("Score Distributions")
        st.pyplot(plt.gcf())

        if scenario != ALL_KEY and len(subset) > 1:
            temp_dfs = []
            for name, df in subset.items():
                df: pd.DataFrame  # type: ignore
                # Create a temporary DataFrame with only 'example_id' and 'score'
                for given_val, col in zip((model, scenario), ("model", "scenario")):
                    if given_val == analyser.concat_key:
                        df["example_id"] = df["example_id"] + df[col]
                temp_dfs.append(
                    df[["example_id", "score"]]
                    .rename(columns={"score": name})
                    .set_index("example_id")
                )

            scores_df = pd.concat(temp_dfs, axis=1)
            # Calculate correlation matrix
            corr_matrix = scores_df.corr()
            plt.figure(figsize=(10, 8))
            sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm")
            plt.title("Score correlations")
            st.pyplot(plt.gcf())
            plt.clf()

            plt.figure(figsize=(10, 8))
            pairplot = sns.pairplot(scores_df, diag_kind="kde")
            for i, col_y in enumerate(scores_df.columns):
                for j, col_x in enumerate(scores_df.columns):
                    if i < j:
                        ax = pairplot.axes[i][j]
                        ax.clear()
                        sns.histplot(
                            scores_df[[col_x, col_y]],
                            ax=ax,
                            x=col_x,
                            y=col_y,
                        )

            st.pyplot(plt.gcf())
            plt.clf()


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
        chosen_metrics,
        efficiency=chosen_dimension == Dimension.EFFICIENCY,
        micro=index_micro,
    )

    st.dataframe(table, use_container_width=True)

    if st.button("Log current table as LaTeX"):
        latex = format_table_for_latex(table, lower_is_better)
        logger.info("Table:\n%s", latex)

    with st.expander("Open to analyse results"):
        setup_analysis(chosen_metrics)

    logger.info("App built!")


if __name__ == "__main__":
    # pylint: disable=no-value-for-parameter
    setup_app()
