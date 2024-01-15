from collections import defaultdict

import streamlit as st

from danoliterate.evaluation.analysis.dimensions import Dimension
from danoliterate.evaluation.artifact_integration import get_scores_wandb
from danoliterate.evaluation.leaderboard.metric_parsing import default_choices, extract_metrics
from danoliterate.evaluation.leaderboard.table import build_leaderboard_table

ALL_KEY = "Each Individually"
WANDB_PROJECT = "nlgenda"
WANDB_ENTITY = "sorenmulli"


def group_models_by_metrics(models):
    metrics_to_models = defaultdict(list)
    for model, metrics in models.items():
        metrics_key = tuple(metric.short_name for metric in metrics)
        metrics_to_models[metrics_key].append(model)
    return metrics_to_models


@st.cache_data
def fetch_scores_cached():
    return get_scores_wandb(WANDB_PROJECT, WANDB_ENTITY)


st.set_page_config("Danoliterate Leaderboard", page_icon="🇩🇰")
# https://discuss.streamlit.io/t/remove-made-with-streamlit-from-bottom-of-app/1370/17
hide_streamlit_style = """
        <style>
        [data-testid="stToolbar"] {visibility: hidden !important;}
        footer {visibility: hidden !important;}
        </style>
        """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

st.title("Danoliterate GLLM Leaderboard")
st.warning(
    "The benchmark is a beta version and results are subject to change. Especially toxicity, robustness and fairness are experimental solutions.",
    icon="🤖",
)
"""
Go to the `start` page to get more details about what is going on.
Note that the table can be expanded.
"""

print("Fetching scores ...")
scores = fetch_scores_cached()

chosen_dimension = st.selectbox(
    "Evaluation Dimension", Dimension, format_func=lambda dim: dim.value
)
show_missing = st.checkbox("Show models with missing values")
all_models = sorted(
    list({scoring.execution_metadata.model_cfg["name"] for scoring in scores.scorings})
)
excluded_models = st.multiselect("Select models to exclude", all_models)
chosen_models = [model for model in all_models if model not in excluded_models]

index_micro = st.selectbox("Index Average", ["Micro Avg.", "Macro Avg."]) == "Micro Avg."
metric_structure = extract_metrics(scores, chosen_dimension, chosen_models=chosen_models)
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

print("Building leaderboard table ...")
table, _ = build_leaderboard_table(
    chosen_metrics,
    efficiency=chosen_dimension == Dimension.EFFICIENCY,
    micro=index_micro,
    show_missing=show_missing,
)

st.dataframe(table, use_container_width=True)

print("App built!")
