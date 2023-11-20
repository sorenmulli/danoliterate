"""
Should be run as streamlit application
"""
import logging

import hydra
import hydra.core.global_hydra
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from omegaconf import DictConfig

from nlgenda.evaluation.analysis.analyser import Analyser
from nlgenda.infrastructure.constants import CONFIG_DIR

logger = logging.getLogger(__name__)

ALL_KEY = "All"
COMBINED_KEY = "Concatenated"


@st.cache_data
def fetch_analyser_cached(_cfg: DictConfig):
    return Analyser(_cfg)


# TODO: Move to config dir more elegantly
# pylint: disable=too-many-locals
@hydra.main(config_path=f"../../{CONFIG_DIR}", config_name="master", version_base=None)
def setup_app(cfg: DictConfig):
    hydra.core.global_hydra.GlobalHydra.instance().clear()
    analyser = fetch_analyser_cached(cfg)
    # Streamlit interface
    st.title("Data Analysis Application")

    # Let the user select the parameters for analysis
    model = st.selectbox(
        "Select a Model", options=[ALL_KEY, COMBINED_KEY, *analyser.options_dict["model"]], index=0
    )
    scenario = st.selectbox(
        "Select a Scenario",
        options=[ALL_KEY, COMBINED_KEY, *analyser.options_dict["scenario"]],
        index=0,
    )
    metric = st.selectbox(
        "Select a Metric",
        options=[ALL_KEY, COMBINED_KEY, *analyser.options_dict["metric"]],
        index=0,
    )

    # Button to perform action
    if st.button("Analyse"):
        # Call the get_subset function with user inputs
        subset = analyser.get_subset(
            metric=None if metric == ALL_KEY else metric,
            model=None if model == ALL_KEY else model,
            scenario=None if scenario == ALL_KEY else scenario,
        )

        summary_stats = pd.DataFrame()
        for name, df in subset.items():
            # Add these statistics to the summary_stats DataFrame with the name as the column header
            summary_stats[name] = df["score"].describe()
        st.dataframe(summary_stats)

        plt.figure(figsize=(10, 6))
        for name, df in subset.items():
            sns.histplot(df["score"], kde=True, label=name)
        plt.legend(title="Dataset")
        plt.title("Score Distributions")
        st.pyplot(plt.gcf())

        if scenario != ALL_KEY and len(subset) > 1:
            temp_dfs = []
            for name, df in subset.items():
                # Create a temporary DataFrame with only 'example_id' and 'score'
                for given_val, col in zip(
                    (metric, model, scenario), ("metric", "model", "scenario")
                ):
                    if given_val == COMBINED_KEY:
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
            g = sns.pairplot(scores_df, diag_kind="kde")
            for i in range(len(scores_df.columns)):
                for j in range(len(scores_df.columns)):
                    if i < j:
                        ax = g.axes[i][j]
                        ax.clear()
                        sns.histplot(
                            scores_df[[scores_df.columns[j], scores_df.columns[i]]],
                            ax=ax,
                            x=scores_df.columns[j],
                            y=scores_df.columns[i],
                        )

            st.pyplot(plt.gcf())
            plt.clf()


if __name__ == "__main__":
    # pylint: disable=no-value-for-parameter
    setup_app()
