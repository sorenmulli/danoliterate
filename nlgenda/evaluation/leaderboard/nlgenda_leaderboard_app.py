"""
Should be run as streamlit application
"""
import logging

import hydra
import streamlit as st
from omegaconf import DictConfig

from nlgenda.evaluation.artifact_integration import get_scores_wandb
from nlgenda.evaluation.leaderboard.table import build_leaderboard_table
from nlgenda.infrastructure.constants import CONFIG_DIR

logger = logging.getLogger(__name__)


# TODO: Move to config dir more elegantly
@hydra.main(config_path=f"../../{CONFIG_DIR}", config_name="master", version_base=None)
def setup_app(cfg: DictConfig):
    st.title("NLGenDa Leaderboard")

    logger.info("Fetching scores ...")
    scores = get_scores_wandb(cfg.wandb.project, cfg.wandb.entity)

    logger.info("Building leaderboard table ...")
    table, helps = build_leaderboard_table(scores)

    # Use columns config = None to hide specific rows such as choosing metrics
    st.dataframe(
        table,
        column_config={
            name: st.column_config.Column(help=helps.get(name)) for name in table.columns
        },
    )
    logger.info("App built!")


if __name__ == "__main__":
    # pylint: disable=no-value-for-parameter
    setup_app()
