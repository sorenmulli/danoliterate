"""
Should be run as streamlit application
"""
import logging

import hydra
import streamlit as st
from omegaconf import DictConfig

from nlgenda.infrastructure.constants import CONFIG_DIR
from nlgenda.leaderboard.data import get_results_wandb
from nlgenda.leaderboard.table import build_leaderboard_table

logger = logging.getLogger(__name__)


# TODO: Move to config dir more elegantly
@hydra.main(config_path=f"../{CONFIG_DIR}", config_name="master", version_base=None)
def setup_app(cfg: DictConfig):
    st.title("NLGenDa Leaderboard")

    logger.info("Fetching results ...")
    results = get_results_wandb(cfg.wandb.project, cfg.wandb.entity)

    logger.info("Building leaderboard table ...")
    table = build_leaderboard_table(results)

    st.table(table)
    logger.info("App built!")


if __name__ == "__main__":
    # pylint: disable=no-value-for-parameter
    setup_app()
