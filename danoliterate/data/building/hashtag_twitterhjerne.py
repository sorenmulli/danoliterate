import pandas as pd
from datasets import Dataset
from omegaconf import DictConfig

from danoliterate.data.building.hub import push
from danoliterate.infrastructure.logging import format_config, logger


def create_hashtag_twitterhjerne(cfg: DictConfig):
    logger.debug("Running with arguments: %s", format_config(cfg))
    # Don't go public with source links
    df = pd.read_csv(cfg.databuild.collected_csv, index_col=None).drop(columns="Source")
    dataset = Dataset.from_pandas(df)
    if cfg.databuild.hub.push:
        push(dataset, cfg.databuild.hub)
