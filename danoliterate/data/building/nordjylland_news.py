import pandas as pd
from datasets import Dataset, load_dataset
from omegaconf import DictConfig

from danoliterate.data.building.hub import push
from danoliterate.infrastructure.logging import format_config, logger


def limit_dataset(dataset: Dataset, databuild_cfg: DictConfig) -> Dataset:
    df = pd.DataFrame(dataset)
    # Save original index
    df["ind"] = df.index
    df = df[df["text_len"] < databuild_cfg.max_text_chars]
    assert len(df) > databuild_cfg.n_to_keep
    logger.info(
        "Dataset length was reduced from %i to %i due to max text characters=%i",
        len(dataset),
        len(df),
        databuild_cfg.max_text_chars,
    )
    df = df.sample(databuild_cfg.n_to_keep, random_state=1887)
    logger.info("Dataset was subsampled to %i examples", len(df))
    return Dataset.from_pandas(df, preserve_index=False)


def create_nordjylland_news(cfg: DictConfig):
    logger.debug("Running with arguments: %s", format_config(cfg))

    logger.info("Acquiring full Nordjylland News testset.")
    dataset = load_dataset("alexandrainst/nordjylland-news-summarization", split="test")

    logger.info("Limiting dataset")
    dataset = limit_dataset(dataset, cfg.databuild)

    if cfg.databuild.hub.push:
        push(dataset, cfg.databuild.hub)
