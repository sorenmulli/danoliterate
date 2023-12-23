from datasets import Dataset
from omegaconf import DictConfig

from danoliterate.infrastructure.logging import logger


def push(dataset: Dataset, cfg: DictConfig, config_name="default"):
    private_string = " (privately) " if cfg.private else ""
    if (
        input(f"Do you want to push the dataset to HF hub {cfg.target}{private_string}? [y/N]")
        .lower()
        .strip()
        != "y"
    ):
        return logger.warning("Aborting push due to user rejection")
    logger.info("Pushing dataset to HF hub %s%s...", cfg.target, private_string)
    dataset.push_to_hub(cfg.target, private=cfg.private, config_name=config_name)
