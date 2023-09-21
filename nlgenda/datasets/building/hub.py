import logging

from datasets import Dataset

logger = logging.getLogger(__name__)


def push(dataset: Dataset, target: str, private: bool, config_name="default"):
    private_string = " (privately) " if private else ""
    if (
        input(f"Do you want to push the dataset to HF hub {target}{private_string}? [y/N]")
        .lower()
        .strip()
        != "y"
    ):
        return logger.warning("Aborting push due to user rejection")
    logger.info("Pushing dataset to HF hub %s%s...", target, private_string)
    dataset.push_to_hub(target, private=private, config_name=config_name)
