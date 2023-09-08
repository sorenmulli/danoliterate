import logging

from omegaconf import DictConfig

from nlgenda.infrastructure import format_config


def evaluate(cfg: DictConfig):
    logging.info("Running evaluation with arguments: %s", format_config(cfg))
