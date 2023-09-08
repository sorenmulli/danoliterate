import logging

import hydra
from omegaconf import DictConfig

from nlgenda.evaluation import evaluate
from nlgenda.infrastructure import CONFIG_DIR


@hydra.main(config_path=CONFIG_DIR, config_name="master", version_base=None)
def hydra_entry(cfg: DictConfig) -> None:
    match cfg.do:
        case "evaluate":
            evaluate(cfg)
        case _:
            logging.getLogger(__name__).error(
                "Unsupported do=%s. Currently, only evaluate is supported", cfg.do
            )
            raise ValueError


if __name__ == "__main__":
    # pylint: disable=no-value-for-parameter
    hydra_entry()
