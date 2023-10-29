import logging

import hydra
from omegaconf import DictConfig

from nlgenda.datasets.building.citizenship_test_da import create_citizen_da
from nlgenda.datasets.building.hyggeswag import create_hyggeswag
from nlgenda.datasets.building.nordjylland_news import create_nordjylland_news
from nlgenda.datasets.building.prompt_answer_da import create_prompt_answer_da
from nlgenda.evaluation.analysis import score
from nlgenda.evaluation.execution import evaluate
from nlgenda.infrastructure import CONFIG_DIR
from nlgenda.training import train_lm


@hydra.main(config_path=CONFIG_DIR, config_name="master", version_base=None)
def hydra_entry(cfg: DictConfig) -> None:
    match cfg.do:
        case "evaluate":
            evaluate(cfg)
        case "databuild":
            match cfg.databuild.type:
                case "prompt-answer":
                    create_prompt_answer_da(cfg)
                case "citizenship-test":
                    create_citizen_da(cfg)
                case "hyggeswag":
                    create_hyggeswag(cfg)
                case "nordjylland-news":
                    create_nordjylland_news(cfg)
                case _:
                    logging.error(
                        "Unsupported databuild.type=%s. "
                        "'prompt-answer', 'citizenship-test', 'hyggeswag' are supported",
                        cfg.do,
                    )
                    raise ValueError("Unsupported databuild type")
        case "train":
            train_lm(cfg)
        case "score":
            score(cfg)
        case _:
            logging.error(
                "Unsupported do=%s. 'evaluate', 'databuild', 'train', 'score' are supported", cfg.do
            )
            raise ValueError("Unsupported do")


if __name__ == "__main__":
    # pylint: disable=no-value-for-parameter
    hydra_entry()
