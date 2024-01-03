import hydra
from omegaconf import DictConfig

from danoliterate.infrastructure import CONFIG_DIR
from danoliterate.infrastructure.logging import logger


# pylint: disable=import-outside-toplevel
@hydra.main(config_path=CONFIG_DIR, config_name="master", version_base=None)
def hydra_entry(cfg: DictConfig) -> None:
    match cfg.do:
        case "evaluate":
            from danoliterate.evaluation.execution.evaluator import evaluate

            evaluate(cfg)
        case "databuild":
            match cfg.databuild.type:
                case "prompt-answer":
                    from danoliterate.data.building.prompt_answer_da import create_prompt_answer_da

                    create_prompt_answer_da(cfg)
                case "citizenship-test":
                    from danoliterate.data.building.citizenship_test_da import create_citizen_da

                    create_citizen_da(cfg)
                case "hyggeswag":
                    from danoliterate.data.building.hyggeswag import create_hyggeswag

                    create_hyggeswag(cfg)
                case "nordjylland-news":
                    from danoliterate.data.building.nordjylland_news import create_nordjylland_news

                    create_nordjylland_news(cfg)
                case "da-gym-2000":
                    from danoliterate.data.building.da_gym_2000 import create_da_gym_200

                    create_da_gym_200(cfg)
                case "da-cloze-self-test":
                    from danoliterate.data.building.da_cloze_self_test import (
                        create_da_cloze_self_test,
                    )

                    create_da_cloze_self_test(cfg)
                case "hashtag-twitterhjerne":
                    from danoliterate.data.building.hashtag_twitterhjerne import (
                        create_hashtag_twitterhjerne,
                    )

                    create_hashtag_twitterhjerne(cfg)
                case _:
                    logger.error("Unsupported databuild.type=%s.", cfg.do)
                    raise ValueError("Unsupported databuild type")
        case "train":
            from danoliterate.training import train_lm

            train_lm(cfg)
        case "score":
            from danoliterate.evaluation.analysis.scorer import score

            score(cfg)
        case "inspect":
            from danoliterate.evaluation.analysis.inspection import inspect

            inspect(cfg)
        case _:
            logger.error(
                "Unsupported do=%s. 'evaluate', 'databuild', 'train', 'score' are supported", cfg.do
            )
            raise ValueError("Unsupported do")


if __name__ == "__main__":
    try:
        # pylint: disable=no-value-for-parameter
        hydra_entry()
    except ImportError as error:
        logger.warning(
            "%s\nGot above ImportError.\n"
            "You might need to install the [full] version of the package with "
            "pip install danoliterate[full]",
            error,
        )
        raise
