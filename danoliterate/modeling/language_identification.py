import fasttext
from omegaconf import DictConfig

from danoliterate.infrastructure.logging import logger

DANISH = "da"


class LanguageIdentifier:
    def __init__(self, cfg: DictConfig):
        path = cfg.model_paths.fasttext
        try:
            self.model = fasttext.load_model(path)
        except ValueError as error:
            logger.error(
                msg := f"{path} did not contain fasttext model. "
                "Download the model by `make fasttext`"
            )
            raise FileNotFoundError(msg) from error

    def predict(self, text: str) -> str:
        # TODO: Should I do sentence level prediction instead?
        text = text.replace("\n", "")
        return self.model.predict(text)[0][0].replace("__label__", "")
