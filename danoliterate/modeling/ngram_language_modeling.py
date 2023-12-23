import re

import kenlm
from omegaconf import DictConfig

from danoliterate.infrastructure.logging import logger


class NgramLm:
    norm_pattern = re.compile(r"[\W_]+")

    def __init__(self, cfg: DictConfig):
        path = cfg.model_paths.dsl3gram
        try:
            self.model = kenlm.Model(path)
        except FileNotFoundError as error:
            logger.error(
                msg := f"{path} did not contain DSL 3gram model. "
                "Download the model by `make dsl3gram`"
            )
            raise FileNotFoundError(msg) from error

    def predict(self, text: str, normalize_text=True, normalize_length=True) -> float:
        if normalize_text:
            text = self._normalize(text)
        return self.model.score(text) / (len(text.split()) if normalize_length else 1)

    def _normalize(self, text: str) -> str:
        normed = []
        for word in text.split():
            word = self.norm_pattern.sub("", word)
            if not word:
                continue
            normed.append(word.lower().replace("é", "e"))
        return " ".join(normed)
