from abc import ABC, abstractmethod
from typing import Callable, Iterator

import augmenty
import spacy
from augmenty.character.replace import create_keystroke_error_augmenter_v1
from augmenty.util import Example
from dacy.datasets import danish_names, female_names, male_names, muslim_names
from omegaconf import DictConfig
from spacy.language import Language

from danoliterate.infrastructure.logging import logger


class Augmenter(ABC):
    @abstractmethod
    def __call__(self, text: str) -> str:
        ...

    @property
    @abstractmethod
    def description(self) -> str:
        ...

    @property
    @abstractmethod
    def key(self) -> str:
        ...


class AugmentyBasedAugmenter(Augmenter, ABC):
    spacy_model = "da_core_news_sm"

    def __init__(self):
        self.nlp = spacy.load(self.spacy_model)
        self.augmenter = self.create_augmenter()

    def __call__(self, text: str) -> str:
        try:
            return list(augmenty.texts([text], augmenter=self.augmenter, nlp=self.nlp))[0]
        except ValueError as error:
            logger.warning(
                "%s\nCould not augment text due to above error. Using it in non-augmented form. Text was:\n%s.",
                error,
                text,
            )
            return text

    @abstractmethod
    def create_augmenter(self) -> Callable[[Language, Example], Iterator[Example]]:
        ...


class KeystrokeErrorAdder(AugmentyBasedAugmenter):
    p = 0.1

    def create_augmenter(self) -> Callable[[Language, Example], Iterator[Example]]:
        return create_keystroke_error_augmenter_v1(self.p, keyboard="da_qwerty_v1")

    @property
    def description(self):
        return f"{self.p:.0%} random adjacent keystroke errors"

    @property
    def key(self):
        return "keystroke-error"


class NameInserter(AugmentyBasedAugmenter, ABC):
    def __init__(self):
        self.names = self.get_names()
        super().__init__()

    @abstractmethod
    def get_names(self) -> dict[str, list[str]]:
        ...

    def create_augmenter(self) -> Callable[[Language, Example], Iterator[Example]]:
        return augmenty.create_per_replace_augmenter_v1(
            self.names,
            patterns=[
                ["first_name"],
                ["first_name", "last_name"],
                ["first_name", "first_name", "last_name"],
            ],
            level=1,
            person_tag="PER",
        )


class MaleNameInserter(NameInserter):
    def get_names(self) -> dict[str, list[str]]:
        return male_names()

    @property
    def description(self):
        return "All person entities replaced with male names"

    @property
    def key(self):
        return "male-inserted"


class FemaleNameInserter(NameInserter):
    def get_names(self) -> dict[str, list[str]]:
        return female_names()

    @property
    def description(self):
        return "All person entities replaced with female names"

    @property
    def key(self):
        return "female-inserted"


class DanishNameInserter(NameInserter):
    def get_names(self) -> dict[str, list[str]]:
        return danish_names()

    @property
    def description(self):
        return "All person entities replaced with Danish names"

    @property
    def key(self):
        return "danish-inserted"


class MuslimNameInserter(NameInserter):
    def get_names(self) -> dict[str, list[str]]:
        return muslim_names()

    @property
    def description(self):
        return "All person entities replaced with Muslim names"

    @property
    def key(self):
        return "muslim-inserted"


def get_augmenters(cfg: DictConfig) -> list[Augmenter]:
    augmenters: list[Augmenter] = []
    if cfg.evaluation.robustness_augment:
        augmenters.extend((KeystrokeErrorAdder(),))
    if cfg.evaluation.fairness_augment:
        augmenters.extend(
            (
                MaleNameInserter(),
                FemaleNameInserter(),
                DanishNameInserter(),
                MuslimNameInserter(),
            )
        )
    return augmenters
