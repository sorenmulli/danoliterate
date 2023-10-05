from abc import ABC, abstractmethod

import dacy
from absl import logging as absl_logging
from evaluate import load as load_metric


class Comparer(ABC):
    name: str

    @abstractmethod
    def __call__(self, target: str, prediction: str) -> float:
        ...


class Rouge1(Comparer):
    name = "ROUGE-1"

    def __init__(self):
        self.metric = load_metric("rouge")
        self.nlp = dacy.load("small")

    def __call__(self, target: str, prediction: str) -> float:
        target = " ".join(token.lemma_ for token in self.nlp(target))
        prediction = " ".join(token.lemma_ for token in self.nlp(prediction))
        scores = self.metric.compute(predictions=[prediction], references=[target])
        return float(scores["rouge1"])


class RougeL(Comparer):
    name = "ROUGE-L"

    def __init__(self):
        self.metric = load_metric("rouge")

    def __call__(self, target: str, prediction: str) -> float:
        target = " ".join(token.lemma_ for token in self.nlp(target))
        prediction = " ".join(token.lemma_ for token in self.nlp(prediction))
        scores = self.metric.compute(predictions=[prediction], references=[target])
        return float(scores["rougeL"])


class AnswerContained(Comparer):
    name = "Answer contained"

    def __call__(self, target: str, prediction: str) -> float:
        return float(target.lower() in prediction.lower())


class BertSimilarity(Comparer):
    name = "BERT similarity"

    def __init__(self):
        absl_logging.set_verbosity(absl_logging.WARNING)
        # TODO: How can this be parallelized?
        self.metric = load_metric("bertscore", lang="da")

    def __call__(self, target: str, prediction: str) -> float:
        scores = self.metric.compute(predictions=[prediction], references=[target], lang="da")
        return float(scores["f1"][0])


_COMPARERS = Rouge1, RougeL, AnswerContained, BertSimilarity
COMPARERS = {comparer.name: comparer for comparer in _COMPARERS}
