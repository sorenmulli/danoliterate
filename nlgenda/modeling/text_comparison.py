from abc import ABC, abstractmethod
from typing import Optional

import spacy
from absl import logging as absl_logging
from evaluate import load as load_metric


class Comparer(ABC):
    name: str

    nlp: Optional[spacy.language.Language] = None

    @abstractmethod
    def __call__(self, targets: list[str], predictions: list[str]) -> list[float]:
        ...

    def lemmatize(self, texts: list[str], batch_size=1000) -> list[str]:
        assert self.nlp is not None, "You must first load Spacy before doing lemmatization"
        out = []
        for doc in self.nlp.pipe(texts, batch_size=batch_size):
            out.append(" ".join(token.lemma_ for token in doc))
        return out


class Rouge1(Comparer):
    name = "ROUGE-1"

    def __init__(self):
        self.metric = load_metric("rouge")
        self.nlp = spacy.load("da_core_news_sm")

    def __call__(self, targets: list[str], predictions: list[str]) -> list[float]:
        targets = self.lemmatize(targets)
        predictions = self.lemmatize(predictions)
        scores = self.metric.compute(
            predictions=predictions, references=targets, use_aggregator=False
        )
        return [float(score) for score in scores["rouge1"]]


class RougeL(Comparer):
    name = "ROUGE-L"

    def __init__(self):
        self.metric = load_metric("rouge")
        self.nlp = spacy.load("da_core_news_sm")

    def __call__(self, targets: list[str], predictions: list[str]) -> list[float]:
        targets = self.lemmatize(targets)
        predictions = self.lemmatize(predictions)
        scores = self.metric.compute(
            predictions=predictions, references=targets, use_aggregator=False
        )
        return [float(score) for score in scores["rougeL"]]


class BertSimilarity(Comparer):
    name = "BERT similarity"
    encoder = "chcaa/dfm-encoder-large-v1"

    def __init__(self):
        absl_logging.set_verbosity(absl_logging.WARNING)
        self.metric = load_metric("bertscore")

    def __call__(self, targets: list[str], predictions: list[str]) -> list[float]:
        scores = self.metric.compute(predictions=predictions, references=targets, lang="da")
        return [float(score) for score in scores["f1"]]


class ClassChoiceParser(Comparer):
    name = "Parsing of chosen class"

    def __call__(self, targets: list[str], predictions: list[str]) -> list[float]:
        all_classes = set(targets)
        scores = []
        for target, pred in zip(targets, predictions, strict=True):
            class_counts = {cla: pred.count(cla) for cla in all_classes}
            # If the true class is the only mentioned: Score is 1
            # If a class is the only mentioned: Score i 0
            # If both true true and some wrong classes are mentioned
            total_mentions = sum(class_counts.values())
            scores.append(class_counts[target] / total_mentions if total_mentions else 0.0)
        return scores


_COMPARERS = Rouge1, RougeL, BertSimilarity, ClassChoiceParser
COMPARERS = {comparer.name: comparer for comparer in _COMPARERS}
