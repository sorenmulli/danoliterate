from abc import ABC, abstractmethod
from collections import OrderedDict
from typing import Optional, Type

import spacy
from absl import logging as absl_logging
from evaluate import load as load_metric


class Comparer(ABC):
    key: str
    name: str
    max_cache_size = 100_000

    cutoff: Optional[float] = 1.5

    nlp: Optional[spacy.language.Language] = None

    def __init__(self) -> None:
        # Make this some datastructure with a max size
        self.cache: OrderedDict[tuple[str, str], float] = OrderedDict()

    def __call__(self, targets: list[str], predictions: list[str]) -> list[float]:
        results: list[Optional[float]] = []
        if self.cutoff is not None:
            self.cutoff_texts(targets, predictions)
        for target, pred in zip(targets, predictions, strict=True):
            key = (target, pred)
            if key in self.cache:
                results.append(self.cache[key])
                self.cache.move_to_end(key)
            else:
                results.append(None)
        to_predict = [
            (target, pred)
            for target, pred, res in zip(targets, predictions, results, strict=True)
            if res is None
        ]
        new_results = self.predict(
            [target for target, _ in to_predict], [pred for _, pred in to_predict]
        )
        for (target, pred), res in zip(to_predict, new_results, strict=True):
            self.cache[target, pred] = res
            if len(self.cache) > self.max_cache_size:
                self.cache.popitem(last=False)
        for i, old_res in enumerate(results):
            if old_res is None:
                results[i] = new_results.pop(0)
        return results  # type: ignore

    @abstractmethod
    def predict(self, targets: list[str], predictions: list[str]) -> list[float]:
        ...

    def lemmatize(self, texts: list[str], batch_size=1000) -> list[str]:
        assert self.nlp is not None, "You must first load Spacy before doing lemmatization"
        out = []
        for doc in self.nlp.pipe(texts, batch_size=batch_size):
            out.append(" ".join(token.lemma_ for token in doc))
        return out

    def cutoff_texts(self, targets: list[str], predictions: list[str]) -> list[str]:
        assert self.cutoff is not None, "Cutoff value is not set."
        new_predictions = []
        for target, pred in zip(targets, predictions, strict=True):
            if len(pred_words := pred.split()) > (
                cutoff_len := int(self.cutoff * len(target.split()))
            ):
                new_predictions.append(" ".join(pred_words[:cutoff_len]))
            else:
                new_predictions.append(pred)
        return new_predictions


class Rouge1(Comparer):
    key = "rouge-1"
    name = "ROUGE-1"

    def __init__(self):
        super().__init__()
        self.metric = load_metric("rouge")
        self.nlp = spacy.load("da_core_news_sm")

    def predict(self, targets: list[str], predictions: list[str]) -> list[float]:
        targets = self.lemmatize(targets)
        predictions = self.lemmatize(predictions)
        scores = self.metric.compute(
            predictions=predictions, references=targets, use_aggregator=False
        )
        return [float(score) for score in scores["rouge1"]]


class RougeL(Comparer):
    key = "rouge-l"
    name = "ROUGE-L"

    def __init__(self):
        super().__init__()
        self.metric = load_metric("rouge")
        self.nlp = spacy.load("da_core_news_sm")

    def predict(self, targets: list[str], predictions: list[str]) -> list[float]:
        targets = self.lemmatize(targets)
        predictions = self.lemmatize(predictions)
        scores = self.metric.compute(
            predictions=predictions, references=targets, use_aggregator=False
        )
        return [float(score) for score in scores["rougeL"]]


class BertSimilarity(Comparer):
    key = "bert-sim"
    name = "BERT similarity"
    encoder = "chcaa/dfm-encoder-large-v1"

    def __init__(self):
        super().__init__()
        absl_logging.set_verbosity(absl_logging.WARNING)
        self.metric = load_metric("bertscore")

    def predict(self, targets: list[str], predictions: list[str]) -> list[float]:
        scores = self.metric.compute(predictions=predictions, references=targets, lang="da")
        return [float(score) for score in scores["f1"]]


class ClassChoiceParser(Comparer):
    key = "chosen-parsing"
    name = "Parsing of chosen option"

    cutoff = None

    def predict(self, targets: list[str], predictions: list[str]) -> list[float]:
        all_classes = set(targets)
        scores = []
        for target, pred in zip(targets, predictions, strict=True):
            class_counts = {cla: pred.count(cla) for cla in all_classes}
            # If the true class is the only mentioned: Score is 1
            # If a class is the only mentioned: Score is 0
            # If both true and some wrong classes are mentioned, choose most mentioned
            total_mentions = sum(class_counts.values())
            scores.append(class_counts[target] / total_mentions if total_mentions else 0.0)
        return scores


_COMPARERS: tuple[Type[Comparer], ...] = Rouge1, RougeL, BertSimilarity, ClassChoiceParser
COMPARERS: dict[str, Type[Comparer]] = {comparer.name: comparer for comparer in _COMPARERS}
