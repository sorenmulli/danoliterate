import random
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, fields
from typing import Optional

import torch

from danoliterate.evaluation.results import ExecutionExample


def set_deterministic(seed=0):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class WrongInferenceError(RuntimeError):
    ...


@dataclass
class QueriedCall:
    id_: str = field(init=False)

    def __post_init__(self):
        self.id_ = str(uuid.uuid1())


@dataclass
class QueriedGenerateCall(QueriedCall):
    prompt: str

    result: Optional[str] = None
    do_query_score = False

    def query_score(self, score: Optional[float] = None):
        query = QueriedGenerateScoreCall(result=score)
        query.id_ = self.id_
        self.do_query_score = True
        return query


@dataclass
class QueriedGenerateScoreCall(QueriedCall):
    result: Optional[float] = None


@dataclass
class QueriedLikelihoodCall(QueriedCall):
    prompt: str
    target: str

    result: Optional[float] = None


class ModelInference(ABC):
    generate_queries: dict[str, QueriedGenerateCall]
    generate_score_queries: dict[str, QueriedGenerateScoreCall]
    likelihood_queries: dict[str, QueriedLikelihoodCall]

    def __init__(self):
        self.generate_queries = {}
        self.generate_score_queries = {}
        self.likelihood_queries = {}

    @property
    @abstractmethod
    def can_do_lm(self) -> bool:
        ...

    @property
    @abstractmethod
    def can_do_nlg(self) -> bool:
        ...

    # These are pseudo abstract methods as at least one of them should be overwritten
    # so I keep the arguments named
    # pylint: disable=unused-argument
    def generate_texts(self, prompts: list[str]) -> list[tuple[str, Optional[float]]]:
        if not self.can_do_nlg:
            raise WrongInferenceError(
                f"Cannot generate text with {self} which does not support NLG."
            )
        raise NotImplementedError(f"{type(self)} is missing implementation of `generate_text`.")

    # pylint: disable=unused-argument
    def likelihoods(self, prompt_and_targets: list[tuple[str, str]]) -> list[float]:
        if not self.can_do_lm:
            raise WrongInferenceError(
                f"Cannot calculate likelihood with {self} which does not support LM."
            )
        raise NotImplementedError(f"{type(self)} is missing implementation of `likelihood`.")

    def query_generate_text(self, prompt: str):
        query = QueriedGenerateCall(prompt)
        assert self.can_do_nlg
        self.generate_queries[query.id_] = query
        return query

    def query_likelihood(self, prompt: str, target: str):
        query = QueriedLikelihoodCall(prompt, target)
        assert self.can_do_lm
        self.likelihood_queries[query.id_] = query
        return query

    # We want to ignore pylint about the pseudo-abstract methods "generate_text" and "likelihood"
    # pylint: disable=assignment-from-no-return
    def answer_queries(self, queried_examples: list[ExecutionExample]) -> list[ExecutionExample]:
        if self.generate_queries:
            generations = self.generate_texts(
                [query.prompt for query in self.generate_queries.values()]
            )
            for gquery, (generation, score) in zip(self.generate_queries.values(), generations):
                gquery.result = generation
                if gquery.do_query_score:
                    self.generate_score_queries[gquery.id_] = gquery.query_score(score)

        if self.likelihood_queries:
            likelihoods = self.likelihoods(
                [(query.prompt, query.target) for query in self.likelihood_queries.values()]
            )
            for lquery, likelihood in zip(self.likelihood_queries.values(), likelihoods):
                lquery.result = likelihood

        results = [self._populate_with_answers(example) for example in queried_examples]
        assert (
            not self.generate_queries
            and not self.generate_score_queries
            and not self.likelihood_queries
        )

        return results

    def _populate_with_answers(self, example: ExecutionExample) -> ExecutionExample:
        for field_ in fields(example):
            name = field_.name
            value = getattr(example, name)
            for query_type, queries in zip(
                (QueriedGenerateCall, QueriedGenerateScoreCall, QueriedLikelihoodCall),
                (self.generate_queries, self.generate_score_queries, self.likelihood_queries),
            ):
                if isinstance(value, query_type):
                    setattr(example, name, queries.pop(value.id_).result)  # type: ignore
                elif isinstance(value, list):
                    if all(isinstance(elem, query_type) for elem in value):
                        setattr(
                            example,
                            name,
                            [queries.pop(elem.id_).result for elem in value],  # type: ignore
                        )
        return example


class ConstantBaseline(ModelInference):
    def generate_texts(self, prompts: list[str]) -> list[tuple[str, Optional[float]]]:
        return [("a b c d e f g h i j k l m n o p q r s t u v w x y z æ ø å", 0) for _ in prompts]

    def likelihoods(self, prompt_and_targets: list[tuple[str, str]]) -> list[float]:
        return [0] * len(prompt_and_targets)

    @property
    def can_do_lm(self) -> bool:
        return True

    @property
    def can_do_nlg(self) -> bool:
        return True
