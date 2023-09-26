from abc import ABC
from dataclasses import dataclass
from typing import Optional


class EvaluationExample(ABC):
    ...


@dataclass
class MultichoiceExample(EvaluationExample):
    prompt: str
    options: list[str]
    label: int

    prediction: Optional[int] = None
