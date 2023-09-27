from abc import ABC
from dataclasses import dataclass
from typing import Optional

# TODO: Consider dataclass inheritance


@dataclass
class EvaluationExample(ABC):
    prompt: str
    id_: str

    index_label: Optional[int] = None
    options: Optional[list[str]] = None

    generated_text: Optional[str] = None
    index_prediction: Optional[int] = None

    options_model_scores: Optional[list[float]] = None
