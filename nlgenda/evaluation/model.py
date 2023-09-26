import logging
from enum import Enum
from typing import Generator

from transformers import AutoModelForCausalLM, AutoTokenizer

from nlgenda.evaluation.example import EvaluationExample, MultichoiceExample
from nlgenda.modeling.large_language_modeling import perplexity

logger = logging.getLogger(__name__)


class InferenceType(Enum):
    HUGGINGFACE = "hf"


class MultichoicePredictType(Enum):
    LIKELIHOOD = "likelihood"
    OUTPUT_ROUGE1 = "output-rouge1"


class EvaluatorModel:
    def __init__(self, name: str, path: str, inference_type: str, multichoice_predict: str):
        self.name = name
        self.path = path
        self.inference_type = InferenceType(inference_type)
        self.multichoice_predict = MultichoicePredictType(multichoice_predict)

        match self.inference_type:
            case InferenceType.HUGGINGFACE:
                self.model = AutoModelForCausalLM.from_pretrained(self.path)
                self.tokenizer = AutoTokenizer.from_pretrained(self.path)

    def argmin_perplexity(self, example: MultichoiceExample) -> int:
        assert self.inference_type == InferenceType.HUGGINGFACE
        perplexities = [
            perplexity(example.prompt, option, self.tokenizer, self.model)
            for option in example.options
        ]
        return perplexities.index(min(perplexities))

    def predict_multichoice_example(self, example: MultichoiceExample) -> MultichoiceExample:
        match self.multichoice_predict:
            case MultichoicePredictType.LIKELIHOOD:
                example.prediction = self.argmin_perplexity(example)
                return example

            case MultichoicePredictType.OUTPUT_ROUGE1:
                raise NotImplementedError

    def generate_results(
        self, examples: Generator[EvaluationExample, None, None]
    ) -> Generator[EvaluationExample, None, None]:
        # TODO: Allow batching
        for example in examples:
            if isinstance(example, MultichoiceExample):
                yield self.predict_multichoice_example(example)
            else:
                logger.error("Example class %s unknown for evaluator model", str(type(example)))
                raise ValueError("Example was unknown class")
