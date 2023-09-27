from abc import ABC, abstractmethod
from enum import Enum

import torch
from transformers import pipeline


class WrongInferenceError(RuntimeError):
    ...


class InferenceMethod(Enum):
    LM = "lm"
    NLG = "nlg"


class ModelInference(ABC):
    @property
    @abstractmethod
    def can_do_lm(self) -> bool:
        ...

    @property
    @abstractmethod
    def can_do_nlg(self) -> bool:
        ...

    @property
    @abstractmethod
    def inference_method(self) -> InferenceMethod:
        ...

    # These are pseudo abstract methods as at least one of them should be overwritten
    # so I keep the arguments named
    # pylint: disable=unused-argument
    def generate_text(self, prompt: str) -> str:
        if not self.can_do_nlg:
            raise WrongInferenceError(
                f"Cannot generate text with {self} which does not support NLG."
            )
        raise NotImplementedError(f"{type(self)} is missing implementation of `generate_text`.")

    # pylint: disable=unused-argument
    def likelihood(self, prompt: str, target: str) -> float:
        if not self.can_do_lm:
            raise WrongInferenceError(
                f"Cannot calculate likelihood with {self} which does not support LM."
            )
        raise NotImplementedError(f"{type(self)} is missing implementation of `likelihood`.")


class HuggingfaceCausalLm(ModelInference):
    ignore_target_idx = -100

    def __init__(self, inference_method: str, hf_key: str):
        self._inference_method = InferenceMethod(inference_method)
        self.pipeline = pipeline("text-generation", model=hf_key)

    @property
    def can_do_lm(self) -> bool:
        return True

    @property
    def can_do_nlg(self) -> bool:
        return True

    @property
    def inference_method(self) -> InferenceMethod:
        return self._inference_method

    def generate_text(self, prompt: str) -> str:
        return self.pipeline(prompt)[0]["generated_text"][len(prompt) :]

    def likelihood(self, prompt: str, target: str) -> float:
        encodings = self.pipeline.tokenizer(prompt, text_target=target, return_tensors="pt")

        input_ids = torch.cat((encodings.input_ids, encodings.labels), dim=1)
        target_ids = input_ids.clone()

        # Ignore prompt in likelihood computation
        target_ids[:, : encodings.input_ids.size(1)] = self.ignore_target_idx

        with torch.no_grad():
            outputs = self.pipeline.model(input_ids, labels=target_ids)

        # Loss is negative log likelihood so convert to likelihood
        return torch.exp(-outputs.loss).item()
