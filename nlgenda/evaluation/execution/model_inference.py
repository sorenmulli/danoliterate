import json
import logging
import os
from abc import ABC, abstractmethod
from typing import Optional

import openai
import torch
from transformers import AutoModelForCausalLM, pipeline

from nlgenda.modeling.load_model import from_pretrained_hf_hub_no_disk

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

logger = logging.getLogger(__name__)


class WrongInferenceError(RuntimeError):
    ...


class ModelInference(ABC):
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

    def __init__(self, hf_key: str, download_no_cache=True):
        model_cls = AutoModelForCausalLM
        model = (
            from_pretrained_hf_hub_no_disk(hf_key, model_cls)
            if download_no_cache
            else model_cls.from_pretrained(hf_key)
        )
        self.pipeline = pipeline("text-generation", model=model, device=DEVICE, tokenizer=hf_key)

    def generate_text(self, prompt: str) -> str:
        return self.pipeline(prompt)[0]["generated_text"][len(prompt) :]

    def likelihood(self, prompt: str, target: str) -> float:
        encodings = self.pipeline.tokenizer(prompt, text_target=target, return_tensors="pt")

        input_ids = torch.cat((encodings.input_ids, encodings.labels), dim=1)
        target_ids = input_ids.clone()

        # Ignore prompt in likelihood computation
        target_ids[:, : encodings.input_ids.size(1)] = self.ignore_target_idx

        with torch.no_grad():
            outputs = self.pipeline.model(input_ids.to(DEVICE), labels=target_ids.to(DEVICE))

        # Loss is negative log likelihood so convert to likelihood
        return torch.exp(-outputs.loss).item()

    @property
    def can_do_lm(self) -> bool:
        return True

    @property
    def can_do_nlg(self) -> bool:
        return True


# TODO: Default to temperature = 0
# TODO: Save maximal info from the result
# TODO: Make tenacious
class OpenAiAPI(ModelInference):
    secret_file = "secret.json"
    api_key_str = "OPENAI_API_KEY"

    def __init__(self, model_key: str, api_key: Optional[str] = None):
        self.model_key = model_key
        if not api_key:
            api_key = os.getenv(self.api_key_str)
        if not api_key:
            if os.path.isfile(self.secret_file):
                with open(self.secret_file, "r", encoding="utf-8") as file:
                    api_key = json.load(file).get(self.api_key_str)
        if not api_key:
            logger.error(
                "Not given API key and did not find %s in env or in %s",
                self.api_key_str,
                self.secret_file,
            )
        openai.api_key = api_key

    def generate_text(self, prompt: str) -> str:
        if "turbo" in self.model_key or "gpt-4" in self.model_key:
            completion = openai.ChatCompletion.create(
                model=self.model_key, messages=[{"role": "user", "content": prompt}]
            )
            return completion.choices[0].message.content
        completion = openai.Completion.create(model=self.model_key, prompt=prompt)
        return completion.choices[0].text

    @property
    def can_do_lm(self) -> bool:
        return False

    @property
    def can_do_nlg(self) -> bool:
        return True
