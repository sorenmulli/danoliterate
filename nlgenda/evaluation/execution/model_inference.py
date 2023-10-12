import json
import logging
import os
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, fields
from typing import Optional

import openai
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, pipeline

from nlgenda.evaluation.results import ExecutionExample
from nlgenda.modeling.load_model import from_pretrained_hf_hub_no_disk

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

logger = logging.getLogger(__name__)


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

    generated: Optional[str] = None


@dataclass
class QueriedLikelihoodCall(QueriedCall):
    prompt: str
    target: str

    likelihood: Optional[float] = None


class ModelInference(ABC):
    generate_queries: dict[str, QueriedGenerateCall]
    likelihood_queries: dict[str, QueriedLikelihoodCall]

    def __init__(self):
        self.generate_queries = {}
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
    def generate_texts(self, prompts: list[str]) -> list[str]:
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
            for gquery, generation in zip(self.generate_queries.values(), generations):
                gquery.generated = generation

        if self.likelihood_queries:
            likelihoods = self.likelihoods(
                [(query.prompt, query.target) for query in self.likelihood_queries.values()]
            )
            for lquery, likelihood in zip(self.likelihood_queries.values(), likelihoods):
                lquery.likelihood = likelihood

        results = [self._populate_with_answers(example) for example in queried_examples]
        assert not self.generate_queries and not self.likelihood_queries

        return results

    def _populate_with_answers(self, example: ExecutionExample) -> ExecutionExample:
        for field_ in fields(example):
            name = field_.name
            value = getattr(example, name)
            if isinstance(value, QueriedGenerateCall):
                setattr(example, name, self.generate_queries.pop(value.id_))
            elif isinstance(value, QueriedLikelihoodCall):
                setattr(example, name, self.likelihood_queries.pop(value.id_))
            elif isinstance(value, list):
                if all(isinstance(elem, QueriedGenerateCall) for elem in value):
                    setattr(example, name, [self.generate_queries.pop(elem.id_) for elem in value])
                if all(isinstance(elem, QueriedLikelihoodCall) for elem in value):
                    setattr(
                        example, name, [self.likelihood_queries.pop(elem.id_) for elem in value]
                    )
        return example


def _maybe_raise_oom(error: RuntimeError, batch_size: int):
    if "alloc" not in str(error):
        raise error
    if batch_size == 1:
        logger.error("Reached batch size of 1 and still couldn't forward pass batch.")
        raise error


class HuggingfaceCausalLm(ModelInference):
    ignore_target_idx = -100

    def __init__(self, hf_key: str, batch_size=1, download_no_cache=True):
        super().__init__()

        model_cls = AutoModelForCausalLM
        model = (
            from_pretrained_hf_hub_no_disk(hf_key, model_cls)
            if download_no_cache
            else model_cls.from_pretrained(hf_key)
        )
        self.batch_size = batch_size
        self.pipeline = pipeline("text-generation", model=model, device=DEVICE, tokenizer=hf_key)
        self.pipeline.tokenizer.pad_token_id = model.config.eos_token_id

    def generate_texts(self, prompts: list[str]) -> list[str]:
        out: list[str] = []
        batch_size = self.batch_size
        for i in tqdm(range(0, len(prompts), batch_size)):
            batch_completed = False
            while not batch_completed:
                batch = prompts[i : i + batch_size]
                try:
                    out.extend(self.pipeline(batch, batch_size=batch_size))
                    batch_completed = True
                except RuntimeError as error:
                    _maybe_raise_oom(error, batch_size)
                    logger.warning(
                        "Batch size %i was too large, lowering to %i", batch_size, batch_size // 2
                    )
                    batch_size = batch_size // 2
        return out

    def _compute_likelihoods(self, logits: torch.Tensor, target_ids: torch.Tensor) -> list[float]:
        loss_fn = torch.nn.CrossEntropyLoss(reduction='none')

        # Only consider losses from positions that aren't ignored (using the mask)
        active_loss = target_ids != self.ignore_target_idx
        active_logits = logits[active_loss]
        active_labels = target_ids[active_loss]

        losses = loss_fn(active_logits, active_labels)

        # Reshape the losses to be of shape (batch_size, sequence_length)
        losses_reshaped = torch.zeros_like(target_ids, dtype=torch.float)
        losses_reshaped[active_loss] = losses

        # Sum losses over the sequence length
        summed_losses = losses_reshaped.sum(dim=-1)

        # Convert negative log likelihoods to likelihoods
        return torch.exp(-summed_losses).tolist()

    def likelihoods(self, prompt_and_targets: list[tuple[str, str]]) -> list[float]:
        out: list[float] = []
        batch_size = self.batch_size

        for i in tqdm(range(0, len(prompt_and_targets), batch_size)):
            batch_completed = False
            while not batch_completed:
                batch = prompt_and_targets[i : i + batch_size]
                try:
                    encodings = self.pipeline.tokenizer(
                        [item[0] for item in batch],
                        text_target=[item[1] for item in batch],
                        return_tensors="pt",
                        padding=True,
                    )

                    input_ids = torch.cat((encodings.input_ids, encodings.labels), dim=1)

                    # Ignore prompt in likelihood computation
                    target_ids = input_ids.clone()
                    target_ids[:, : encodings.input_ids.size(1)] = self.ignore_target_idx

                    with torch.no_grad():
                        logits = self.pipeline.model(input_ids.to(DEVICE)).logits
                        out.extend(self._compute_likelihoods(logits, target_ids))

                    batch_completed = True
                except RuntimeError as error:
                    _maybe_raise_oom(error, batch_size)
                    logger.warning(
                        "Batch size %i was too large, lowering to %i", batch_size, batch_size // 2
                    )
                    batch_size = batch_size // 2

        return out

    @property
    def can_do_lm(self) -> bool:
        return True

    @property
    def can_do_nlg(self) -> bool:
        return True


# TODO: Default to temperature = 0
# TODO: Save maximal info from the result
# TODO: Make tenacious
# TODO: Threading?
class OpenAiAPI(ModelInference):
    secret_file = "secret.json"
    api_key_str = "OPENAI_API_KEY"

    def __init__(self, model_key: str, api_key: Optional[str] = None):
        super().__init__()

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

    def generate_texts(self, prompts: list[str]) -> list[str]:
        out = []
        for prompt in tqdm(prompts):
            if "turbo" in self.model_key or "gpt-4" in self.model_key:
                completion = openai.ChatCompletion.create(
                    model=self.model_key, messages=[{"role": "user", "content": prompt}]
                )
                return completion.choices[0].message.content
            completion = openai.Completion.create(model=self.model_key, prompt=prompt)
            out.append(completion.choices[0].text)
        return out

    @property
    def can_do_lm(self) -> bool:
        return False

    @property
    def can_do_nlg(self) -> bool:
        return True
