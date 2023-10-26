import json
import logging
import os
import random
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, fields
from typing import Optional

import openai
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.tokenization_utils_base import VERY_LARGE_INTEGER

from nlgenda.evaluation.results import ExecutionExample
from nlgenda.modeling.load_model import from_pretrained_hf_hub_no_disk

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

logger = logging.getLogger(__name__)


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
    # TODO: Should be set at scenario level
    max_new_tokens = 256
    default_max_length = 1024

    # TODO: Allow setting download no cache from config
    def __init__(self, hf_key: str, batch_size=1, download_no_cache=True):
        super().__init__()

        model_cls = AutoModelForCausalLM
        self.model = (
            from_pretrained_hf_hub_no_disk(hf_key, model_cls)
            if download_no_cache
            else model_cls.from_pretrained(hf_key)
        )
        self.model.to(DEVICE)
        self.model.eval()
        self.batch_size = batch_size
        self.tokenizer = AutoTokenizer.from_pretrained(hf_key)
        self.tokenizer.pad_token_id = self.model.config.eos_token_id
        self.tokenizer.padding_side = "left"

    @property
    def model_max_length(self):
        if (max_len := self.tokenizer.model_max_length) < VERY_LARGE_INTEGER:
            return max_len
        for candidate_key in (
            "model_max_length",
            "max_position_embeddings",
            "n_positions",
        ):
            if (max_len := self.model.config.get(candidate_key, None)) is not None:
                return max_len
        logger.warning(
            "Could not detect model max length for %s, defaulting to %i",
            self.model,
            self.default_max_length,
        )
        return self.default_max_length

    def generate_texts(self, prompts: list[str]) -> list[str]:
        # See also
        # https://huggingface.co/docs/transformers/llm_tutorial
        out: list[str] = []
        batch_size = self.batch_size
        pbar = tqdm(total=len(prompts))
        i = 0
        while i < len(prompts):
            batch_completed = False
            while not batch_completed:
                batch = prompts[i : i + batch_size]
                try:
                    # TODO: Extract parameters nicely in a generationconfig
                    model_inputs = self.tokenizer(
                        batch,
                        return_tensors="pt",
                        padding=True,
                        max_length=self.model_max_length - self.max_new_tokens,
                        truncation=True,
                    ).to(DEVICE)
                    with torch.no_grad():
                        generated = self.model.generate(
                            **model_inputs,
                            max_new_tokens=self.max_new_tokens,
                            do_sample=False,
                            temperature=0,
                        )
                    texts = self.tokenizer.batch_decode(generated, skip_special_tokens=True)
                    new_texts = [text[len(prompt) :] for text, prompt in zip(texts, prompts)]
                    out.extend(new_texts)
                    pbar.update(batch_size)
                    batch_completed = True
                    i += batch_size
                except RuntimeError as error:
                    _maybe_raise_oom(error, batch_size)
                    logger.warning(
                        "Batch size %i was too large, lowering to %i", batch_size, batch_size // 2
                    )
                    batch_size = batch_size // 2
        pbar.close()
        return out

    def _compute_likelihoods(self, logits: torch.Tensor, target_ids: torch.Tensor) -> list[float]:
        loss_fn = torch.nn.CrossEntropyLoss(reduction="none", ignore_index=self.ignore_target_idx)
        # Move vocab dimension last as we do classification over these
        batch_logits = logits.permute(0, 2, 1)
        # Shift to do next-token prediction
        target_ids = target_ids[:, 1:]
        batch_logits = batch_logits[:, :, :-1]
        losses = loss_fn(batch_logits, target_ids)
        # Sum losses over the sequence length
        summed_lls = -losses.sum(dim=-1)
        return summed_lls.tolist()

    def likelihoods(self, prompt_and_targets: list[tuple[str, str]]) -> list[float]:
        out: list[float] = []
        batch_size = self.batch_size
        pbar = tqdm(total=len(prompt_and_targets))
        i = 0
        while i < len(prompt_and_targets):
            batch_completed = False
            while not batch_completed:
                batch = prompt_and_targets[i : i + batch_size]
                try:
                    encodings = self.tokenizer(
                        [item[0] for item in batch],
                        text_target=[item[1] for item in batch],
                        return_tensors="pt",
                        padding=True,
                    )

                    input_ids = torch.cat((encodings.input_ids, encodings.labels), dim=1)

                    # Ignore prompt in likelihood computation
                    target_ids = input_ids.clone()
                    target_ids[:, : encodings.input_ids.size(1)] = self.ignore_target_idx

                    # Do hole truncation
                    if (input_size := input_ids.size(1)) > self.model_max_length:
                        logger.warning(
                            "Example was too long: %i tokens > %i max tokens. "
                            "Left-truncating to max tokens.",
                            input_size,
                            self.model_max_length,
                        )
                        input_ids = input_ids[:, -self.model_max_length :]
                        target_ids = target_ids[:, -self.model_max_length :]

                    with torch.no_grad():
                        logits = self.model(input_ids.to(DEVICE)).logits.cpu()
                        out.extend(self._compute_likelihoods(logits, target_ids))

                    batch_completed = True
                    pbar.update(batch_size)
                    i += batch_size
                except RuntimeError as error:
                    _maybe_raise_oom(error, batch_size)
                    logger.warning(
                        "Batch size %i was too large, lowering to %i", batch_size, batch_size // 2
                    )
                    batch_size = batch_size // 2
        pbar.close()
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
