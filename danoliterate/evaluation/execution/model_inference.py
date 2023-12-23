import json
import os
import random
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Optional

import openai
import torch
from openai.openai_object import OpenAIObject
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.tokenization_utils_base import VERY_LARGE_INTEGER

from danoliterate.evaluation.results import ExecutionExample
from danoliterate.infrastructure.logging import logger
from danoliterate.modeling.load_model import from_pretrained_hf_hub_no_disk

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
            if (max_len := getattr(self.model.config, candidate_key, None)) is not None:
                return max_len
        logger.warning(
            "Could not detect model max length for %s, defaulting to %i",
            self.model,
            self.default_max_length,
        )
        return self.default_max_length

    def generate_texts(self, prompts: list[str]) -> list[tuple[str, Optional[float]]]:
        out: list[tuple[str, Optional[float]]] = []
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
                        return_token_type_ids=False,
                    ).to(DEVICE)
                    with torch.inference_mode():
                        outputs = self.model.generate(
                            **model_inputs,
                            max_new_tokens=self.max_new_tokens,
                            do_sample=False,
                            temperature=0,
                            return_dict_in_generate=True,
                            output_scores=True,
                        )
                    input_id_len = model_inputs.input_ids.shape[1]
                    texts = self.tokenizer.batch_decode(
                        outputs.sequences[:, input_id_len:],
                        skip_special_tokens=True,
                    )
                    scores = self.model.compute_transition_scores(
                        outputs.sequences, outputs.scores, normalize_logits=True
                    ).mean(axis=1)
                    out.extend(
                        (text, float(score)) for text, score in zip(texts, scores, strict=True)
                    )
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


class OpenAiAPI(ModelInference):
    cache: dict[str, dict]
    is_chat: bool

    secret_file = "secret.json"
    api_key_str = "OPENAI_API_KEY"
    api_retries = 5

    def __init__(self, model_key: str, api_call_cache: str, api_key: Optional[str] = None, seed=1):
        super().__init__()

        self.model_key = model_key
        self.is_chat = "turbo" in self.model_key or "gpt-4" in self.model_key
        self.seed = seed

        if not api_key:
            api_key = os.getenv(self.api_key_str)
        if not api_key:
            if Path(self.secret_file).is_file():
                with open(self.secret_file, "r", encoding="utf-8") as file:
                    api_key = json.load(file).get(self.api_key_str)
        if not api_key:
            logger.error(
                "Not given API key and did not find %s in env or in %s",
                self.api_key_str,
                self.secret_file,
            )
        openai.api_key = api_key

        self.load_cache(api_call_cache)
        self.completion_args = {
            "seed": seed,
            "temperature": 0,
            # TODO: Should be set at scenario level
            "max_tokens": 256,
        }

    def generate_texts(self, prompts: list[str]) -> list[tuple[str, Optional[float]]]:
        for prompt in tqdm(prompts):
            if prompt in self.cache:
                continue
            completion = self.call_completion(prompt)
            self.cache_add(prompt, completion)

        out: list[tuple[str, Optional[float]]] = []
        for prompt in prompts:
            generated_dict = self.cache[prompt]
            answer = generated_dict["choices"][0]
            generated = answer["message"]["content"] if self.is_chat else answer["text"]
            # We have no scores from API
            out.append((generated, None))
        return out

    def call_completion(self, prompt: str):
        for i in range(self.api_retries):
            try:
                if self.is_chat:
                    return openai.ChatCompletion.create(
                        model=self.model_key,
                        messages=[{"role": "user", "content": prompt}],
                        **self.completion_args,
                    )
                return openai.Completion.create(
                    model=self.model_key,
                    prompt=prompt,
                    **self.completion_args,
                )
            except (
                openai.error.ServiceUnavailableError,
                openai.error.APIError,
                openai.error.RateLimitError,
                openai.error.Timeout,
                openai.error.APIConnectionError,
                openai.error.TryAgain,
            ) as openai_error:
                if i + 1 == self.api_retries:
                    logger.error("Retried %i times, failed to get connection.", self.api_retries)
                    raise
                retry_time = i + 1
                logger.warning(
                    "Got connectivity error %s, retrying in %i seconds...", openai_error, retry_time
                )
                time.sleep(retry_time)

    @property
    def can_do_lm(self) -> bool:
        return False

    @property
    def can_do_nlg(self) -> bool:
        return True

    def cache_add(self, prompt: str, completion: OpenAIObject):
        completion_dict = completion.to_dict_recursive()
        with open(self.api_call_cache, "a", encoding="utf-8") as file:
            file.write(json.dumps({"prompt": prompt, "completion": completion_dict}) + "\n")
        self.cache[prompt] = completion_dict

    def load_cache(self, location):
        self.cache = {}
        self.api_call_cache = Path(location) / f"{self.model_key}.json"
        if self.api_call_cache.exists():
            with open(self.api_call_cache, "r", encoding="utf-8") as file:
                for line in file.readlines():
                    result = json.loads(line)
                    self.cache[result["prompt"]] = result["completion"]
            logger.info(
                "Loaded %i results from cache %s. Delete file to recompute.",
                len(self.cache),
                self.api_call_cache,
            )
        else:
            self.api_call_cache.parent.mkdir(parents=True, exist_ok=True)
