# pylint: disable=import-outside-toplevel,undefined-variable
# mypy: disable-error-code="name-defined"
import json
import logging
import os
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Optional

import requests
from tqdm import tqdm

from danoliterate.evaluation.execution.model_inference import ModelInference
from danoliterate.infrastructure.logging import logger


class ApiInference(ModelInference, ABC):
    cache: dict[str, dict]
    secret_file = "secret.json"
    api_retries = 5

    def __init__(self, model_key: str, api_call_cache: str):
        super().__init__()
        self.model_key = model_key
        self.load_cache(api_call_cache)

    def cache_add(self, prompt: str, completion: dict):
        with open(self.api_call_cache, "a", encoding="utf-8") as file:
            file.write(json.dumps({"prompt": prompt, "completion": completion}) + "\n")
        self.cache[prompt] = completion

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

    def generate_texts(self, prompts: list[str]) -> list[tuple[str, Optional[float]]]:
        for prompt in tqdm(prompts):
            if prompt in self.cache:
                continue
            completion = self.call_completion(prompt)
            self.cache_add(prompt, completion)

        out: list[tuple[str, Optional[float]]] = []
        for prompt in prompts:
            out.append(self.extract_answer(self.cache[prompt]))
        return out

    @abstractmethod
    def call_completion(self, prompt: str) -> dict:
        ...

    @abstractmethod
    def extract_answer(self, generated_dict: dict) -> tuple[str, Optional[float]]:
        ...


class OpenAiApi(ApiInference):
    is_chat: bool

    api_key_str = "OPENAI_API_KEY"

    def __init__(self, model_key: str, api_call_cache: str, api_key: Optional[str] = None, seed=1):
        super().__init__(model_key, api_call_cache)
        try:
            import openai
        except ImportError as error:
            raise ImportError("To use OpenaiApi, you must install openai") from error
        self.is_chat = "instruct" not in self.model_key and (
            "turbo" in self.model_key or "gpt-4" in self.model_key
        )
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

        self.completion_args = {
            "seed": seed,
            "temperature": 0,
            # TODO: Should be set at scenario level
            "max_tokens": 256,
        }

    def extract_answer(self, generated_dict: dict) -> tuple[str, Optional[float]]:
        answer = generated_dict["choices"][0]
        # We have no scores from OpenAI API
        return answer["message"]["content"] if self.is_chat else answer["text"], None

    def call_completion(self, prompt: str) -> dict:
        for i in range(self.api_retries):
            try:
                if self.is_chat:
                    completion = openai.ChatCompletion.create(
                        model=self.model_key,
                        messages=[{"role": "user", "content": prompt}],
                        **self.completion_args,
                    )
                else:
                    completion = openai.Completion.create(
                        model=self.model_key,
                        prompt=prompt,
                        **self.completion_args,
                    )
                return completion.to_dict_recursive()
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
        raise ValueError("Retries must be > 0")

    @property
    def can_do_lm(self) -> bool:
        return False

    @property
    def can_do_nlg(self) -> bool:
        return True


class GoogleApi(ApiInference):
    api_retries = 10

    def __init__(self, model_key: str, api_call_cache: str):
        super().__init__(model_key, api_call_cache)
        try:
            import google.auth
            import vertexai

            # pylint: disable=unused-import
            from google.api_core.exceptions import ResourceExhausted
            from google.cloud.aiplatform_v1beta1 import SafetySetting
            from vertexai.generative_models._generative_models import GenerationConfig

            # pylint: disable=unused-import
            from vertexai.preview.generative_models import (
                GenerationResponse,
                GenerativeModel,
                HarmCategory,
            )
        except ImportError as error:
            raise ImportError(
                "To use GoogleApi, you must install google-cloud-aiplatform"
            ) from error
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "google-secret.json"
        creds, project_id = google.auth.default()
        vertexai.init(project=project_id, credentials=creds)
        self.model = GenerativeModel("gemini-pro")
        self.config = GenerationConfig(
            temperature=0.0,
            # TODO: Should be set at scenario level
            max_output_tokens=256,
        )
        self.safety_settings = {
            cat: SafetySetting.HarmBlockThreshold.BLOCK_NONE for cat in HarmCategory
        }

    def extract_answer(self, generated_dict: dict) -> tuple[str, Optional[float]]:
        # Google does not give scores
        return (
            generated_dict["candidates"][0]["text"] if len(generated_dict["candidates"]) else "",
            None,
        )

    def call_completion(self, prompt: str) -> dict:
        for i in range(self.api_retries):
            try:
                completion: "GenerationResponse" = self.model.generate_content(
                    prompt, generation_config=self.config, safety_settings=self.safety_settings
                )
                out: dict[str, Any] = {}
                out["candidates"] = [
                    {"text": _get_google_text(cand), "finish_reason": cand.finish_reason.value}
                    for cand in completion.candidates
                ]
                try:
                    out["usage"] = {
                        # pylint: disable=protected-access
                        key: getattr(completion._raw_response.usage_metadata, key)
                        for key in ("prompt_token_count", "candidates_token_count")
                    }
                except (KeyError, AttributeError) as error:
                    logger.warning("Could not get usage due to error %s", error)
                return out
            except ResourceExhausted as api_error:
                if i + 1 == self.api_retries:
                    logger.error("Retried %i times, failed to get connection.", self.api_retries)
                    raise
                retry_time = i + 4
                logger.warning(
                    "Got connectivity error %s, retrying in %i seconds...", api_error, retry_time
                )
                time.sleep(retry_time)
        raise ValueError("Retries must be > 0")

    @property
    def can_do_lm(self) -> bool:
        return False

    @property
    def can_do_nlg(self) -> bool:
        return True


class DanskGptAPi(ApiInference):
    api_retries = 10

    def extract_answer(self, generated_dict: dict) -> tuple[str, Optional[float]]:
        return (generated_dict.get("text"), None)  # type: ignore

    def call_completion(self, prompt: str) -> dict:
        for i in range(self.api_retries):
            try:
                response = requests.post(self.model_key, data={"prompt": prompt}, timeout=120)
                response.raise_for_status()
                return response.json()
            except requests.exceptions.RequestException as api_error:
                if i + 1 == self.api_retries:
                    logger.error("Retried %i times, failed to get connection.", self.api_retries)
                    raise
                retry_time = i + 1
                logger.warning(
                    "Got connectivity error %s, retrying in %i seconds...", api_error, retry_time
                )
                time.sleep(retry_time)
        raise ValueError("Retries must be > 0")

    @property
    def can_do_nlg(self) -> bool:
        return True

    @property
    def can_do_lm(self) -> bool:
        return False


class AnthropicApi(ApiInference):
    api_retries = 10
    api_key_str = "ANTHROPIC_API_KEY"

    def __init__(self, model_key: str, api_call_cache: str):
        super().__init__(model_key, api_call_cache)
        try:
            import anthropic
        except ImportError as error:
            raise ImportError("To use AnthropicApi, you must install anthropic") from error
        with open(self.secret_file, "r", encoding="utf-8") as file:
            try:
                api_key = json.load(file)[self.api_key_str]
            except KeyError as error:
                raise KeyError(
                    f"Secret file {self.secret_file} lacked Anthropic API key {self.api_key_str}"
                ) from error
        self.client = anthropic.Anthropic(api_key=api_key)
        # Avoid spam
        logging.getLogger("httpx").setLevel(logging.WARNING)

        self.completion_args = {
            "temperature": 0,
            "model": self.model_key,
            # TODO: Should be set at scenario level
            "max_tokens": 256,
        }

    def extract_answer(self, generated_dict: dict) -> tuple[str, Optional[float]]:
        # We have no scores from Anthropic API
        return generated_dict["content"][0]["text"], None

    def call_completion(self, prompt: str) -> dict:
        for i in range(self.api_retries):
            try:
                message = self.client.messages.create(
                    messages=[{"role": "user", "content": prompt}],
                    **self.completion_args,
                )  # type: ignore
                return message.dict()
            except (
                anthropic.APIStatusError,
                anthropic.APITimeoutError,
                anthropic.APIConnectionError,
                anthropic.RateLimitError,
                anthropic.InternalServerError,
            ) as error:
                if i + 1 == self.api_retries:
                    logger.error("Retried %i times, failed to get connection.", self.api_retries)
                    raise
                retry_time = i + 1
                logger.warning(
                    "Got connectivity error %s, retrying in %i seconds...", error, retry_time
                )
                time.sleep(retry_time)
        raise ValueError("Retries must be > 0")

    @property
    def can_do_nlg(self) -> bool:
        return True

    @property
    def can_do_lm(self) -> bool:
        return False


class GroqApi(ApiInference):
    api_key_str = "GROQ_API_KEY"

    def __init__(self, model_key: str, api_call_cache: str):
        super().__init__(model_key, api_call_cache)
        try:
            import groq
        except ImportError as error:
            raise ImportError("To use GroqApi, you must install groq") from error
        with open(self.secret_file, "r", encoding="utf-8") as file:
            try:
                api_key = json.load(file)[self.api_key_str]
            except KeyError as error:
                raise KeyError(
                    f"Secret file {self.secret_file} lacked Groq API key {self.api_key_str}"
                ) from error
        self.client = groq.Groq(api_key=api_key)
        # Avoid spam
        logging.getLogger("httpx").setLevel(logging.WARNING)

        self.completion_args = {
            "temperature": 0,
            "model": self.model_key,
            # TODO: Should be set at scenario level
            "max_tokens": 256,
        }

    def extract_answer(self, generated_dict: dict) -> tuple[str, Optional[float]]:
        # We have no scores from Groq API
        return generated_dict["choices"][0]["message"]["content"], None

    def call_completion(self, prompt: str) -> dict:
        for i in range(self.api_retries):
            try:
                message = self.client.chat.completions.create(
                    messages=[{"role": "user", "content": prompt}],
                    **self.completion_args,
                )  # type: ignore
                return message.dict()
            except (
                groq.APIStatusError,
                groq.APIConnectionError,
                groq.APITimeoutError,
                groq.RateLimitError,
                groq.InternalServerError,
            ) as error:
                if i + 1 == self.api_retries:
                    logger.error("Retried %i times, failed to get connection.", self.api_retries)
                    raise
                retry_time = i + 1
                logger.warning(
                    "Got connectivity error %s, retrying in %i seconds...", error, retry_time
                )
                time.sleep(retry_time)
        raise ValueError("Retries must be > 0")

    @property
    def can_do_nlg(self) -> bool:
        return True

    @property
    def can_do_lm(self) -> bool:
        return False


def _get_google_text(candidate):
    try:
        return candidate.text
    except ValueError as error:
        logger.warning("Could not extract text from candidate %s due to error %s", candidate, error)
        return ""
