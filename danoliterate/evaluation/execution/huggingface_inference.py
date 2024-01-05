from typing import Optional

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.tokenization_utils_base import VERY_LARGE_INTEGER

from danoliterate.evaluation.execution.model_inference import ModelInference
from danoliterate.infrastructure.logging import logger
from danoliterate.modeling.load_model import from_pretrained_hf_hub_no_disk

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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

    def __init__(self, hf_key: str, batch_size=1, auto_device_map=False, download_no_cache=False):
        super().__init__()

        init_kwargs = {}
        if auto_device_map:
            init_kwargs["device_map"] = "auto"
        model_cls = AutoModelForCausalLM
        self.model = (
            from_pretrained_hf_hub_no_disk(hf_key, model_cls)
            if download_no_cache
            else model_cls.from_pretrained(hf_key, **init_kwargs)
        )
        if not auto_device_map:
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
                    if self.tokenizer.chat_template is not None:
                        batch = self.apply_chat_template(batch)
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
        torch.cuda.empty_cache()
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
        torch.cuda.empty_cache()
        return out

    def apply_chat_template(self, texts: list[str]) -> list[str]:
        return [
            self.tokenizer.apply_chat_template(
                [{"role": "user", "content": text}], tokenize=False, add_generation_prompt=True
            )
            for text in texts
        ]

    @property
    def can_do_lm(self) -> bool:
        return True

    @property
    def can_do_nlg(self) -> bool:
        return True
