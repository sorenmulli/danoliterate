import random
import warnings
from functools import partial

import torch
from datasets import IterableDataset, interleave_datasets, load_dataset
from omegaconf import DictConfig
from transformers import PreTrainedTokenizerBase
from trl.trainer import ConstantLengthDataset


def get_streaming_data(cfg: DictConfig) -> dict[str, IterableDataset]:
    data_paths = []
    for data_path in cfg.datasets:
        name = None
        if ":" in data_path:
            data_path, name = data_path.split(":")
        data_paths.append(load_dataset(data_path, name=name, split="train", streaming=True))

    dataset = interleave_datasets(data_paths).shuffle(seed=cfg.seed).select_columns(cfg.text_col)
    datasets = {
        "test": dataset.take(cfg.test_examples),
        "val": dataset.skip(cfg.test_examples).take(cfg.validation_examples),
        "train": dataset.skip(cfg.test_examples + cfg.validation_examples),
    }
    return datasets


def tokenize_batch(
    examples: dict[str, list[str]], tokenizer: PreTrainedTokenizerBase, cfg: DictConfig
) -> dict[str, list[int]]:
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokens = tokenizer(
        examples[cfg.text_col],
        truncation=True,
        max_length=cfg.context_tokens,
        return_overflowing_tokens=True,
        return_length=True,
    )
    return {
        "input_ids": [
            input_ids
            for length, input_ids in zip(tokens["length"], tokens["input_ids"])
            if length == cfg.context_tokens
        ]
    }


def tokenize_datasets(
    datasets: dict[str, IterableDataset], tokenizer: PreTrainedTokenizerBase, cfg: DictConfig
) -> dict[str, IterableDataset]:
    map_func = partial(tokenize_batch, tokenizer=tokenizer, cfg=cfg.train.data)
    return {
        split: dataset.map(map_func, batched=True, remove_columns=dataset.column_names)
        for split, dataset in datasets.items()
    }


# pylint: disable=abstract-method
class ConstantLengthDatasetRandomSubsequence(ConstantLengthDataset):
    def __init__(self, *args, one_seq_per_example: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self.one_seq_per_example = one_seq_per_example

    # Mostly a copy from parent class
    def __iter__(self):
        iterator = iter(self.dataset)
        more_examples = True
        while more_examples:
            buffer, buffer_len = [], 0
            while True:
                if buffer_len >= self.max_buffer_size:
                    break
                try:
                    buffer.append(self.formatting_func(next(iterator)))
                    buffer_len += len(buffer[-1])
                except StopIteration:
                    if self.infinite:
                        iterator = iter(self.dataset)
                        warnings.warn(
                            "The dataset reached end and the iterator is reset to the start."
                        )
                    else:
                        more_examples = False
                        break
            tokenized_inputs = self.tokenizer(buffer, truncation=False)["input_ids"]
            all_token_ids = []
            examples = []
            for tokenized_input in tokenized_inputs:
                # Below is change implemented: Take a random subsequence for each example
                if self.one_seq_per_example and len(tokenized_input) > (self.seq_length - 1):
                    random_start = random.randint(0, len(tokenized_input) - (self.seq_length - 1))
                    tokenized_input = tokenized_input[
                        random_start : random_start + self.seq_length - 1
                    ]
                all_token_ids.extend(tokenized_input + [self.concat_token_id])
            for i in range(0, len(all_token_ids), self.seq_length):
                input_ids = all_token_ids[i : i + self.seq_length]
                if len(input_ids) == self.seq_length:
                    examples.append(input_ids)
            if self.shuffle:
                random.shuffle(examples)
            for example in examples:
                self.current_size += 1
                yield {
                    "input_ids": torch.LongTensor(example),
                    "labels": torch.LongTensor(example),
                }
