import random
import warnings
from functools import partial
from pathlib import Path
from typing import Optional

import torch
from datasets import Dataset, IterableDataset, interleave_datasets, load_dataset, load_from_disk
from omegaconf import DictConfig
from tqdm import tqdm
from transformers import PreTrainedTokenizerBase
from trl.trainer import ConstantLengthDataset

from danoliterate.infrastructure.logging import logger


def get_streaming_data(cfg: DictConfig) -> dict[str, Dataset | IterableDataset]:
    data_paths = []
    for data_path in cfg.datasets:
        name = None
        if ":" in data_path:
            data_path, name = data_path.split(":")
        data_paths.append(load_dataset(data_path, name=name, split="train", streaming=True))

    dataset = interleave_datasets(data_paths, seed=cfg.seed).shuffle(
        seed=cfg.seed, buffer_size=cfg.buffer_size
    )
    if (splits_path := Path(cfg.splits_path)).exists():
        datasets = load_splits(splits_path)
        if cfg.save_splits:
            raise RuntimeError(
                f"Splits already exist at {cfg.splits_path}, "
                "set train.data.save_splits=false or delete it."
            )
    elif cfg.save_splits:
        datasets = create_splits(cfg, dataset, splits_path)
    else:
        raise RuntimeError(
            f"Splits did not exist at {cfg.splits_path}, "
            "set train.data.save_splits=true to create new ones."
        )
    # Make sure that ordering was the same when the splits were generated
    example_iter = iter(dataset)
    i = 0
    while i < 10:
        if datasets["test"][i] != next(example_iter):
            raise ValueError(
                "Difference between order of current iterable dataset and "
                "the dataset used to create the loaded splits. "
                "Risk of training on test! Fix seed or implement explicit test data skipping"
            )
        i += 1

    # Shuffle train without seed to aboid same training order between runs
    datasets["train"] = dataset.skip(cfg.test_examples + cfg.validation_examples).shuffle(
        buffer_size=cfg.buffer_size
    )
    return datasets


def create_splits(
    cfg: DictConfig, dataset: IterableDataset, splits_path: Path
) -> dict[str, Dataset]:
    split_datasets = {
        "test": dataset.take(cfg.test_examples),
        "val": dataset.skip(cfg.test_examples).take(cfg.validation_examples),
    }
    out_datasets = {}
    for split, iterable_dataset in split_datasets.items():
        out_datasets[split] = Dataset.from_list(
            list(tqdm(iterable_dataset, desc=f"Generating split={split!r}"))
        )
        out_datasets[split].save_to_disk(out := splits_path / split)
        logger.info("Saved dataset split %s to %s", split, out)
    return out_datasets


def load_splits(path: Path) -> dict[str, Dataset]:
    datasets = {}
    for split in "test", "val":
        datasets[split] = load_from_disk(out := path / split)
        logger.info("Loaded dataset split %s from %s", split, out)
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
    def __init__(
        self,
        *args,
        one_seq_per_example: bool = False,
        save_data_debug: Optional[Path] = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.one_seq_per_example = one_seq_per_example
        self.save_data_debug = save_data_debug

    # pylint: disable=too-many-branches
    def __iter__(self):  # This function is mostly a copy from parent class
        iterator = iter(self.dataset)
        more_examples = True
        while more_examples:
            buffer, buffer_len = [], 0
            while True:
                if buffer_len >= self.max_buffer_size:
                    break
                try:
                    data_sample = next(iterator)
                    formatted_sample = self.formatting_func(data_sample)
                    # Skip empty strings
                    if not formatted_sample:
                        continue
                    buffer.append(formatted_sample)
                    if self.save_data_debug is not None:
                        with open(self.save_data_debug, "a", encoding="utf-8") as file:
                            file.write(str(hash(formatted_sample)) + "\n")
                    buffer_len += len(formatted_sample)
                except StopIteration:
                    if self.infinite:
                        iterator = iter(self.dataset)
                        warnings.warn(
                            "The dataset reached end and the iterator is reset to the start."
                        )
                    else:
                        more_examples = False
                        break
            if self.one_seq_per_example:
                for i, text in enumerate(buffer):
                    # Computational hack to avoid tokenizing long examples
                    # that would just in a moment be subsampled
                    if len(text) > (max_len := self.seq_length * 5):
                        random_start = random.randint(0, len(text) - max_len)
                        buffer[i] = text[random_start : random_start + max_len]

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
