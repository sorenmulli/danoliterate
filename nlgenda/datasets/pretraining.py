from functools import partial

from datasets import IterableDataset, interleave_datasets, load_dataset
from omegaconf import DictConfig
from transformers import PreTrainedTokenizerBase


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
