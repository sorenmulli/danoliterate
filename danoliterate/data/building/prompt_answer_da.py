import json
from collections import defaultdict
from typing import DefaultDict, Optional

import pandas as pd
from datasets import Dataset, DatasetDict, concatenate_datasets, load_dataset
from omegaconf import DictConfig
from tqdm import tqdm

from danoliterate.data.building.hub import push
from danoliterate.data.statistics import text_stats
from danoliterate.infrastructure.logging import format_config, logger
from danoliterate.modeling.language_identification import DANISH, LanguageIdentifier


def deduplicate(dataset: Dataset) -> Dataset:
    text_to_example_id: DefaultDict[str, Optional[str]] = defaultdict(lambda: None)

    def _map_function(examples):
        for idx, (prompt, answer) in enumerate(zip(examples["prompt"], examples["answer"])):
            key = prompt + answer
            if text_to_example_id[key] is None:
                text_to_example_id[key] = examples["id"][idx]
        return examples

    dataset = dataset.map(_map_function, batched=True)

    def _filter_function(example):
        return example["id"] == text_to_example_id[example["prompt"] + example["answer"]]

    deduplicated_dataset = dataset.filter(_filter_function)
    logger.info("Removed %i during deduplication", len(dataset) - len(deduplicated_dataset))
    return deduplicated_dataset


def normalise_zetavg_processed(cfg: DictConfig, dataset: DatasetDict) -> Dataset:
    logger.info("Original dataset stats:\n%s", dataset["train"])
    logger.info(
        "Conversations marked as Danish: %s",
        len(dataset["train"].filter(lambda example: example["lang"] == DANISH)),
    )

    li_model = LanguageIdentifier(cfg)

    new_dataset: dict[str, list[str]] = {
        "prompt": [],
        "answer": [],
        "source": [],
        "id": [],
    }
    missing_start = 0
    wrong_form = 0
    wrong_language = 0

    for conversation in tqdm(dataset["train"]):
        # Extract the first two messages: Prompt and answer
        if len(conversation["conversations"]) < 2:
            missing_start += 1
            continue
        if not (
            (human := conversation["conversations"][0])["from"] == "human"
            and (gpt := conversation["conversations"][1])["from"] == "gpt"
        ):
            wrong_form += 1
            continue
        if not li_model.predict(human["markdown"]) == li_model.predict(gpt["markdown"]) == DANISH:
            wrong_language += 1
            continue

        new_dataset["prompt"].append(human["markdown"])
        new_dataset["answer"].append(gpt["markdown"])
        new_dataset["source"].append("sharegpt")
        new_dataset["id"].append(conversation["id"])

    logger.info(
        "%i examples did not follow prompt/answer structure. "
        "%i examples did not have human first then GPT. "
        "%i examples had wrong language.",
        missing_start,
        wrong_form,
        wrong_language,
    )
    return Dataset.from_pandas(pd.DataFrame(new_dataset))


def normalise_oastt1(_: DictConfig, dataset_dict: DatasetDict) -> Dataset:
    dataset = concatenate_datasets([dataset_dict["train"], dataset_dict["validation"]])
    logger.info("Original dataset stats:\n%s", dataset)

    da_messages = dataset.filter(lambda example: example["lang"] == DANISH)
    logger.info("Messages marked as Danish: %s", len(da_messages))

    new_dataset: dict[str, list[str]] = {
        "prompt": [],
        "answer": [],
        "source": [],
        "id": [],
    }

    message_dict = {message["message_id"]: message for message in da_messages}
    for message_id, message in message_dict.items():
        if message["parent_id"] is None:
            continue
        if (parent := message_dict.get(message["parent_id"])) is None:
            continue

        new_dataset["prompt"].append(parent["text"])
        new_dataset["answer"].append(message["text"])
        new_dataset["source"].append("oastt1")
        new_dataset["id"].append(message_id)

    return Dataset.from_pandas(pd.DataFrame(new_dataset))


NORMALISERS = {
    "zetavg/ShareGPT-Processed": normalise_zetavg_processed,
    "OpenAssistant/oasst1": normalise_oastt1,
}


def describe_da_dataset(dataset: Dataset):
    logger.info(
        "Created dataset with statistics:\n%s",
        json.dumps(text_stats(dataset, ["prompt", "answer"]), indent=4),
    )


def create_prompt_answer_da(cfg: DictConfig):
    logger.debug("Running with arguments: %s", format_config(cfg))
    logger.info("Creating prompt-answer-da from sources: %s", ", ".join(NORMALISERS.keys()))

    da_datasets = []
    for source, normaliser in NORMALISERS.items():
        logger.info("Processing %s.", source)
        source_dataset = load_dataset(source)
        da_dataset = normaliser(cfg, source_dataset)
        describe_da_dataset(da_dataset)
        da_datasets.append(da_dataset)

    logger.info("Combining datasets.")
    combined_da_dataset = deduplicate(concatenate_datasets(da_datasets))
    combined_da_dataset = combined_da_dataset.shuffle(1887)
    describe_da_dataset(combined_da_dataset)
    if cfg.databuild.hub.push:
        push(combined_da_dataset, cfg.databuild.hub)
