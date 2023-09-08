import os

from datasets import Dataset


def text_stats(dataset: Dataset, text_cols: list[str]) -> dict[str, int]:
    def _count(example: dict[str, str | int]) -> dict[str, str | int]:
        text = " ".join(str(example[text_col]) for text_col in text_cols)
        example["chars"] = len(text)
        example["words"] = len(text.split())
        return example

    with_counts = dataset.map(_count, num_proc=os.cpu_count())
    return {
        "num_examples": len(dataset),
        "num_characters": sum(with_counts["chars"]),
        "num_words": sum(with_counts["words"]),
    }
