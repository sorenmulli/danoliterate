import os

import matplotlib.pyplot as plt
import pandas as pd
from datasets import Dataset, load_dataset
from tqdm import tqdm


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


def hyggeswag_categories(side_effects=True):
    source_dataset = load_dataset("hellaswag", split="validation")
    dataset = load_dataset("sorenmulli/hyggeswag", split="train")
    activities = {(row["source_id"], row["ind"]): row["activity_label"] for row in source_dataset}
    activity_labels = []
    for row in tqdm(dataset):
        key = (row["source_id"], row["ind"])
        activity_label = activities[key]
        activity_labels.append(activity_label)
    df = pd.DataFrame(dataset)
    df["activity"] = activity_labels
    if side_effects:
        num_unique_activities = len(df["activity"].unique())
        print(f"Number of unique activities: {num_unique_activities}")

        top_activities = df["activity"].value_counts()
        print(f"Top activities:\n{top_activities.head()}")

        top_activities.head(10).plot(kind="bar")
        plt.show()
    return df
