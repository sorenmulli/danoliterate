import pandas as pd
from datasets import Dataset, load_dataset
from omegaconf import DictConfig

from danoliterate.data.building.hub import push
from danoliterate.infrastructure.logging import format_config, logger


def run_extract(cfg: DictConfig, hellaswag: Dataset):
    # Want to translate each ending, create rows for each
    original_df = pd.DataFrame(hellaswag)
    df = original_df.explode("endings")

    # Also a row for translating the context
    na_rows = original_df.copy()
    na_rows["endings"] = pd.NA
    df = pd.concat([df, na_rows])
    df.loc[df["endings"].notnull(), "ctx"] = pd.NA  # type: ignore
    df = df.reset_index(drop=True)
    df["id"] = df["source_id"] + "-" + df["ind"].astype("string")

    # Know whether row was ctx or ending and join rows
    df["type"] = df.apply(lambda row: "ctx" if pd.notna(row["ctx"]) else "endings", axis=1)
    df["english_text"] = df["ctx"].combine_first(df["endings"])

    # Add an empty column "danish_text"
    df["danish_text"] = pd.NA

    # Rearrange columns and sort rows
    df = df[["id", "type", "english_text", "danish_text"]]
    df = df.sort_values(by=["id", "type"], ascending=[True, True])

    df.to_csv(out := cfg.databuild.translation.todo_file, index=False)
    logger.info("Saved extraction to %s", out)


def run_building(cfg: DictConfig, hellaswag: Dataset):
    in_path = cfg.databuild.translation.done_file
    try:
        df = pd.read_csv(in_path)
    except FileNotFoundError:
        logger.warning("Found no translated dataset at %s. Exiting.", in_path)
        return
    # Only need examples with Danish
    df = df.dropna(subset=["danish_text"]).drop(columns=["english_text"])

    # Undo the flattening by matching the context and the four options
    df["ending_num"] = df.groupby("id").cumcount().where(df["type"] == "endings", "ctx")
    df = (
        df.pivot(index="id", columns="ending_num", values="danish_text")
        .rename(columns={"ctx": "ctx", 1: "option-0", 2: "option-1", 3: "option-2", 4: "option-3"})
        .reset_index()
    )

    # Assert that no Danish translations are mssing
    for idx, row in df.iterrows():
        if row[["ctx", "option-0", "option-1", "option-2", "option-3"]].notna().sum() < 5:
            logger.error("Example %i has missing Danish text:\n%s", idx, row)
            raise ValueError("Incomplete Danish text.")

    # Add correct option
    original_df = pd.DataFrame(hellaswag)
    original_df["id"] = original_df["source_id"] + "-" + original_df["ind"].astype("string")
    id_to_label = dict(zip(original_df["id"], original_df["label"]))
    df["correct"] = df.id.apply(id_to_label.get).astype(int)

    # Go back to previous ID and order columns
    df["source_id"], df["ind"] = zip(*df["id"].apply(lambda id_: id_.rsplit("-", 1)).tolist())
    df["ind"] = df["ind"].astype(int)
    df = df.drop(columns=["id"])
    df = df[["ctx"] + [col for col in df if col != "ctx"]]  # type: ignore

    dataset = Dataset.from_pandas(df)
    if cfg.databuild.hub.push:
        push(dataset, cfg.databuild.hub)


def create_hyggeswag(cfg: DictConfig):
    logger.debug("Running with arguments: %s", format_config(cfg))

    logger.info("Extracting examples from English hellaswag.")
    hellaswag = load_dataset("hellaswag", split="validation")
    run_extract(cfg, hellaswag)

    logger.info("Building from translated examples.")
    run_building(cfg, hellaswag)
