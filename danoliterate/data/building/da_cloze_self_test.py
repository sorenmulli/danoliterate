from pathlib import Path

import pandas as pd
from bs4 import BeautifulSoup
from datasets import Dataset
from omegaconf import DictConfig

from danoliterate.data.building.hub import push
from danoliterate.infrastructure.logging import format_config, logger

ENCODING_ERROR_FIXES = {
    "Ã¦": "æ",
    "Ã¸": "ø",
    "Ã¥": "å",
    "â��": "'",
}


def parse_cloze_tests(path: str):
    cloze_tests = []
    with open(path, "r", encoding="utf-8") as file:
        for text in file.readlines():
            for wrong, fix in ENCODING_ERROR_FIXES.items():
                text = text.replace(wrong, fix)
            cloze_tests.append(text)
    examples: list[tuple[str, list[str]]] = []
    for cloze_test in cloze_tests:
        soup = BeautifulSoup(cloze_test, "html.parser")
        answers = soup.find_all("span", class_=lambda cla: "answer" in cla)
        answer_tuples: dict[str, list[str]] = {}
        for answer in answers:
            identifier = answer.get("onclick").split(",")[1].strip(")")
            text = answer.get_text()
            if identifier not in answer_tuples:
                answer_tuples[identifier] = []
            answer_tuples[identifier].append(text)
        cloze_items = soup.find_all("span", class_="item")
        for i, item in enumerate(cloze_items):
            item.replace_with(f"{{{i}}}")
        examples.append((soup.get_text().strip(), list(answer_tuples.values())))  # type: ignore
    return examples


def save_for_prediction(tests: list[tuple[str, list[str]]], out_path: Path):
    rows: list[dict[str, str | int]] = []
    for i, (text, clozes) in enumerate(tests):
        for j, options in enumerate(clozes):
            rows.append(
                {
                    "text-idx": i,
                    "cloze-idx": j,
                    "text": text,
                    "correct": "",
                    **dict(enumerate(options)),  # type: ignore
                }
            )
    pd.DataFrame(rows).to_csv(out_path, index=False)


def build_with_predictions(tests: list[tuple[str, list[str]]], pred_path: Path):
    pred_df = pd.read_csv(pred_path, index_col=None)
    examples = []
    for i, (text, clozes) in enumerate(tests):
        text_df = pred_df[pred_df["text-idx"] == i]
        for j, options in enumerate(clozes):
            text = text_df.iloc[j].text.format(
                *[
                    (text_df.iloc[cloze_idx][str(correct)] if cloze_idx != j else "{cloze}")
                    for cloze_idx, correct in zip(
                        text_df["cloze-idx"], text_df["correct"], strict=True
                    )
                ]
            )
            examples.append(
                {
                    "text-idx": i,
                    "cloze-idx": j,
                    "text": text,
                    "correct": text_df.iloc[j]["correct"],
                    **{f"option-{i}": option for i, option in enumerate(options)},
                }
            )
    return pd.DataFrame(examples)


def create_da_cloze_self_test(cfg: DictConfig):
    logger.debug("Running with arguments: %s", format_config(cfg))
    tests = parse_cloze_tests(cfg.databuild.html_list_file)
    save_for_prediction(tests, out_path := Path(cfg.databuild.prediction.todo_file))
    logger.info("Saved for prediction to %s", out_path)
    if (done_path := Path(cfg.databuild.prediction.done_file)).exists():
        logger.info("Reading prediction from %s", done_path)
        df = build_with_predictions(tests, done_path)
        dataset = Dataset.from_pandas(df)
        if cfg.databuild.hub.push:
            push(dataset, cfg.databuild.hub)
