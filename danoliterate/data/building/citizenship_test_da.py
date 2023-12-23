import re
import time
from dataclasses import dataclass
from functools import partial
from io import BytesIO
from typing import Match, Optional

import pandas as pd
import requests
from datasets import Dataset, load_dataset
from omegaconf import DictConfig
from pypdf import PdfReader

from danoliterate.data.building.hub import push
from danoliterate.infrastructure.logging import format_config, logger
from danoliterate.modeling.ngram_language_modeling import NgramLm

QUESTION_PATTERN = r"^\d+\."
ANSWER_PATTERN = r"^[A-Z]:"
LINEBREAK_PATTERN = r"(\w)-\s"

TOFIX_COLS = ["wrong", "fixed", "score_diff"]


@dataclass
class CitizenshipTest:
    name: str
    url: str
    correct_url: str


# SEE https://siri.dk/nyheder/?categorizations=9115

HEADER_REMOVE_LINES = "Indfødsretspr øven", "Medborgerskabsprøven", "Indfødsretsprøven"
FIRST_QUESTIONPAGE = 2

TESTS = [
    CitizenshipTest(
        "Indfødsretsprøven Maj 2023",
        "https://siri.dk/media/9945/indfoedsretsproeven-2023-05.pdf",
        "https://siri.dk/media/9946/indfoedsretsproeven-2023-05-retteark.pdf",
    ),
    CitizenshipTest(
        "Medborgerskabsprøven Maj 2023",
        "https://siri.dk/media/9947/medborgerskabsproeven-2023-05.pdf",
        "https://siri.dk/media/9948/medborgerskabsproeven-2023-05-retteark.pdf",
    ),
    CitizenshipTest(
        "Indfødsretsprøven November 2022",
        "https://siri.dk/media/9901/indfoedsretsproeven-2022-11.pdf",
        "https://siri.dk/media/9902/indfoedsretsproeven-2022-11-retteark.pdf",
    ),
    CitizenshipTest(
        "Medborgerskabsprøven November 2022",
        "https://siri.dk/media/9904/medborgerskabsproeven-2022-11.pdf",
        "https://siri.dk/media/9905/medborgerskabsproeven-2022-11-retteark.pdf",
    ),
    CitizenshipTest(
        "Infødsretsprøven Juni 2022",
        "https://siri.dk/media/9551/indfoedsretsproeve-s22.pdf",
        "https://siri.dk/media/9555/retteark_indfoedsretsproeve22.pdf",
    ),
    CitizenshipTest(
        "Medborgerskabsprøven Juni 2022",
        "https://siri.dk/media/9553/medborgerskabsproeven-s22.pdf",
        "https://siri.dk/media/9550/retteark-medborgerskabsproeve-s22.pdf",
    ),
    CitizenshipTest(
        "Indfødsretsprøven November 2021",
        "https://siri.dk/media/9521/indfoedsretsproeven-vinter-2021.pdf",
        "https://siri.dk/media/9527/indfoedsretsproeven-vinterterminen-2021-retteark.pdf",
    ),
    CitizenshipTest(
        "Medborgerskabsprøven November 2021",
        "https://siri.dk/media/9523/medborgerskabsproeven-vinter-2021.pdf",
        "https://siri.dk/media/9522/medborgerskabsproeven-vinter-2021-retteark.pdf",
    ),
    CitizenshipTest(
        "Indfødsretsprøven Juni 2021",
        "https://siri.dk/media/9476/indfoedsretsproeven-sommer-2021.pdf",
        "https://siri.dk/media/9477/indfoedsretsproeven-sommer-2021-retteark.pdf",
    ),
    CitizenshipTest(
        "Medborgerskabsprøven Juni 2021",
        "https://siri.dk/media/9478/medborgerskabsproeven-sommer-2021.pdf",
        "https://siri.dk/media/9479/medborgerskabsproeven-sommer-2021-retteark.pdf",
    ),
    CitizenshipTest(
        "Indfødsretsprøven November 2020",
        "https://siri.dk/media/9480/indfoedsretsproeven-nov-2020.pdf",
        "https://siri.dk/media/9481/indfoedsretsproeven-nov-2020-retteark.pdf",
    ),
    CitizenshipTest(
        "Medborgerskabsprøven November 2020",
        "https://siri.dk/media/9485/medborgerskabsproeven-vinter-2020.pdf",
        "https://siri.dk/media/9486/medborgerskabsproeven-vinter-2020-retteark.pdf",
    ),
    CitizenshipTest(
        "Indfødsretsprøven Juni 2020",
        "https://siri.dk/media/9490/indfoedsretsproeven-juni-2020.pdf",
        "https://siri.dk/media/9491/indfoedsretsproeven-juni-2020-retteark.pdf",
    ),
    CitizenshipTest(
        "Medborgerskabsprøven Juni 2020",
        "https://siri.dk/media/9492/medborgerskabsproeven-sommer-2020.pdf",
        "https://siri.dk/media/9493/medborgerskabsproeven-sommer-2020-retteark.pdf",
    ),
    CitizenshipTest(
        "Indfødsretsprøven November 2019",
        "https://siri.dk/media/9497/indfoedsretsproeven-nov-2019.pdf",
        "https://siri.dk/media/9498/indfoedsretsproeven-nov-2019-retteark.pdf",
    ),
    CitizenshipTest(
        "Medborgerskabsprøven November 2019",
        "https://siri.dk/media/9499/medborgeskabsproeven-nov-2019.pdf",
        "https://siri.dk/media/9500/medborgeskabsproeven-nov-2019-retteark.pdf",
    ),
    CitizenshipTest(
        "Indfødsretsprøven Juni 2019",
        "https://siri.dk/media/9501/indfoedsretsproeven-sommer-2019.pdf",
        "https://siri.dk/media/9502/indfoedsretsproeven-sommer-2019-retteark.pdf",
    ),
    CitizenshipTest(
        "Medborgerskabsprøven Juni 2019",
        "https://siri.dk/media/9503/medborgerskabsproeven-sommer-2019.pdf",
        "https://siri.dk/media/9504/medborgerskabsproeven-sommer-2019-retteark.pdf",
    ),
]

RAW_CONFIG_NAME = "raw"
CLEANED_CONFIG_NAME = "default"


def question_match(line: str) -> Optional[Match[str]]:
    return re.match(QUESTION_PATTERN, line)


def answer_match(line: str) -> Optional[Match[str]]:
    return re.match(ANSWER_PATTERN, line)


def get_pdf_content(url: str, retries=3, wait=1) -> BytesIO:
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        return BytesIO(response.content)
    except requests.exceptions.ConnectionError as err:
        if not retries:
            raise err
        logger.warning("Retrying after connection error:\n%s", str(err))
        time.sleep(wait)
        return get_pdf_content(url, retries - 1)


def merge_question_lines(lines: list[str]) -> list[str]:
    merged = []
    for line in lines:
        if question_match(line) or answer_match(line):
            merged.append(line)
        else:
            merged[-1] += " " + line
    return merged


def build_from_lines(lines: list[str]) -> pd.DataFrame:
    examples: list[dict[str, str | int]] = []
    open_example: Optional[dict[str, str | int]] = None
    for line in lines:
        if match := question_match(line):
            if open_example is not None:
                examples.append(open_example)
            open_example = {
                "question": line[match.span()[1] :].strip(),
                "index": int(match.group()[:-1]),
            }
        if match := answer_match(line):
            assert open_example is not None
            open_example[f"option-{match.group()[0]}"] = line[2:].strip()
    assert open_example is not None
    examples.append(open_example)
    examples = sorted(examples, key=lambda ex: ex["index"])
    return pd.DataFrame(examples)


def get_questions_and_answers(test: CitizenshipTest) -> pd.DataFrame:
    reader = PdfReader(get_pdf_content(test.url))
    lines: list[str] = []
    # pylint: disable=not-an-iterable
    for page in reader.pages[FIRST_QUESTIONPAGE:]:
        lines.extend(
            line.strip()
            for line in page.extract_text().split("\n")
            if line.strip() and all(remove_line not in line for remove_line in HEADER_REMOVE_LINES)
        )
    lines = merge_question_lines(lines)
    lines = [" ".join(line.split()) for line in lines]
    return build_from_lines(lines)


def add_correct_options(test: CitizenshipTest, df: pd.DataFrame):
    reader = PdfReader(get_pdf_content(test.correct_url))
    lines: list[str] = []
    for page in reader.pages:
        lines.extend(line.strip() for line in page.extract_text().split("\n") if line.strip())
    indices, letters = zip(
        *[
            (int("".join(filter(str.isdigit, line))), line[-1])
            for line in lines
            if line.split()[0].isdigit()
        ]
    )
    assert list(indices) == list(df["index"])
    df["correct"] = letters
    for _, row in df.iterrows():
        assert not pd.isna(row[f"option-{row['correct']}"])


def run_scrape(cfg: DictConfig):
    dfs = []
    for test in TESTS:
        logger.info("Building %s.", test.name)
        df = get_questions_and_answers(test)
        add_correct_options(test, df)
        df["origin"] = test.name
        dfs.append(df)

    dataset = Dataset.from_pandas(pd.concat(dfs, ignore_index=True))

    if cfg.databuild.hub.push:
        push(
            dataset,
            cfg,
            config_name=RAW_CONFIG_NAME,
        )


def simple_clean(text: str) -> str:
    return (
        re.sub(LINEBREAK_PATTERN, r"\1", text)
        .replace(" ?", "?")
        .replace(" -", "-")
        .replace(" ,", ",")
        .replace(" .", ".")
    )


def calc_fix_candidates(nlm: NgramLm, text: str) -> list[dict[str, str | float]]:
    candidates: list[dict[str, str | float]] = []
    words = text.split()
    original_score = nlm.predict(text)
    for i in range(len(words) - 1):
        joined_word = words[i] + words[i + 1]
        new_score = nlm.predict(" ".join(words[:i] + [joined_word] + words[i + 2 :]))
        if (diff := new_score - original_score) > 0:
            candidates.append(
                {"wrong": words[i] + " " + words[i + 1], "fixed": joined_word, "score_diff": diff}
            )
    return candidates


def clean_example(
    example: dict[str, str], cfg: DictConfig, nlm: NgramLm, to_replace: dict[str, str]
):
    for name, text in example.items():
        if name in {"index", "source"}:
            continue
        if text is not None:
            cleaned = simple_clean(text)
            for wrong, fixed in to_replace.items():
                cleaned = cleaned.replace(wrong, fixed)
            example[name] = cleaned

            if cfg.databuild.calc_lm:
                candidates = calc_fix_candidates(nlm, cleaned)
                if candidates:
                    pd.DataFrame(candidates, columns=TOFIX_COLS).to_csv(
                        cfg.databuild.scored_tofix, mode="a", header=False, index=False
                    )
    return example


def reorder_idx_to_last(example: dict[str, str]) -> dict[str, str]:
    example["index"] = example.pop("index")
    return example


def run_clean(cfg: DictConfig):
    raw_dataset: Dataset = load_dataset(
        cfg.databuild.hub.target, RAW_CONFIG_NAME, download_mode="force_redownload"
    )["train"]

    nlm = NgramLm(cfg)

    to_replace_df = pd.read_csv(cfg.databuild.verified_tofix)
    to_replace = dict(zip(to_replace_df["wrong"], to_replace_df["fixed"]))
    if cfg.databuild.calc_lm:
        pd.DataFrame(columns=TOFIX_COLS).to_csv(cfg.databuild.scored_tofix, index=False)

    cleaned_dataset = (
        raw_dataset.map(partial(clean_example, cfg=cfg, nlm=nlm, to_replace=to_replace))
        .map(reorder_idx_to_last)
        .shuffle(1887)
    )
    if cfg.databuild.calc_lm:
        pd.read_csv(cfg.databuild.scored_tofix).sort_values(
            by="score_diff", ascending=False
        ).drop_duplicates("wrong", keep="first").to_csv(cfg.databuild.scored_tofix, index=False)

    if cfg.databuild.hub.push:
        push(cleaned_dataset, cfg.databuild.hub, config_name=CLEANED_CONFIG_NAME)


def create_citizen_da(cfg: DictConfig):
    logger.debug("Running with arguments: %s", format_config(cfg))
    if cfg.databuild.scrape:
        logger.info("Scraping dataset from SIRI.")
        run_scrape(cfg)
    if cfg.databuild.clean:
        logger.info("Cleaning dataset from HF hub.")
        run_clean(cfg)
