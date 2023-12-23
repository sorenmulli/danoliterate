import pandas as pd
import requests
from bs4 import BeautifulSoup
from datasets import Dataset
from omegaconf import DictConfig

from danoliterate.data.building.hub import push
from danoliterate.infrastructure.logging import format_config, logger

QUESTIONS_URL = "http://static.uvm.dk/Publikationer/2000/laesetekster/2/hel.htm"
ANSWERS_URL = "http://static.uvm.dk/Publikationer/2000/laesetekster/1/2.htm"
TEXTS = (
    "http://static.uvm.dk/Publikationer/2000/laesetekster/3/2.htm",
    "http://static.uvm.dk/Publikationer/2000/laesetekster/3/3.htm",
    "http://static.uvm.dk/Publikationer/2000/laesetekster/3/4.htm",
)
TITLE_STR = "Opgaver til"
QUESTION_STR = "De rigtige svar: "

TEXTUAL_CONTEXTS: tuple[list[slice], ...] = (
    [
        slice(1, 2),
        slice(24, 26),
        slice(30, 50),
        slice(1, 5),
        slice(23, 25),
        slice(33, 58),
        slice(25, 27),
        slice(33, 42),
        slice(33, 41),
        slice(47, 53),
        slice(53, 58),
        slice(40, 57),
        slice(53, 64),
        slice(53, 64),
        slice(52, 55),
    ],
    [slice(1, 5), slice(1, 5), slice(7, 9), slice(4, 5), slice(10, 13), slice(10, 15)],
    [],
)


def get_questions() -> pd.DataFrame:
    response = requests.get(QUESTIONS_URL, timeout=10)
    response.raise_for_status()

    data = []
    soup = BeautifulSoup(response.text, "html.parser")
    task_titles = soup.find_all("h2", text=lambda t: t is not None and TITLE_STR in t)

    # Process each task title and its questions
    for task_title in task_titles:
        current_title = task_title.get_text(strip=True).replace(TITLE_STR, "").strip()
        if current_title[0].islower():
            current_title = current_title.capitalize()
        next_element = task_title.find_next_sibling()

        while next_element and next_element.name != "h2":
            if next_element.name == "p" and next_element.strong:
                question_text = next_element.strong.get_text(strip=True)
                options = [
                    " ".join(opt.get_text(strip=True).split()[1:])
                    for opt in next_element.find_next_siblings("p", limit=4)
                ]

                # Check if we have exactly 4 options, otherwise skip
                assert len(options) == 4

                if "Figur 10.6" not in question_text:
                    # Manual, hacky fix of italic
                    question_text = question_text.replace(
                        "harmindstfornemmelse", "har mindst fornemmelse"
                    )
                    index = int(question_text.split(".")[0])
                    question_text = " ".join(question_text.split()[1:])
                    # Manual, even more hacky fix of a text across multiple *strongs*
                    if "figur 10.6" in question_text:
                        question_text = (
                            "Næringsstofferne skal fra tarmen ud til cellerne i kroppen."
                            " På vejen passerer næringsstofferne disse organer i denne rækkefølge:"
                        )
                    data.append(
                        {
                            "question": question_text,
                            "option-A": options[0],
                            "option-B": options[1],
                            "option-C": options[2],
                            "option-D": options[3],
                            "task_title": current_title,
                            "index": index,
                        }
                    )
            next_element = next_element.find_next_sibling()
    return pd.DataFrame(data)


def add_contexts(df: pd.DataFrame):
    contexts = []
    for url, context_ranges in zip(TEXTS, TEXTUAL_CONTEXTS, strict=True):
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        paragraphs = [paragraph.text for paragraph in soup.find_all("p")]
        for context_range in context_ranges:
            contexts.append(" ".join(" ".join(paragraphs[context_range]).strip().split()))
    df["context"] = pd.Series(contexts)


def add_correct_options(df: pd.DataFrame):
    response = requests.get(ANSWERS_URL, timeout=10)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")
    answer_lists = soup.find_all("p", text=lambda t: t is not None and QUESTION_STR in t)
    correct_options = []
    for answer_list_obj in answer_lists:
        for answer in (
            answer_list_obj.get_text(strip=True)
            .replace(QUESTION_STR, "")
            .replace(".", "")
            .split(",")
        ):
            correct_options.append(answer.split(":")[-1].strip().upper())
    df["correct"] = pd.Series(correct_options)


def create_da_gym_200(cfg: DictConfig):
    logger.debug("Running with arguments: %s", format_config(cfg))
    df = get_questions()
    add_contexts(df)
    add_correct_options(df)
    dataset = Dataset.from_pandas(df)
    if cfg.databuild.hub.push:
        push(dataset, cfg.databuild.hub)
