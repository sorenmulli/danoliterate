from collections import Counter
from typing import Callable, Optional

TextCompareFun = Optional[Callable[[str, str], float]]


class UnknownComparisonKey(KeyError):
    ...


def compare_rouge1_f1_unique(target: str, prediction: str) -> float:
    target_counts = Counter(set(target.lower().split()))
    prediction_counts = Counter(set(prediction.lower().split()))
    overlap = sum((target_counts & prediction_counts).values())

    # Avoid division by zero
    if not overlap:
        return 0

    precision = overlap / sum(prediction_counts.values())
    recall = overlap / sum(target_counts.values())

    if not precision + recall:
        return 0
    return 2 * (precision * recall) / (precision + recall)


COMPARE_FUNCTIONS = {
    "rouge1-f1-unq": compare_rouge1_f1_unique,
}


def get_compare_fun(key: str) -> TextCompareFun:
    try:
        return COMPARE_FUNCTIONS[key]
    except KeyError as error:
        raise UnknownComparisonKey(
            f"Text comparison key '{key}' not one of {', '.join(COMPARE_FUNCTIONS.keys())}"
        ) from error
