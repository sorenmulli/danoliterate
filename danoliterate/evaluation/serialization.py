from dataclasses import _MISSING_TYPE, fields
from typing import Any, Optional, Union

from danoliterate.infrastructure.logging import logger

OutDictType = dict[str, Union[str, int, float, bool, None, "OutDictType", list["OutDictType"]]]

TASK_TYPE_RENAMES = {
    "hyggeswag": "default-mc",
    "citizenship-test": "default-mc-letter-options",
    "prompt-similarity": "default-answer-similarity",
    "default-mc-letter-context": "default-mc-letter-context-and-options",
}

SCENARIO_NAME_TO_TASK_TYPE = {
    "Angry Tweets": "angry-tweets",
}
SCENARIO_NAME_AND_TYPE_TO_TASK_TYPE: dict[tuple[str, Optional[str]], str] = {
    ("HyggeSwag", "free-generation"): "default-mc",
    ("Da. Cloze Self Test", "free-generation"): "cloze",
    ("Citizenship Test", "free-generation"): "default-mc-letter-options",
    ("Da. Gym 2000", "free-generation"): "default-mc-letter-context",
}
MODEL_KEY_TO_NEW_NAME = {
    "gpt-4": "OpenAI GPT 4",
    "gpt-4-1106-preview": "OpenAI GPT 4 Turbo",
}


def apply_backcomp_fixes_execution_result_metadata(input_args: OutDictType):
    warnings = []
    model_cfg: OutDictType = input_args["model_cfg"]  # type: ignore
    if (new_name := MODEL_KEY_TO_NEW_NAME.get(model_cfg.get("path"))) is not None:
        model_cfg["name"] = new_name
    scenario_cfg: OutDictType = input_args["scenario_cfg"]  # type: ignore
    task_type: str = scenario_cfg["task"]["type"]  # type: ignore
    scenario_name: str = scenario_cfg["name"]  # type: ignore
    scenario_type: Optional[str] = scenario_cfg.get("type")  # type: ignore
    if (desired_task_type := SCENARIO_NAME_TO_TASK_TYPE.get(scenario_name)) is not None or (
        desired_task_type := SCENARIO_NAME_AND_TYPE_TO_TASK_TYPE.get((scenario_name, scenario_type))
    ) is not None:
        if task_type != desired_task_type:
            scenario_cfg["task"]["type"] = desired_task_type  # type: ignore
            warnings.append(f"Changed {scenario_cfg['name']} task type to {desired_task_type}")

    elif (new_name := TASK_TYPE_RENAMES.get(task_type)) is not None:
        scenario_cfg["task"]["type"] = new_name  # type: ignore
        warnings.append(f"Renamed {task_type} to {new_name}.")
    if warnings:
        logger.debug("Performed backwards compatability fixing:\n%s", ",".join(warnings))


RENAME_OPTIONS = {
    "negative": "negativ",
    "positive": "positiv",
}


def apply_backcomp_fixes_execution_example(input_args: dict):
    if (opts := input_args.get("options")) is not None:
        input_args["options"] = [RENAME_OPTIONS.get(opt, opt) for opt in opts]


METRICS_TO_BE_PUT_TO_FRONT = [
    "Prediction odd-one-out frequency (BERT similarity)",
    "Prediction odd-one-out frequency (ROUGE-L)",
    "Prediction odd-one-out frequency (ROUGE-1)",
]


def apply_backcomp_reordering_metric_results(input_args: list[dict]) -> list[dict]:
    name_to_res = {metric["short_name"]: metric for metric in input_args}
    if len(name_to_res) != len(input_args):
        raise ValueError("Duplicate metrics")
    for metric in METRICS_TO_BE_PUT_TO_FRONT[::-1]:
        if metric in name_to_res:
            name_to_res = {metric: name_to_res.pop(metric), **name_to_res}
    return list(name_to_res.values())


def fix_args_for_dataclass(dataclass: Any, input_args: OutDictType):
    class_fields = {field.name: field for field in fields(dataclass)}
    input_keys = set(input_args.keys())

    # Given arguments not corresponding to fields in the dataclass
    extra_keys = input_keys - class_fields.keys()
    # Required fields in the data class that are not given in argument
    missing_keys = set()
    for key in class_fields.keys() - input_keys:
        if isinstance(class_fields[key].default, _MISSING_TYPE):
            missing_keys.add(key)

    if extra_keys:
        logger.debug(
            "Found extra keys in serialized data class that are not in the object. "
            "Will ignore their values. "
            "Extra keys: %s.",
            ", ".join(extra_keys),
        )
    if missing_keys:
        logger.debug(
            "Required fields in data class object were not in serialized data class. "
            "Will instantiate as None. "
            "Missing keys: %s.",
            ", ".join(missing_keys),
        )

    for key in extra_keys:
        del input_args[key]
    for key in missing_keys:
        input_args[key] = None
