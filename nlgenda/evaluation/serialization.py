import logging
from dataclasses import _MISSING_TYPE, fields
from typing import Any, Union

logger = logging.getLogger(__name__)

OutDictType = dict[str, Union[str, int, float, bool, None, "OutDictType", list["OutDictType"]]]

TASK_TYPE_RENAMES = {
    "hyggeswag": "default-mc",
    "citizenship-test": "default-mc-letter-options",
    "prompt-similarity": "default-answer-similarity",
}


def apply_backcomp_fixes_execution_result_metadata(input_args: OutDictType):
    warnings = []
    scenario_cfg: OutDictType = input_args["scenario_cfg"]  # type: ignore # type: ignore
    task_type: str = scenario_cfg["task"]["type"]  # type: ignore
    if scenario_cfg["name"] == "Angry Tweets" and task_type != "angry-tweets":
        scenario_cfg["task"]["type"] = "angry-tweets"  # type: ignore
        warnings.append("Changed Angry Tweets task type to angry-tweets.")
    elif (new_name := TASK_TYPE_RENAMES.get(task_type)) is not None:
        scenario_cfg["task"]["type"] = new_name  # type: ignore
        warnings.append(f"Renamed {task_type} to {new_name}.")
    if warnings:
        logger.debug("Performed backwards compatability fixing:\n%s", ",".join(warnings))


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
