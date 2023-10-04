import logging
from dataclasses import _MISSING_TYPE, fields
from typing import Any, Union

logger = logging.getLogger(__name__)

OutDictType = dict[str, Union[str, int, float, bool, None, "OutDictType", list["OutDictType"]]]


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
