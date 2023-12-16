from omegaconf import DictConfig

from danoliterate.evaluation.analysis.metrics import Metric

# We need to be sure that all metrics, inferences, and tasks are registered so we import first
# isort: off
# pylint: disable=unused-import
import danoliterate.evaluation.registries.metrics
import danoliterate.evaluation.registries.inferences
import danoliterate.evaluation.registries.tasks

# isort: on
from danoliterate.evaluation.registries.registration import (
    inference_registry,
    inference_unsupported_metrics_registry,
    metric_registry,
    task_registry,
    task_supported_metrics_registry,
)
from danoliterate.evaluation.serialization import OutDictType


class UnknownMetric(KeyError):
    """A metric key was given without a registered metric"""


class UnknownInference(KeyError):
    """A modle inference key was given without a registered model inference"""


class UnknownTask(KeyError):
    """A task key was given without a registered task inference"""


def get_compatible_metrics(task: str, inference: str, scenario_cfg: OutDictType) -> list[Metric]:
    try:
        task_supported = task_supported_metrics_registry[task]
    except KeyError as error:
        raise UnknownTask(f"No task registered with task type {task}") from error
    try:
        inference_unsupported = inference_unsupported_metrics_registry[inference]
    except KeyError as error:
        raise UnknownInference(
            f"No inference registered with model inference type {inference}"
        ) from error
    metric_keys = [metric for metric in task_supported if metric not in inference_unsupported]
    metrics = []
    for metric_key in metric_keys:
        try:
            metrics.append(metric_registry[metric_key](scenario_cfg))
        except KeyError as error:
            raise UnknownMetric(f"No metric registered with metric key {metric_key}") from error
    return metrics


def get_task_runner(scenario_cfg: DictConfig):
    task_name = scenario_cfg.task.type
    try:
        return task_registry[task_name](scenario_cfg)
    except KeyError as error:
        raise UnknownTask(f"No task registered with scenario.task.type {task_name}") from error


def get_inference(cfg: DictConfig):
    inference_name = cfg.model.inference.type
    try:
        return inference_registry[inference_name](cfg)
    except KeyError as error:
        raise UnknownInference(
            f"No inference registered with model.inference.type {inference_name}"
        ) from error
