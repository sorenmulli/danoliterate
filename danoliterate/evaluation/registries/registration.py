from typing import Callable

from omegaconf import DictConfig

from danoliterate.evaluation.analysis.dimensions import Dimension
from danoliterate.evaluation.analysis.metrics import Metric
from danoliterate.evaluation.execution.model_inference import ModelInference
from danoliterate.evaluation.execution.task_runner import TaskRunner
from danoliterate.evaluation.serialization import OutDictType

MetricFunctionType = Callable[[OutDictType], Metric]

metric_registry: dict[str, MetricFunctionType] = {}
metric_dimension_registry: dict[str, Dimension] = {}


def register_metric(
    metric_name: str,
    dimension: Dimension = Dimension.CAPABILITY,
) -> Callable[[MetricFunctionType], MetricFunctionType]:
    def decorator(func: MetricFunctionType) -> MetricFunctionType:
        if metric_name in metric_registry:
            raise ValueError(f"Evaluation metric {metric_name} registered more than once!")
        metric_registry[metric_name] = func
        metric_dimension_registry[metric_name] = dimension
        return func

    return decorator


TaskFunctionType = Callable[[DictConfig], TaskRunner]

task_registry: dict[str, TaskFunctionType] = {}
task_supported_metrics_registry: dict[str, list[str]] = {}


def register_task(
    task_name: str, metrics: list[str]
) -> Callable[[TaskFunctionType], TaskFunctionType]:
    def decorator(func: TaskFunctionType) -> TaskFunctionType:
        if task_name in task_registry:
            raise ValueError(f"Task {task_name} registered more than once!")
        task_registry[task_name] = func
        task_supported_metrics_registry[task_name] = metrics
        return func

    return decorator


InferenceFunctionType = Callable[[DictConfig], ModelInference]

inference_registry: dict[str, InferenceFunctionType] = {}
inference_unsupported_metrics_registry: dict[str, list[str]] = {}


def register_inference(
    inference_name: str,
    unsupported_metrics: list[str],
) -> Callable[[InferenceFunctionType], InferenceFunctionType]:
    def decorator(func: InferenceFunctionType) -> InferenceFunctionType:
        if inference_name in inference_registry:
            raise ValueError(f"Model inference {inference_name} registered more than once!")
        inference_registry[inference_name] = func
        inference_unsupported_metrics_registry[inference_name] = unsupported_metrics
        return func

    return decorator
