from typing import Callable

from omegaconf import DictConfig

from nlgenda.evaluation.execution.model_inference import ModelInference
from nlgenda.evaluation.execution.task_runner import TaskRunner

TaskFunctionType = Callable[[DictConfig], TaskRunner]

task_registry: dict[str, TaskFunctionType] = {}


def register_task(task_name: str) -> Callable[[TaskFunctionType], TaskFunctionType]:
    def decorator(func: TaskFunctionType) -> TaskFunctionType:
        if task_name in task_registry:
            raise ValueError(f"Task {task_name} registered more than once!")
        task_registry[task_name] = func
        return func

    return decorator


InferenceFunctionType = Callable[[DictConfig], ModelInference]

inference_registry: dict[str, InferenceFunctionType] = {}


def register_inference(
    inference_name: str,
) -> Callable[[InferenceFunctionType], InferenceFunctionType]:
    def decorator(func: InferenceFunctionType) -> InferenceFunctionType:
        if inference_name in inference_registry:
            raise ValueError(f"Model inference {inference_name} registered more than once!")
        inference_registry[inference_name] = func
        return func

    return decorator
