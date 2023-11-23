from omegaconf import DictConfig

# We need to be sure that all inferences and tasks are registered so we import first
# pylint: disable=unused-import
import nlgenda.evaluation.registries.inferences
import nlgenda.evaluation.registries.tasks
from nlgenda.evaluation.registries.registration import inference_registry, task_registry


def get_task_runner(scenario_cfg: DictConfig):
    task_name = scenario_cfg.task.type
    try:
        return task_registry[task_name](scenario_cfg)
    except KeyError as error:
        raise ValueError(f"No task registered with scenario.task.type {task_name}") from error


def get_inference(cfg: DictConfig):
    inference_name = cfg.model.inference.type
    try:
        return inference_registry[inference_name](cfg)
    except KeyError as error:
        raise ValueError(
            f"No inference registered with model.inference.type {inference_name}"
        ) from error
