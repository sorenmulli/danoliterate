from omegaconf import DictConfig

from nlgenda.evaluation.registries.registration import register_task
from nlgenda.evaluation.task_runner import MultichoiceRunner, TaskRunner


@register_task("hyggeswag")
def get_hyggeswag(_: DictConfig) -> TaskRunner:
    return MultichoiceRunner(prompt_feature="ctx", id_features=("source_id", "ind"))
