from omegaconf import DictConfig

from nlgenda.evaluation.execution.task_runner import (
    AnswerSimilarityRunner,
    MultichoiceRunner,
    MultichoiceRunnerLetterOptions,
    TaskRunner,
)
from nlgenda.evaluation.registries.registration import register_task


# TODO: Remove, currently kept as backwards comp with old executions
@register_task("hyggeswag")
@register_task("default-mc")
def get_mc(scenario_cfg: DictConfig) -> TaskRunner:
    kwargs = {}
    for feature in "prompt_feature", "id_features":
        if (config_value := scenario_cfg.task.get(feature)) is not None:
            kwargs[feature] = config_value
    return MultichoiceRunner(**kwargs)


# TODO: Remove, currently kept as backwards comp with old executions
@register_task("citizenship-test")
@register_task("default-mc-letter-options")
def get_mc_letter_options(scenario_cfg: DictConfig) -> TaskRunner:
    kwargs = {}
    for feature in "prompt_feature", "id_features":
        if (config_value := scenario_cfg.task.get(feature)) is not None:
            kwargs[feature] = config_value
    return MultichoiceRunnerLetterOptions(**kwargs)


# TODO: Remove, currently kept as backwards comp with old executions
@register_task("prompt-similarity")
@register_task("default-answer-similarity")
def get_answer_similarity(scenario_cfg: DictConfig) -> TaskRunner:
    kwargs = {}
    for feature in "prompt_feature", "answer_feature", "id_features":
        if (config_value := scenario_cfg.task.get(feature)) is not None:
            kwargs[feature] = config_value
    return AnswerSimilarityRunner(**kwargs)
