from omegaconf import DictConfig

from nlgenda.evaluation.execution.task_runner import (
    AnswerSimilarityRunner,
    ClozeRunnerWithOptions,
    GptNerRunner,
    MultichoiceRunner,
    MultichoiceRunnerLetterOptions,
    MultichoiceRunnerLetterWithContext,
    MultiChoiceRunnerSameOptions,
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


@register_task("default-mc-letter-context")
def get_mc_letter_context(scenario_cfg: DictConfig) -> TaskRunner:
    kwargs = {}
    for feature in "prompt_feature", "id_features", "context_feature":
        if (config_value := scenario_cfg.task.get(feature)) is not None:
            kwargs[feature] = config_value
    return MultichoiceRunnerLetterWithContext(**kwargs)


@register_task("default-mc-same-options")
def get_mc_same_options(scenario_cfg: DictConfig) -> TaskRunner:
    kwargs = {}
    for feature in "all_labels", "prompt_feature", "label_feature", "id_features":
        if (config_value := scenario_cfg.task.get(feature)) is not None:
            kwargs[feature] = config_value
    return MultiChoiceRunnerSameOptions(**kwargs)


@register_task("cloze-showing-options")
def get_cloze_showing_options(scenario_cfg: DictConfig) -> TaskRunner:
    kwargs = {}
    for feature in "prompt_feature", "id_features", "cloze_mask_key", "cloze_mask_replaced":
        if (config_value := scenario_cfg.task.get(feature)) is not None:
            kwargs[feature] = config_value
    return ClozeRunnerWithOptions(**kwargs)


# TODO: Remove, currently kept as backwards comp with old executions
@register_task("prompt-similarity")
@register_task("default-answer-similarity")
def get_answer_similarity(scenario_cfg: DictConfig) -> TaskRunner:
    kwargs = {}
    for feature in "prompt_feature", "answer_feature", "id_features":
        if (config_value := scenario_cfg.task.get(feature)) is not None:
            kwargs[feature] = config_value
    return AnswerSimilarityRunner(**kwargs)


@register_task("gpt-ner")
def get_gpt_ner(scenario_cfg: DictConfig) -> TaskRunner:
    kwargs = {}
    for feature in (
        "entity_types",
        "prompt_feature",
        "token_feature",
        "label_feature",
        "few_shot_format",
        "num_examples",
        "id_features",
    ):
        if (config_value := scenario_cfg.task.get(feature)) is not None:
            kwargs[feature] = config_value
    return GptNerRunner(**kwargs)
