from typing import Callable

from omegaconf import DictConfig

from danoliterate.evaluation.execution.task_runner import (
    AnswerSimilarityRunner,
    ClozeRunnerWithOptions,
    GptNerRunner,
    MultiAnswerSimilarityRunner,
    MultichoiceRunner,
    MultichoiceRunnerLetterOptions,
    MultichoiceRunnerLetterWithContext,
    MultichoiceRunnerLetterWithContextAndOptions,
    MultichoiceRunnerLetterWithOptions,
    MultiChoiceRunnerSameOptions,
    MultichoiceRunnerWithOptions,
    TaskRunner,
)

TaskFunctionType = Callable[[DictConfig], TaskRunner]

task_registry: dict[str, TaskFunctionType] = {}
task_supported_metrics_registry: dict[str, list[str]] = {}


class UnknownTask(KeyError):
    """A task key was given without a registered task inference"""


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


MC_STANDARD_METRICS = [
    "text-similarity-bert-sim",
    "text-similarity-rouge-l",
    "text-similarity-rouge-1",
    "max-likelihood-accuracy",
    "max-likelihood-f1",
    "max-similarity-accuracy-bert-sim",
    "max-similarity-accuracy-rouge-l",
    "max-similarity-accuracy-rouge-1",
    "max-similarity-f1-bert-sim",
    "max-similarity-f1-rouge-l",
    "max-similarity-f1-rouge-1",
    "likelihood-brier",
    "likelihood-ece",
]
MC_SHOWING_OPTIONS_METRICS = [
    "max-similarity-accuracy-chosen-parsing",
    "max-similarity-f1-chosen-parsing",
    "max-likelihood-accuracy",
    "max-likelihood-f1",
    "likelihood-brier",
    "likelihood-ece",
]


@register_task(
    "default-mc",
    metrics=MC_STANDARD_METRICS,
)
def get_mc(scenario_cfg: DictConfig) -> TaskRunner:
    kwargs = {}
    for feature in "prompt_feature", "id_features":
        if (config_value := scenario_cfg.task.get(feature)) is not None:
            kwargs[feature] = config_value
    return MultichoiceRunner(**kwargs)


@register_task(
    "default-mc-options",
    metrics=MC_SHOWING_OPTIONS_METRICS,
)
def get_mc_options(scenario_cfg: DictConfig) -> TaskRunner:
    kwargs = {}
    for feature in "prompt_feature", "id_features":
        if (config_value := scenario_cfg.task.get(feature)) is not None:
            kwargs[feature] = config_value
    return MultichoiceRunnerWithOptions(**kwargs)


@register_task("default-mc-letter-options", metrics=MC_STANDARD_METRICS)
def get_mc_letter_options(scenario_cfg: DictConfig) -> TaskRunner:
    kwargs = {}
    for feature in "prompt_feature", "id_features":
        if (config_value := scenario_cfg.task.get(feature)) is not None:
            kwargs[feature] = config_value
    return MultichoiceRunnerLetterOptions(**kwargs)


@register_task("default-mc-letter-options-showing", metrics=MC_SHOWING_OPTIONS_METRICS)
def get_mc_letter_options_showing(scenario_cfg: DictConfig) -> TaskRunner:
    kwargs = {}
    for feature in "prompt_feature", "id_features":
        if (config_value := scenario_cfg.task.get(feature)) is not None:
            kwargs[feature] = config_value
    return MultichoiceRunnerLetterWithOptions(**kwargs)


@register_task("default-mc-same-options", metrics=MC_STANDARD_METRICS)
def get_mc_same_options(scenario_cfg: DictConfig) -> TaskRunner:
    kwargs = {}
    for feature in "all_labels", "prompt_feature", "label_feature", "id_features":
        if (config_value := scenario_cfg.task.get(feature)) is not None:
            kwargs[feature] = config_value
    return MultiChoiceRunnerSameOptions(**kwargs)


@register_task("default-mc-letter-context", metrics=MC_STANDARD_METRICS)
def get_mc_letter_context(scenario_cfg: DictConfig) -> TaskRunner:
    kwargs = {}
    for feature in "prompt_feature", "id_features", "context_feature":
        if (config_value := scenario_cfg.task.get(feature)) is not None:
            kwargs[feature] = config_value
    return MultichoiceRunnerLetterWithContext(**kwargs)


@register_task("default-mc-letter-context-and-options", metrics=MC_SHOWING_OPTIONS_METRICS)
def get_mc_letter_context_options(scenario_cfg: DictConfig) -> TaskRunner:
    kwargs = {}
    for feature in "prompt_feature", "id_features", "context_feature":
        if (config_value := scenario_cfg.task.get(feature)) is not None:
            kwargs[feature] = config_value
    return MultichoiceRunnerLetterWithContextAndOptions(**kwargs)


@register_task("cloze", metrics=MC_STANDARD_METRICS)
@register_task(
    "cloze-showing-options",
    metrics=MC_SHOWING_OPTIONS_METRICS,
)
def get_cloze_showing_options(scenario_cfg: DictConfig) -> TaskRunner:
    kwargs = {}
    for feature in "prompt_feature", "id_features", "cloze_mask_key", "cloze_mask_replaced":
        if (config_value := scenario_cfg.task.get(feature)) is not None:
            kwargs[feature] = config_value
    return ClozeRunnerWithOptions(**kwargs)


@register_task("angry-tweets", metrics=MC_SHOWING_OPTIONS_METRICS)
def get_angry_tweets(scenario_cfg: DictConfig) -> TaskRunner:
    return get_mc_same_options(scenario_cfg)


@register_task(
    "default-answer-similarity",
    metrics=[
        "text-similarity-bert-sim",
        "text-similarity-rouge-l",
        "text-similarity-rouge-1",
        "offensive-prob",
    ],
)
def get_answer_similarity(scenario_cfg: DictConfig) -> TaskRunner:
    kwargs = {}
    for feature in "prompt_feature", "answer_feature", "id_features":
        if (config_value := scenario_cfg.task.get(feature)) is not None:
            kwargs[feature] = config_value
    return AnswerSimilarityRunner(**kwargs)


@register_task(
    "multi-answer-similarity",
    metrics=[
        "avg-text-similarity-bert-sim",
        "avg-text-similarity-rouge-l",
        "avg-text-similarity-rouge-1",
        "odd-one-out-accuracy-bert-sim",
        "odd-one-out-accuracy-rouge-l",
        "odd-one-out-accuracy-rouge-1",
        "max-text-similarity-bert-sim",
        "max-text-similarity-rouge-l",
        "max-text-similarity-rouge-1",
        "min-text-similarity-bert-sim",
        "min-text-similarity-rouge-l",
        "min-text-similarity-rouge-1",
        "offensive-prob",
    ],
)
def get_multi_answer_similarity(scenario_cfg: DictConfig) -> TaskRunner:
    kwargs = {}
    for feature in "prompt_feature", "answer_feature", "id_features":
        if (config_value := scenario_cfg.task.get(feature)) is not None:
            kwargs[feature] = config_value
    return MultiAnswerSimilarityRunner(**kwargs)


@register_task("gpt-ner", metrics=["gpt-ner"])
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
        "n_examples",
    ):
        if (config_value := scenario_cfg.task.get(feature)) is not None:
            kwargs[feature] = config_value
    return GptNerRunner(**kwargs)


def get_task_runner(scenario_cfg: DictConfig):
    task_name = scenario_cfg.task.type
    try:
        return task_registry[task_name](scenario_cfg)
    except KeyError as error:
        raise UnknownTask(f"No task registered with scenario.task.type {task_name}") from error
