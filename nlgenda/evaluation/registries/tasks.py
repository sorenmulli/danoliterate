from omegaconf import DictConfig

from nlgenda.evaluation.registries.registration import register_task
from nlgenda.evaluation.task_runner import (
    AnswerSimilarityRunner,
    MultichoiceRunner,
    MultichoiceRunnerLetterOptions,
    TaskRunner,
)


@register_task("default-mc")
def get_mc(_: DictConfig) -> TaskRunner:
    return MultichoiceRunner()


@register_task("hyggeswag")
def get_hyggeswag(_: DictConfig) -> TaskRunner:
    return MultichoiceRunner(prompt_feature="ctx", id_features=("source_id", "ind"))


@register_task("default-mc-letter-options")
def get_mc_letter_options(_: DictConfig) -> TaskRunner:
    return MultichoiceRunnerLetterOptions()


@register_task("citizenship-test")
def get_citizenship_test(_: DictConfig) -> TaskRunner:
    return MultichoiceRunnerLetterOptions(
        prompt_feature="question", id_features=("origin", "index")
    )


@register_task("default-answer-similarity")
def get_answer_similarity(_: DictConfig) -> TaskRunner:
    return AnswerSimilarityRunner()
