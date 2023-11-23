import logging
from typing import Generator

from datasets import Dataset, DownloadMode, load_dataset
from omegaconf import DictConfig

from nlgenda.evaluation.artifact_integration import send_result_wandb, setup_short_run
from nlgenda.evaluation.execution.model_inference import ModelInference, set_deterministic
from nlgenda.evaluation.registries.get import get_inference, get_task_runner
from nlgenda.evaluation.results import ExecutionExample, ExecutionResult
from nlgenda.infrastructure import format_config

logger = logging.getLogger(__name__)


class Evaluator:
    def __init__(self, cfg: DictConfig, scenario_cfg: DictConfig, model_inference: ModelInference):
        logger.info("Evaluating %s on %s.", cfg.model.name, scenario_cfg.name)
        self.result = ExecutionResult.from_config(cfg, scenario_cfg)

        logger.info("Setting execution seed to %i", cfg.evaluation.seed)
        set_deterministic(cfg.evaluation.seed)

        logger.info("Setting up scenario ...")
        self.task_runner = get_task_runner(scenario_cfg)
        # TODO: Consider splits
        self.dataset: Dataset = load_dataset(
            scenario_cfg.path,
            split=scenario_cfg.get("dataset_split", "train"),
        )

        self.train_dataset = None
        if self.task_runner.is_few_shot:
            assert scenario_cfg.dataset_split != "train"
            self.train_dataset: Dataset = load_dataset(
                scenario_cfg.path,
                split="train",
            )

        self.model_inference = model_inference

        self.wandb = setup_short_run(self.result.name, "eval", cfg.wandb)
        self.scenario_cfg: DictConfig = scenario_cfg

    def run(self):
        logger.info("Initializing example generators ...")
        examples = self.generate_examples()
        queried_results = self.generate_results(examples)

        logger.info("Executing result loop ...")
        for result in self.model_inference.answer_queries(list(queried_results)):
            self.result.examples.append(result)

        logger.info("Finished result loop.")

    def save_results(self):
        out = self.result.save_locally()
        logger.info("Result was saved locally to %s.", out)
        if self.wandb is not None:
            send_result_wandb(self.result, self.wandb)
            logger.info("Sucessfully sent result to W&B.")

    def generate_examples(self) -> Generator[ExecutionExample, None, None]:
        for i, data_example in enumerate(self.dataset):
            args = dict(
                row=data_example,
                pre_prompt=self.scenario_cfg.get("pre_prompt", ""),
                post_prompt=self.scenario_cfg.get("post_prompt", ""),
                idx=i,
            )
            if self.task_runner.is_few_shot:
                assert self.train_dataset is not None
                args["few_shot_dataset"] = self.train_dataset
            if self.task_runner.can_build_multi_examples:
                yield from self.task_runner.build_multi_examples(**args)
            else:
                yield self.task_runner.build_example(**args)

    def generate_results(
        self, examples: Generator[ExecutionExample, None, None]
    ) -> Generator[ExecutionExample, None, None]:
        for eval_example in examples:
            yield self.task_runner.get_prediction(eval_example, self.model_inference)


def evaluate(cfg: DictConfig):
    logger.debug("Running evaluation with arguments: %s", format_config(cfg))
    logger.info("Setting up model ...")
    model_inference = get_inference(cfg)

    logger.info("Model set up. Evaluating on %i scenarios.", len(cfg.scenarios))
    for scenario_cfg in cfg.scenarios.values():
        evaluator = Evaluator(cfg, scenario_cfg, model_inference)
        evaluator.run()
        evaluator.save_results()
