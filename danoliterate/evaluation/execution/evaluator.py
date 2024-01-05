from time import perf_counter
from typing import Generator, Optional

from datasets import Dataset, load_dataset
from omegaconf import DictConfig

from danoliterate.evaluation.artifact_integration import send_result_wandb, setup_short_run
from danoliterate.evaluation.execution.augmentation import Augmenter, get_augmenters
from danoliterate.evaluation.execution.eval_types import EVALUATION_TYPES
from danoliterate.evaluation.execution.model_inference import ModelInference, set_deterministic
from danoliterate.evaluation.registries.inferences import get_inference
from danoliterate.evaluation.registries.tasks import get_task_runner
from danoliterate.evaluation.results import ExecutionExample, ExecutionResult
from danoliterate.infrastructure import format_config
from danoliterate.infrastructure.logging import logger


class Evaluator:
    def __init__(
        self,
        cfg: DictConfig,
        scenario_cfg: DictConfig,
        model_inference: ModelInference,
        augmenter: Optional[Augmenter] = None,
    ):
        logger.info(
            "Evaluating %s on %s%s.",
            cfg.model.name,
            scenario_cfg.name,
            "" if augmenter is None else f" with augmenter: {augmenter.description}",
        )
        self.result = ExecutionResult.from_config(cfg, scenario_cfg, augmenter)
        self.augmenter = augmenter

        logger.info("Setting execution seed to %i", cfg.evaluation.seed)
        set_deterministic(cfg.evaluation.seed)

        logger.info("Setting up scenario ...")
        self.task_runner = get_task_runner(scenario_cfg)
        self.task_runner.set_augmenter(augmenter)
        # TODO: Consider splits
        self.dataset: Dataset = load_dataset(
            scenario_cfg.path,
            split=scenario_cfg.get("dataset_split", "train"),
        )

        self.train_dataset: Optional[Dataset] = None
        if self.task_runner.is_few_shot:
            assert scenario_cfg.dataset_split != "train"
            self.train_dataset = load_dataset(
                scenario_cfg.path,
                split="train",
            )

        self.model_inference = model_inference

        self.wandb = setup_short_run(self.result.name, "eval", cfg.wandb)
        self.scenario_cfg: DictConfig = scenario_cfg
        if (eval_type := self.scenario_cfg.get("type")) is not None:
            assert eval_type in EVALUATION_TYPES, f"scenario.type must be one of {EVALUATION_TYPES}"

    def run(self):
        logger.info("Initializing example generators ...")
        examples = self.generate_examples()
        start_time = perf_counter()
        queried_results = self.generate_results(examples)

        logger.info("Executing result loop ...")
        for result in self.model_inference.answer_queries(list(queried_results)):
            self.result.examples.append(result)
        self.result.metadata.total_inference_seconds = total_time = perf_counter() - start_time
        logger.info("Finished result loop in %.0f seconds.", total_time)

    def save_results(self):
        out = self.result.save_locally()
        logger.info("Result was saved locally to %s.", out)
        if self.wandb is not None:
            send_result_wandb(self.result, self.wandb)
            logger.info("Sucessfully sent result to W&B.")

    def generate_examples(self) -> Generator[ExecutionExample, None, None]:
        for i, data_example in enumerate(self.dataset):
            args = {
                "row": data_example,
                "pre_prompt": self.scenario_cfg.get("pre_prompt", ""),
                "post_prompt": self.scenario_cfg.get("post_prompt", ""),
                "idx": i,
            }
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
        augmenters = get_augmenters(cfg)
        if augmenters:
            logger.info(
                "Instantiated %i augmenters, will run augmentation versions first.", len(augmenters)
            )
        for augmenter in augmenters:
            evaluator = Evaluator(cfg, scenario_cfg, model_inference, augmenter=augmenter)
            evaluator.run()
            evaluator.save_results()
        if cfg.evaluation.get("skip_unaugmented"):
            continue
        evaluator = Evaluator(cfg, scenario_cfg, model_inference)
        evaluator.run()
        evaluator.save_results()
