import logging
from typing import Generator

from datasets import Dataset, load_dataset
from omegaconf import DictConfig
from tqdm import tqdm

from nlgenda.evaluation.artifact_integration import send_result_wandb, setup_short_run
from nlgenda.evaluation.registries.get import get_inference, get_task_runner
from nlgenda.evaluation.results import ExecutionExample, ExecutionResult
from nlgenda.infrastructure import format_config

logger = logging.getLogger(__name__)


class Evaluator:
    def __init__(self, cfg: DictConfig):
        logger.info("Evaluating %s on %s.", cfg.model.name, cfg.scenario.name)
        self.result = ExecutionResult.from_config(cfg)

        logger.info("Setting up scenario ...")
        self.task_runner = get_task_runner(cfg.scenario)
        # TODO: Consider splits
        self.dataset: Dataset = load_dataset(cfg.scenario.path, split="train")

        logger.info("Setting up model ...")
        self.model_inference = get_inference(cfg.model)

        self.wandb = setup_short_run(self.result.name, "eval", cfg.wandb)

    def run(self):
        logger.info("Initializing example generators ...")
        examples = self.generate_examples()
        results = self.generate_results(examples)

        logger.info("Executing result loop ...")
        for result in tqdm(results, total=len(self.dataset)):
            self.result.examples.append(result)

        logger.info("Finished result loop.")

    def save_results(self):
        if self.wandb is not None:
            send_result_wandb(self.result, self.wandb)
            logger.info("Sucessfully sent result to W&B.")

        out = self.result.save_locally()
        logger.info("Result was saved locally to %s.", out)

    def generate_examples(self) -> Generator[ExecutionExample, None, None]:
        for data_example in self.dataset:
            yield self.task_runner.build_example(data_example)

    def generate_results(
        self, examples: Generator[ExecutionExample, None, None]
    ) -> Generator[ExecutionExample, None, None]:
        for eval_example in examples:
            yield self.task_runner.get_prediction(eval_example, self.model_inference)


def evaluate(cfg: DictConfig):
    logger.debug("Running evaluation with arguments: %s", format_config(cfg))

    evaluator = Evaluator(cfg)
    evaluator.run()
    evaluator.save_results()
