import logging
from typing import Generator

from datasets import Dataset, load_dataset
from omegaconf import DictConfig
from tqdm import tqdm

from nlgenda.evaluation.example import EvaluationExample
from nlgenda.evaluation.registries.get import get_inference, get_task_runner
from nlgenda.infrastructure import format_config
from nlgenda.modeling.text_comparison import get_compare_fun

logger = logging.getLogger(__name__)


# TODO: Move some of the storage into an EvaluationResult
# pylint: disable=too-many-instance-attributes
class Evaluator:
    def __init__(self, cfg: DictConfig):
        logger.info("Evaluating %s on %s.", cfg.model.name, cfg.scenario.name)
        self.scenario_name = cfg.scenario.name
        self.model_name = cfg.model.name

        self.text_compare = (
            get_compare_fun(cfg.scenario.compare) if cfg.scenario.compare is not None else None
        )

        logger.info("Setting up scenario ...")
        self.task_runner = get_task_runner(cfg.scenario)
        # TODO: Consider splits
        self.dataset: Dataset = load_dataset(cfg.scenario.path, split="train")

        logger.info("Setting up model ...")
        self.model_inference = get_inference(cfg.model)

        self.results: list[EvaluationExample] = []
        self.result_db = cfg.evaluation.result_db

    def run(self):
        logger.info("Initializing example generators ...")
        examples = self.generate_examples()
        results = self.generate_results(examples)

        logger.info("Executing result loop ...")
        for result in tqdm(results, total=len(self.dataset)):
            self.results.append(result)

        logger.info("Finished result loop.")

    def save_results(self):
        raise NotImplementedError

    def generate_examples(self) -> Generator[EvaluationExample, None, None]:
        for data_example in self.dataset:
            yield self.task_runner.build_example(data_example)

    def generate_results(
        self, examples: Generator[EvaluationExample, None, None]
    ) -> Generator[EvaluationExample, None, None]:
        for eval_example in examples:
            yield self.task_runner.get_prediction(
                eval_example, self.model_inference, self.text_compare
            )


def evaluate(cfg: DictConfig):
    logger.debug("Running evaluation with arguments: %s", format_config(cfg))

    evaluator = Evaluator(cfg)
    evaluator.run()
    evaluator.save_results()
