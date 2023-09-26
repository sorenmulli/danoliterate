import logging

from omegaconf import DictConfig
from tqdm import tqdm

from nlgenda.evaluation.evaluator import Evaluator
from nlgenda.evaluation.model import EvaluatorModel
from nlgenda.evaluation.scenario import EvaluatorScenario
from nlgenda.infrastructure import format_config

logger = logging.getLogger(__name__)


def evaluate(cfg: DictConfig):
    logger.debug("Running evaluation with arguments: %s", format_config(cfg))
    logger.info("Evaluating %s on %s.", cfg.model.name, cfg.scenario.name)

    logger.info("Setting up evaluator ...")
    evaluator = Evaluator(**cfg.evaluation)

    logger.info("Setting up scenario ...")
    scenario = EvaluatorScenario(**cfg.scenario)

    logger.info("Setting up model ...")
    model = EvaluatorModel(**cfg.model)

    logger.info("Initializing generators ...")
    examples = scenario.generate_examples()
    results = model.generate_results(examples)

    logger.info("Running result loop")
    for result in tqdm(results, total=len(scenario)):
        evaluator.receive_result(result)
    evaluator.finish()
