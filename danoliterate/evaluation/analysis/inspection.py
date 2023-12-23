from collections import defaultdict
from pathlib import Path

import pandas as pd
from omegaconf import DictConfig

from danoliterate.evaluation.artifact_integration import get_results_wandb
from danoliterate.evaluation.results import ExecutionResult
from danoliterate.infrastructure.logging import format_config, logger


# TODO: Also add scores and correct results
def output_for_inspection(cfg: DictConfig, name: str, results: list[ExecutionResult]):
    df = pd.DataFrame({"prompt": [ex.prompt for ex in results[0].examples]})
    for res in results:
        df[res.metadata.model_cfg["name"]] = pd.Series([ex.generated_text for ex in res.examples])
    short_name = (
        name.lower().replace(" ", "-").replace(".", "-").replace("#", "-").replace("--", "-")
    )
    df.to_csv(out := Path(cfg.evaluation.local_results) / f"inspect-{short_name}.csv")
    logger.info("Saved results to %s", out)


def inspect(cfg: DictConfig):
    logger.debug("Running inspection with arguments: %s", format_config(cfg))
    results = get_results_wandb(
        cfg.wandb.project,
        cfg.wandb.entity,
        cache_file=cfg.wandb.artifact_cache,
        cache_update=cfg.wandb.cache_update,
    )
    # Do not save out augmented results
    results = [res for res in results if res.metadata.augmenter_key is None]
    scenario_groups = defaultdict(list)
    for res in results:
        scenario_groups[res.metadata.scenario_cfg["name"]].append(res)
    for name, group in scenario_groups.items():
        output_for_inspection(cfg, name, group)  # type: ignore
