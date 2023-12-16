from pathlib import Path

import hydra


def run_dir() -> Path:
    return Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)


def run_name() -> str:
    return hydra.core.hydra_config.HydraConfig.get().job.name
