import json
import logging
from typing import Optional

from git import InvalidGitRepositoryError, Repo
from omegaconf import DictConfig, OmegaConf

from danoliterate.infrastructure.constants import REPO_PATH

TORCH_IMPORTED = True
try:
    import torch
except ImportError:
    TORCH_IMPORTED = False


logger = logging.getLogger("danoliterate")


def format_config(cfg: DictConfig) -> str:
    return json.dumps(OmegaConf.to_container(cfg), indent=1)


def commit_hash() -> Optional[str]:
    try:
        return str(Repo(REPO_PATH).head.commit)
    except InvalidGitRepositoryError:
        return None


def get_compute_unit_string() -> str:
    if not TORCH_IMPORTED:
        return "CPU"
    if torch.cuda.is_available():
        return torch.cuda.get_device_name(0)
    return "CPU"


def maybe_setup_wandb_logging_run(name: str, job_type: str, wandb_cfg: DictConfig):
    if wandb_cfg.enabled:
        try:
            # pylint: disable=import-outside-toplevel
            import wandb
        except ImportError as error:
            raise ImportError("You need to install wandb to run with wandb.enabled=true") from error
        wandb.init(
            name=name,
            job_type=job_type,
            entity=wandb_cfg.entity,
            project=wandb_cfg.project,
        )
