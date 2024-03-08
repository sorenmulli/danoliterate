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


logger = logging.getLogger(__name__)


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
