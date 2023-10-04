import json
from typing import Optional

from git import InvalidGitRepositoryError, Repo
from omegaconf import DictConfig, OmegaConf

from nlgenda.infrastructure.constants import REPO_PATH


def format_config(cfg: DictConfig) -> str:
    return json.dumps(OmegaConf.to_container(cfg), indent=1)


def commit_hash() -> Optional[str]:
    try:
        return str(Repo(REPO_PATH).head.commit)
    except InvalidGitRepositoryError:
        return None
