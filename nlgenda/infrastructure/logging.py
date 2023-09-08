import json

from omegaconf import DictConfig, OmegaConf


def format_config(cfg: DictConfig) -> str:
    return json.dumps(OmegaConf.to_container(cfg), indent=1)
