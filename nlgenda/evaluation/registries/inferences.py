from omegaconf import DictConfig

from nlgenda.evaluation.model_inference import HuggingfaceCausalLm
from nlgenda.evaluation.registries.registration import register_inference


@register_inference("hf-causal")
def get_hf_causal(cfg: DictConfig) -> HuggingfaceCausalLm:
    return HuggingfaceCausalLm(cfg.inference.method, cfg.path)
