from omegaconf import DictConfig

from nlgenda.evaluation.model_inference import HuggingfaceCausalLm, OpenAiAPI
from nlgenda.evaluation.registries.registration import register_inference


@register_inference("hf-causal")
def get_hf_causal(cfg: DictConfig) -> HuggingfaceCausalLm:
    return HuggingfaceCausalLm(cfg.inference.method, cfg.path)


@register_inference("openai-api")
def get_openai_api(cfg: DictConfig) -> OpenAiAPI:
    return OpenAiAPI(cfg.path)
