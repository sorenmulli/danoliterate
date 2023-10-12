from omegaconf import DictConfig

from nlgenda.evaluation.execution.model_inference import HuggingfaceCausalLm, OpenAiAPI
from nlgenda.evaluation.registries.registration import register_inference


@register_inference("hf-causal")
def get_hf_causal(cfg: DictConfig) -> HuggingfaceCausalLm:
    return HuggingfaceCausalLm(cfg.path, batch_size=cfg.batch_size)


@register_inference("openai-api")
def get_openai_api(cfg: DictConfig) -> OpenAiAPI:
    return OpenAiAPI(cfg.path)
