from omegaconf import DictConfig

from nlgenda.evaluation.execution.model_inference import HuggingfaceCausalLm, OpenAiAPI
from nlgenda.evaluation.registries.registration import register_inference


@register_inference("hf-causal")
def get_hf_causal(cfg: DictConfig) -> HuggingfaceCausalLm:
    return HuggingfaceCausalLm(
        cfg.model.path, batch_size=cfg.model.batch_size, download_no_cache=cfg.download_no_cache
    )


@register_inference("openai-api")
def get_openai_api(cfg: DictConfig) -> OpenAiAPI:
    return OpenAiAPI(cfg.path)
