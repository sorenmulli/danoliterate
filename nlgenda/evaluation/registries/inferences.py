from omegaconf import DictConfig

from nlgenda.evaluation.execution.model_inference import (
    HuggingfaceCausalLm,
    OpenAiAPI,
    ReflectiveBaseline,
)
from nlgenda.evaluation.registries.registration import register_inference


@register_inference("baseline", unsupported_metrics=[])
def get_baseline(_: DictConfig) -> ReflectiveBaseline:
    return ReflectiveBaseline()


@register_inference("hf-causal", unsupported_metrics=[])
def get_hf_causal(cfg: DictConfig) -> HuggingfaceCausalLm:
    return HuggingfaceCausalLm(
        cfg.model.path, batch_size=cfg.model.batch_size, download_no_cache=cfg.download_no_cache
    )


@register_inference(
    "openai-api",
    unsupported_metrics=[
        "max-likelihood-accuracy",
        "max-likelihood-f1",
        "likelihood-brier",
        "likelihood-ece",
    ],
)
def get_openai_api(cfg: DictConfig) -> OpenAiAPI:
    return OpenAiAPI(cfg.model.path, cfg.evaluation.api_call_cache)
