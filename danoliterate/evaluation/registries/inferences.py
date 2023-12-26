from omegaconf import DictConfig

from danoliterate.evaluation.execution.api_inference import GoogleApi, OpenAiApi
from danoliterate.evaluation.execution.huggingface_inference import HuggingfaceCausalLm
from danoliterate.evaluation.execution.model_inference import ConstantBaseline
from danoliterate.evaluation.registries.registration import register_inference


@register_inference("baseline", unsupported_metrics=[])
def get_baseline(_: DictConfig) -> ConstantBaseline:
    return ConstantBaseline()


@register_inference("hf-causal", unsupported_metrics=[])
def get_hf_causal(cfg: DictConfig) -> HuggingfaceCausalLm:
    return HuggingfaceCausalLm(
        cfg.model.path,
        batch_size=cfg.model.batch_size,
        download_no_cache=cfg.download_no_cache,
        auto_device_map=cfg.model.inference.get("auto_device_map"),
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
def get_openai_api(cfg: DictConfig) -> OpenAiApi:
    return OpenAiApi(cfg.model.path, cfg.evaluation.api_call_cache)


@register_inference(
    "google-api",
    unsupported_metrics=[
        "max-likelihood-accuracy",
        "max-likelihood-f1",
        "likelihood-brier",
        "likelihood-ece",
    ],
)
def get_google_api(cfg: DictConfig) -> GoogleApi:
    return GoogleApi(cfg.model.path, cfg.evaluation.api_call_cache)
