from typing import Callable

from omegaconf import DictConfig

from danoliterate.evaluation.execution.api_inference import DanskGptAPi, GoogleApi, OpenAiApi
from danoliterate.evaluation.execution.huggingface_inference import HuggingfaceCausalLm
from danoliterate.evaluation.execution.model_inference import ConstantBaseline, ModelInference

InferenceFunctionType = Callable[[DictConfig], ModelInference]

inference_registry: dict[str, InferenceFunctionType] = {}
inference_unsupported_metrics_registry: dict[str, list[str]] = {}


class UnknownInference(KeyError):
    """A modle inference key was given without a registered model inference"""


def register_inference(
    inference_name: str,
    unsupported_metrics: list[str],
) -> Callable[[InferenceFunctionType], InferenceFunctionType]:
    def decorator(func: InferenceFunctionType) -> InferenceFunctionType:
        if inference_name in inference_registry:
            raise ValueError(f"Model inference {inference_name} registered more than once!")
        inference_registry[inference_name] = func
        inference_unsupported_metrics_registry[inference_name] = unsupported_metrics
        return func

    return decorator


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


@register_inference(
    "danskgpt-api",
    unsupported_metrics=[
        "max-likelihood-accuracy",
        "max-likelihood-f1",
        "likelihood-brier",
        "likelihood-ece",
    ],
)
def get_danskgpt_api(cfg: DictConfig) -> DanskGptAPi:
    return DanskGptAPi(cfg.model.path, cfg.evaluation.api_call_cache)


def get_inference(cfg: DictConfig):
    inference_name = cfg.model.inference.type
    try:
        return inference_registry[inference_name](cfg)
    except KeyError as error:
        raise UnknownInference(
            f"No inference registered with model.inference.type {inference_name}"
        ) from error
