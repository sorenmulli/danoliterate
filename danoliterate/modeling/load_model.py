import gc
import json
from functools import partial
from io import BytesIO
from typing import Any

import requests
import torch
from huggingface_hub import hf_hub_url
from huggingface_hub.file_download import build_hf_headers
from requests import HTTPError
from safetensors.torch import load_file as safe_load_file
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForCausalLM, GenerationConfig, PreTrainedModel
from transformers.modeling_utils import _load_state_dict_into_model as hf_state_dict_load
from transformers.modeling_utils import no_init_weights
from transformers.models.auto.auto_factory import _get_model_class as hf_get_model_class

from danoliterate.infrastructure.logging import logger


def download(url: str, chunk_size=8192):
    response = requests.get(url, stream=True, timeout=10, headers=build_hf_headers())
    response.raise_for_status()
    total_size = int(response.headers.get("content-length", 0))
    progress_bar = tqdm(total=total_size, unit="iB", unit_scale=True)

    model_data = bytearray()
    for chunk in response.iter_content(chunk_size=chunk_size):
        progress_bar.update(len(chunk))
        model_data.extend(chunk)

    progress_bar.close()
    return BytesIO(model_data)


def get_single(_: str, checkpoint_file: BytesIO, filename: str):
    loader = (
        safe_load_file
        if filename.endswith("safetensors")
        else partial(torch.load, map_location="cpu")
    )
    return loader(checkpoint_file)


def get_sharded(model_key: str, index_file: BytesIO, _: str):
    index = json.load(index_file)
    shard_files = list(set(index["weight_map"].values()))

    state_dict = {}

    for shard_filename in shard_files:
        shard_url = hf_hub_url(model_key, shard_filename)
        shard_file = download(shard_url)
        shard_state_dict = get_single(model_key, shard_file, shard_filename)
        state_dict.update(shard_state_dict)

        # Don't keep each shard in memory
        del shard_state_dict
        gc.collect()
    return state_dict


# TODO: Safetensor still does not work; requires to read from disk
SINGLE_FILE_TO_HANDLER = {
    # "model.safetensors.index.json": get_sharded,
    # "model.safetensors": get_single,
    "pytorch_model.bin.index.json": get_sharded,
    "pytorch_model.bin": get_single,
}


def load_state_dict(model: PreTrainedModel, state_dict: dict[str, Any]) -> PreTrainedModel:
    prefix = model.base_model_prefix + "."
    errors = hf_state_dict_load(model, state_dict, prefix)
    if errors:
        error_msg = ", ".join(errors)
        logger.error("Error loading state dict for model:\n%s\nErrors: %s", str(model), error_msg)
        raise KeyError(f"State dict loading errors: {error_msg}")
    return model


def from_pretrained_hf_hub_no_disk(
    model_key: str,
    model_class=AutoModelForCausalLM,
) -> PreTrainedModel:
    config = AutoConfig.from_pretrained(model_key)
    # pylint: disable=protected-access
    pretrained_model_cls = hf_get_model_class(config, model_class._model_mapping)
    with no_init_weights():
        pretrained_model = model_class.from_config(config)

    state_dict = None
    for filename, handler in SINGLE_FILE_TO_HANDLER.items():
        try:
            url = hf_hub_url(model_key, filename)
            file = download(url)
            state_dict = handler(model_key, file, filename)
            break
        except HTTPError as error:
            logger.debug("No model found in %s due to %s, continuing ...", filename, str(error))

    if state_dict is None:
        logger.error(
            "No handleable checkpoint (out of %s) found in %s.",
            ", ".join(SINGLE_FILE_TO_HANDLER.keys()),
            model_key,
        )
        raise FileNotFoundError("Could not find model checkpoint.")

    # pylint: disable=protected-access
    model, *_ = pretrained_model_cls._load_pretrained_model(
        pretrained_model,
        state_dict,
        list(state_dict.keys()),
        None,
        model_key,
    )
    model.tie_weights()
    model.eval()
    if model.can_generate():
        try:
            model.generation_config = GenerationConfig.from_pretrained(model_key)
        except (OSError, TypeError):
            pass
    return model
