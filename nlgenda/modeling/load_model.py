import gc
import json
import logging
from io import BytesIO
from typing import Any

import requests
import torch
from huggingface_hub import hf_hub_url
from huggingface_hub.file_download import build_hf_headers
from requests import HTTPError
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForCausalLM, PreTrainedModel
from transformers.modeling_utils import _load_state_dict_into_model as hf_state_dict_load
from transformers.modeling_utils import no_init_weights

logger = logging.getLogger(__name__)


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


def get_single(_: str, checkpoint_file: BytesIO):
    return torch.load(checkpoint_file, map_location="cpu")


def get_sharded(model_key: str, index_file: BytesIO):
    index = json.load(index_file)
    shard_files = list(set(index["weight_map"].values()))

    state_dict = {}

    for shard_filename in shard_files:
        shard_url = hf_hub_url(model_key, shard_filename)
        shard_file = download(shard_url)
        shard_state_dict = get_single(model_key, shard_file)
        state_dict.update(shard_state_dict)

        # Don't keep each shard in memory
        del shard_state_dict
        gc.collect()
    return state_dict


# TODO: Add functionality for loading from safetensors, see #16

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

    state_dict = None
    for filename, handler in SINGLE_FILE_TO_HANDLER.items():
        try:
            url = hf_hub_url(model_key, filename)
            file = download(url)
            state_dict = handler(model_key, file)
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

    with no_init_weights():
        model = model_class.from_config(config)
    return load_state_dict(model, state_dict)


# if __name__ == "__main__":
#     # example_model = from_pretrained_hf_hub_no_disk("hf-internal-testing/tiny-random-gpt2")
#     example_model = from_pretrained_hf_hub_no_disk("jonatanklosko/test-tiny-gpt2-sharded")
#     breakpoint()
