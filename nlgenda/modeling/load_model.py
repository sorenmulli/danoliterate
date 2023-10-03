import gc
import json
from io import BytesIO

import requests
import torch
from huggingface_hub import hf_hub_url
from huggingface_hub.file_download import build_hf_headers
from requests import HTTPError
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForCausalLM, PreTrainedModel
from transformers.modeling_utils import _load_state_dict_into_model as hf_state_dict_load
from transformers.modeling_utils import no_init_weights


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

        del shard_state_dict
        gc.collect()
    return state_dict


# TODO: Safetensors
SINGLE_FILE_TO_HANDLER = {
    # "model.safetensors.index.json": get_sharded,
    "pytorch_model.bin.index.json": get_sharded,
    # "model.safetensors": get_single,
    "pytorch_model.bin": get_single,
}
#     loader = safe_load_file if load_safe else partial(torch.load, map_location="cpu")


def load_state_dict(model: PreTrainedModel, state_dict) -> PreTrainedModel:
    prefix = model.base_model_prefix + "."
    errors = hf_state_dict_load(model, state_dict, prefix)
    if errors:
        raise RuntimeError(",".join(errors))
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
        except HTTPError as error:
            print(error)

    if state_dict is None:
        raise FileNotFoundError("AHHH")

    with no_init_weights():
        model = model_class.from_config(config)
    return load_state_dict(model, state_dict)


# if __name__ == "__main__":
#     # example_model = from_pretrained_hf_hub_no_disk("hf-internal-testing/tiny-random-gpt2")
#     example_model = from_pretrained_hf_hub_no_disk("jonatanklosko/test-tiny-gpt2-sharded")
#     breakpoint()
