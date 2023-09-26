import torch
from transformers import AutoModelForCausalLM, PreTrainedTokenizer

IGNORE_TARGET_IDX = -100

# TODO: Should probably handle models like objects instead
# TODO: Handle too long sequences
# TODO: How do we handle special tokens?
# TODO: Device


def perplexity(
    context: str, target: str, tokenizer: PreTrainedTokenizer, model: AutoModelForCausalLM
) -> float:
    encodings = tokenizer(context, text_target=target, return_tensors="pt")
    input_ids = torch.cat((encodings.input_ids, encodings.labels), dim=1)
    target_ids = input_ids.clone()
    target_ids[:, : encodings.input_ids.size(1)] = -100
    with torch.no_grad():
        outputs = model(input_ids, labels=target_ids)
    return torch.exp(outputs.loss).item()
