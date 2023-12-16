import re
from pathlib import Path

from omegaconf import DictConfig
from peft import LoraConfig, PeftModelForCausalLM, TaskType, get_peft_model
from transformers import PreTrainedModel

from danoliterate.infrastructure.runs import run_dir


def setup_lora(model: PreTrainedModel, lora_cfg: DictConfig) -> PeftModelForCausalLM:
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=lora_cfg.r,
        lora_alpha=lora_cfg.alpha,
        lora_dropout=lora_cfg.dropout,
        target_modules=list(lora_cfg.target_modules),
    )
    peft_model = get_peft_model(model, peft_config)
    return peft_model


def _checkpoint_sort_key(checkpoint: Path) -> float:
    _match = re.search(r"-([\d]+)", str(checkpoint))
    return float(_match.group(1)) if _match else float("inf")


def resume_lora(model: PreTrainedModel, _: DictConfig) -> PeftModelForCausalLM:
    checkpoints = sorted(run_dir().glob("checkpoint-*"), key=_checkpoint_sort_key)
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoints found for PEFT at {run_dir()}")
    checkpoint = checkpoints[0]
    return PeftModelForCausalLM.from_pretrained(model, checkpoint, is_trainable=True)
