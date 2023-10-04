from omegaconf import DictConfig
from peft import LoraConfig, PeftModelForCausalLM, TaskType, get_peft_model
from transformers import PreTrainedModel


def setup_lora(model: PreTrainedModel, cfg: DictConfig) -> PeftModelForCausalLM:
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=cfg.r,
        lora_alpha=cfg.alpha,
        lora_dropout=cfg.dropout,
        target_modules=list(cfg.target_modules),
    )
    peft_model = get_peft_model(model, peft_config)
    return peft_model
