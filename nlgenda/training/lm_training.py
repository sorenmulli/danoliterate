import logging
from pathlib import Path

import torch
import wandb
from omegaconf import DictConfig, OmegaConf
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

from nlgenda.datasets.pretraining import get_streaming_data, tokenize_datasets
from nlgenda.infrastructure.logging import format_config
from nlgenda.infrastructure.runs import run_dir, run_name
from nlgenda.modeling.load_model import from_pretrained_hf_hub_no_disk
from nlgenda.training.efficiency import setup_lora

logger = logging.getLogger(__name__)


def get_arguments(cfg: DictConfig, wandb_enabled: bool):
    return TrainingArguments(
        run_dir(),
        # Training steps
        max_steps=cfg.steps,
        warmup_steps=cfg.warmup_steps,
        # Optimisation
        optim=cfg.optimizer,
        learning_rate=cfg.lr,
        lr_scheduler_type=cfg.scheduler,
        weight_decay=cfg.weight_decay,
        # Batch sizes
        per_device_train_batch_size=cfg.batch_size,
        gradient_accumulation_steps=cfg.accumulation,
        per_device_eval_batch_size=cfg.eval_batch_size,
        eval_accumulation_steps=cfg.eval_accumulation,
        auto_find_batch_size=True,
        # Compute
        fp16=cfg.fp16 if torch.cuda.is_available() else False,
        # Evaluation
        load_best_model_at_end=cfg.eval,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to=["wandb"] if wandb_enabled else [],
        eval_steps=cfg.eval_every,
        save_steps=cfg.eval_every,
        save_total_limit=cfg.save_limit,
        logging_steps=cfg.log_every,
        evaluation_strategy="steps" if cfg.eval else "no",
        save_strategy="steps",
        logging_strategy="steps",
        # Data loading
        dataloader_num_workers=cfg.data.workers,
    )


def train_lm(cfg: DictConfig):
    logger.debug("Running with arguments: %s", format_config(cfg))

    logger.info("Setting up model and tokenizer from %s ...", cfg.train.base_model)
    model_cls = AutoModelForCausalLM
    model = (
        from_pretrained_hf_hub_no_disk(cfg.train.base_model, model_cls)
        if cfg.download_no_cache
        else model_cls.from_pretrained(cfg.train.base_model)
    )
    tokenizer = AutoTokenizer.from_pretrained(cfg.train.base_model)
    logger.info("Loaded model with %.1f M parameters.", model.num_parameters() / 1e6)
    logger.info("Loaded tokenizer with vocabulary size %i.", tokenizer.vocab_size)

    if cfg.train.lora.enabled:
        model = setup_lora(model, cfg.train.lora)
        logger.info(
            "Set up PEFT with LoRa.\n - Trainable parameters:\t%i\n - Total parameters:\t\t%i",
            *model.get_nb_trainable_parameters()
        )

    logger.info("Setting up streaming datasets %s ...", " + ".join(cfg.train.data.datasets))
    datasets = get_streaming_data(cfg.train.data)
    datasets = tokenize_datasets(datasets, tokenizer, cfg)
    logger.info(
        "Datasets set up! %i were skipped for test, %i are validation, rest train. "
        "Splits were divided after shuffle with seed=%i",
        cfg.train.data.test_examples,
        cfg.train.data.validation_examples,
        cfg.train.data.seed,
    )

    collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    logger.info("Setting up trainer%s...", " and W&B integration" if cfg.wandb.enabled else "")
    if cfg.wandb.enabled:
        wandb.init(
            name=run_name(),
            entity=cfg.wandb.entity,
            project=cfg.wandb.project,
            reinit=True,
            dir=run_dir(),
            job_type="train",
            config=OmegaConf.to_container(cfg),  # type: ignore
        )
    train_args = get_arguments(cfg.train, wandb_enabled=cfg.wandb.enabled)
    trainer = Trainer(
        model,
        train_args,
        tokenizer=tokenizer,
        train_dataset=datasets["train"],
        eval_dataset=datasets["val"] if cfg.train.eval else None,
        data_collator=collator,
    )

    logger.info(
        "Everything ready! Here we go %s🚀🚀🚀", "(resuming previous run) " if cfg.train.resume else ""
    )
    result = trainer.train(resume_from_checkpoint=cfg.train.resume)
    output_dir = Path(trainer.args.output_dir) / "checkpoint-best"
    trainer.save_model(output_dir=output_dir)
    logger.info("Training complete 🎆🎆🎆. Saved to %s. Got result:\n%s", output_dir, result)
