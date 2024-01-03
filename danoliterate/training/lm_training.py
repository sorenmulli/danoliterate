from pathlib import Path

import torch
import wandb
from accelerate import Accelerator
from omegaconf import DictConfig, OmegaConf
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)
from transformers import logging as transformers_logging
from trl import SFTTrainer

from danoliterate.data.pretraining import (
    ConstantLengthDatasetRandomSubsequence,
    get_streaming_data,
    tokenize_datasets,
)
from danoliterate.infrastructure.logging import format_config, logger
from danoliterate.infrastructure.runs import run_dir, run_name
from danoliterate.modeling.load_model import from_pretrained_hf_hub_no_disk
from danoliterate.training.efficiency import resume_lora, setup_lora


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
        bf16=cfg.bf16 if torch.cuda.is_available() else False,
        # Evaluation
        evaluation_strategy="steps" if cfg.eval else "no",
        eval_steps=cfg.eval_every,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        # Saving
        save_strategy="steps",
        save_steps=cfg.save_every,
        save_total_limit=cfg.save_limit,
        load_best_model_at_end=cfg.eval,
        # Logging
        report_to=["wandb"] if wandb_enabled else [],
        logging_steps=cfg.log_every,
        logging_strategy="steps",
        # Data loading
        dataloader_num_workers=cfg.data.workers,
        ignore_data_skip=True,
    )


def train_lm(cfg: DictConfig):
    logger.debug("Running with arguments: %s", format_config(cfg))
    transformers_logging.set_verbosity_info()

    logger.info("Setting up model and tokenizer from %s ...", cfg.train.base_model)
    model_cls = AutoModelForCausalLM
    model = (
        from_pretrained_hf_hub_no_disk(cfg.train.base_model, model_cls)
        if cfg.download_no_cache
        else model_cls.from_pretrained(cfg.train.base_model)
    )
    if cfg.train.reinit:
        # pylint: disable=protected-access
        model = model.apply(model._init_weights)
        logger.info("Randomly reinitialized model paramters")

    logger.info("Loaded model with %.1f M parameters.", model.num_parameters() / 1e6)
    tokenizer = AutoTokenizer.from_pretrained(cfg.train.base_model)
    logger.info("Loaded tokenizer with vocabulary size %i.", tokenizer.vocab_size)

    if cfg.train.lora.enabled:
        assert not cfg.train.use_sft
        model = (
            resume_lora(model, cfg.train.lora)
            if cfg.train.resume
            else setup_lora(model, cfg.train.lora)
        )
        logger.info(
            "Set up PEFT with LoRa.\n - Trainable parameters:\t%i\n - Total parameters:\t\t%i",
            *model.get_nb_trainable_parameters(),
        )

    logger.info("Setting up streaming datasets %s ...", " + ".join(cfg.train.data.datasets))
    datasets = get_streaming_data(cfg.train.data)
    if cfg.train.use_sft:
        tokenizer.padding_side = "right"
        datasets = {
            name: ConstantLengthDatasetRandomSubsequence(
                tokenizer,
                dataset,
                cfg.train.data.text_col,
                seq_length=cfg.train.data.context_tokens,
                shuffle=name == "train",
                one_seq_per_example=name == "train",
                save_data_debug=None
                if cfg.train.data.debug_data is None
                else Path(cfg.train.data.debug_data) / f"{name}.txt",
                # Increase buffer
                num_of_sequences=4096,
            )
            for name, dataset in datasets.items()
        }
    else:
        datasets = tokenize_datasets(datasets, tokenizer, cfg)
    logger.info(
        "Datasets set up! %i were skipped for test, %i are validation, rest train. "
        "Splits were divided after shuffle with seed=%i",
        cfg.train.data.test_examples,
        cfg.train.data.validation_examples,
        cfg.train.data.seed,
    )

    logger.info("Setting up trainer%s...", " and W&B integration" if cfg.wandb.enabled else "")
    if cfg.wandb.enabled and Accelerator().is_main_process:
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
    if cfg.train.use_sft:
        trainer = SFTTrainer(
            model,
            train_args,
            train_dataset=datasets["train"],
            eval_dataset=datasets["val"] if cfg.train.eval else None,
            max_seq_length=cfg.train.data.context_tokens,
            dataset_text_field=cfg.train.data.text_col,
        )
    else:
        collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
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
