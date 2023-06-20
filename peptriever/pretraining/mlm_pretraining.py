import os
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import torch.optim
from torch.utils.data import DataLoader
from transformers import BertConfig, BertForMaskedLM, Trainer, TrainingArguments

from peptriever.acceleration import compile_model
from peptriever.pretraining.config import PretrainingConfig
from peptriever.pretraining.load_tokenizer import load_local_tokenizer
from peptriever.pretraining.mlm_dataset import MLMCollator, MaskedLMDataset
from peptriever.reproducibility import set_seed


@dataclass
class MLMConfig:
    batch_size: int = 24
    epochs: int = 20
    lr: float = 1e-5
    steps_interval: int = 10_000
    resume_training: Optional[str] = None
    n_workers: int = os.cpu_count() or 1


def pretrain_mlm_multiple_models(config: PretrainingConfig, mlm_config: MLMConfig):
    now = datetime.utcnow().isoformat()
    model_names = []
    for max_length in config.max_lengths:
        model_name = f"base_protein_{max_length}_{now}"
        pretrain_mlm_single_model(
            config=config,
            mlm_config=mlm_config,
            model_name=model_name,
            max_length=max_length,
        )
        model_names.append(model_name)
    return model_names


def pretrain_mlm_single_model(config, mlm_config: MLMConfig, model_name, max_length):
    set_seed(seed=999)
    model = get_model(
        config, max_length=max_length, resume_training=mlm_config.resume_training
    )

    model_path = config.models_dir / model_name
    trainer_args = TrainingArguments(
        output_dir=model_path,
        do_train=True,
        do_eval=True,
        overwrite_output_dir=True,
        save_steps=mlm_config.steps_interval,
        evaluation_strategy="steps",
        eval_steps=mlm_config.steps_interval,
        logging_dir=os.path.join("tensorboard", model_name),
        logging_steps=mlm_config.steps_interval,
        per_device_train_batch_size=mlm_config.batch_size,
        per_device_eval_batch_size=mlm_config.batch_size,
        data_seed=999,
        learning_rate=mlm_config.lr,
        num_train_epochs=mlm_config.epochs,
        save_total_limit=2,
        dataloader_num_workers=mlm_config.n_workers,
        report_to=["tensorboard"],
    )
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=mlm_config.lr)
    trainer = Trainer(
        model=model,
        args=trainer_args,
        train_dataset=MaskedLMDataset(
            seq_repo_name=config.hf_sequence_repo_name, train_part="train"
        ),
        eval_dataset=MaskedLMDataset(
            seq_repo_name=config.hf_sequence_repo_name, train_part="val"
        ),
        data_collator=MLMCollator(
            tokenizer_path=config.tokenizer_path,
            mask_token=config.mask_token,
            max_length=max_length,
        ),
        optimizers=(
            optimizer,
            torch.optim.lr_scheduler.ConstantLR(optimizer=optimizer, factor=1.0),
        ),
    )

    try:
        trainer.train()
    except KeyboardInterrupt:
        print("Stopped manually")

    model.save_pretrained(model_path)


def get_model(config, max_length, resume_training: Optional[str]):
    if resume_training is None:
        tokenizer = load_local_tokenizer(config.tokenizer_path)
        bert_vocab_size = len(tokenizer.all_special_tokens) + tokenizer.vocab_size
        bert_config = BertConfig(
            vocab_size=bert_vocab_size,
            max_position_embeddings=max_length,
            pad_token_id=tokenizer.pad_token_id,
        )
        model = BertForMaskedLM(config=bert_config)
        model = compile_model(model)
    else:
        model = BertForMaskedLM.from_pretrained(
            str(config.models_dir / resume_training)
        )
    return model


def get_dataloaders(tokenizer_path, seq_repo_name, max_length, mask_token, batch_size):
    n_workers = os.cpu_count() or 1
    train_dataset = MaskedLMDataset(seq_repo_name=seq_repo_name, train_part="train")
    train_collator = MLMCollator(
        tokenizer_path=tokenizer_path,
        mask_token=mask_token,
        max_length=max_length,
    )
    train_dataloader = DataLoader(
        train_dataset,
        num_workers=n_workers,
        shuffle=True,
        collate_fn=train_collator,
        batch_size=batch_size,
    )
    val_dataset = MaskedLMDataset(seq_repo_name=seq_repo_name, train_part="val")
    val_collator = MLMCollator(
        tokenizer_path=tokenizer_path,
        mask_token=mask_token,
        max_length=max_length,
    )
    val_dataloader = DataLoader(
        val_dataset,
        num_workers=n_workers,
        shuffle=False,
        collate_fn=val_collator,
        batch_size=batch_size,
    )
    return train_dataloader, val_dataloader


if __name__ == "__main__":
    pretrain_mlm_multiple_models(config=PretrainingConfig(), mlm_config=MLMConfig())
