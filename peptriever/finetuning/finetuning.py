import os
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Tuple

import torch.optim
import torchinfo
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizerFast

from peptriever.acceleration import compile_model
from peptriever.finetuning.binding_dataset import BindingCollator, BindingDataset
from peptriever.finetuning.config import FinetuningConfig
from peptriever.finetuning.loss import EuclideanCombinedMLMMarginLoss
from peptriever.finetuning.lr_scheduler import get_lr_scheduler
from peptriever.finetuning.peptriever_trainiing_session import (
    PeptrieverTrainingSession,
    SessionParams,
)
from peptriever.model.bert_embedding import BertEmbeddingConfig, BertForEmbedding
from peptriever.model.bi_encoder import BiEncoderConfig, BiEncoderWithMaskedLM


@dataclass
class TrainParams:
    batch_size: int = 16
    epochs: int = 80
    lr: float = 1e-5
    grad_accumulation: int = 1
    n_data_workers: int = os.cpu_count() or 1
    resume_training: Optional[str] = None
    pretrained_weights: Optional[Tuple[str, str]] = None
    warmup: int = 10
    cooldown: int = 50


def finetune_protein_binding_model(config: FinetuningConfig, train_params: TrainParams):
    collator = BindingCollator(
        hf_tokenizer_repo=config.hf_tokenizer_repo,
        max_length1=config.tokenizer1_max_length,
        max_length2=config.tokenizer2_max_length,
        mask_token=config.mask_token,
    )
    model = setup_model(
        config=config,
        tokenizer=collator.tokenizer,
        resume_training=train_params.resume_training,
        pretrained_weights=train_params.pretrained_weights,
    )
    model = compile_model(model)
    torchinfo.summary(model)

    train_data, val_data = setup_dataloaders(
        config=config, train_params=train_params, collator=collator
    )

    optimizer = torch.optim.Adam(params=model.parameters(), lr=train_params.lr)
    scheduler = get_lr_scheduler(
        optimizer=optimizer,
        warmup=train_params.warmup,
        cooldown=train_params.cooldown,
        epochs=train_params.epochs,
    )
    now = datetime.utcnow().isoformat()
    model_name = f"peptriever_{now}"

    loss = _get_loss(config.distance_function)
    output_path = config.models_path / model_name

    session = PeptrieverTrainingSession(
        model=model,
        model_name=model_name,
        loss_func=loss,
        optimizer=optimizer,
        lr_scheduler=scheduler,
        dist_metric=config.distance_function,
        session_params=SessionParams(train_params.grad_accumulation),
        output_path=config.models_path,
    )

    try:
        session.train(
            train_data=train_data, val_data=val_data, epochs=train_params.epochs
        )
    except KeyboardInterrupt:
        print("stopped manually")

    session.save(output_path=output_path)
    return model_name


def _get_loss(dist_func):
    if dist_func == "euclidean":
        loss = EuclideanCombinedMLMMarginLoss()
    else:
        raise NotImplementedError(dist_func)
    return loss


def setup_dataloaders(config: FinetuningConfig, train_params: TrainParams, collator):
    train_dataset, val_dataset = get_datasets(config.hf_binding_data_repo_id)
    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=train_params.batch_size,
        shuffle=True,
        num_workers=train_params.n_data_workers,
        collate_fn=collator,
    )
    val_dataloader = DataLoader(
        dataset=val_dataset,
        batch_size=train_params.batch_size,
        shuffle=False,
        num_workers=train_params.n_data_workers,
        collate_fn=collator,
    )
    return train_dataloader, val_dataloader


def get_datasets(repo_id):
    train_dataset = BindingDataset(repo_id=repo_id, train_part="train")
    val_dataset = BindingDataset(repo_id=repo_id, train_part="val")
    return train_dataset, val_dataset


def setup_model(
    config: FinetuningConfig,
    tokenizer: PreTrainedTokenizerFast,
    pretrained_weights: Optional[Tuple[str, str]] = None,
    resume_training: Optional[str] = None,
):
    if pretrained_weights is not None and resume_training is not None:
        raise ValueError(
            "Pretrained weights and resume_training were provided at the same time"
        )

    bert_config = _get_bert_config(config, tokenizer)
    model = BiEncoderWithMaskedLM(config=bert_config)
    if resume_training is not None:
        full_path = config.models_path / resume_training
        model = BiEncoderWithMaskedLM.from_pretrained(str(full_path))

    if pretrained_weights is not None:
        pretrain_path1 = str(config.models_path / pretrained_weights[0])
        model.bert1 = load_bert_embeddings_from_mlm_pretrained(
            pretrained_path=pretrain_path1,
            output_dims=config.encoded_sequence_dims,
            max_length=bert_config.max_length1,
        )
        pretrain_path2 = str(config.models_path / pretrained_weights[1])
        model.bert2 = load_bert_embeddings_from_mlm_pretrained(
            pretrained_path=pretrain_path2,
            output_dims=config.encoded_sequence_dims,
            max_length=bert_config.max_length2,
        )
    return model


def load_bert_embeddings_from_mlm_pretrained(
    pretrained_path: str, output_dims: int, max_length: int
):
    conf = BertEmbeddingConfig.from_pretrained(pretrained_path)
    conf.n_output_dims = output_dims
    conf.max_position_embeddings = max_length
    bert = BertForEmbedding.from_pretrained(str(pretrained_path), config=conf)
    return bert


def _get_bert_config(config: FinetuningConfig, tokenizer: PreTrainedTokenizerFast):
    bert_vocab_size = len(tokenizer.all_special_tokens) + tokenizer.vocab_size
    return BiEncoderConfig(
        vocab_size=bert_vocab_size,
        max_length1=config.tokenizer1_max_length,
        max_length2=config.tokenizer2_max_length,
        pad_token_id=tokenizer.pad_token_id,
        n_output_dims=config.encoded_sequence_dims,
        distance_func=config.distance_function,
    )
