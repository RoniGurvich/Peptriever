import os
from pathlib import Path
from typing import Iterator

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, processors, trainers
from tokenizers.normalizers import NFKC


def build_bpe_tokenizer(
    docs: Iterator[str],
    vocab_size: int,
    output_path: Path,
    unk_token: str,
    pad_token: str,
    mask_token: str,
):
    tokenizer = Tokenizer(models.BPE(unk_token=unk_token))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.post_processor = processors.ByteLevel(trim_offsets=True)
    tokenizer.normalizer = NFKC()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=2,
        special_tokens=[unk_token, pad_token, mask_token],
    )
    tokenizer.train_from_iterator(iterator=docs, trainer=trainer)
    pad_id = tokenizer.token_to_id(pad_token)
    tokenizer.enable_padding(pad_id=pad_id, pad_token=pad_token)
    save_tokenizer(output_path, tokenizer)


def save_tokenizer(output_path: Path, tokenizer: Tokenizer):
    os.makedirs(output_path, exist_ok=True)
    tokenizer.save(os.path.join(output_path, "tokenizer.json"))
    tokenizer.model.save(str(output_path))
