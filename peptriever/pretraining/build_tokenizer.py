from pathlib import Path

from datasets import load_dataset

from peptriever.pretraining.build_bpe_tokenizer import build_bpe_tokenizer
from peptriever.pretraining.config import PretrainingConfig


def build_tokenizer(seq_repo: str, vocab_size: int, output_path: Path):
    dataset = load_dataset(seq_repo)
    docs = iterate_sequences(dataset)
    build_bpe_tokenizer(
        docs=docs,
        vocab_size=vocab_size,
        output_path=output_path,
        unk_token=config.unk_token,
        pad_token=config.pad_token,
        mask_token=config.mask_token,
    )


def iterate_sequences(dataset):
    for row in dataset["train"].to_iterable_dataset():
        if row["train_part"] in ("train", "val"):
            yield row["sequence"]


if __name__ == "__main__":
    config = PretrainingConfig()
    build_tokenizer(
        seq_repo=config.hf_sequence_repo_name,
        vocab_size=config.vocab_size,
        output_path=config.tokenizer_path,
    )
