from pathlib import Path

import torch
from torch.utils.data import Dataset

from peptriever.pdb_seq_lookup import get_pdb_lookup
from peptriever.pretraining.load_tokenizer import load_local_tokenizer


class MaskedLMDataset(Dataset):
    def __init__(
        self,
        seq_repo_name: str,
        train_part: str,
    ):
        self.sequence_map = get_pdb_lookup(seq_repo_name, train_part)
        self.keys = sorted(self.sequence_map.keys())

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, index):
        key = self.keys[index]
        value = self.sequence_map[key]
        return value


class MLMCollator:
    def __init__(
        self,
        tokenizer_path: Path,
        mask_token: str,
        max_length: int,
        mask_ratio: float = 0.15,
        mask_token_ratio: float = 0.8,
        random_token_ratio: float = 0.1,
    ):
        self.mask_token = mask_token
        self.tokenizer = self.get_tokenizer(tokenizer_path)
        self.max_length = max_length
        self.mask_ratio = mask_ratio
        self.mask_token_ratio = mask_token_ratio
        self.random_token_ratio = random_token_ratio
        self._mask_token_id = None

    @staticmethod
    def get_tokenizer(tokenizer_path):
        return load_local_tokenizer(tokenizer_path=tokenizer_path)

    def get_mask_token_id(self):
        if self._mask_token_id is None:
            self._mask_token_id = self.tokenizer.encode(self.mask_token)[0]
        return self._mask_token_id

    def __call__(self, batch):
        encoded = self.tokenizer.batch_encode_plus(
            batch,
            add_special_tokens=True,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        masked_tokens, labels = self.mask_encoded_batch(encoded)
        encoded["input_ids"] = masked_tokens
        encoded["labels"] = labels
        return encoded

    def mask_encoded_batch(self, encoded):
        masked_tokens = encoded.input_ids.clone()

        input_shape = encoded.input_ids.shape

        mask_proba = torch.full(input_shape, self.mask_ratio) * encoded.attention_mask
        relevant_inds = torch.bernoulli(mask_proba).bool()

        masked_inds = (
            torch.bernoulli(torch.full(input_shape, self.mask_token_ratio)).bool()
            & relevant_inds
        )
        random_inds = (
            torch.bernoulli(
                torch.full(
                    input_shape, self.random_token_ratio / (1 - self.mask_token_ratio)
                )
            ).bool()
            & relevant_inds
            & ~masked_inds
        )
        random_tokens = torch.randint(
            int(len(self.tokenizer)), input_shape, dtype=masked_tokens.dtype
        )
        masked_tokens[masked_inds] = self.get_mask_token_id()
        masked_tokens[random_inds] = random_tokens[random_inds]

        labels = encoded.input_ids.clone()
        labels[~relevant_inds] = -100

        return masked_tokens, labels
