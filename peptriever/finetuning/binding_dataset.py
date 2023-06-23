from typing import List

from datasets import load_dataset
from torch.utils.data import Dataset
from transformers import AutoTokenizer

from peptriever.pretraining.mlm_dataset import MLMCollator


class BindingDataset(Dataset):
    def __init__(self, repo_id: str, train_part: str):
        full_dataset = load_dataset(repo_id)["train"]
        pairs = _filter_train_part_pairs(full_dataset, train_part)
        self.pairs = pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index: int):
        pair = self.pairs[index]

        pep = pair["peptide"]
        prot = pair["receptor"]
        return {
            "x1": pep,
            "x2": prot,
        }


def _filter_train_part_pairs(full_dataset, train_part):
    pairs = []
    for row in full_dataset.to_iterable_dataset():
        if row["train_part"] == train_part:
            pairs.append(row)
    return pairs


class BindingCollator(MLMCollator):
    def __init__(
        self,
        hf_tokenizer_repo,
        max_length1,
        max_length2,
        mask_token: str,
        mask_ratio: float = 0.15,
        mask_token_ratio: float = 0.8,
        random_token_ratio: float = 0.1,
    ):
        super().__init__(
            tokenizer_path=hf_tokenizer_repo,
            mask_token=mask_token,
            max_length=-1,
            mask_ratio=mask_ratio,
            mask_token_ratio=mask_token_ratio,
            random_token_ratio=random_token_ratio,
        )
        self.max_length1, self.max_length2 = max_length1, max_length2

    @staticmethod
    def get_tokenizer(tokenizer_path):
        return AutoTokenizer.from_pretrained(tokenizer_path)

    def __call__(self, batch: List[dict]):
        texts1, texts2 = zip(*[(sample["x1"], sample["x2"]) for sample in batch])
        x1 = self._encode(texts1, self.max_length1)
        x2 = self._encode(texts2, self.max_length2)
        x1_masked, labels1 = self.mask_encoded_batch(x1)
        x2_masked, labels2 = self.mask_encoded_batch(x2)
        x1["input_ids"] = x1_masked
        x2["input_ids"] = x2_masked
        return {
            "x1": x1,
            "x2": x2,
            "labels1": labels1,
            "labels2": labels2,
        }

    def _encode(self, texts, max_length):
        return self.tokenizer.batch_encode_plus(
            texts,
            add_special_tokens=True,
            truncation=True,
            max_length=max_length,
            padding="max_length",
            return_tensors="pt",
        )
