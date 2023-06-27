from typing import List

from transformers import AutoTokenizer


class SequenceCollator:
    def __init__(self, hf_tokenizer_repo, max_length):
        self.tokenizer = AutoTokenizer.from_pretrained(hf_tokenizer_repo)
        self.max_length = max_length

    def __call__(self, batch: List[dict]):
        texts = [sample["sequence"] for sample in batch]
        encoded = self._encode(texts, self.max_length)
        return {"encoded": encoded, "batch_data": batch}

    def _encode(self, texts, max_length):
        return self.tokenizer.batch_encode_plus(
            texts,
            add_special_tokens=True,
            truncation=True,
            max_length=max_length,
            padding="max_length",
            return_tensors="pt",
        )
