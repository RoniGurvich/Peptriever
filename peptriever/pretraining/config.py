from dataclasses import dataclass
from pathlib import Path
from typing import Tuple


@dataclass
class PretrainingConfig:
    hf_user_id: str = "ronig"

    dataset_path: Path = Path("/data/training_sets/external/pdb")
    tokenizer_path: Path = Path("/data/models/protein_sequence_tokenizer_mlm")
    models_dir: Path = Path("/data/models")

    vocab_size: int = 1024
    max_lengths: Tuple[int, int] = (300, 30)

    unk_token = "<unk>"
    pad_token = "<pad>"
    mask_token = "<mask>"

    @property
    def tokenizer_id(self):
        return f"pdb_bpe_tokenizer_{self.vocab_size}_mlm"

    @property
    def hf_tokenizer_repo(self):
        return f"{self.hf_user_id}/{self.tokenizer_id}"

    @property
    def hf_sequence_repo_name(self):
        return f"{self.hf_user_id}/pdb_sequences"
