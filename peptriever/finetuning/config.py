from dataclasses import dataclass
from pathlib import Path


@dataclass
class FinetuningConfig:
    datasets_path: Path = Path("/data/training_sets/external/")
    hf_pdb_seq_repo: str = "ronig/pdb_sequences"
    hf_tokenizer_repo: str = "ronig/pdb_bpe_tokenizer_1024_mlm"
    mask_token = "<mask>"
    hf_binding_data_repo_id: str = "ronig/protein_binding_sequences"
    distance_function: str = "euclidean"
    tokenizer1_max_length = 30
    tokenizer2_max_length = 300

    encoded_sequence_dims: int = 128
    models_path: Path = Path("/data/models")

    @property
    def data_path(self):
        return self.datasets_path / "binding"

    @property
    def pdb_path(self):
        return self.datasets_path / "pdb"

    @property
    def pos_benchmark_name(self):
        return self.data_path / "pos_benchmark.csv"

    @property
    def neg_benchmark_name(self):
        return self.data_path / "neg_benchmark.csv"
