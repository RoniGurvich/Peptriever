from dataclasses import dataclass
from pathlib import Path


@dataclass
class IndexingConfig:
    n_trees: int = 10
    pep_threshold: int = 20

    hf_pdb_seq_repo: str = "ronig/pdb_sequences"
    hf_tokenizer_repo: str = "ronig/pdb_bpe_tokenizer_1024_mlm"
    max_length1: int = 30
    max_length2: int = 300

    models_path: Path = Path("/data/models")
    encoded_sequence_dims: int = 128
    distance_function: str = "euclidean"

    min_indexed_entries: int = 1_000

    local_indexes_path: Path = Path("/data/indexes/")

    hf_binding_dataset_repo: str = "ronig/protein_binding_sequences"
    hf_model_repo: str = "ronig/protein_biencoder"
    hf_index_repo: str = "ronig/protein_index"
    hf_demo_space: str = "ronig/protein_binding_search"
