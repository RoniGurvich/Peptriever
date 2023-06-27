from dataclasses import dataclass
from pathlib import Path


@dataclass
class IndexingConfig:
    hf_pdb_seq_repo: str = "ronig/pdb_sequences"
    hf_tokenizer_repo: str = "ronig/pdb_bpe_tokenizer_1024_mlm"
    max_length1: int = 30
    max_length2: int = 300

    models_path: Path = Path("/data/models")
    encoded_sequence_dims: int = 128
    distance_function: str = "euclidean"

    min_indexed_entries: int = 1_000

    hf_binding_dataset_repo: str = "ronig/protein_binding_sequences"
    hf_model_repo: str = "ronig/protein_biencoder"
    hf_demo_space: str = "ronig/protein_binding_search"

    milvus_uri = "https://in03-ddab8e9a5a09fcc.api.gcp-us-west1.zillizcloud.com"
    milvus_db_name: str = "Protein"
    milvus_collection_name: str = "Peptriever"
    milvus_insert_batch_size: int = 1000
