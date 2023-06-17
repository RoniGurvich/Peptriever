from dataclasses import dataclass
from pathlib import Path


@dataclass
class PDBDataConfig:
    dataset_path: Path = Path("/data/training_sets/external/pdb")
    fasta_fname: str = "pdb_seqres.txt"
    sequences_fname: str = "sequences.csv"
    uniprot_fname: str = "uniprot_metadata.tsv"
    test_ratio: float = 0.1
    val_ratio: float = 0.1
    hf_sequence_repo_name: str = "ronig/pdb_sequences"

    @property
    def uniprot_meta_path(self):
        return self.dataset_path / self.uniprot_fname

    @property
    def sequence_dataset_path(self):
        return self.dataset_path / self.sequences_fname

    @property
    def fasta_dataset_path(self):
        return self.dataset_path / self.fasta_fname
