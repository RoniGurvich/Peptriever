from dataclasses import dataclass
from pathlib import Path


@dataclass
class BindingDataConfig:
    datasets_path: Path = Path("/data/training_sets/external/")
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    hf_pdb_repo: str = "ronig/pdb_sequences"
    hf_dataset_repo: str = "ronig/protein_binding_sequences"

    @property
    def binding_path(self):
        return self.datasets_path / "binding"

    @property
    def pdb_path(self):
        return self.datasets_path / "pdb"

    @property
    def training_set_path(self):
        return self.binding_path / "binding.csv"
