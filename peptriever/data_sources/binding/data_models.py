from dataclasses import dataclass


@dataclass
class BindingEntry:
    protein_pdb_name: str
    peptide_pdb_name: str
    protein_pdb_chain: str
    peptide_pdb_chain: str
    train_part: str


@dataclass
class BindingTrainingSample(BindingEntry):
    peptide: str
    receptor: str
