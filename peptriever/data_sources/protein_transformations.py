import hashlib
from typing import List

from Bio.Data.IUPACData import protein_letters_3to1
from Bio.PDB import standard_aa_names
from Bio.PDB.Residue import Residue


def protein_train_part(protein_name, test_ratio, val_ratio):
    protein_seed = seed_string(protein_name.lower())
    if protein_seed > 1 - test_ratio:
        train_part = "test"
    elif protein_seed > 1 - test_ratio - val_ratio:
        train_part = "val"
    else:
        train_part = "train"
    return train_part


def seed_string(str_: str, max_value: int = 10**8) -> float:
    return (
        int(hashlib.sha256(str_.encode("utf-8")).hexdigest(), 16)
        % max_value
        / max_value
    )


def convert_seq(res_names: List[str]):
    return "".join([protein_letters_3to1[res.title()] for res in res_names])


def residues_to_aa_codes(residues: List[Residue]):
    return [
        residue.get_resname()
        for residue in residues
        if residue is not None and residue.get_resname() in standard_aa_names
    ]
