import os
from pathlib import Path

from Bio.PDB import PDBParser
from Bio.PDB.PDBExceptions import PDBConstructionException

from peptriever.data_sources.protein_transformations import (
    convert_seq,
    residues_to_aa_codes,
)


class BindingPDB:
    def __init__(self, data_path: Path, sep="|"):
        self.data_path = data_path
        self.sep = sep

    @property
    def models_dir(self):
        return self.data_path / "pepbdb"

    def get_sequences(self, protein_name: str):
        pep_seq = get_sequence(self.models_dir / protein_name / "peptide.pdb")
        rec_seq = get_sequence(self.models_dir / protein_name / "receptor.pdb")
        return {"peptide": pep_seq, "rec": rec_seq}


def get_sequence(fname: str):
    struct = get_struct(fname)
    if struct is None:
        return None
    aa = residues_to_aa_codes(struct.get_residues())
    seq = convert_seq(aa)
    return seq


def get_struct(fname: str):
    structure = None
    protein_name = ""
    if os.path.isfile(fname):
        parser = PDBParser(PERMISSIVE=True, QUIET=True)
        try:
            structure = parser.get_structure(protein_name, fname)
        except (PDBConstructionException, KeyError, AttributeError, ValueError):
            pass
    return structure
