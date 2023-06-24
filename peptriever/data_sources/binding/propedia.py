import csv
from pathlib import Path
from typing import Iterator

from tqdm import tqdm

from peptriever.data_sources.binding.data_models import BindingTrainingSample
from peptriever.data_sources.binding.train_partitioner import TrainParitioner


def get_propedia_training_samples(
    data_path: Path, train_partitioner: TrainParitioner
) -> Iterator[dict]:
    msg = "processing prppedia"
    for row in tqdm(_iterate_complexes(data_path / "complex.csv"), desc=msg):
        pdb_name = row["PDB"].lower()
        protein_chain = row["Receptor Chain"]
        peptide_chain = row["Peptide Chain"]
        pep_sequence, receptor_sequence = _strip_outer_x(row)
        peptide_too_short = len(pep_sequence) < 5
        if any(
            [
                "X" in pep_sequence,
                "X" in receptor_sequence,
                peptide_too_short,
            ]
        ):
            continue

        train_part = train_partitioner.get_partition(
            pdb_id=pdb_name, peptide_chain=peptide_chain, protein_chain=protein_chain
        )

        yield BindingTrainingSample(
            protein_pdb_name=pdb_name,
            peptide_pdb_name=pdb_name,
            peptide_pdb_chain=peptide_chain,
            protein_pdb_chain=protein_chain,
            train_part=train_part,
            peptide=pep_sequence,
            receptor=receptor_sequence,
        ).__dict__


def _iterate_complexes(fname):
    with open(fname, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        yield from reader


def _strip_outer_x(row):
    pep_sequence = row["Peptide Sequence"].strip("X")
    receptor_sequence = row["Receptor Sequence"].strip("X")
    return pep_sequence, receptor_sequence
