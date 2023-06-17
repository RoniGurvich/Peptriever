import functools
from pathlib import Path
from typing import Dict, Iterable, Iterator, Optional

import pandas as pd
from tqdm import tqdm

from peptriever.concurrency import execute_multiprocess
from peptriever.data_sources.binding.binding_pdb import BindingPDB
from peptriever.data_sources.binding.data_models import (
    BindingEntry,
    BindingTrainingSample,
)
from peptriever.data_sources.binding.train_partitioner import TrainParitioner


def get_huang_training_samples(data_path, train_partitioner: TrainParitioner):
    protein_names = list(read_huang_metadata(data_path))
    binding_entries = train_part_split(
        protein_names=protein_names, train_partitioner=train_partitioner
    )
    pdb = BindingPDB(data_path=data_path)
    msg = "processing huang"
    training_samples = []
    for training_sample in execute_multiprocess(
        func=functools.partial(read_training_sample, pdb=pdb),
        inputs=(
            {"binding_entry": entry}
            for entry in tqdm(binding_entries, total=len(protein_names), desc=msg)
        ),
    ):
        if training_sample is not None:
            training_samples.append(training_sample)
    return training_samples


def read_huang_metadata(data_path: Path) -> Iterator[Dict[str, str]]:
    fname = data_path / "peptidelist.txt"
    df = pd.read_table(
        fname, sep=" ", skipinitialspace=True, names=[str(n) for n in range(11)]
    )
    for _, row in df.iterrows():
        yield {
            "pdb_id": row["0"],
            "peptide_chain": row["1"],
            "protein_chain": row["4"],
        }


def train_part_split(
    protein_names: Iterable[Dict[str, str]], train_partitioner: TrainParitioner
):
    for pdb_metadata in protein_names:
        pdb_id = pdb_metadata["pdb_id"]
        peptide_chain = pdb_metadata["peptide_chain"]
        protein_chain = pdb_metadata["protein_chain"]
        train_part = train_partitioner.get_partition(
            pdb_id=pdb_id, peptide_chain=peptide_chain, protein_chain=protein_chain
        )
        yield BindingEntry(
            protein_pdb_name=pdb_id,
            peptide_pdb_name=pdb_id,
            peptide_pdb_chain=peptide_chain,
            protein_pdb_chain=protein_chain,
            train_part=train_part,
        )


def read_training_sample(
    binding_entry: BindingEntry, pdb: BindingPDB
) -> Optional[dict]:
    pdb_id = "_".join([binding_entry.protein_pdb_name, binding_entry.peptide_pdb_chain])
    sequences = pdb.get_sequences(pdb_id)
    peptide = sequences["peptide"]
    receptor = sequences["rec"]
    sample = None
    if peptide is not None and receptor is not None and peptide and receptor:
        sample = BindingTrainingSample(
            peptide=peptide, receptor=receptor, **binding_entry.__dict__
        ).__dict__
    return sample
