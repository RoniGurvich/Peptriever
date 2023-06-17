import csv
import json
import os.path
from pathlib import Path

from tqdm import tqdm

from peptriever.data_sources.binding.binding_pdb import get_sequence
from peptriever.data_sources.binding.data_models import BindingTrainingSample
from peptriever.data_sources.binding.train_partitioner import TrainParitioner


def get_yapp_training_samples(data_path: Path, train_partitioner: TrainParitioner):
    meta = read_yapp_meta(data_path)
    cache_path = data_path / "yapp_seqs.json"
    cache = read_json(cache_path)

    msg = "processing yapp"
    training_samples = []
    for meta_entry in tqdm(meta, total=len(meta), desc=msg):
        training_sample = get_training_sample(
            meta_entry=meta_entry,
            data_path=data_path,
            train_partitioner=train_partitioner,
            cache=cache,
        )

        if training_sample is not None:
            training_samples.append(training_sample)
            cache[
                cache_key(
                    pdb_id=training_sample["protein_pdb_name"],
                    protein_chain=training_sample["protein_pdb_chain"],
                )
            ] = training_sample["receptor"]
    write_json(cache_path, cache)
    return training_samples


def read_json(cache_path):
    cache = {}
    if os.path.isfile(cache_path):
        with open(cache_path, encoding="utf-8") as f:
            cache = json.load(f)
    return cache


def write_json(cache_path, cache):
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(cache, f)


def get_training_sample(data_path, meta_entry, train_partitioner, cache):
    pdb_id = meta_entry["PDB ID"].lower()
    peptide_chain = meta_entry["Peptide Chain ID"]
    protein_chain = meta_entry["Protein Chain ID"]
    peptide = meta_entry["Peptide sequence"]
    protein = read_protein_sequence_with_cache(data_path, pdb_id, protein_chain, cache)
    training_sample = None
    if peptide is not None and protein is not None and peptide and protein:
        train_part = train_partitioner.get_partition(
            pdb_id=pdb_id, peptide_chain=peptide_chain, protein_chain=protein_chain
        )
        training_sample = BindingTrainingSample(
            protein_pdb_name=pdb_id,
            peptide_pdb_name=pdb_id,
            peptide_pdb_chain=peptide_chain,
            protein_pdb_chain=protein_chain,
            train_part=train_part,
            peptide=peptide,
            receptor=protein,
        ).__dict__
    return training_sample


def read_protein_sequence_with_cache(data_path, pdb_id, protein_chain, cache: dict):
    protein = cache.get(cache_key(pdb_id, protein_chain))
    if protein is None:
        protein_pdb_name = str(data_path / "protein" / f"{pdb_id}.{protein_chain}.pdb")
        protein = get_sequence(protein_pdb_name)
    return protein


def cache_key(pdb_id, protein_chain):
    return f"{pdb_id}.{protein_chain}"


def read_yapp_meta(data_path):
    yapp_meta_fname = data_path / "yappcdann.csv"
    with open(yapp_meta_fname, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)
