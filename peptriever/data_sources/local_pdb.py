import collections
import json
import os
from pathlib import Path

import pandas as pd


def get_pdb_to_gene(pdb_dir: Path):
    cache_path = pdb_dir / "pdb_to_gene.json"
    mapping = _maybe_get_pdb_to_gene_from_cache(cache_path)
    if mapping is not None:
        return mapping

    data = read_genes_file(pdb_dir)
    mapping = collections.defaultdict(set)
    for _, row in data.iterrows():
        pdb = row["PDB"]
        gene_id = row["GENE_ID"]
        chain = row["CHAIN"]
        mapping[(pdb, chain)].update([gene_id])

    sorted_mapping = {key: sorted(value) for key, value in mapping.items()}
    _cache_pdb_to_gene(cache_path, sorted_mapping)

    return sorted_mapping


def read_genes_file(pdb_dir: Path) -> pd.DataFrame:
    fname = pdb_dir / "pdb_chain_ensembl.csv"
    with open(fname, encoding="utf-8") as f:
        next(iter(f))
        data = pd.read_csv(f)[["PDB", "CHAIN", "GENE_ID"]].drop_duplicates()
    return data


def _maybe_get_pdb_to_gene_from_cache(cache_path):
    mapping = None
    if os.path.isfile(cache_path):
        with open(cache_path, encoding="utf-8") as f:
            serialized_mapping = json.load(f)
            mapping = {
                tuple(key.split(",")): value
                for key, value in serialized_mapping.items()
            }
    return mapping


def _cache_pdb_to_gene(cache_path, sorted_mapping):
    serialized_keys_mapping = {}
    for tuple_key, value in sorted_mapping.items():
        serialized_key = ",".join(str(k) for k in tuple_key)
        serialized_keys_mapping[serialized_key] = value
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(serialized_keys_mapping, f)
