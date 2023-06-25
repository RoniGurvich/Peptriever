import csv
from pathlib import Path

import pandas as pd
from Bio import SeqIO
from tqdm import tqdm

from peptriever.data_sources.pdb_sequences.config import PDBDataConfig
from peptriever.data_sources.protein_transformations import protein_train_part


def extract_pdb_sequences(config: PDBDataConfig):
    uniprot_lookup = get_uniprot_metadata(config.uniprot_meta_path)
    with open(config.sequence_dataset_path, "w", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=(
                "name",
                "chain",
                "genes",
                "organism",
                "sequence",
                "train_part",
                "uniprot_id"
            ),
        )
        writer.writeheader()

        for rec in tqdm(iterate_fasta(config.fasta_dataset_path)):
            record = process_fasta_entry(
                record=rec,
                uniprot_lookup=uniprot_lookup,
                val_ratio=config.val_ratio,
                test_ratio=config.test_ratio,
            )
            writer.writerow(record)


def get_uniprot_metadata(uniprot_meta_path: Path):
    metadata = pd.read_csv(str(uniprot_meta_path), delimiter="\t")[
        ["From", "Protein names", "Gene Names", "Organism", "Entry"]
    ]
    meta_lookup = {}
    for row in metadata.to_dict("records"):
        meta_lookup[row.pop("From")] = row
    return meta_lookup


def process_fasta_entry(record, uniprot_lookup, val_ratio, test_ratio):
    record = record.copy()
    pdb_name = record["name"]
    meta = uniprot_lookup.get(pdb_name)
    if meta is None:
        genes = ""
        organism = ""
        uniprot_id = ""
    else:
        genes = meta["Gene Names"]
        organism = meta["Organism"]
        uniprot_id = meta["Entry"]
        if not isinstance(genes, str):
            genes = ""
    train_part = protein_train_part(
        protein_name=pdb_name,
        test_ratio=test_ratio,
        val_ratio=val_ratio,
    )
    record["train_part"] = train_part
    record["genes"] = genes
    record["organism"] = organism.title()
    record["uniprot_id"] = uniprot_id
    return record


def iterate_fasta(fasta_path):
    for seq_record in SeqIO.parse(fasta_path, "fasta"):
        seq = str(seq_record.seq)
        name, chain = seq_record.id.split("_")
        yield {
            "name": name.lower(),
            "chain": chain,
            "sequence": seq,
        }


if __name__ == "__main__":
    extract_pdb_sequences(config=PDBDataConfig())
