import csv
from pathlib import Path
from typing import Set, Tuple


def read_excluded_pdbs(data_path: Path) -> Set[Tuple[str, str, str, str]]:
    pos_benchmark_fname = data_path / "pos_benchmark.csv"
    neg_benchmark_fname = data_path / "neg_benchmark.csv"

    excluded_chains = set()

    with open(pos_benchmark_fname, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            excluded_chains.update(
                [(row["pdb_id"], row["prot_chain"], row["pdb_id"], row["pep_chain"])]
            )
    with open(neg_benchmark_fname, encoding="utf=8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            excluded_chains.update(
                [
                    (
                        row["pdb_id_prot"],
                        row["prot_chain"],
                        row["pdb_id_pep"],
                        row["pep_chain"],
                    ),
                ]
            )

    return excluded_chains
